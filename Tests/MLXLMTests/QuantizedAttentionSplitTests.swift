import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// Parity for the small-M causal row split in `quantizedScaledDotProductAttention`.
///
/// The split runs each of the M query rows as an independent single-row attention over
/// its causal key prefix (the fast qmv path). It must produce the same result as
/// (a) the unsplit multi-row quantized path (kill switch MLXZ_QKV_ROWSPLIT=0 — compared
/// here by invoking the same math via a full-length boolean mask equivalent), and
/// (b) an fp16 reference computed on the dequantized K/V with an offset causal mask.
final class QuantizedAttentionSplitTests: XCTestCase {

    /// fp16 reference: dequantize K/V and run standard SDPA with an offset causal mask.
    private func reference(
        queries: MLXArray,
        keys: MLXArray, values: MLXArray,
        scale: Float, groupSize: Int, bits: Int
    ) -> MLXArray {
        let (qL, kL) = (queries.dim(2), keys.dim(2))
        // Offset causal: query row i attends keys [0, kL - qL + i].
        let qIdx = MLXArray(0 ..< qL) + MLXArray(kL - qL)
        let kIdx = MLXArray(0 ..< kL)
        let allowed = greaterEqual(
            expandedDimensions(qIdx, axis: -1), expandedDimensions(kIdx, axis: -2))
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale,
            mask: .array(allowed))
    }

    func testRowSplitMatchesReferenceAndUnsplit() {
        MLXRandom.seed(3)
        let (B, nQHeads, nKVHeads, D) = (1, 8, 2, 64)
        let groupSize = 64
        let bits = 4

        for (ctx, m) in [(96, 2), (96, 4), (200, 8)] {
            let kL = ctx + m
            let keys = MLXRandom.normal([B, nKVHeads, kL, D])
            let values = MLXRandom.normal([B, nKVHeads, kL, D])
            let queries = MLXRandom.normal([B, nQHeads, m, D])
            let scale = Float(pow(Double(D), -0.5))

            let qk = quantized(keys, groupSize: groupSize, bits: bits)
            let qv = quantized(values, groupSize: groupSize, bits: bits)
            let dqKeys = dequantized(
                qk.wq, scales: qk.scales, biases: qk.biases, groupSize: groupSize, bits: bits)
            let dqValues = dequantized(
                qv.wq, scales: qv.scales, biases: qv.biases, groupSize: groupSize, bits: bits)

            // Split path (M in 2...8, causal → row split active by default).
            let split = quantizedScaledDotProductAttention(
                queries: queries,
                quantizedKeys: (qk.wq, qk.scales, qk.biases),
                quantizedValues: (qv.wq, qv.scales, qv.biases),
                scale: scale, mask: .causal, groupSize: groupSize, bits: bits)

            // Unsplit path: identical math via the general branch (boolean array mask
            // equal to the offset causal mask — .array masks never take the split).
            let qIdx = MLXArray(0 ..< m) + MLXArray(kL - m)
            let kIdx = MLXArray(0 ..< kL)
            let allowed = greaterEqual(
                expandedDimensions(qIdx, axis: -1), expandedDimensions(kIdx, axis: -2))
            let unsplit = quantizedScaledDotProductAttention(
                queries: queries,
                quantizedKeys: (qk.wq, qk.scales, qk.biases),
                quantizedValues: (qv.wq, qv.scales, qv.biases),
                scale: scale, mask: .array(allowed), groupSize: groupSize, bits: bits)

            let ref = reference(
                queries: queries, keys: dqKeys, values: dqValues,
                scale: scale, groupSize: groupSize, bits: bits)

            eval(split, unsplit, ref)
            XCTAssertEqual(split.shape, [B, nQHeads, m, D], "ctx=\(ctx) m=\(m)")
            let vsUnsplit = abs(split - unsplit).max().item(Float.self)
            let vsRef = abs(split - ref).max().item(Float.self)
            // Same quantized inputs, same math, different kernel path → tight tolerance.
            XCTAssertLessThan(vsUnsplit, 2e-3, "split vs unsplit (ctx=\(ctx) m=\(m))")
            // Quantized matmul vs dequantize-then-matmul: fp accumulation differences only.
            XCTAssertLessThan(vsRef, 2e-2, "split vs fp16 reference (ctx=\(ctx) m=\(m))")
        }
    }

    /// M=1 and M>8 must not take the split (shape/behavior unchanged).
    func testSplitBoundaries() {
        MLXRandom.seed(4)
        let (B, heads, D) = (1, 4, 64)
        for m in [1, 9] {
            let kL = 64 + m
            let keys = MLXRandom.normal([B, heads, kL, D])
            let values = MLXRandom.normal([B, heads, kL, D])
            let queries = MLXRandom.normal([B, heads, m, D])
            let qk = quantized(keys, groupSize: 64, bits: 4)
            let qv = quantized(values, groupSize: 64, bits: 4)
            let out = quantizedScaledDotProductAttention(
                queries: queries,
                quantizedKeys: (qk.wq, qk.scales, qk.biases),
                quantizedValues: (qv.wq, qv.scales, qv.biases),
                scale: 0.125, mask: .causal, groupSize: 64, bits: 4)
            eval(out)
            XCTAssertEqual(out.shape, [B, heads, m, D])
        }
    }
}
