import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Parity tests for BatchKVCache: a left-padded multi-sequence batch must produce, for each row,
/// attention output identical to running that sequence ALONE through KVCacheSimple. This is the
/// make-or-break correctness gate for batched/continuous decoding — if the mask or per-row offsets
/// are wrong, padded positions leak into attention and rows diverge.
@Suite struct BatchKVCacheTests {

    /// Random q/k/v with shape (B,H,S,D).
    private func qkv(B: Int, H: Int, S: Int, D: Int, seed: UInt64) -> (MLXArray, MLXArray, MLXArray) {
        MLXRandom.seed(seed)
        let q = MLXRandom.normal([B, H, S, D]) * 0.5
        let k = MLXRandom.normal([B, H, S, D]) * 0.5
        let v = MLXRandom.normal([B, H, S, D]) * 0.5
        eval(q, k, v)
        return (q, k, v)
    }

    /// The prefill causal mask for a left-padded batch must, for each row, allow exactly the
    /// causal+real positions and block the padded front. We verify the boolean mask equals a
    /// hand-built per-row reference (this is where padding bugs would show up).
    @Test func leftPaddedPrefillMaskIsCorrect() {
        // lengths 3 and 5 → maxLen 5, left padding [2, 0].
        let lengths = [3, 5]
        let maxLen = lengths.max()!
        let leftPad = lengths.map { maxLen - $0 }

        let batch = BatchKVCache(leftPadding: leftPad)
        let mask = createCausalMask(n: maxLen, offset: 0, leftPadding: batch.leftPadding)
        // shape (B, 1, maxLen, maxLen)
        #expect(mask.dim(0) == 2)
        #expect(mask.dim(2) == maxLen && mask.dim(3) == maxLen)
        let m = mask.asArray(Bool.self)  // flat [B*1*maxLen*maxLen]

        func at(_ b: Int, _ i: Int, _ j: Int) -> Bool { m[((b * maxLen) + i) * maxLen + j] }
        for (b, len) in lengths.enumerated() {
            let pad = maxLen - len
            for i in 0 ..< maxLen {
                for j in 0 ..< maxLen {
                    // Correct mask: query i attends key j iff j <= i (causal) AND j >= pad (real).
                    let expected = (j <= i) && (j >= pad)
                    #expect(at(b, i, j) == expected, "mask[\(b)][\(i)][\(j)] wrong (len=\(len))")
                }
            }
        }
    }

    /// After prefilling a left-padded batch, the cached keys/values for each row's REAL positions
    /// must equal the original tokens (no corruption from the shared buffer / padding).
    @Test func cachedRowsMatchInput() {
        let H = 2, D = 16
        let lengths = [3, 5]
        let maxLen = lengths.max()!
        let leftPad = lengths.map { maxLen - $0 }
        let (_, kFull, vFull) = qkv(B: 2, H: H, S: maxLen, D: D, seed: 7)

        let batch = BatchKVCache(leftPadding: leftPad)
        let (ck, cv) = batch.update(keys: kFull, values: vFull)
        eval(ck, cv)
        #expect(ck.shape == [2, H, maxLen, D])
        // Real region of each row round-trips unchanged.
        for (row, len) in lengths.enumerated() {
            let start = maxLen - len
            let inK = kFull[row ..< (row + 1), 0..., start..., 0...]
            let outK = ck[row ..< (row + 1), 0..., start..., 0...]
            #expect(allClose(inK, outK, atol: 1e-5).item(Bool.self), "row \(row) keys corrupted")
            let inV = vFull[row ..< (row + 1), 0..., start..., 0...]
            let outV = cv[row ..< (row + 1), 0..., start..., 0...]
            #expect(allClose(inV, outV, atol: 1e-5).item(Bool.self), "row \(row) values corrupted")
        }
    }

    /// Decode-step mask after prefill: a single new query (n=1) attends all real positions of its
    /// row plus the new token, and nothing padded.
    @Test func decodeStepMaskIgnoresPadding() {
        let lengths = [2, 4]
        let maxLen = lengths.max()!
        let leftPad = lengths.map { maxLen - $0 }
        let batch = BatchKVCache(leftPadding: leftPad)
        let (_, k, v) = qkv(B: 2, H: 1, S: maxLen, D: 8, seed: 11)
        _ = batch.update(keys: k, values: v)  // bufferIndex = maxLen
        // One decode step: n=1, offset=bufferIndex.
        let mask = batch.makeMask(n: 1, windowSize: nil, returnArray: true)
        guard case let .array(arr) = mask else { #expect(Bool(false), "expected array mask"); return }
        // shape (B,1,1,bufferIndex+1)
        #expect(arr.dim(0) == 2)
        let cols = arr.dim(3)
        #expect(cols == maxLen + 1)
        let flat = arr.asArray(Bool.self)
        for (b, len) in lengths.enumerated() {
            let pad = maxLen - len
            for j in 0 ..< cols {
                let expected = j >= pad  // real positions + the new token (j up to maxLen)
                #expect(flat[b * cols + j] == expected, "decode mask[\(b)][\(j)] wrong")
            }
        }
    }

    @Test func perSequenceOffsetStartsAtNegativeLeftPadding() {
        let c = BatchKVCache(leftPadding: [2, 0, 5])
        // mlx-lm: offset = -left_padding.
        #expect(c.seqOffset.asArray(Int32.self) == [-2, 0, -5])
        #expect(c.leftPadding.asArray(Int32.self) == [2, 0, 5])
    }

    @Test func updateAdvancesPerSequenceOffset() {
        let c = BatchKVCache(leftPadding: [1, 0])
        let (_, k, v) = qkv(B: 2, H: 1, S: 4, D: 8, seed: 3)
        _ = c.update(keys: k, values: v)
        // After feeding 4 padded positions, each row's offset advanced by 4 from its start.
        #expect(c.seqOffset.asArray(Int32.self) == [-1 + 4, 0 + 4])  // [3, 4]
        #expect(c.offset == 4)  // shared buffer index
    }

    @Test func filterDropsRowsAndShrinksPadding() {
        // Two rows, left padding [3, 0]. Drop row 0 → remaining row had 0 padding, but the shared
        // buffer still carries the dropped row's leading region; filter shrinks it.
        let c = BatchKVCache(leftPadding: [3, 0])
        let (_, k, v) = qkv(B: 2, H: 1, S: 6, D: 8, seed: 5)
        _ = c.update(keys: k, values: v)
        c.filter(batchIndices: MLXArray([Int32(1)]))  // keep only row 1
        #expect(c.seqOffset.asArray(Int32.self) == [6])  // row1 offset unchanged (0 + 6)
        #expect(c.leftPadding.asArray(Int32.self) == [0])
        // state keys should now have batch dim 1.
        let s = c.state
        #expect(s[0].dim(0) == 1)
    }

    @Test func extendAppendsRows() {
        let a = BatchKVCache(leftPadding: [0])
        let b = BatchKVCache(leftPadding: [0])
        let (_, ka, va) = qkv(B: 1, H: 1, S: 4, D: 8, seed: 1)
        let (_, kb, vb) = qkv(B: 1, H: 1, S: 4, D: 8, seed: 2)
        _ = a.update(keys: ka, values: va)
        _ = b.update(keys: kb, values: vb)
        a.extend(b)
        #expect(a.state[0].dim(0) == 2)  // two rows now
        #expect(a.seqOffset.dim(0) == 2)
    }
}
