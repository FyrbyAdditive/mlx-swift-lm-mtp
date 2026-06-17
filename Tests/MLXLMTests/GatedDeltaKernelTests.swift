import Foundation
import MLX
import XCTest

@testable import MLXLLM

/// Parity tests for the fused gated-delta Metal kernel vs the ops-based sequential reference.
/// The kernel runs the whole recurrence on-GPU (one dispatch); the ops loop dispatches per
/// timestep. They MUST be numerically equivalent — this is what makes it safe to use the kernel
/// for fast prefill (the fix for ~60 tok/s sequential-scan prefill on the hybrid SSM backbone).
/// The MLXVLM Qwen35 (the 27B) now uses the same kernel; this guards both.
final class GatedDeltaKernelTests: XCTestCase {

    /// Build random inputs with the given shapes. Hk may be < Hv (grouped heads, repeat factor).
    private func makeInputs(B: Int, T: Int, Hk: Int, Hv: Int, Dk: Int, Dv: Int)
        -> (q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray)
    {
        MLXRandom.seed(1234)
        let q = MLXRandom.normal([B, T, Hk, Dk]) * 0.1
        let k = MLXRandom.normal([B, T, Hk, Dk]) * 0.1
        let v = MLXRandom.normal([B, T, Hv, Dv]) * 0.1
        // g is a per-(B,T,Hv) decay in (0,1); beta a per-(B,T,Hv) gate in (0,1).
        let g = sigmoid(MLXRandom.normal([B, T, Hv]))
        let beta = sigmoid(MLXRandom.normal([B, T, Hv]))
        let state = MLXRandom.normal([B, Hv, Dv, Dk]) * 0.1
        eval(q, k, v, g, beta, state)
        return (q, k, v, g, beta, state)
    }

    private func maxDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        (a.asType(.float32) - b.asType(.float32)).abs().max().item(Float.self)
    }

    func testKernelMatchesOps_singleHead() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "kernel needs GPU")
        let (q, k, v, g, beta, state) = makeInputs(B: 1, T: 16, Hk: 2, Hv: 2, Dk: 32, Dv: 32)
        let (yK, sK) = gatedDeltaKernel(q: q, k: k, v: v, g: g, beta: beta, state: state)
        let (yO, sO) = gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state)
        eval(yK, sK, yO, sO)
        XCTAssertEqual(yK.shape, yO.shape)
        XCTAssertLessThan(maxDiff(yK, yO), 2e-3, "kernel/ops output diverged")
        XCTAssertLessThan(maxDiff(sK, sO), 2e-3, "kernel/ops state diverged")
    }

    func testKernelMatchesOps_groupedHeads() throws {
        // Hv > Hk: the kernel maps hv->hk internally; the ops path repeats q/k. Must still match.
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "kernel needs GPU")
        let (q, k, v, g, beta, state) = makeInputs(B: 2, T: 24, Hk: 2, Hv: 8, Dk: 64, Dv: 64)
        let (yK, sK) = gatedDeltaKernel(q: q, k: k, v: v, g: g, beta: beta, state: state)
        let (yO, sO) = gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state)
        eval(yK, sK, yO, sO)
        XCTAssertLessThan(maxDiff(yK, yO), 3e-3)
        XCTAssertLessThan(maxDiff(sK, sO), 3e-3)
    }

    func testKernelMatchesOps_singleTokenDecode() throws {
        // T=1 is the decode path; kernel and ops must agree there too.
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "kernel needs GPU")
        let (q, k, v, g, beta, state) = makeInputs(B: 1, T: 1, Hk: 4, Hv: 4, Dk: 128, Dv: 128)
        let (yK, sK) = gatedDeltaKernel(q: q, k: k, v: v, g: g, beta: beta, state: state)
        let (yO, sO) = gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state)
        eval(yK, sK, yO, sO)
        XCTAssertLessThan(maxDiff(yK, yO), 2e-3)
        XCTAssertLessThan(maxDiff(sK, sO), 2e-3)
    }

    /// Splitting a sequence and running it as two consecutive chunks (threading state through) must
    /// equal running the whole sequence at once — the property the MTP nConfirmed verify split and
    /// prefix-cache restore both rely on.
    func testChunkedEqualsWholeSequence() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "kernel needs GPU")
        let (q, k, v, g, beta, state) = makeInputs(B: 1, T: 20, Hk: 2, Hv: 4, Dk: 64, Dv: 64)
        let (yFull, sFull) = gatedDeltaKernel(q: q, k: k, v: v, g: g, beta: beta, state: state)

        let split = 7
        let (y1, s1) = gatedDeltaKernel(
            q: q[0..., 0 ..< split], k: k[0..., 0 ..< split], v: v[0..., 0 ..< split],
            g: g[0..., 0 ..< split], beta: beta[0..., 0 ..< split], state: state)
        let (y2, s2) = gatedDeltaKernel(
            q: q[0..., split...], k: k[0..., split...], v: v[0..., split...],
            g: g[0..., split...], beta: beta[0..., split...], state: s1)
        let yChunked = concatenated([y1, y2], axis: 1)
        eval(yFull, sFull, yChunked, s2)
        XCTAssertLessThan(maxDiff(yFull, yChunked), 3e-3, "chunked != whole sequence")
        XCTAssertLessThan(maxDiff(sFull, s2), 3e-3, "chunked state != whole-sequence state")
    }
}
