import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// The losslessness-critical math, tested without any model.
///
/// `testOutputDistributionMatchesTarget` is the M3 statistical gate: speculative sampling
/// (draft from q, accept w.p. min(1, p/q), residual-resample on reject) must produce an
/// EXACT sample from p — verified by chi-square against several (p, q) pairs including
/// adversarial ones.
final class SpeculativeVerifierTests: XCTestCase {

    // MARK: - greedyAcceptCount

    func testGreedyAcceptCount() {
        XCTAssertEqual(SpeculativeVerifier.greedyAcceptCount(draft: [1, 2, 3], targetArgmax: [1, 2, 3, 9]), 3)
        XCTAssertEqual(SpeculativeVerifier.greedyAcceptCount(draft: [1, 2, 3], targetArgmax: [1, 9, 3, 4]), 1)
        XCTAssertEqual(SpeculativeVerifier.greedyAcceptCount(draft: [7], targetArgmax: [8, 8]), 0)
        XCTAssertEqual(SpeculativeVerifier.greedyAcceptCount(draft: [], targetArgmax: [1]), 0)
    }

    // MARK: - confidenceKeepCount

    func testConfidenceKeepCount() {
        // survival [0.9, 0.8, 0.5]: cumulative 0.9, 0.72, 0.36
        XCTAssertEqual(SpeculativeVerifier.confidenceKeepCount(survival: [0.9, 0.8, 0.5], threshold: 0.5), 2)
        XCTAssertEqual(SpeculativeVerifier.confidenceKeepCount(survival: [0.9, 0.8, 0.5], threshold: 0.95), 0)
        XCTAssertEqual(SpeculativeVerifier.confidenceKeepCount(survival: [0.9, 0.8, 0.5], threshold: 0.1), 3)
        XCTAssertEqual(SpeculativeVerifier.confidenceKeepCount(survival: [], threshold: 0.5), 0)
    }

    // MARK: - truncateProbs

    func testTruncateProbsIdentityFastPath() {
        let p = MLXArray([0.5, 0.3, 0.2] as [Float])
        let out = SpeculativeVerifier.truncateProbs(p, topP: 1.0, topK: 0)
        XCTAssertEqual(abs(out - p).max().item(Float.self), 0)
    }

    func testTruncateProbsTopK() {
        let p = MLXArray([0.4, 0.3, 0.2, 0.1] as [Float])
        let out = SpeculativeVerifier.truncateProbs(p, topP: 1.0, topK: 2).asArray(Float.self)
        // Keep {0.4, 0.3}, renormalized to {4/7, 3/7}.
        XCTAssertEqual(out[0], 4.0 / 7.0, accuracy: 1e-5)
        XCTAssertEqual(out[1], 3.0 / 7.0, accuracy: 1e-5)
        XCTAssertEqual(out[2], 0, accuracy: 1e-6)
        XCTAssertEqual(out[3], 0, accuracy: 1e-6)
    }

    func testTruncateProbsTopP() {
        let p = MLXArray([0.5, 0.3, 0.15, 0.05] as [Float])
        // topP 0.7: cumulative-before values 0, 0.5, 0.8 → nucleus {0.5, 0.3}.
        let out = SpeculativeVerifier.truncateProbs(p, topP: 0.7, topK: 0).asArray(Float.self)
        XCTAssertEqual(out[0], 0.5 / 0.8, accuracy: 1e-5)
        XCTAssertEqual(out[1], 0.3 / 0.8, accuracy: 1e-5)
        XCTAssertEqual(out[2], 0, accuracy: 1e-6)
        // Always keeps at least top-1 even with tiny topP.
        let top1 = SpeculativeVerifier.truncateProbs(p, topP: 0.01, topK: 0).asArray(Float.self)
        XCTAssertEqual(top1[0], 1.0, accuracy: 1e-5)
    }

    // MARK: - sampledAccept (hand-computed cases; injected sampler captures its input)

    /// p/q over vocab 4 at 3 positions + bonus row. Draft [0, 1, 2].
    private func makePQ() -> (p: MLXArray, q: MLXArray) {
        let p = MLXArray(
            [
                0.7, 0.1, 0.1, 0.1,  // pos 0: p(draft=0)=0.7
                0.1, 0.2, 0.6, 0.1,  // pos 1: p(draft=1)=0.2
                0.1, 0.1, 0.7, 0.1,  // pos 2: p(draft=2)=0.7
                0.4, 0.3, 0.2, 0.1,  // bonus row
            ] as [Float], [4, 4])
        let q = MLXArray(
            [
                0.7, 0.1, 0.1, 0.1,  // pos 0: q == p → always accept
                0.1, 0.8, 0.05, 0.05,  // pos 1: q(1)=0.8 ≫ p(1)=0.2 → accept ratio 0.25
                0.1, 0.1, 0.7, 0.1,
            ] as [Float], [3, 4])
        return (p, q)
    }

    func testSampledAcceptAllAcceptedSamplesBonus() {
        let (p, q) = makePQ()
        var sampledFrom: [Float] = []
        let (n, repl) = SpeculativeVerifier.sampledAccept(
            targetProbs: p, draftTokens: [0, 1, 2], draftProbs: q,
            uniforms: MLXArray([0.5, 0.2, 0.5] as [Float]),  // 0.2 < 0.25 → pos 1 accepts
            sample: { dist in
                sampledFrom = dist.asArray(Float.self)
                return 3
            })
        XCTAssertEqual(n, 3)
        XCTAssertEqual(repl, 3)
        // Bonus sampled from the target's row L (=3).
        XCTAssertEqual(sampledFrom, [0.4, 0.3, 0.2, 0.1])
    }

    func testSampledAcceptRejectionResamplesResidual() {
        let (p, q) = makePQ()
        var sampledFrom: [Float] = []
        let (n, repl) = SpeculativeVerifier.sampledAccept(
            targetProbs: p, draftTokens: [0, 1, 2], draftProbs: q,
            uniforms: MLXArray([0.5, 0.9, 0.0] as [Float]),  // 0.9 ≥ 0.25 → pos 1 rejects
            sample: { dist in
                sampledFrom = dist.asArray(Float.self)
                return 2
            })
        XCTAssertEqual(n, 1)
        XCTAssertEqual(repl, 2)
        // Residual at pos 1: max(p−q, 0) = [0, 0, 0.55, 0.05] → normalized [0, 0, 11/12, 1/12].
        XCTAssertEqual(sampledFrom[0], 0, accuracy: 1e-6)
        XCTAssertEqual(sampledFrom[1], 0, accuracy: 1e-6)
        XCTAssertEqual(sampledFrom[2], 0.55 / 0.6, accuracy: 1e-5)
        XCTAssertEqual(sampledFrom[3], 0.05 / 0.6, accuracy: 1e-5)
    }

    func testSampledAcceptRatioAboveOneAlwaysAccepts() {
        let (p, q) = makePQ()
        // Position 0: p == q → ratio 1; uniform 0.999… < 1 accepts. Rejection can only
        // happen where q overshoots p.
        let (n, _) = SpeculativeVerifier.sampledAccept(
            targetProbs: p, draftTokens: [0, 1, 2], draftProbs: q,
            uniforms: MLXArray([0.9999, 0.2, 0.9999] as [Float]),
            sample: { _ in 0 })
        XCTAssertEqual(n, 3)
    }

    // MARK: - M3 statistical gate: the committed token is an exact sample from p

    /// One speculative round at a single position: draft x ~ q, accept w.p. min(1, p/q),
    /// residual-resample on reject. The committed token's distribution must equal p.
    private func chiSquare(p: [Float], q: [Float], draws: Int) -> Double {
        let vocab = p.count
        let pArr = MLXArray(p, [1, vocab])  // row 0 = verify position (bonus row unused)
        let pFull = concatenated([pArr, pArr], axis: 0)  // sampledAccept expects L+1 rows
        let qArr = MLXArray(q, [1, vocab])

        // Vectorized draft draws + accept uniforms for speed; the accept/residual path runs
        // through the REAL sampledAccept per draw.
        MLXRandom.seed(1234)
        let draftDraws = MLXRandom.categorical(log(qArr + 1e-20), count: draws)
            .reshaped([draws]).asArray(Int32.self)
        let uniforms = MLXRandom.uniform(low: Float(0), high: Float(1), [draws])
            .asArray(Float.self)

        var counts = [Int](repeating: 0, count: vocab)
        for i in 0 ..< draws {
            let (n, repl) = SpeculativeVerifier.sampledAccept(
                targetProbs: pFull, draftTokens: [draftDraws[i]], draftProbs: qArr,
                uniforms: MLXArray([uniforms[i]]))
            let committed = n == 1 ? draftDraws[i] : repl
            counts[Int(committed)] += 1
        }
        var chi2 = 0.0
        for t in 0 ..< vocab {
            let expected = Double(p[t]) * Double(draws)
            guard expected > 0 else { continue }
            let d = Double(counts[t]) - expected
            chi2 += d * d / expected
        }
        return chi2
    }

    func testOutputDistributionMatchesTarget() {
        // vocab 8, df=7 → chi2 critical value at p=0.01 is 18.475.
        let critical = 18.475
        let draws = 4000
        let pairs: [(p: [Float], q: [Float])] = [
            // q close to p (the trained-drafter regime)
            ([0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05],
             [0.35, 0.25, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]),
            // adversarial: q piles mass on a token p dislikes
            ([0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05],
             [0.02, 0.02, 0.8, 0.04, 0.03, 0.03, 0.03, 0.03]),
            // near-uniform p, spiky q
            ([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
             [0.9, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01]),
        ]
        for (i, pair) in pairs.enumerated() {
            let chi2 = chiSquare(p: pair.p, q: pair.q, draws: draws)
            XCTAssertLessThan(
                chi2, critical,
                "pair \(i): committed-token distribution diverges from target p (chi2=\(chi2))")
        }
    }
}
