// Speculative-decoding acceptance math (Leviathan/Chen 2023), model-free and unit-testable.
//
// Greedy (temperature 0): accept the longest prefix where the draft matches the target's
// argmax exactly — output is target-greedy by construction.
//
// Sampling (temperature > 0): token i is accepted with probability min(1, p(x)/q(x));
// the first rejection resamples from the residual norm(max(0, p−q)) and discards the rest
// of the block; if all L tokens are accepted, a bonus token is sampled from the target's
// distribution at position L. This preserves the target's sampling distribution EXACTLY.
//
// Losslessness-critical invariant: p and q must have undergone IDENTICAL temperature and
// top-p/top-k truncation (`truncateProbs` used on both sides with the same parameters).

import Foundation
import MLX

public enum SpeculativeVerifier {

    /// Greedy: length of the accepted draft prefix (positionwise exact match).
    public static func greedyAcceptCount(draft: [Int32], targetArgmax: [Int32]) -> Int {
        var n = 0
        while n < draft.count && n < targetArgmax.count && draft[n] == targetArgmax[n] {
            n += 1
        }
        return n
    }

    /// Confidence trim: the longest draft prefix whose cumulative survival probability
    /// stays ≥ `threshold` (paper Eq. 7–8: prefix survival aⱼ = ∏ᵢ≤ⱼ cᵢ).
    public static func confidenceKeepCount(survival: [Float], threshold: Float) -> Int {
        var cumulative: Float = 1
        var keep = 0
        for (i, c) in survival.enumerated() {
            cumulative *= c
            if cumulative < threshold { break }
            keep = i + 1
        }
        return keep
    }

    /// Top-k then top-p (nucleus) truncation of a probability distribution over the last
    /// axis, renormalized. Must be applied identically to the draft q and the target p.
    /// Identity fast-path when `topP ≥ 1 && topK ≤ 0` (plain temperature sampling/greedy).
    public static func truncateProbs(
        _ probs: MLXArray, topP: Float = 1.0, topK: Int = 0
    ) -> MLXArray {
        if topP >= 1.0 && topK <= 0 { return probs }
        var p = probs
        let v = p.dim(-1)
        if topK > 0 && topK < v {
            // kth-largest value per row (ascending sort → index v−topK); keep ≥ it.
            let kth = sorted(p, axis: -1)[.ellipsis, v - topK].expandedDimensions(axis: -1)
            p = which(p .>= kth, p, MLXArray(Float(0)))
        }
        if topP < 1.0 {
            let sortedDesc = negative(sorted(negative(p), axis: -1))
            let csum = cumsum(sortedDesc, axis: -1)
            // In the nucleus while the cumulative mass BEFORE a token is still < topP
            // (always keeps at least the top-1 token). Ties at the boundary are kept.
            let inNucleus = (csum - sortedDesc) .< topP
            let inf = MLXArray(Float.infinity)
            let cutoff = which(inNucleus, sortedDesc, inf).min(axis: -1, keepDims: true)
            p = which(p .>= cutoff, p, MLXArray(Float(0)))
        }
        return p / maximum(p.sum(axis: -1, keepDims: true), 1e-12)
    }

    /// Categorical sample from a probability row (the default residual/bonus sampler).
    public static func categoricalSample(_ probs: MLXArray) -> Int32 {
        MLXRandom.categorical(log(probs + 1e-20)).item(Int32.self)
    }

    /// Sampled-mode acceptance for one verified block.
    ///
    /// - Parameters:
    ///   - targetProbs: [L+1, V] target distributions at the verify positions, already
    ///     temperature-scaled and truncated.
    ///   - draftTokens: the L drafted tokens.
    ///   - draftProbs: [L, V] draft distributions the tokens were sampled from (same
    ///     truncation as the target's).
    ///   - uniforms: [L] pre-drawn U(0,1) accept draws (injectable for deterministic tests).
    ///   - sample: categorical sampler for the residual/bonus draw (injectable).
    /// - Returns: accepted count n and the committed replacement token (residual sample at
    ///   the rejected position, or the bonus token when n == L).
    public static func sampledAccept(
        targetProbs: MLXArray,
        draftTokens: [Int32],
        draftProbs: MLXArray,
        uniforms: MLXArray,
        sample: (MLXArray) -> Int32 = categoricalSample
    ) -> (accepted: Int, replacement: Int32) {
        let L = draftTokens.count
        let rows = MLXArray(Array(0 ..< Int32(L)))
        let idx = MLXArray(draftTokens)
        let pDraft = targetProbs[rows, idx]  // target prob of each drafted token
        let qDraft = draftProbs[rows, idx]   // draft prob it was sampled from
        let accepted = uniforms .< minimum(MLXArray(Float(1)), pDraft / maximum(qDraft, 1e-9))
        // Accepted-prefix length: cumprod stops at the first rejection.
        let n = cumprod(accepted.asType(.int32)).sum().item(Int.self)
        if n < L {
            var residual = maximum(targetProbs[n] - draftProbs[n], MLXArray(Float(0)))
            residual = residual / maximum(residual.sum(), 1e-9)
            return (n, sample(residual))
        }
        return (n, sample(targetProbs[L]))
    }
}
