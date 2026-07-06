// Target-model surface required by DSpark speculative decoding.
//
// The DSpark drafter conditions on the TARGET's intermediate hidden states: the residual
// stream captured after each layer in the drafter checkpoint's `target_layer_ids`
// (pre-final-norm), concatenated on the feature axis. Verification and drafting share one
// forward: the verify pass of round N produces the taps that become drafter context for
// round N+1, so a conforming model just returns both.

import Foundation
import MLX

public protocol DSparkTargetModel: LanguageModel {
    /// Full forward with hidden-state taps.
    /// - Returns: logits (B, S, V) and the tapped residual-stream states concatenated on
    ///   the feature axis (B, S, H·tapLayers.count), in `tapLayers` order.
    func forwardWithTaps(
        _ inputs: MLXArray, cache: [KVCache]?, tapLayers: [Int]
    ) -> (logits: MLXArray, taps: MLXArray)

    /// Prefill variant: warms the cache and returns ONLY the taps. Implementations should
    /// avoid materializing the LM head over the chunk (with MLX laziness, simply not
    /// consuming the logits suffices).
    func prefillWithTaps(_ inputs: MLXArray, cache: [KVCache]?, tapLayers: [Int]) -> MLXArray
}

/// Whether a loaded model can serve as a DSpark target for a drafter expecting
/// `numTargetLayers` layers and `hiddenSize` features per tap.
public func dsparkTargetCompatible(
    _ model: any LanguageModel, config: DSparkConfiguration
) -> Bool {
    model is any DSparkTargetModel
}
