// Native Multi-Token-Prediction (MTP) self-speculative decoding.
//
// A model that carries an MTP head (e.g. Qwen3.5/3.6 `qwen3_5_mtp` checkpoints) can draft its
// own future tokens and verify them in a single backbone forward pass, accepting via
// rejection sampling so the output distribution exactly matches non-speculative decoding.

import Foundation
import MLX

/// A `LanguageModel` that exposes a native MTP head for self-speculative decoding.
public protocol MTPSpeculativeModel: LanguageModel {
    /// True when the loaded weights include an MTP head.
    var hasMTP: Bool { get }

    /// Fresh KVCache list for the MTP layers (empty if no head).
    func makeMTPCache() -> [KVCache]

    /// Backbone forward returning logits AND the pre-norm hidden state.
    /// `nConfirmed` (> 0 and < sequence length) makes recurrent/SSM layers snapshot their state
    /// after the confirmed prefix so it can be rolled back if the draft token is rejected.
    func backboneWithHidden(
        _ inputs: MLXArray, cache: [KVCache]?, nConfirmed: Int
    ) -> (logits: MLXArray, hidden: MLXArray)

    /// Run the MTP head on a pre-norm hidden state + next-token ids → logits (B, N, vocab).
    func mtpForward(
        _ hiddenStates: MLXArray, nextTokenIds: MLXArray, cache: [KVCache?]?
    ) -> MLXArray
}
