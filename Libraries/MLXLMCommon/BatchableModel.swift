// Continuous-batching seam: models that can decode a left-padded BATCH of sequences in one
// forward pass conform to this. The batch engine (in the app layer) drives prefill/decode through
// `batchForward` and builds per-layer batched caches via `newBatchCache`.
//
// Both the text Qwen3.5 backbone and the VLM Qwen3.5 backbone conform; the VLM uses the text-only
// path (Copilot/agent requests carry no images). MTP/speculative models intentionally do NOT
// conform — speculative decode is single-sequence and routes to the existing path.

import Foundation
import MLX

public protocol BatchableModel: AnyObject {
    /// Per-layer caches for a left-padded batch of `leftPadding.count` sequences. `leftPadding[b]`
    /// is the number of pad positions at the front of row b. Attention layers get a `BatchKVCache`,
    /// recurrent (Mamba/GatedDeltaNet) layers get a batched `MambaCache`.
    func newBatchCache(leftPadding: [Int]) -> [KVCache]

    /// Forward a `(B, seqLen)` Int32 token array against batched `cache`, returning `(B, seqLen,
    /// vocab)` logits. Masks (causal+left-padding for attention, left-padding for SSM) are derived
    /// from the caches' per-sequence offsets.
    func batchForward(_ tokens: MLXArray, cache: [KVCache]) -> MLXArray
}
