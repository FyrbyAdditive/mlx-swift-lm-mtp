// DSpark drafter (DeepSeek, arXiv:2606.19348) — Swift MLX port of the reference inference
// path (deepseek-ai/DeepSpec; ARahim3/mlx-dspark, both MIT, attribution in ACKNOWLEDGMENTS).
//
// EAGLE-style cross-attention drafter: Q comes from the draft block ([pending] + mask
// tokens), K/V from concat([fused target context, block]) with the context K/V cached
// per layer, append-only (only committed tokens enter; never trimmed during decode).
// Block attention is BIDIRECTIONAL (no mask): each position's hidden state depends on the
// whole k-wide block, so the backbone always runs at full block width even when only the
// first `cap` positions are drafted/verified.
//
// Module attribute keys match the HF checkpoint 1:1 so bf16 weights load unrenamed.
// Qwen3 drafter family only (separate v_proj, default rope, llama 2-norm, silu, no
// softcap); the gemma4 drafter family lands with the Gemma4 target milestone.

import Foundation
import MLX
import MLXNN

/// Rank-256 previous-token logit correction: `logits += w2(w1[prev])`.
public class DSparkMarkovHead: Module {
    @ModuleInfo(key: "markov_w1") var w1: Embedding
    @ModuleInfo(key: "markov_w2") var w2: Linear

    init(_ config: DSparkConfiguration) {
        _w1.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.markovRank)
        _w2.wrappedValue = Linear(config.markovRank, config.vocabularySize, bias: false)
    }

    public func prevEmbeddings(_ tokenIds: MLXArray) -> MLXArray { w1(tokenIds) }
    public func stepBias(_ tokenIds: MLXArray) -> MLXArray { w2(w1(tokenIds)) }
}

/// Per-position prefix-survival head: linear on [block hidden ; markov embedding of the
/// previous drafted token] (sigmoid applied by the caller).
public class DSparkConfidenceHead: Module {
    @ModuleInfo(key: "proj") var proj: Linear

    init(inputDim: Int) {
        _proj.wrappedValue = Linear(inputDim, 1, bias: true)
    }

    public func callAsFunction(_ features: MLXArray) -> MLXArray {
        proj(features).squeezed(axis: -1)
    }
}

/// Cross-attention: Q from the draft block, K/V from [cached target context, block].
/// The context cache holds already-projected (roped K, raw V) — appended via
/// `updateContext`, read (never grown) by `attend`.
class DSparkAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    init(_ config: DSparkConfiguration) {
        self.nHeads = config.attentionHeads
        self.nKVHeads = config.attnKVHeads
        self.headDim = config.attnHeadDim
        self.scale = config.attnScale

        let h = config.hiddenSize
        let bias = config.attentionBias
        _qProj.wrappedValue = Linear(h, nHeads * headDim, bias: bias)
        _kProj.wrappedValue = Linear(h, nKVHeads * headDim, bias: bias)
        _vProj.wrappedValue = Linear(h, nKVHeads * headDim, bias: bias)
        _oProj.wrappedValue = Linear(nHeads * headDim, h, bias: bias)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta, scale: 1)
    }

    /// Project x → (normed K, V), shaped [B, kvHeads, S, headDim]. K is NOT yet roped.
    private func kv(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let (B, S) = (x.dim(0), x.dim(1))
        let k = kNorm(kProj(x).reshaped(B, S, nKVHeads, headDim)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, S, nKVHeads, headDim).transposed(0, 2, 1, 3)
        return (k, v)
    }

    /// Append newly committed fused-context positions to this layer's ctx cache.
    /// K is roped at its absolute position; V is not roped (reference semantics).
    func updateContext(_ fusedNew: MLXArray, cache: KVCacheSimple) {
        let (k, v) = kv(fusedNew)
        _ = cache.update(keys: rope(k, offset: cache.offset), values: v)
    }

    /// One block attention: `hidden` [B, k, H] at absolute position `blockOffset`.
    /// No mask — the block attends the whole context plus every block position.
    func attend(_ hidden: MLXArray, blockOffset: Int, cache: KVCacheSimple) -> MLXArray {
        let (B, qLen) = (hidden.dim(0), hidden.dim(1))
        var q = qNorm(qProj(hidden).reshaped(B, qLen, nHeads, headDim)).transposed(0, 2, 1, 3)
        q = rope(q, offset: blockOffset)

        var (kBlk, vBlk) = kv(hidden)
        kBlk = rope(kBlk, offset: blockOffset)

        let ctx = cache.state
        let keys = ctx.isEmpty ? kBlk : concatenated([ctx[0], kBlk], axis: 2)
        let values = ctx.isEmpty ? vBlk : concatenated([ctx[1], vBlk], axis: 2)

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: keys, values: values, scale: scale, mask: .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, qLen, -1)
        return oProj(out)
    }
}

class DSparkMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ config: DSparkConfiguration) {
        let (h, i) = (config.hiddenSize, config.intermediateSize)
        _gate.wrappedValue = Linear(h, i, bias: false)
        _up.wrappedValue = Linear(h, i, bias: false)
        _down.wrappedValue = Linear(i, h, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class DSparkDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DSparkAttention
    let mlp: DSparkMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: DSparkConfiguration) {
        _selfAttn.wrappedValue = DSparkAttention(config)
        self.mlp = DSparkMLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, blockOffset: Int, cache: KVCacheSimple) -> MLXArray {
        let h = x + selfAttn.attend(inputLayerNorm(x), blockOffset: blockOffset, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class DSparkDrafter: Module {
    public let config: DSparkConfiguration
    public var blockSize: Int { config.blockSize }
    public var maskTokenId: Int { config.maskTokenId }

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    let layers: [DSparkDecoderLayer]
    let norm: RMSNorm
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    @ModuleInfo(key: "markov_head") var markovHead: DSparkMarkovHead?
    @ModuleInfo(key: "confidence_head") var confidenceHead: DSparkConfidenceHead?

    public var hasConfidenceHead: Bool { confidenceHead != nil }

    public init(_ config: DSparkConfiguration) {
        precondition(
            config.family == .qwen3,
            "DSparkDrafter currently supports the qwen3 drafter family only")
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _fc.wrappedValue = Linear(
            config.targetLayerIds.count * config.hiddenSize, config.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.layers = (0 ..< config.hiddenLayers).map { _ in DSparkDecoderLayer(config) }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        _markovHead.wrappedValue = config.markovRank > 0 ? DSparkMarkovHead(config) : nil
        if config.enableConfidenceHead {
            let inDim = config.hiddenSize
                + (config.confidenceHeadWithMarkov ? config.markovRank : 0)
            _confidenceHead.wrappedValue = DSparkConfidenceHead(inputDim: inDim)
        } else {
            _confidenceHead.wrappedValue = nil
        }
    }

    /// One append-only context cache per layer. `KVCacheSimple` gives trim (prefix-cache
    /// restore) and copy (snapshots) for free.
    public func makeContextCaches() -> [KVCacheSimple] {
        layers.map { _ in KVCacheSimple() }
    }

    public func embed(_ ids: MLXArray) -> MLXArray {
        config.embedScale == 1.0 ? embedTokens(ids) : embedTokens(ids) * config.embedScale
    }

    /// Fuse the target's tapped hidden states (concatenated on the feature axis) into the
    /// drafter's hidden space.
    public func fuseTarget(_ targetHiddenCat: MLXArray) -> MLXArray {
        hiddenNorm(fc(targetHiddenCat))
    }

    /// Append newly committed target positions to every layer's context cache.
    /// `targetHiddenCat` is [B, S, tapCount·H] for exactly the committed tokens; each
    /// cache's offset must already equal those tokens' absolute start position.
    public func updateContext(_ targetHiddenCat: MLXArray, ctxCaches: [KVCacheSimple]) {
        let fused = fuseTarget(targetHiddenCat)
        for (layer, cache) in zip(layers, ctxCaches) {
            layer.selfAttn.updateContext(fused, cache: cache)
        }
    }

    /// Run the block backbone: `noiseEmbedding` [B, k, H] (embedded [pending] + mask ids)
    /// at absolute position `blockOffset`. Returns final-normed hidden states [B, k, H].
    public func backbone(
        _ noiseEmbedding: MLXArray, blockOffset: Int, ctxCaches: [KVCacheSimple]
    ) -> MLXArray {
        var h = noiseEmbedding
        for (layer, cache) in zip(layers, ctxCaches) {
            h = layer(h, blockOffset: blockOffset, cache: cache)
        }
        return norm(h)
    }

    public func computeLogits(_ hidden: MLXArray) -> MLXArray {
        var logits = lmHead(hidden)
        if let cap = config.finalLogitSoftcapping {
            logits = tanh(logits / cap) * cap
        }
        return logits
    }

    /// Greedy semi-autoregressive block sampling: position i's logits get the Markov bias
    /// of the token drafted at i−1 (position 0 conditions on `firstPrevToken` = the pending
    /// committed token). Sequential by construction; every op stays on the GPU.
    public func sampleBlock(_ baseLogits: MLXArray, firstPrevToken: Int32) -> MLXArray {
        let k = baseLogits.dim(0)
        guard let markovHead else { return argMax(baseLogits, axis: -1) }
        var tokens: [MLXArray] = []
        var prev = MLXArray([firstPrevToken])
        for i in 0 ..< k {
            let step = baseLogits[i] + markovHead.stepBias(prev)[0]
            let next = argMax(step, axis: -1, keepDims: true)
            tokens.append(next)
            prev = next
        }
        return concatenated(tokens)
    }

    /// Sampled semi-autoregressive block for speculative SAMPLING: each position draws
    /// from its temperature-scaled, top-p/top-k-truncated distribution q_i. Returns the
    /// drafted tokens and the q distributions they were sampled from — the verifier needs
    /// them for the min(1, p/q) accept test and residual resampling. Truncating q with the
    /// SAME parameters as the target keeps acceptance from collapsing under nucleus
    /// sampling; losslessness comes from the target side. Sequential (Markov bias for
    /// position i depends on the token sampled at i−1).
    public func sampleBlockProbs(
        _ baseLogits: MLXArray, firstPrevToken: Int32,
        temperature: Float, topP: Float = 1.0, topK: Int = 0
    ) -> (tokens: MLXArray, probs: MLXArray) {
        let k = baseLogits.dim(0)
        let invT = 1 / temperature
        var tokens: [MLXArray] = []
        var probs: [MLXArray] = []
        var prev = MLXArray([firstPrevToken])
        for i in 0 ..< k {
            var logits = baseLogits[i]
            if let markovHead { logits = logits + markovHead.stepBias(prev)[0] }
            let q = SpeculativeVerifier.truncateProbs(
                softmax(logits * invT, axis: -1), topP: topP, topK: topK)
            probs.append(q)
            let next = categorical(log(q + 1e-20)).reshaped([1])
            tokens.append(next)
            prev = next
        }
        return (concatenated(tokens), stacked(probs, axis: 0))
    }

    /// Confidence logits for each block position (sigmoid → conditional survival
    /// probability). `blockHidden` [k, H]; `prevTokenIds` [k] = [pending, draft[0..k-2]].
    public func confidenceLogits(_ blockHidden: MLXArray, prevTokenIds: MLXArray) -> MLXArray? {
        guard let confidenceHead else { return nil }
        let features: MLXArray
        if config.confidenceHeadWithMarkov, let markovHead {
            features = concatenated(
                [blockHidden, markovHead.prevEmbeddings(prevTokenIds)], axis: -1)
        } else {
            features = blockHidden
        }
        return confidenceHead(features)
    }
}
