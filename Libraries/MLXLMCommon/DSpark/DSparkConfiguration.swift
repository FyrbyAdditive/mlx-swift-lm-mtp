// DSpark drafter configuration — decoded from the drafter checkpoint's config.json.
//
// DSpark (DeepSeek, arXiv:2606.19348, MIT) is a semi-autoregressive speculative drafter:
// an EAGLE-style cross-attention backbone over fused target hidden states drafts a block
// of tokens in one pass, a rank-256 Markov head adds previous-token corrections, and a
// confidence head predicts per-position survival for draft truncation.
//
// Two drafter families share the inference path (mirrors the reference implementations,
// deepseek-ai/DeepSpec and ARahim3/mlx-dspark, both MIT):
//   - qwen3:  standard GQA (separate v_proj, no v_norm), default rope, llama 2-norm
//             layers, silu MLP, no logit softcap.
//   - gemma4: k_eq_v attention, v_norm, partial rope, sandwich norms + layer_scalar,
//             gelu MLP, logit softcap. (Decoded here; drafter support lands with the
//             Gemma4 target milestone.)

import Foundation

public struct DSparkConfiguration: Decodable, Sendable {
    public enum Family: String, Sendable {
        case qwen3
        case gemma4
    }

    // core dims
    public var hiddenSize: Int
    public var vocabularySize: Int
    public var hiddenLayers: Int
    public var intermediateSize: Int
    public var rmsNormEps: Float

    // attention
    public var attentionHeads: Int
    public var kvHeads: Int
    public var globalKVHeads: Int
    public var headDim: Int
    public var globalHeadDim: Int
    public var attentionKEqualsV: Bool
    public var attentionBias: Bool

    // rope
    public var ropeTheta: Float
    public var partialRotaryFactor: Float

    // dspark specifics
    public var blockSize: Int
    public var maskTokenId: Int
    public var targetLayerIds: [Int]
    public var numTargetLayers: Int

    // markov + confidence
    public var markovRank: Int
    public var markovHeadType: String
    public var enableConfidenceHead: Bool
    public var confidenceHeadWithMarkov: Bool

    // logits
    public var finalLogitSoftcapping: Float?

    public var modelType: String

    public var family: Family { modelType.contains("qwen3") ? .qwen3 : .gemma4 }

    /// Head dim used by the drafter's own attention (gemma4 uses the global-attention dims).
    public var attnHeadDim: Int { family == .gemma4 ? globalHeadDim : headDim }
    public var attnKVHeads: Int {
        family == .gemma4 && attentionKEqualsV ? globalKVHeads : kvHeads
    }
    public var attnScale: Float {
        family == .qwen3 ? pow(Float(attnHeadDim), -0.5) : 1.0
    }
    /// Gemma scales token embeddings by sqrt(hidden); qwen does not.
    public var embedScale: Float {
        family == .gemma4 ? Float(Double(hiddenSize).squareRoot()) : 1.0
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case vocabularySize = "vocab_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case globalKVHeads = "num_global_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case attentionKEqualsV = "attention_k_eq_v"
        case attentionBias = "attention_bias"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case blockSize = "block_size"
        case maskTokenId = "mask_token_id"
        case targetLayerIds = "target_layer_ids"
        case numTargetLayers = "num_target_layers"
        case markovRank = "markov_rank"
        case markovHeadType = "markov_head_type"
        case enableConfidenceHead = "enable_confidence_head"
        case confidenceHeadWithMarkov = "confidence_head_with_markov"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case modelType = "model_type"
    }

    /// The checkpoint's rope settings live either flat (`rope_theta`) or nested under
    /// `rope_parameters` (qwen3: `{rope_theta}`; gemma4: `{full_attention: {rope_theta,
    /// partial_rotary_factor}}`).
    private struct RopeParameters: Decodable {
        var ropeTheta: Float?
        var partialRotaryFactor: Float?
        var fullAttention: Inner?
        struct Inner: Codable {
            var ropeTheta: Float?
            var partialRotaryFactor: Float?
            enum CodingKeys: String, CodingKey {
                case ropeTheta = "rope_theta"
                case partialRotaryFactor = "partial_rotary_factor"
            }
        }
        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
            case fullAttention = "full_attention"
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_text"
        self.hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        self.vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        self.hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        self.rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        self.globalKVHeads = try c.decodeIfPresent(Int.self, forKey: .globalKVHeads) ?? 1
        let hidden = self.hiddenSize
        let heads = self.attentionHeads
        self.headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? (hidden / heads)
        self.globalHeadDim = try c.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        self.attentionKEqualsV = try c.decodeIfPresent(Bool.self, forKey: .attentionKEqualsV) ?? false
        self.attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false

        let rope = try c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
        let flatTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
        self.ropeTheta = rope?.fullAttention?.ropeTheta ?? rope?.ropeTheta ?? flatTheta ?? 1_000_000
        self.partialRotaryFactor =
            rope?.fullAttention?.partialRotaryFactor ?? rope?.partialRotaryFactor ?? 0.25

        self.blockSize = try c.decode(Int.self, forKey: .blockSize)
        self.maskTokenId = try c.decode(Int.self, forKey: .maskTokenId)
        self.targetLayerIds = try c.decode([Int].self, forKey: .targetLayerIds)
        self.numTargetLayers = try c.decodeIfPresent(Int.self, forKey: .numTargetLayers) ?? 36
        self.markovRank = try c.decodeIfPresent(Int.self, forKey: .markovRank) ?? 256
        self.markovHeadType = try c.decodeIfPresent(String.self, forKey: .markovHeadType) ?? "vanilla"
        self.enableConfidenceHead =
            try c.decodeIfPresent(Bool.self, forKey: .enableConfidenceHead) ?? true
        self.confidenceHeadWithMarkov =
            try c.decodeIfPresent(Bool.self, forKey: .confidenceHeadWithMarkov) ?? true
        self.finalLogitSoftcapping = try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
    }
}
