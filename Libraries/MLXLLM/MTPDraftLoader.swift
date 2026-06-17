// Load a standalone MTP drafter checkpoint into a full backbone model (draft-model style).
//
// mlx-community publishes Qwen3.5/3.6 MTP heads as separate ~hundreds-of-MB checkpoints
// (e.g. `Qwen3.6-27B-MTP-4bit`) that contain only the MTP head weights (`fc`, one transformer
// layer, the pre-fc norms, and the head norm) and share the target model's embeddings + lm_head.
// This attaches such a head to an already-loaded `Qwen35Model` so it can self-speculate.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum MTPDraftLoader {
    /// Attach + load a drafter checkpoint into `model`'s MTP head.
    ///
    /// - Parameters:
    ///   - model: a full backbone already loaded (registered as `qwen3_5`).
    ///   - drafterDirectory: directory of the standalone MTP drafter checkpoint.
    ///   - quantization: quantization to apply to the head (match the drafter's config).
    public static func attach(
        to model: any MTPSpeculativeModel,
        drafterDirectory: URL,
        quantization: BaseConfiguration.Quantization?
    ) throws {
        guard let module = model as? Module else {
            throw MTPDraftError.configHasNoMTP
        }
        model.attachMTPHead()
        guard model.hasMTP else {
            throw MTPDraftError.configHasNoMTP
        }

        // Load the drafter's safetensors.
        var weights = [String: MLXArray]()
        let enumerator = FileManager.default.enumerator(
            at: drafterDirectory, includingPropertiesForKeys: nil)!
        for case let url as URL in enumerator where url.pathExtension == "safetensors" {
            for (k, v) in try loadArrays(url: url) { weights[k] = v }
        }
        guard !weights.isEmpty else { throw MTPDraftError.noWeights }

        // The drafter keys are the head's own structure (`fc.*`, `layers.0.*`, `pre_fc_norm_*`,
        // `norm.*`). Map them to the attached head's parameter path: `language_model.mtp.*`.
        var mapped = [String: MLXArray]()
        for (k, v) in weights {
            // Skip any embed/lm_head the drafter might carry — the head shares the backbone's.
            if k.hasPrefix("embed_tokens") || k.hasPrefix("lm_head") { continue }
            mapped["language_model.mtp.\(k)"] = v
        }

        // Quantize the attached head's Linears to match the drafter (4-bit etc.), but only those
        // for which the drafter actually carries `.scales` (quantized) weights.
        if let quantization {
            quantize(model: module) { path, _ in
                guard path.hasPrefix("language_model.mtp.") else { return nil }
                return mapped["\(path).scales"] != nil ? quantization.asTuple : nil
            }
        }

        // Apply just the MTP head parameters (verify none-missing within the head).
        let parameters = ModuleParameters.unflattened(mapped)
        try module.update(parameters: parameters, verify: [])
        eval(module)
    }
}

public enum MTPDraftError: Error {
    case configHasNoMTP
    case noWeights
}
