// Load a standalone DSpark drafter checkpoint (deepseek-ai/dspark_*_block7).
//
// Unlike the MTP drafter (a head grafted onto the target's backbone), the DSpark drafter
// is a self-contained model with its own embeddings and lm_head; only its context
// (fused target hidden states) comes from the target at runtime. The official checkpoints
// are bf16 with key names matching `DSparkDrafter`'s module structure 1:1; the drafter
// runs every speculative round, so by default it is quantized on load to 4-bit
// (~6.9 GB bf16 → ~1.8 GB) — correctness is unaffected (the target verifies every
// token), only acceptance length can shift on near-ties.

import Foundation
import MLX
import MLXNN

public enum DSparkDraftLoader {
    /// Load a drafter from a checkpoint directory. `quantBits: nil` keeps bf16 (parity tests).
    public static func load(
        directory: URL, quantBits: Int? = 4, groupSize: Int = 64
    ) throws -> DSparkDrafter {
        let configURL = directory.appendingPathComponent("config.json")
        let config = try JSONDecoder().decode(
            DSparkConfiguration.self, from: Data(contentsOf: configURL))
        guard config.family == .qwen3 else {
            throw DSparkLoadError.unsupportedFamily(config.modelType)
        }
        let drafter = DSparkDrafter(config)

        var weights = [String: MLXArray]()
        let enumerator = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: nil)!
        for case let url as URL in enumerator where url.pathExtension == "safetensors" {
            for (k, v) in try loadArrays(url: url) { weights[k] = v }
        }
        guard !weights.isEmpty else { throw DSparkLoadError.noWeights(directory.path) }

        // Keys load 1:1 — diagnose any mismatch loudly BEFORE update() fails opaquely.
        let modelKeys = Set(drafter.parameters().flattened().map { $0.0 })
        let checkpointKeys = Set(weights.keys)
        let missing = modelKeys.subtracting(checkpointKeys).sorted()
        let unexpected = checkpointKeys.subtracting(modelKeys).sorted()
        guard missing.isEmpty, unexpected.isEmpty else {
            throw DSparkLoadError.keyMismatch(
                missing: Array(missing.prefix(8)), unexpected: Array(unexpected.prefix(8)))
        }

        try drafter.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])

        if let bits = quantBits {
            // Quantize Linear/Embedding weights; norms stay full precision (reference:
            // mlx-dspark `load_drafter`, nn.quantize default predicate).
            quantize(model: drafter) { _, module in
                (module is Linear || module is Embedding) ? (groupSize, bits) : nil
            }
        }
        eval(drafter)
        return drafter
    }
}

public enum DSparkLoadError: Error, CustomStringConvertible {
    case unsupportedFamily(String)
    case noWeights(String)
    case keyMismatch(missing: [String], unexpected: [String])

    public var description: String {
        switch self {
        case .unsupportedFamily(let t):
            return "DSpark drafter family \(t) is not supported yet (qwen3 only)"
        case .noWeights(let dir):
            return "no .safetensors found in DSpark drafter directory \(dir)"
        case .keyMismatch(let missing, let unexpected):
            return "DSpark drafter key mismatch — missing: \(missing) unexpected: \(unexpected)"
        }
    }
}
