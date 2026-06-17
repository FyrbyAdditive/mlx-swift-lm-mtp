import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

/// THE make-or-break test for batched concurrency: a left-padded B>1 batch forward of the hybrid
/// Qwen3.5 (GatedDeltaNet SSM + interleaved full-attention) must produce, for each row, logits
/// identical to a solo B=1 forward of that row's real (unpadded) prompt. If the SSM mask / conv
/// window / per-sequence attention offset are wrong, padded positions leak and rows diverge.
/// Uses a tiny random-weight model — parity is a property of the math, not the weights.
final class Qwen35BatchParityTests: XCTestCase {

    private func makeModel() -> Qwen35TextModel {
        // Tiny hybrid config: full_attention_interval 2 → mix of GatedDeltaNet + attention layers.
        let json = """
        {
          "model_type": "qwen3_5_text",
          "hidden_size": 64,
          "num_hidden_layers": 4,
          "intermediate_size": 128,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "linear_num_value_heads": 8,
          "linear_num_key_heads": 4,
          "linear_key_head_dim": 32,
          "linear_value_head_dim": 32,
          "linear_conv_kernel_dim": 4,
          "rms_norm_eps": 1e-6,
          "vocab_size": 256,
          "head_dim": 16,
          "rope_theta": 1000.0,
          "partial_rotary_factor": 0.5,
          "max_position_embeddings": 4096,
          "tie_word_embeddings": true,
          "full_attention_interval": 2
        }
        """
        let config = try! JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
        let model = Qwen35TextModel(config)
        MLXRandom.seed(0)
        eval(model)
        return model
    }

    /// Logits (last position) of a solo forward of `tokens`.
    private func soloLogits(_ model: Qwen35TextModel, _ tokens: [Int32]) -> MLXArray {
        let input = MLXArray(tokens)[.newAxis, .ellipsis]  // (1, L)
        let cache = model.newCache(parameters: nil)
        let out = model(input, cache: cache)  // (1, L, vocab)
        let last = out[0, -1, 0...]
        eval(last)
        return last
    }

    func testBatchedPrefillMatchesSoloPerRow() {
        let model = makeModel()
        // Two prompts of different lengths.
        let p0: [Int32] = [5, 12, 7]
        let p1: [Int32] = [9, 2, 30, 14, 6]
        let maxLen = max(p0.count, p1.count)
        let leftPad = [maxLen - p0.count, maxLen - p1.count]  // [2, 0]

        // Left-pad each row to maxLen (pad token 0).
        func leftPadded(_ t: [Int32]) -> [Int32] { Array(repeating: 0, count: maxLen - t.count) + t }
        let batchTokens = [leftPadded(p0), leftPadded(p1)]
        let batchInput = MLXArray(batchTokens.flatMap { $0 }, [2, maxLen])

        // newBatchCache already constructs caches with leftPadding configured (do NOT call
        // prepare again — it would double the padding).
        let batchCache = model.newBatchCache(leftPadding: leftPad)
        let batchOut = model(batchInput, cache: batchCache)  // (2, maxLen, vocab)
        eval(batchOut)

        // Compare each row's LAST real position against a solo run.
        let solo0 = soloLogits(model, p0)
        let solo1 = soloLogits(model, p1)
        let batchRow0 = batchOut[0, maxLen - 1, 0...]  // last position = last real token of p0
        let batchRow1 = batchOut[1, maxLen - 1, 0...]
        eval(batchRow0, batchRow1)

        let d0 = (batchRow0 - solo0).abs().max().item(Float.self)
        let d1 = (batchRow1 - solo1).abs().max().item(Float.self)
        // Tight tolerance: with per-row RoPE offsets (BatchPositionedKVCache) the batched row should
        // match the solo run to numerical noise, not just approximately.
        XCTAssertLessThan(d0, 5e-4, "row 0 batched logits diverge from solo (max abs diff \(d0))")
        XCTAssertLessThan(d1, 5e-4, "row 1 batched logits diverge from solo (max abs diff \(d1))")
    }

    /// Greedy DECODE parity: prefill + several decode steps. Each batched row's greedy token
    /// sequence must equal a solo greedy run of that prompt. This is what the engine relies on —
    /// the cache must advance correctly per row across steps (offsets, conv/ssm recurrence).
    func testBatchedGreedyDecodeMatchesSolo() {
        let model = makeModel()
        let p0: [Int32] = [5, 12, 7]
        let p1: [Int32] = [9, 2, 30, 14, 6]
        let steps = 5

        // Solo greedy for a prompt.
        func soloGreedy(_ prompt: [Int32]) -> [Int32] {
            let cache = model.newCache(parameters: nil)
            var toks: [Int32] = []
            var cur = MLXArray(prompt)[.newAxis, .ellipsis]
            for _ in 0 ..< steps {
                let out = model(cur, cache: cache)
                let next = argMax(out[0, -1, 0...], axis: -1).item(Int32.self)
                toks.append(next)
                cur = MLXArray([next])[.newAxis, .ellipsis]
            }
            return toks
        }
        let solo0 = soloGreedy(p0)
        let solo1 = soloGreedy(p1)

        // Batched greedy.
        let maxLen = max(p0.count, p1.count)
        let leftPad = [maxLen - p0.count, maxLen - p1.count]
        func lpad(_ t: [Int32]) -> [Int32] { Array(repeating: 0, count: maxLen - t.count) + t }
        let cache = model.newBatchCache(leftPadding: leftPad)
        var cur = MLXArray([lpad(p0), lpad(p1)].flatMap { $0 }, [2, maxLen])
        var batch0: [Int32] = []
        var batch1: [Int32] = []
        for _ in 0 ..< steps {
            let out = model.batchForward(cur, cache: cache)  // (2, S, vocab)
            let lastLogits = out[0..., -1, 0...]             // (2, vocab)
            let next = argMax(lastLogits, axis: -1)          // (2,)
            eval(next)
            let n = next.asArray(Int32.self)
            batch0.append(n[0]); batch1.append(n[1])
            cur = next[0..., .newAxis]                       // (2, 1)
        }

        XCTAssertEqual(batch0, solo0, "row 0 batched greedy decode != solo")
        XCTAssertEqual(batch1, solo1, "row 1 batched greedy decode != solo")
    }
}
