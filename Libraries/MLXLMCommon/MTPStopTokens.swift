// Stop-token handling for the native MTP decode loop. Extracted as pure, model-free logic so it
// can be unit-tested without loading a model (the decode loop in MTPGenerate.swift runs inside a
// Task with a live ModelContext, which a fast unit test can't construct).

import Foundation

/// Builds the set of token ids that terminate generation, from all the same sources the standard
/// `generate(...)` loop uses: the model config's EOS ids, the tokenizer's EOS id, and any extra
/// EOS *strings* (e.g. `<|im_end|>`) resolved to ids via the tokenizer.
///
/// Without this the MTP loop ran to `maxTokens`, sailing past `<|im_end|>`/`<|endoftext|>` so the
/// model role-played both sides of the chat (the "infinite waffle").
public enum MTPStopTokens {
    /// - Parameters:
    ///   - eosTokenIds: numeric EOS ids from the model configuration.
    ///   - tokenizerEOSTokenId: the tokenizer's EOS id, if any.
    ///   - extraEOSTokens: extra EOS token *strings* from the configuration.
    ///   - tokenToId: resolves an EOS string to its id (nil if the tokenizer doesn't know it).
    public static func build(
        eosTokenIds: Set<Int>,
        tokenizerEOSTokenId: Int?,
        extraEOSTokens: Set<String>,
        tokenToId: (String) -> Int?
    ) -> Set<Int> {
        var ids = eosTokenIds
        if let tokenizerEOSTokenId {
            ids.insert(tokenizerEOSTokenId)
        }
        for token in extraEOSTokens {
            if let id = tokenToId(token) {
                ids.insert(id)
            }
        }
        return ids
    }
}
