// Pure, model-free logic for cross-request KV-cache reuse on the MTP path. Kept separate from the
// decode loop so the cache-correctness invariants can be unit-tested without loading a model.

import Foundation

/// Output box for `mtpGenerate`, filled once on clean completion (left untouched on
/// cancellation/error). Carries the prompt tokens (so the engine can pick the next snapshot point)
/// and, when a snapshot was requested, copies of the model+MTP caches taken mid-prefill plus the
/// exact token prefix they encode. A reference box (not a closure) so the non-Sendable `KVCache`
/// arrays never cross the generation `Task` boundary as `sending` values; written only inside the
/// serialized generation task and read by the caller afterward.
public final class MTPCacheResult: @unchecked Sendable {
    /// The full prompt token sequence of the just-completed request.
    public var promptTokens: [Int32]?
    /// Snapshot caches taken at `snapshotTokens.count` tokens (nil if no snapshot was requested).
    public var snapshotModelCache: [KVCache]?
    public var snapshotMtpCache: [KVCache]?
    /// The exact token prefix the snapshot caches encode.
    public var snapshotTokens: [Int32]?
    public init() {}
}

public enum MTPCacheReuse {

    /// Length of the longest common prefix of two token sequences.
    public static func commonPrefixLength(_ a: [Int32], _ b: [Int32]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n, a[i] == b[i] { i += 1 }
        return i
    }

    /// Decide how many leading tokens of `newTokens` a cache **snapshot** (which encodes exactly
    /// `snapshotTokens`) can be reused for.
    ///
    /// The hybrid backbone has SSM (Mamba) layers whose recurrent state cannot be rewound, so reuse
    /// requires the snapshot to encode *exactly* a prefix of the new prompt: every token the
    /// snapshot holds must match, and there must be a non-empty suffix left to generate from.
    /// Unlike a live cache (which accrues generated tokens and so rarely stays a clean prefix), a
    /// snapshot is deliberately taken at a stable boundary (see `snapshotPoint`). Returns the number
    /// of reusable leading tokens (== snapshot length) or 0 for "prefill from scratch".
    public static func reuseCount(snapshotTokens: [Int32], newTokens: [Int32], minReuse: Int = 16)
        -> Int
    {
        guard snapshotTokens.count >= minReuse, snapshotTokens.count < newTokens.count else {
            return 0
        }
        for i in 0 ..< snapshotTokens.count where snapshotTokens[i] != newTokens[i] {
            return 0
        }
        return snapshotTokens.count
    }

    /// The token position at which to snapshot the current prefill so a *future* request can reuse
    /// it: the longest prefix this prompt shares with the previous prompt (their common prefix is
    /// the stable region — e.g. a constant system prompt — that recurs across turns). Returns 0
    /// when there's nothing worth snapshotting (no previous prompt or a too-short overlap).
    public static func snapshotPoint(previousTokens: [Int32], currentTokens: [Int32], minReuse: Int = 16)
        -> Int
    {
        let l = commonPrefixLength(previousTokens, currentTokens)
        return l >= minReuse ? l : 0
    }
}
