// Prompt-lookup (n-gram) drafting — drafter-free draft proposals for copy runs
// (quoting, code edits, tool-call echoes). Port of mlx-dspark's NGramIndex (MIT):
// when the current suffix n-gram already occurred earlier in the sequence, propose the
// tokens that followed the LATEST earlier occurrence and let the target verify them like
// any other draft. Hybrid defaults (minN 4): 4-grams almost never fire spuriously, while
// genuine copying has them in abundance; a miss costs nothing (the round proceeds as
// usual). The two most recent occurrences are kept per n-gram so a query can skip the
// occurrence that is the current suffix itself.

import Foundation

public final class NGramIndex {
    let minN: Int
    let maxN: Int
    let maxDraft: Int
    private var tokens: [Int32] = []
    /// n-gram → (previous end position, latest end position); "end position" is the index
    /// right AFTER the n-gram, i.e. where its continuation starts.
    private var index: [[Int32]: (prev: Int?, latest: Int)] = [:]

    public init(minN: Int = 4, maxN: Int = 5, maxDraft: Int = 6) {
        self.minN = max(1, minN)
        self.maxN = max(self.minN, maxN)
        self.maxDraft = max(1, maxDraft)
    }

    public var count: Int { tokens.count }

    public func extend(_ newTokens: [Int32]) {
        for t in newTokens {
            tokens.append(t)
            let end = tokens.count
            for n in minN ... maxN where end >= n {
                let ng = Array(tokens[(end - n) ..< end])
                let prev = index[ng]
                index[ng] = (prev?.latest, end)
            }
        }
    }

    /// Draft tokens continuing the latest EARLIER occurrence of the current suffix n-gram
    /// (longest n first). Empty = no confident match; do a normal round (zero miss cost).
    public func propose() -> [Int32] {
        let end = tokens.count
        for n in stride(from: maxN, through: minN, by: -1) {
            guard end >= n, let hit = index[Array(tokens[(end - n) ..< end])] else { continue }
            let pos = hit.latest != end ? hit.latest : hit.prev  // skip the suffix itself
            guard let pos else { continue }
            let draft = Array(tokens[pos ..< min(pos + maxDraft, tokens.count)])
            if !draft.isEmpty { return draft }
        }
        return []
    }
}
