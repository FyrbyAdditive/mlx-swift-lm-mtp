import Foundation
import MLXLMCommon
import XCTest

/// Tests the pure cache-reuse decision logic for the MTP path: which prefix of a new prompt a
/// snapshot can be reused for, and where to take the next snapshot. Getting these wrong would
/// silently corrupt generation (reusing a cache against a mismatched prefix) or never reuse at all.
final class MTPCacheReuseTests: XCTestCase {

    // MARK: commonPrefixLength

    func testCommonPrefixLength() {
        XCTAssertEqual(MTPCacheReuse.commonPrefixLength([1, 2, 3, 4], [1, 2, 9, 4]), 2)
        XCTAssertEqual(MTPCacheReuse.commonPrefixLength([1, 2, 3], [1, 2, 3, 4, 5]), 3)
        XCTAssertEqual(MTPCacheReuse.commonPrefixLength([1, 2, 3], [1, 2, 3]), 3)
        XCTAssertEqual(MTPCacheReuse.commonPrefixLength([], [1, 2]), 0)
        XCTAssertEqual(MTPCacheReuse.commonPrefixLength([9], [1, 2]), 0)
    }

    // MARK: reuseCount — a snapshot encodes EXACTLY snapshotTokens; reuse needs it to be a strict
    // prefix of the new prompt with a non-empty suffix to generate.

    func testReusesSnapshotThatIsAStrictPrefix() {
        let snap: [Int32] = Array(1...100)          // e.g. a cached system prompt
        let new: [Int32] = Array(1...120)           // same prompt + new user tokens
        XCTAssertEqual(MTPCacheReuse.reuseCount(snapshotTokens: snap, newTokens: new), 100)
    }

    func testNoReuseOnDivergence() {
        let snap: [Int32] = Array(1...100)
        var new = Array<Int32>(1...100); new[50] = 999; new.append(101)
        XCTAssertEqual(MTPCacheReuse.reuseCount(snapshotTokens: snap, newTokens: new), 0)
    }

    func testNoReuseWhenNoSuffixToGenerate() {
        let snap: [Int32] = Array(1...100)
        XCTAssertEqual(MTPCacheReuse.reuseCount(snapshotTokens: snap, newTokens: snap), 0)  // equal
        XCTAssertEqual(
            MTPCacheReuse.reuseCount(snapshotTokens: snap, newTokens: Array(1...50)), 0)    // shorter
    }

    func testNoReuseBelowMinReuse() {
        // A tiny snapshot isn't worth the bookkeeping; require at least `minReuse` tokens.
        XCTAssertEqual(
            MTPCacheReuse.reuseCount(snapshotTokens: [1, 2, 3], newTokens: [1, 2, 3, 4], minReuse: 16),
            0)
        XCTAssertEqual(
            MTPCacheReuse.reuseCount(snapshotTokens: Array(1...20), newTokens: Array(1...30), minReuse: 16),
            20)
    }

    func testNoReuseWithEmptySnapshot() {
        XCTAssertEqual(MTPCacheReuse.reuseCount(snapshotTokens: [], newTokens: [1, 2, 3]), 0)
    }

    // MARK: snapshotPoint — where to snapshot the current prefill so a FUTURE request can reuse it
    // (the region shared with the previous prompt, i.e. the recurring system prompt).

    func testSnapshotPointIsCommonPrefixWithPrevious() {
        let prev: [Int32] = Array(1...100) + [500, 501]      // system + turn-1 tail
        let curr: [Int32] = Array(1...100) + [600, 601, 602] // same system + turn-2 tail
        XCTAssertEqual(MTPCacheReuse.snapshotPoint(previousTokens: prev, currentTokens: curr), 100)
    }

    func testNoSnapshotWhenNoPrevious() {
        XCTAssertEqual(MTPCacheReuse.snapshotPoint(previousTokens: [], currentTokens: Array(1...50)), 0)
    }

    func testNoSnapshotWhenOverlapTooSmall() {
        XCTAssertEqual(
            MTPCacheReuse.snapshotPoint(previousTokens: [1, 2, 3], currentTokens: [1, 2, 9], minReuse: 16),
            0)
    }

    /// Three-turn round trip: turn 2 snapshots the system prompt (its overlap with turn 1); turn 3,
    /// which shares that system prompt, reuses the snapshot.
    func testThreeTurnRoundTrip() {
        let system: [Int32] = Array(1...80)
        let turn1 = system + [900]                 // system + user1
        let turn2 = system + [901, 902]            // system + (user1+gen1 wrapped) + user2 …shape varies
        // Turn 2 snapshots at its common prefix with turn 1 = the system prompt.
        let snapAt = MTPCacheReuse.snapshotPoint(previousTokens: turn1, currentTokens: turn2)
        XCTAssertEqual(snapAt, 80)
        let snapshotTokens = Array(turn2.prefix(snapAt))

        // Turn 3 shares the system prompt → reuse the whole snapshot.
        let turn3 = system + [903, 904, 905]
        XCTAssertEqual(
            MTPCacheReuse.reuseCount(snapshotTokens: snapshotTokens, newTokens: turn3),
            80)
    }
}
