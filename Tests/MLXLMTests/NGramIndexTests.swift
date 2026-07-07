import Foundation
import XCTest

@testable import MLXLMCommon

final class NGramIndexTests: XCTestCase {
    func testProposesContinuationOfEarlierOccurrence() {
        let idx = NGramIndex(minN: 4, maxN: 5, maxDraft: 6)
        // ... 1 2 3 4 5 6 7 8 ... then the suffix repeats 1 2 3 4 → propose 5 6 7 8...
        idx.extend([9, 1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 1, 2, 3, 4])
        XCTAssertEqual(idx.propose(), [5, 6, 7, 8, 20, 21])
    }

    func testNoMatchReturnsEmpty() {
        let idx = NGramIndex(minN: 4, maxN: 5, maxDraft: 6)
        idx.extend([1, 2, 3, 4, 5, 6, 7, 8])
        XCTAssertEqual(idx.propose(), [])
    }

    func testPrefersLatestEarlierOccurrence() {
        let idx = NGramIndex(minN: 4, maxN: 4, maxDraft: 2)
        // 1 2 3 4 occurs twice with different continuations; the LATEST earlier one wins.
        idx.extend([1, 2, 3, 4, 50, 51, 1, 2, 3, 4, 60, 61, 1, 2, 3, 4])
        XCTAssertEqual(idx.propose(), [60, 61])
    }

    func testShortSequenceSafe() {
        let idx = NGramIndex(minN: 4, maxN: 5, maxDraft: 6)
        idx.extend([1, 2])
        XCTAssertEqual(idx.propose(), [])
    }
}
