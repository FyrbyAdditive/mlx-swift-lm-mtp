import Foundation
import MLXLMCommon
import XCTest

/// Tests the MTP stop-token set construction. Regression coverage for the "infinite waffle" bug:
/// the MTP decode loop had no EOS detection, so generation ran to maxTokens and the model
/// role-played both sides of the chat past `<|im_end|>`/`<|endoftext|>`.
final class MTPStopTokensTests: XCTestCase {

    func testCombinesAllSources() {
        // Qwen-style: a tokenizer EOS, a config EOS id, and extra EOS strings resolved via the
        // tokenizer's vocab.
        let vocab = ["<|im_end|>": 151645, "<|endoftext|>": 151643]
        let ids = MTPStopTokens.build(
            eosTokenIds: [100],
            tokenizerEOSTokenId: 151643,
            extraEOSTokens: ["<|im_end|>"],
            tokenToId: { vocab[$0] })
        XCTAssertEqual(ids, [100, 151643, 151645])
    }

    func testNilTokenizerEOSIsSkipped() {
        let ids = MTPStopTokens.build(
            eosTokenIds: [7],
            tokenizerEOSTokenId: nil,
            extraEOSTokens: [],
            tokenToId: { _ in nil })
        XCTAssertEqual(ids, [7])
    }

    func testUnknownExtraEOSStringIsSkipped() {
        // An extra EOS string the tokenizer doesn't know must not crash or insert a bogus id.
        let ids = MTPStopTokens.build(
            eosTokenIds: [],
            tokenizerEOSTokenId: 2,
            extraEOSTokens: ["<|not_in_vocab|>"],
            tokenToId: { _ in nil })
        XCTAssertEqual(ids, [2])
    }

    func testDeduplicatesOverlappingSources() {
        // The same id arriving from multiple sources collapses (it's a Set).
        let ids = MTPStopTokens.build(
            eosTokenIds: [2],
            tokenizerEOSTokenId: 2,
            extraEOSTokens: ["</s>"],
            tokenToId: { _ in 2 })
        XCTAssertEqual(ids, [2])
    }

    func testEmptyWhenNoSources() {
        let ids = MTPStopTokens.build(
            eosTokenIds: [], tokenizerEOSTokenId: nil, extraEOSTokens: [], tokenToId: { _ in nil })
        XCTAssertTrue(ids.isEmpty)
    }

    /// The decode loop checks `stopTokenIds.contains(id)` to stop *before* emitting the token —
    /// this asserts the membership semantics that gate the loop.
    func testMembershipGatesStop() {
        let stop = MTPStopTokens.build(
            eosTokenIds: [], tokenizerEOSTokenId: 151643,
            extraEOSTokens: ["<|im_end|>"], tokenToId: { $0 == "<|im_end|>" ? 151645 : nil })
        XCTAssertTrue(stop.contains(151645))   // <|im_end|> → stop
        XCTAssertTrue(stop.contains(151643))   // <|endoftext|>/tokenizer EOS → stop
        XCTAssertFalse(stop.contains(9707))    // an ordinary content token → keep going
    }
}
