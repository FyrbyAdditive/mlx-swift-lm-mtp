import Foundation
import XCTest

@testable import MLXLMCommon

final class AdaptiveDraftControllerTests: XCTestCase {

    /// Feed the controller and run whatever arm it asks for, with fixed per-arm costs.
    private func run(
        _ c: AdaptiveDraftController, steps: Int,
        specMs: Double, plainMs: Double
    ) -> (drafting: Int, plain: Int) {
        var counts = (drafting: 0, plain: 0)
        for _ in 0 ..< steps {
            switch c.nextArm {
            case .drafting:
                counts.drafting += 1
                c.record(arm: .drafting, msPerToken: specMs)
            case .plain:
                counts.plain += 1
                c.record(arm: .plain, msPerToken: plainMs)
            }
        }
        return counts
    }

    func testBootstrapSamplesBothArmsThenPicksWinner() {
        let c = AdaptiveDraftController(probeCadence: 32, probeLength: 6)
        let counts = run(c, steps: 40, specMs: 8, plainMs: 10)
        XCTAssertEqual(c.mode, .drafting)
        XCTAssertGreaterThanOrEqual(counts.plain, 6)  // bootstrap block
        XCTAssertGreaterThan(counts.drafting, counts.plain)
    }

    func testSuspendsWhenDraftingLoses() {
        let c = AdaptiveDraftController()
        _ = run(c, steps: 20, specMs: 13, plainMs: 10)
        XCTAssertEqual(c.mode, .plain, "13ms/tok drafting vs 10 plain must suspend")
    }

    func testHysteresisPreventsFlappingOnNearTies() {
        let c = AdaptiveDraftController()
        _ = run(c, steps: 200, specMs: 10.1, plainMs: 10.0)  // within 3% band
        XCTAssertEqual(c.mode, .drafting, "within hysteresis: keep the current mode")
    }

    func testProbesAndRecoversWhenConditionsImprove() {
        let c = AdaptiveDraftController(probeCadence: 20, probeLength: 4)
        _ = run(c, steps: 40, specMs: 14, plainMs: 10)
        XCTAssertEqual(c.mode, .plain)
        // Conditions change: drafting becomes much faster. The periodic probe must
        // discover it and switch back.
        _ = run(c, steps: 120, specMs: 7, plainMs: 10)
        XCTAssertEqual(c.mode, .drafting, "probe must win drafting back")
    }

    func testSuspendedModeStillMostlyPlain() {
        let c = AdaptiveDraftController(probeCadence: 20, probeLength: 4)
        let counts = run(c, steps: 300, specMs: 15, plainMs: 10)
        XCTAssertEqual(c.mode, .plain)
        // Probes bound the wasted drafting work: cadence 20 + probe 4 → ≤ ~1/5 drafting
        // after bootstrap.
        XCTAssertLessThan(Double(counts.drafting) / Double(counts.plain + counts.drafting), 0.3)
    }

    /// A losing mode is abandoned within ~a window of samples (per-sample decisions),
    /// not at the next probe boundary — the lag that let chat sessions draft for 30+
    /// losing rounds.
    func testSwitchesWithinAWindowWhenModeStartsLosing() {
        let c = AdaptiveDraftController(probeCadence: 32, probeLength: 6)
        _ = run(c, steps: 30, specMs: 8, plainMs: 10)  // drafting winning
        XCTAssertEqual(c.mode, .drafting)
        // Content shifts: drafting now much slower. Must suspend within ~window samples.
        for _ in 0 ..< 8 { c.record(arm: .drafting, msPerToken: 16) }
        XCTAssertEqual(c.mode, .plain, "per-sample decisions must abandon a losing mode fast")
    }

    /// Probe cadence backs off while probes confirm the standing mode, bounding
    /// steady-state probe waste when the idle arm keeps losing.
    func testProbeCadenceBacksOff() {
        let c = AdaptiveDraftController(probeCadence: 10, probeLength: 3)
        let counts = run(c, steps: 400, specMs: 15, plainMs: 10)
        XCTAssertEqual(c.mode, .plain)
        // With backoff to 4× cadence, drafting probes are ≤ ~3 per 40 samples steady
        // state; without backoff it would be 3 per 13. Total drafting share stays small.
        XCTAssertLessThan(
            Double(counts.drafting) / Double(counts.drafting + counts.plain), 0.2)
    }

    /// The median window shrugs off a slow first round (per-request warmup) — the exact
    /// failure that made an EWMA controller wrongly suspend and spend most of a request
    /// recovering.
    func testMedianIgnoresSlowWarmupSample() {
        let c = AdaptiveDraftController(probeCadence: 32, probeLength: 6)
        // First drafting sample is a 30ms warmup outlier; the true cost is 9ms.
        c.record(arm: .drafting, msPerToken: 30)
        for _ in 0 ..< 5 { c.record(arm: .drafting, msPerToken: 9) }
        for _ in 0 ..< 6 { c.record(arm: .plain, msPerToken: 12) }
        XCTAssertEqual(c.mode, .drafting, "one warmup outlier must not suspend drafting")
        XCTAssertEqual(c.specEstimate!, 9, accuracy: 0.01)
    }
}
