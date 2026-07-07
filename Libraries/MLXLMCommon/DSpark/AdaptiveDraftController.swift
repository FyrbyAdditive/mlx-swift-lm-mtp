// Adaptive draft on/off controller: drafting only pays when the accepted tokens per
// round outweigh the drafter + multi-token-verify cost, and that balance shifts with
// content (chat acceptance ~2.2/step vs math ~3.0), context length (the drafter's own
// context attention grows), and sustained-load GPU clock drift (the ALU-bound rounds
// slow while bandwidth-bound plain decode doesn't). Rather than model any of that, the
// session MEASURES both arms — ms per committed token — and runs the faster one,
// probing the idle arm periodically so it can win back.
//
// Robustness details learned the hard way:
// - Estimates are the MEDIAN of a small sliding window, not an EWMA: the first round of
//   a request is always slow (cache allocation, prefill→decode transition) and an EWMA
//   seeded from it caused wrong early suspends that took most of a request to unwind.
// - ONE controller is shared across a model's sessions (held by the engine's DSpark
//   runtime, mutated only inside the serialized container) so the bootstrap cost is paid
//   once per process, not once per request.
//
// Output is unaffected either way: a plain step IS plain decode; spec rounds are verified.

import Foundation

public final class AdaptiveDraftController: @unchecked Sendable {
    public enum Arm: Sendable, Equatable {
        case drafting
        case plain
    }

    /// Sliding-window size per arm (median taken over this many recent samples).
    let window: Int
    /// Suspend drafting only when it is at least this much slower than plain.
    let suspendMargin: Double
    /// Resume drafting when it is within this margin of plain (ties draft).
    let resumeMargin: Double
    /// Samples from the active arm between probes of the idle arm.
    let probeCadence: Int
    /// Samples a bootstrap/probe collects from an arm before (re)deciding.
    let probeLength: Int

    public private(set) var mode: Arm = .drafting
    private var probing: (arm: Arm, remaining: Int)? = nil
    private var bootstrapped = false
    private var sinceProbe = 0
    /// Current probe interval: backs off (×2, up to cadence×4) each time a probe
    /// confirms the standing mode, so a consistently losing idle arm is probed rarely;
    /// resets on every mode switch so changed conditions are tracked closely.
    private var cadence: Int
    private(set) var specWindow: [Double] = []
    private(set) var plainWindow: [Double] = []
    private(set) var specSamples = 0
    private(set) var plainSamples = 0

    public init(
        window: Int = 6,
        suspendMargin: Double = 0.08,
        resumeMargin: Double = 0.02,
        probeCadence: Int = 32,
        probeLength: Int = 6
    ) {
        self.window = window
        self.suspendMargin = suspendMargin
        self.resumeMargin = resumeMargin
        self.probeCadence = probeCadence
        self.probeLength = max(3, probeLength)
        self.cadence = probeCadence
    }

    var specEstimate: Double? { median(specWindow) }
    var plainEstimate: Double? { median(plainWindow) }

    private func median(_ xs: [Double]) -> Double? {
        guard !xs.isEmpty else { return nil }
        let s = xs.sorted()
        return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }

    /// Which arm the next decode step should run.
    public var nextArm: Arm {
        if let probing { return probing.arm }
        if !bootstrapped {
            if specSamples < probeLength { return .drafting }
            if plainSamples < probeLength { return .plain }
        }
        return mode
    }

    /// Record one step's cost (milliseconds per committed token) for the arm that ran.
    public func record(arm: Arm, msPerToken: Double) {
        switch arm {
        case .drafting:
            specSamples += 1
            specWindow.append(msPerToken)
            if specWindow.count > window { specWindow.removeFirst() }
        case .plain:
            plainSamples += 1
            plainWindow.append(msPerToken)
            if plainWindow.count > window { plainWindow.removeFirst() }
        }

        if let p = probing, p.arm == arm {
            let remaining = p.remaining - 1
            if remaining <= 0 {
                probing = nil
                sinceProbe = 0
                let before = mode
                decide()
                // Probe confirmed the standing mode → back off; a switch resets tracking.
                cadence = mode == before ? min(cadence * 2, probeCadence * 4) : probeCadence
            } else {
                probing = (arm: p.arm, remaining: remaining)
            }
            return
        }

        if !bootstrapped {
            if specSamples >= probeLength && plainSamples >= probeLength {
                bootstrapped = true
                decide()
            }
            return
        }

        guard arm == mode else { return }
        // Re-decide on EVERY active-arm sample: a mode that starts losing (content or
        // thermal shift) is abandoned within ~a window, not at the next probe boundary.
        // Hysteresis prevents flapping; the idle arm's estimate refreshes via probes.
        let before = mode
        decide()
        if mode != before {
            cadence = probeCadence
            sinceProbe = 0
            return
        }
        sinceProbe += 1
        if sinceProbe >= cadence {
            sinceProbe = 0
            probing = (arm: mode == .drafting ? .plain : .drafting, remaining: probeLength)
        }
    }

    private func decide() {
        guard let spec = specEstimate, let plain = plainEstimate else { return }
        // Asymmetric: suspend only on a CLEAR loss, resume readily. Near-tie workloads
        // (sustained-hot math/code sit within a few % of plain) must not oscillate —
        // drafting's upside in the bursty/interactive regime is large (1.13–1.2×) while
        // a near-tie draft costs ~nothing, so ties default to drafting.
        switch mode {
        case .drafting:
            if spec > plain * (1 + suspendMargin) { mode = .plain }
        case .plain:
            if spec < plain * (1 + resumeMargin) { mode = .drafting }
        }
    }
}
