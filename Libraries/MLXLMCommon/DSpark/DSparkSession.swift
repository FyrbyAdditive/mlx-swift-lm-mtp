// A single DSpark speculative-decode request as a STEPPABLE object (same shape as
// MTPSession, so a scheduler interleaves DSpark and plain requests fairly).
//
// Round structure (reference: mlx-dspark `speculative_generate`, MIT):
//   prefill: target forward WITH taps in chunks; each chunk's taps immediately extend the
//            drafter's context caches; `pending` (first committed token) samples from the
//            last position's logits.
//   round:   1. drafter backbone over [pending] + mask block (bidirectional, full width);
//               lm_head + Markov head over only the first `cap` positions.
//            2. target verifies [pending] + draft in ONE forward (with taps).
//            3. greedy accept = longest exact-argmax prefix; commit accepted + bonus.
//            4. target cache trims the rejected suffix; the drafter context appends the
//               verify taps for the n+1 committed positions (they come FREE from verify).
//
// Invariants: `nCached` == target-cache offset == every drafter ctx cache offset;
// `pending` is committed/emitted but in NEITHER cache (the next verify feeds it).
//
// All MLX work must happen inside the owning `ModelContainer.perform`.
// `@unchecked Sendable`: only touched inside that serialized context.

import Foundation
import MLX

public final class DSparkSession: @unchecked Sendable {
    public enum Phase { case prefilling, decoding, finished }

    private let model: any DSparkTargetModel
    private let drafter: DSparkDrafter
    private let context: ModelContext
    private let parameters: GenerateParameters
    private let isGreedy: Bool
    private let temp: Float
    private let topP: Float
    private let topK: Int
    private let maxTokens: Int
    private let stopTokenIds: Set<Int>
    private let continuation: AsyncStream<Generation>.Continuation
    private let tapLayers: [Int]
    private let blockSize: Int
    private let maskTokenId: Int32
    /// How many of the block's positions are drafted + verified per round (≤ blockSize).
    /// Apple Silicon optimum is 2–3 (verify cost grows per token); the backbone still runs
    /// the full block width — it is bidirectional and was trained that way.
    private let blockCap: Int
    /// Confidence-trim threshold on cumulative survival probability; ≤0 disables.
    private let confidenceThreshold: Float

    private var modelCache: [KVCache]
    private let ctxCaches: [KVCacheSimple]

    // Prompt + prefill bookkeeping (mirrors MTPSession).
    private let promptTokens: MLXArray
    private let promptCount: Int
    private let skipPrefill: Int
    private let snapshotBlock: Int
    private let referenceTokens: [Int32]
    private let result: MTPCacheResult?
    private var prefillY: MLXArray
    private var prefillTotal: Int
    private var prefilled: Int
    private let prefillStep: Int
    private let start = Date()
    private var prefillStart: Date? = nil

    // Reasoning-token budget (same semantics as MTPSession).
    private let reasoningBudget: Int
    private var inThink: Bool
    private var reasoningTokens = 0
    private var thinkForceClosed = false
    private var thinkScanTail = ""
    private var reasoningSeconds: Double?
    private var decodeStart: Date?

    // Adaptive draft on/off: measure both arms (ms per committed token) and run the
    // faster one, probing the idle arm periodically. Kill switch MLXZ_DSPARK_ADAPTIVE=0
    // (always draft). Output is unaffected — a plain step IS plain decode. The controller
    // is SHARED across a model's sessions (injected by the scheduler; only touched inside
    // the serialized container) so bootstrap is paid once per process.
    static let adaptiveEnabled =
        ProcessInfo.processInfo.environment["MLXZ_DSPARK_ADAPTIVE"] != "0"
    private let controller: AdaptiveDraftController

    // Decode loop-carried state.
    /// The last committed+emitted token id — in neither cache; the next verify feeds it.
    private var pending: Int32 = 0
    /// Tokens committed to both caches (== target cache offset == drafter ctx offset).
    private var nCached = 0
    private var ntoks = 0
    private var stopped = false
    private var detokenizer: NaiveStreamingDetokenizer

    public private(set) var phase: Phase = .prefilling

    private var pendingSnapshots: [(tokens: [Int32], model: [KVCache], aux: [KVCache])] = []
    public func takeCapturedSnapshot() -> (tokens: [Int32], model: [KVCache], aux: [KVCache])? {
        guard !pendingSnapshots.isEmpty else { return nil }
        return pendingSnapshots.removeFirst()
    }

    public init(
        model: any DSparkTargetModel,
        drafter: DSparkDrafter,
        context: ModelContext,
        parameters: GenerateParameters,
        promptTokens: MLXArray,
        modelCache: [KVCache],
        ctxCaches: [KVCacheSimple],
        skipPrefill: Int,
        snapshotBlock: Int = 512,
        referenceTokens: [Int32] = [],
        blockCap: Int = 3,
        confidenceThreshold: Float = 0,
        adaptiveController: AdaptiveDraftController? = nil,
        reasoningBudget: Int = 0,
        stopTokenIds: Set<Int>,
        continuation: AsyncStream<Generation>.Continuation,
        result: MTPCacheResult?
    ) {
        self.controller = adaptiveController ?? AdaptiveDraftController()
        self.model = model
        self.drafter = drafter
        self.context = context
        self.parameters = parameters
        self.isGreedy = parameters.temperature == 0
        self.temp = parameters.temperature
        self.topP = parameters.topP
        self.topK = parameters.topK
        self.maxTokens = parameters.maxTokens ?? Int.max
        self.stopTokenIds = stopTokenIds
        self.continuation = continuation
        self.tapLayers = drafter.config.targetLayerIds
        self.blockSize = drafter.blockSize
        self.maskTokenId = Int32(drafter.maskTokenId)
        self.blockCap = max(1, min(blockCap, drafter.blockSize))
        self.confidenceThreshold = confidenceThreshold
        self.modelCache = modelCache
        self.ctxCaches = ctxCaches
        self.promptTokens = promptTokens
        self.promptCount = promptTokens.dim(-1)
        self.skipPrefill = skipPrefill
        self.snapshotBlock = max(1, snapshotBlock)
        self.referenceTokens = referenceTokens
        self.reasoningBudget = reasoningBudget
        self.inThink = reasoningBudget > 0 || Self.decodeDiag
        self.result = result
        self.detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)

        // Prefill covers every prompt token EXCEPT the last; the last runs through
        // `forwardWithTaps` alone at the end of prefill so `pending` samples from a
        // one-position LM head instead of a whole chunk's.
        let y0 = skipPrefill > 0 ? promptTokens[skipPrefill...] : promptTokens
        self.prefillY = y0
        self.prefillTotal = y0.dim(-1)
        self.prefilled = skipPrefill
        self.prefillStep = parameters.prefillStepSize
    }

    // MARK: - Diagnostics ([SPEC], env-gated like MTPSession's [DECODE])

    static let decodeDiag = ProcessInfo.processInfo.environment["MLXZ_DECODE_DIAG"] == "1"
    private var diagDrafterS = 0.0
    private var diagVerifyS = 0.0
    private var diagStepWallS = 0.0
    private var diagSteps = 0
    private var diagPlainSteps = 0
    private var diagDrafted = 0
    private var diagAcceptHist = [Int](repeating: 0, count: 8)
    private var diagPositionDrafted = [Int](repeating: 0, count: 8)

    // MARK: - Prefill

    public func prefillStepOnce() -> Bool {
        if prefillStart == nil { prefillStart = Date() }
        guard prefillTotal > 1 else { finishPrefill(); return false }
        var n = min(prefillStep, prefillTotal - 1)
        if let next = captureBoundaries.filter({ $0 > prefilled }).min(), next < prefilled + n {
            n = next - prefilled
        }
        let chunk = prefillY[0 ..< n]
        // Chunk taps feed the drafter context immediately (a long prompt's fused states
        // never all materialize at once); the LM head is never consumed → never computed.
        let taps = model.prefillWithTaps(
            chunk.expandedDimensions(axis: 0), cache: modelCache, tapLayers: tapLayers)
        drafter.updateContext(taps, ctxCaches: ctxCaches)
        maybeQuantizeKVCache(
            cache: &modelCache, kvBits: parameters.kvBits, kvGroupSize: parameters.kvGroupSize,
            quantizedKVStart: parameters.quantizedKVStart)
        eval(modelCache.map { $0.state }.flatMap { $0 })
        eval(ctxCaches.map { $0.state }.flatMap { $0 })
        prefillY = prefillY[n...]
        prefillTotal -= n
        prefilled += n
        if captureBoundaries.contains(prefilled), !capturedAt.contains(prefilled) {
            capturedAt.insert(prefilled)
            pendingSnapshots.append((
                Array(promptTokensArray.prefix(prefilled)),
                modelCache.map { $0.copy() },
                ctxCaches.map { $0.copy() }))
        }
        if prefillTotal <= 1 { finishPrefill(); return false }
        return true
    }

    /// Same block-aligned snapshot-boundary policy as MTPSession (warm: one at the shared
    /// prefix; cold: a few coarse boundaries).
    private static let coldCaptureCount = 4
    private lazy var captureBoundaries: Set<Int> = {
        let usable = promptCount - 1
        func align(_ x: Int) -> Int { (x / snapshotBlock) * snapshotBlock }
        func valid(_ b: Int) -> Bool { b >= 16 && b > skipPrefill && b < promptCount }

        let shared = MTPCacheReuse.commonPrefixLength(referenceTokens, promptTokensArray)
        if shared >= snapshotBlock {
            let b = align(shared)
            return valid(b) ? [b] : []
        }
        let span = usable - skipPrefill
        guard span >= snapshotBlock else {
            let b = align(usable)
            return valid(b) ? [b] : []
        }
        var set = Set<Int>()
        for i in 1 ... Self.coldCaptureCount {
            let b = align(skipPrefill + span * i / (Self.coldCaptureCount + 1))
            if valid(b) { set.insert(b) }
        }
        let tail = align(usable)
        if valid(tail) { set.insert(tail) }
        return set
    }()

    private lazy var promptTokensArray: [Int32] = promptTokens.asArray(Int32.self)
    private var capturedAt: Set<Int> = []

    private func finishPrefill() {
        // Run the LAST prompt token alone: its logits seed `pending`, its taps complete the
        // drafter context over the full prompt.
        let last = promptTokens[(promptCount - 1)...].expandedDimensions(axis: 0)
        let (logits, taps) = model.forwardWithTaps(
            last, cache: modelCache, tapLayers: tapLayers)
        drafter.updateContext(taps, ctxCaches: ctxCaches)
        let tok = sampleRow(logits[0, -1, 0...])
        eval(tok)
        result?.prefillSeconds = Date().timeIntervalSince(prefillStart ?? start)
        nCached = promptCount
        pending = tok.item(Int32.self)
        phase = .decoding
        if Self.decodeDiag || reasoningBudget > 0 { decodeStart = Date() }
        if emitToken(Int(pending)) || ntoks >= maxTokens { finishDecode() }
    }

    private func sampleRow(_ logits: MLXArray) -> MLXArray {
        if isGreedy { return argMax(logits, axis: -1) }
        let p = SpeculativeVerifier.truncateProbs(
            softmax(logits.asType(.float32) * (1 / temp), axis: -1), topP: topP, topK: topK)
        return categorical(log(p + 1e-20))
    }

    // MARK: - Decode

    /// One full DSpark round: draft → verify → accept → commit. Returns true while more
    /// decode remains. Must be called inside the container.
    public func decodeStepOnce() -> Bool {
        if stopped || ntoks >= maxTokens { finishDecode(); return false }

        if reasoningBudget > 0, inThink, !thinkForceClosed, reasoningTokens >= reasoningBudget {
            forceCloseThink()
            if stopped || ntoks >= maxTokens { finishDecode(); return false }
        }

        // Adaptive arm selection: run plain decode steps when drafting isn't currently
        // paying (the controller measures both arms and probes periodically). Selected
        // BEFORE the diag counters so [DECODE]/[SPEC] stats describe spec rounds only
        // (plain steps are counted separately).
        if Self.adaptiveEnabled, controller.nextArm == .plain {
            return plainStepsOnce(maxSteps: 6)
        }
        if Self.decodeDiag { diagSteps += 1 }
        let stepT0 = Self.decodeDiag ? Date() : nil
        defer { if let stepT0 { diagStepWallS += Date().timeIntervalSince(stepT0) } }
        let roundT0 = Self.adaptiveEnabled ? Date() : nil

        // ---- 1. draft a block (backbone full-width; heads over the first `cap` only) ----
        let t0 = Self.decodeDiag ? Date() : nil
        let cap = blockCap
        var blockIds = [pending]
        blockIds.append(contentsOf: [Int32](repeating: maskTokenId, count: blockSize - 1))
        let noise = drafter.embed(MLXArray(blockIds).expandedDimensions(axis: 0))
        let blockHidden = drafter.backbone(noise, blockOffset: nCached, ctxCaches: ctxCaches)
        let headHidden = blockHidden[0..., 0 ..< cap, 0...]
        let baseLogits = drafter.computeLogits(headHidden)[0]  // [cap, V]

        // Greedy without trimming takes the FUSED path: the drafted tokens never
        // round-trip to the CPU before verify, and the whole round syncs once.
        var draft: [Int32] = []
        var draftGPU: MLXArray? = nil
        var qProbs: MLXArray? = nil
        if isGreedy && !(confidenceThreshold > 0 && drafter.hasConfidenceHead) {
            draftGPU = drafter.sampleBlock(baseLogits, firstPrevToken: pending)
        } else {
            if isGreedy {
                let arr = drafter.sampleBlock(baseLogits, firstPrevToken: pending)
                eval(arr)
                draft = arr.asArray(Int32.self)
            } else {
                let (tokens, probs) = drafter.sampleBlockProbs(
                    baseLogits, firstPrevToken: pending,
                    temperature: temp, topP: topP, topK: topK)
                eval(tokens, probs)
                draft = tokens.asArray(Int32.self)
                qProbs = probs
            }
            if confidenceThreshold > 0, drafter.hasConfidenceHead {
                // Confidence trim (paper Eq. 7–8): keep the longest prefix whose
                // cumulative survival stays above threshold; always propose ≥1 token.
                let prev = cap > 1
                    ? concatenated([MLXArray([pending]), MLXArray(Array(draft.dropLast()))])
                    : MLXArray([pending])
                let conf = sigmoid(
                    drafter.confidenceLogits(headHidden[0], prevTokenIds: prev)!)
                eval(conf)
                let keep = SpeculativeVerifier.confidenceKeepCount(
                    survival: conf.asArray(Float.self), threshold: confidenceThreshold)
                draft = Array(draft.prefix(max(1, keep)))
            }
        }
        if let t0 {
            if let draftGPU { eval(draftGPU) }
            diagDrafterS += Date().timeIntervalSince(t0)
        }

        // ---- 2. verify [pending] + draft in ONE target forward (with taps) ----
        let tv = Self.decodeDiag ? Date() : nil
        let verifyDraft = draftGPU ?? MLXArray(draft)
        let verifyLen = verifyDraft.dim(0)
        let verifyIds = concatenated([MLXArray([pending]), verifyDraft.asType(.int32)])
            .expandedDimensions(axis: 0)
        let (vLogits, vTaps) = model.forwardWithTaps(
            verifyIds, cache: modelCache, tapLayers: tapLayers)

        // ---- 3. accept ----
        let n: Int
        var committed: [Int32]
        if isGreedy {
            // Longest exact-argmax prefix, computed in-graph (single sync for the round).
            let targetArgmax = argMax(vLogits[0], axis: -1).asType(.int32)  // [1+L]
            let match = (verifyDraft.asType(.int32) .== targetArgmax[0 ..< verifyLen])
                .asType(.int32)
            let nArr = cumprod(match).sum()
            eval(nArr, targetArgmax)
            n = nArr.item(Int.self)
            let tt = targetArgmax.asArray(Int32.self)
            let draftHost = draft.isEmpty ? verifyDraft.asArray(Int32.self) : draft
            committed = Array(draftHost.prefix(n))
            committed.append(tt[n])  // bonus (n == L) or correction (n < L)
        } else {
            // Speculative sampling: accept w.p. min(1, p/q); residual-resample on the
            // first reject; bonus from the target when all accept. p gets the SAME
            // temperature + top-p/top-k treatment as q (losslessness-critical).
            let p = SpeculativeVerifier.truncateProbs(
                softmax(vLogits[0].asType(.float32) * (1 / temp), axis: -1),
                topP: topP, topK: topK)
            let uniforms = uniform(Float(0) ..< Float(1), [verifyLen])
            let (accepted, replacement) = SpeculativeVerifier.sampledAccept(
                targetProbs: p, draftTokens: draft, draftProbs: qProbs!, uniforms: uniforms)
            n = accepted
            committed = Array(draft.prefix(n))
            committed.append(replacement)
        }
        if let tv { diagVerifyS += Date().timeIntervalSince(tv) }

        if Self.decodeDiag {
            diagDrafted += verifyLen
            for i in 0 ..< min(verifyLen, 8) {
                diagPositionDrafted[i] += 1
                if i < n { diagAcceptHist[i] += 1 }
            }
        }

        // ---- 4. commit: trim the rejected target suffix; extend drafter ctx from verify taps ----
        let trim = verifyLen - n
        if trim > 0 { trimPromptCache(modelCache, numTokens: trim) }
        drafter.updateContext(vTaps[0..., 0 ..< (n + 1), 0...], ctxCaches: ctxCaches)
        nCached += n + 1
        asyncEval(ctxCaches.map { $0.state }.flatMap { $0 })

        if let roundT0 {
            controller.record(
                arm: .drafting,
                msPerToken: Date().timeIntervalSince(roundT0) * 1000 / Double(committed.count))
        }

        // ---- 5. emit committed tokens (stop token ends mid-block; it is not emitted) ----
        var lastEmitted: Int32? = nil
        for tok in committed {
            if emitToken(Int(tok)) { finishDecode(); return false }
            lastEmitted = tok
            if ntoks >= maxTokens { finishDecode(); return false }
        }
        pending = lastEmitted ?? committed.last!
        return true
    }

    /// A BATCH of plain decode steps (M=1 target forwards — identical math to
    /// non-speculative decode), used while the controller has drafting suspended. The
    /// forwards chain LAZILY (each step's sampled token feeds the next without a host
    /// sync) with one sync for the whole batch — the same pipelining the real plain path
    /// gets — so the controller's plain estimate reflects the true alternative, not a
    /// sync-per-token strawman. Taps still extend the drafter's context so drafting can
    /// resume in sync at any batch boundary.
    private func plainStepsOnce(maxSteps: Int) -> Bool {
        let t0 = Date()
        var tokens: [MLXArray] = []
        var tapsList: [MLXArray] = []
        var input = MLXArray([pending]).expandedDimensions(axis: 0)
        for _ in 0 ..< maxSteps {
            let (logits, taps) = model.forwardWithTaps(
                input, cache: modelCache, tapLayers: tapLayers)
            let tok = sampleRow(logits[0, -1, 0...])
            asyncEval(tok)
            tokens.append(tok)
            tapsList.append(taps)
            input = tok.reshaped([1, 1]).asType(.int32)
        }
        drafter.updateContext(concatenated(tapsList, axis: 1), ctxCaches: ctxCaches)
        asyncEval(ctxCaches.map { $0.state }.flatMap { $0 })
        nCached += maxSteps
        let ids = tokens.map { $0.item(Int32.self) }  // one pipeline sync for the batch
        if Self.decodeDiag { diagPlainSteps += maxSteps }
        let msPerToken = Date().timeIntervalSince(t0) * 1000 / Double(maxSteps)
        for _ in 0 ..< maxSteps {
            controller.record(arm: .plain, msPerToken: msPerToken)
        }
        // Emit with stop/budget handling. On early stop the batch's remaining inputs sit
        // as junk in the caches — harmless, the session is finished (DSpark sessions
        // never share a live cache; prefix snapshots were captured during prefill).
        var lastEmitted: Int32? = nil
        for id in ids {
            if emitToken(Int(id)) { finishDecode(); return false }
            lastEmitted = id
            if ntoks >= maxTokens { finishDecode(); return false }
        }
        pending = lastEmitted ?? ids.last!
        return true
    }

    /// Emit one token's text; true if it was a stop token (not emitted; caller stops).
    private func emitToken(_ id: Int) -> Bool {
        if stopTokenIds.contains(id) { stopped = true; return true }
        detokenizer.append(token: id)
        if let chunk = detokenizer.next() {
            if inThink {
                reasoningTokens += 1
                thinkScanTail += chunk
                if thinkScanTail.contains("</think>") {
                    inThink = false
                    thinkScanTail = ""
                    if let s = decodeStart { reasoningSeconds = Date().timeIntervalSince(s) }
                } else if thinkScanTail.count > 16 {
                    thinkScanTail = String(thinkScanTail.suffix(16))
                }
            }
            continuation.yield(.chunk(chunk))
        }
        ntoks += 1
        return false
    }

    /// Force the model out of `<think>` by feeding a graceful transition (MTPSession's
    /// approach). The transition ids advance the target cache AND the drafter context
    /// (taps come from the same forward); the LAST id becomes `pending` (fed to the next
    /// verify), so all but the last enter the caches.
    private func forceCloseThink() {
        thinkForceClosed = true
        inThink = false
        if let s = decodeStart { reasoningSeconds = Date().timeIntervalSince(s) }
        let transition = "\nConsidering the limited time, I'll answer based on the above.\n</think>\n\n"
        let ids = context.tokenizer.encode(text: transition).map { Int32($0) }
        guard !ids.isEmpty else { return }
        // [pending] + all-but-last transition ids enter the caches; last becomes pending.
        let feed = [pending] + ids.dropLast()
        let (_, taps) = model.forwardWithTaps(
            MLXArray(feed).expandedDimensions(axis: 0), cache: modelCache, tapLayers: tapLayers)
        drafter.updateContext(taps, ctxCaches: ctxCaches)
        eval(ctxCaches.map { $0.state }.flatMap { $0 })
        nCached += feed.count
        for id in ids { detokenizer.append(token: Int(id)) }
        if let chunk = detokenizer.next() { continuation.yield(.chunk(chunk)) }
        pending = ids[ids.count - 1]
    }

    private func finishDecode() {
        guard phase != .finished else { return }
        phase = .finished
        if let tail = detokenizer.next(), !tail.isEmpty { continuation.yield(.chunk(tail)) }
        result?.promptTokens = promptTokensArray
        let elapsed = Date().timeIntervalSince(start)
        let info = GenerateCompletionInfo(
            promptTokenCount: promptCount, generationTokenCount: ntoks,
            promptTime: 0, generationTime: elapsed,
            stopReason: (!stopped && ntoks >= maxTokens) ? .length : .stop)
        continuation.yield(.info(info))
        continuation.finish()
        if Self.decodeDiag, diagSteps > 0 {
            let steps = Double(diagSteps)
            let drafterMs = String(format: "%.1f", diagDrafterS / steps * 1000)
            let verifyMs = String(format: "%.1f", diagVerifyS / steps * 1000)
            let wall = String(format: "%.1f", diagStepWallS / steps * 1000)
            // Spec rounds only: each plain step commits exactly one token.
            let accPerStep = String(format: "%.2f", Double(ntoks - diagPlainSteps) / steps)
            let hist = (0 ..< blockCap).map { i in
                diagPositionDrafted[i] > 0
                    ? String(format: "%.2f", Double(diagAcceptHist[i]) / Double(diagPositionDrafted[i]))
                    : "-"
            }.joined(separator: ",")
            FileHandle.standardError.write(Data(
                "[DECODE] ctx=\(promptCount) steps=\(diagSteps) gen=\(ntoks) drafter=\(drafterMs)ms verify=\(verifyMs)ms STEPWALL=\(wall)ms total=\(String(format: "%.1f", elapsed))s\n".utf8))
            let estS = controller.specEstimate.map { String(format: "%.1f", $0) } ?? "-"
            let estP = controller.plainEstimate.map { String(format: "%.1f", $0) } ?? "-"
            FileHandle.standardError.write(Data(
                "[SPEC] steps=\(diagSteps) plainSteps=\(diagPlainSteps) drafted=\(diagDrafted) cap=\(blockCap) acc/step=\(accPerStep) accHist=[\(hist)] adaptive=\(Self.adaptiveEnabled ? "on" : "off") mode=\(controller.mode == .drafting ? "draft" : "plain") spec=\(estS)ms plain=\(estP)ms\n".utf8))
        }
    }

    public func cancel() {
        guard phase != .finished else { return }
        phase = .finished
        continuation.finish()
    }
}
