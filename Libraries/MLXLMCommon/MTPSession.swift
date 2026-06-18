// A single MTP self-speculative-decode request as a STEPPABLE object, so a scheduler can interleave
// many requests fairly (one decode step each, round-robin) instead of running each to completion
// while others wait. The per-step math is a faithful relocation of `mtpGenerate`'s decode loop body
// (see MTPGenerate.swift) — output for a single session is byte-identical; interleaving only changes
// the ORDER in which different sessions' steps run on the (serialized) GPU, never a session's own
// token sequence. Greedy ⇒ token-identical; sampling ⇒ same marginal (output is backbone-only;
// MTP drafts are verified and replaced on mismatch).
//
// All MLX work must happen inside the owning `ModelContainer.perform` (the scheduler guarantees
// this). `@unchecked Sendable`: only touched inside that serialized context.

import Foundation
import MLX

public final class MTPSession: @unchecked Sendable {
    public enum Phase { case prefilling, decoding, finished }

    private let model: any MTPSpeculativeModel
    private let context: ModelContext
    private let parameters: GenerateParameters
    private let temp: Float
    private let isGreedy: Bool
    private let maxTokens: Int
    private let stopTokenIds: Set<Int>
    private let continuation: AsyncStream<Generation>.Continuation

    // Caches (working copies; the scheduler/engine handles snapshot reuse via `result`).
    private var modelCache: [KVCache]  // var: KV-quantized in place during prefill when kvBits set
    private let mtpCache: [KVCache]

    // Prompt + prefill bookkeeping.
    private let promptTokens: MLXArray
    private let promptCount: Int
    private let skipPrefill: Int
    private let snapshotAt: Int
    /// Token granularity for block-aligned prefix-snapshot capture (configurable; default 512).
    private let snapshotBlock: Int
    /// The most-recent cached prompt's tokens, used to place this request's single snapshot at the
    /// block-aligned length of the SHARED prefix (the recurring system prompt). Empty if none cached.
    private let referenceTokens: [Int32]
    private let result: MTPCacheResult?

    // --- Reasoning-token budget (hard cap on the <think> block) ---
    /// Max reasoning tokens before we FORCE-close `<think>`. ≤0 = uncapped.
    private let reasoningBudget: Int
    /// The model starts inside the template-pre-opened `<think>` block. We count tokens until we see
    /// `</think>` in the decoded text (model self-closed) or we force-close at the budget.
    private var inThink: Bool
    private var reasoningTokens = 0
    private var thinkForceClosed = false
    /// Tail of decoded text, scanned for a self-emitted `</think>` so we stop counting.
    private var thinkScanTail = ""
    /// Wall-clock spent inside the `<think>` block (for the env-gated telemetry). Stamped when the
    /// block closes (self-closed or force-closed); the start is `decodeStart`.
    private var reasoningSeconds: Double?
    private var decodeStart: Date?
    private var prefillY: MLXArray
    private var prefillTotal: Int
    private var prefilled: Int
    private let prefillStep: Int
    private let start = Date()
    private var prefillStart: Date? = nil

    // Decode loop-carried state (mirrors the locals in mtpGenerate's while loop).
    private var y: MLXArray
    private var draftTok: MLXArray? = nil
    private var draftLp = MLXArray(0)
    private var draftAccept = MLXArray(0)
    private var ntoks = 0
    private var stopped = false
    private var detokenizer: NaiveStreamingDetokenizer

    public private(set) var phase: Phase = .prefilling

    /// A prefix snapshot captured during prefill, to be stored for cross-request reuse. Set once
    /// when prefill reaches `snapshotAt`; the scheduler takes it (clearing it) and inserts into the
    /// LRU. `(tokens, modelCache, mtpCache)`.
    /// Snapshots captured during prefill, awaiting insertion into the prefix LRU by the scheduler.
    /// Prefill captures one at each block boundary (so cross-request reuse tracks the shared-prefix
    /// boundary, not an arbitrary tail) plus the final tail; the scheduler drains this each step.
    private var pendingSnapshots: [(tokens: [Int32], model: [KVCache], mtp: [KVCache])] = []
    public func takeCapturedSnapshot() -> (tokens: [Int32], model: [KVCache], mtp: [KVCache])? {
        guard !pendingSnapshots.isEmpty else { return nil }
        return pendingSnapshots.removeFirst()
    }

    public init(
        model: any MTPSpeculativeModel,
        context: ModelContext,
        parameters: GenerateParameters,
        promptTokens: MLXArray,
        modelCache: [KVCache],
        mtpCache: [KVCache],
        skipPrefill: Int,
        snapshotAt: Int,
        snapshotBlock: Int = 512,
        referenceTokens: [Int32] = [],
        reasoningBudget: Int = 0,
        stopTokenIds: Set<Int>,
        continuation: AsyncStream<Generation>.Continuation,
        result: MTPCacheResult?
    ) {
        self.model = model
        self.context = context
        self.parameters = parameters
        self.temp = parameters.temperature
        self.isGreedy = parameters.temperature == 0
        self.maxTokens = parameters.maxTokens ?? Int.max
        self.stopTokenIds = stopTokenIds
        self.continuation = continuation
        self.modelCache = modelCache
        self.mtpCache = mtpCache
        self.promptTokens = promptTokens
        self.promptCount = promptTokens.dim(-1)
        self.skipPrefill = skipPrefill
        self.snapshotAt = snapshotAt
        self.snapshotBlock = max(1, snapshotBlock)
        self.referenceTokens = referenceTokens
        self.reasoningBudget = reasoningBudget
        // Track reasoning when a budget is set (to enforce it) OR when the decode diagnostic is on
        // (to MEASURE the reasoning cost even uncapped). The Qwen template pre-opens `<think>` on the
        // assistant turn, so decode starts inside the reasoning block.
        self.inThink = reasoningBudget > 0 || Self.decodeDiag
        self.result = result
        self.detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)

        let y0 = skipPrefill > 0 ? promptTokens[skipPrefill...] : promptTokens
        self.prefillY = y0
        self.prefillTotal = y0.dim(-1)
        self.prefilled = skipPrefill
        self.prefillStep = parameters.prefillStepSize
        self.y = promptTokens  // overwritten at end of prefill
    }

    // MARK: - Sampling/step helpers (faithful copies of mtpGenerate's closures)

    private func processAndSample(_ logits: MLXArray)
        -> (token: MLXArray, logprobs: MLXArray, lpAccept: MLXArray)
    {
        let logprobs = logits - logSumExp(logits, axis: -1, keepDims: true)
        if isGreedy { return (argMax(logprobs, axis: -1), logprobs, logprobs) }
        let scaled = logprobs * (1 / temp)
        let token = categorical(scaled)
        let lpAccept = scaled - logSumExp(scaled, axis: -1, keepDims: true)
        return (token, logprobs, lpAccept)
    }

    // DECODE DIAGNOSTIC (gated by MLXZ_DECODE_DIAG=1): per-phase wall-time accumulators + step count.
    static let decodeDiag = ProcessInfo.processInfo.environment["MLXZ_DECODE_DIAG"] == "1"
    private var diagBackboneS = 0.0
    private var diagMTPS = 0.0
    private var diagStepWallS = 0.0
    private var diagSteps = 0

    private func stepBackbone(_ y: MLXArray, nPredict: Int, nConfirmed: Int)
        -> (toks: [MLXArray], lps: [MLXArray], accept: [MLXArray], hidden: MLXArray)
    {
        let t0 = Self.decodeDiag ? Date() : nil
        let (logitsAll, hidden) = model.backboneWithHidden(
            y.expandedDimensions(axis: 0), cache: modelCache, nConfirmed: nConfirmed)
        if let t0 { eval(logitsAll, hidden); diagBackboneS += Date().timeIntervalSince(t0) }
        let logits = logitsAll[0..., (logitsAll.dim(1) - nPredict)..., 0...]
        var toks: [MLXArray] = []
        var lps: [MLXArray] = []
        var accept: [MLXArray] = []
        for i in 0 ..< nPredict {
            let (t, lp, alp) = processAndSample(logits[0, i, 0...])
            toks.append(t); lps.append(lp); accept.append(alp)
        }
        return (toks, lps, accept, hidden)
    }

    private func stepMTP(
        hiddenLast: MLXArray, mainTok: MLXArray,
        cacheCommit: (hidden: MLXArray, tok: MLXArray)? = nil
    ) -> (token: MLXArray, logprobs: MLXArray, accept: MLXArray) {
        let hiddenIn: MLXArray
        let nextIds: MLXArray
        if let cacheCommit {
            hiddenIn = concatenated([cacheCommit.hidden, hiddenLast], axis: 1)
            nextIds = concatenated(
                [cacheCommit.tok.reshaped(1, 1), mainTok.reshaped(1, 1)], axis: 1)
        } else {
            hiddenIn = hiddenLast
            nextIds = mainTok.reshaped(1, 1)
        }
        let tmtp = Self.decodeDiag ? Date() : nil
        let mtpLogitsAll = model.mtpForward(
            hiddenIn, nextTokenIds: nextIds, cache: mtpCache.map { $0 as KVCache? })
        if let tmtp { eval(mtpLogitsAll); diagMTPS += Date().timeIntervalSince(tmtp) }
        let mtpLogits = mtpLogitsAll[0..., -1, 0...].squeezed(axis: 0)
        let (t, lp, alp) = processAndSample(mtpLogits)
        return (t, lp, alp)
    }

    private func clearRollback() {
        for c in modelCache where c is MambaCache { (c as! MambaCache).rollbackState = nil }
    }

    private func rollbackDraft() {
        for c in modelCache {
            if let m = c as? MambaCache, let snap = m.rollbackState {
                m.state = [snap.0 ?? m.state[0], snap.1 ?? m.state[1]]
                m.rollbackState = nil
            } else if c.isTrimmable {
                _ = c.trim(1)
            }
        }
    }

    /// Emit a token's text. Returns true if it was a stop token (EOS): caller stops and the token is
    /// NOT detokenized/yielded (matches the standard loop's includeStopToken:false default).
    @discardableResult
    private func emit(_ tokenArray: MLXArray, _ lp: MLXArray) -> Bool {
        let id = tokenArray.item(Int.self)
        if stopTokenIds.contains(id) { stopped = true; return true }
        detokenizer.append(token: id)
        if let chunk = detokenizer.next() {
            // Reasoning-budget tracking: count tokens while inside the pre-opened `<think>` block;
            // if the model self-emits `</think>`, stop counting (it finished reasoning on its own).
            if inThink {
                reasoningTokens += 1
                thinkScanTail += chunk
                if thinkScanTail.contains("</think>") {
                    inThink = false; thinkScanTail = ""
                    if let s = decodeStart { reasoningSeconds = Date().timeIntervalSince(s) }
                } else if thinkScanTail.count > 16 {
                    thinkScanTail = String(thinkScanTail.suffix(16))  // keep enough to span the tag
                }
            }
            continuation.yield(.chunk(chunk))
        }
        _ = lp
        ntoks += 1
        return false
    }

    // MARK: - Steppable prefill + decode

    /// Run ONE prefill chunk. Returns true while more prefill remains. Must be called inside the
    /// container. When prefill completes, transitions to `.decoding`.
    ///
    /// Captures EXACTLY ONE prefix snapshot per request, at `captureBoundary` — the largest
    /// `snapshotBlock`-aligned position < promptLen. Block-aligning the single snapshot lets a FUTURE
    /// request that shares a prefix (same system prompt, different user text) reuse it, while keeping
    /// the cost to one copy per request. (An earlier version captured at EVERY boundary — ~100 copies
    /// of the growing KV for a 55k prompt — which blew RAM to tens of GB; that was the regression.)
    public func prefillStepOnce() -> Bool {
        if prefillStart == nil { prefillStart = Date() }
        guard prefillTotal > 1 else { finishPrefill(); return false }
        var n = min(prefillStep, prefillTotal - 1)
        // Stop the chunk exactly on the next capture boundary so the cache encodes exactly that many
        // tokens when we snapshot.
        if let next = captureBoundaries.filter({ $0 > prefilled }).min(), next < prefilled + n {
            n = next - prefilled
        }
        let chunk = prefillY[0 ..< n]
        // Prefill only needs the cache warmed; the logits are discarded. Use prefillBackbone, which
        // skips the LM head (a wasted [1, chunk, vocab] matmul per chunk). MTP head also not warmed
        // (cold MTP cache self-warms in decode; output is backbone-only — see MTPGenerate.swift).
        _ = model.prefillBackbone(chunk.expandedDimensions(axis: 0), cache: modelCache)
        // Quantize the full-attention layers' KV in place once past `quantizedKVStart` (no-op when
        // kvBits is nil; skips the GatedDeltaNet MambaCache layers). Shrinks the live cache AND the
        // prefix snapshot taken below. QuantizedKVCache supports copy()/trim(), so snapshot reuse and
        // MTP draft rollback still hold.
        maybeQuantizeKVCache(
            cache: &modelCache, kvBits: parameters.kvBits, kvGroupSize: parameters.kvGroupSize,
            quantizedKVStart: parameters.quantizedKVStart)
        eval(modelCache.map { $0.state }.flatMap { $0 })
        prefillY = prefillY[n...]
        prefillTotal -= n
        prefilled += n
        // Snapshot whenever we land exactly on a planned capture boundary (each taken once). Bounded
        // count (≤ captureBoundaries.count); the byte-capped LRU bounds total memory.
        if captureBoundaries.contains(prefilled), !capturedAt.contains(prefilled) {
            capturedAt.insert(prefilled)
            pendingSnapshots.append((
                Array(promptTokensArray.prefix(prefilled)),
                modelCache.map { $0.copy() }, mtpCache.map { $0.copy() }))
        }
        if prefillTotal <= 1 { finishPrefill(); return false }
        return true
    }

    /// The capped, ascending list of token positions to snapshot, each a block-aligned multiple of
    /// `snapshotBlock`, ≥ minReuse, > skipPrefill, < promptLen (a non-empty suffix must remain —
    /// `reuseCount` requires the snapshot be strictly shorter than any prompt sharing it).
    ///
    /// - WARM (we have a `referenceTokens`): ONE snapshot at the block-aligned length of the prefix
    ///   SHARED with the previous prompt — the stable region (system prompt) that recurs. Minimal
    ///   memory, exactly where the next turn will reuse.
    /// - COLD (no reference, first request of a family): we don't yet know where future turns
    ///   diverge, so capture a SMALL bounded set of coarse boundaries spread across the prompt
    ///   (≈ `coldCaptureCount` of them). A diverging follow-up then finds an aligned boundary INSIDE
    ///   the shared region and reuses it — fixing the "turn 2 re-prefills the whole prompt" miss.
    ///   Bounded count + byte-capped LRU keep RAM in check.
    private static let coldCaptureCount = 4
    private lazy var captureBoundaries: Set<Int> = {
        let usable = promptCount - 1  // leave ≥1 token suffix
        func align(_ x: Int) -> Int { (x / snapshotBlock) * snapshotBlock }
        func valid(_ b: Int) -> Bool { b >= 16 && b > skipPrefill && b < promptCount }

        let shared = MTPCacheReuse.commonPrefixLength(referenceTokens, promptTokensArray)
        if shared >= snapshotBlock {
            // Warm: single snapshot at the known shared boundary.
            let b = align(shared)
            return valid(b) ? [b] : []
        }
        // Cold: spread a few coarse boundaries across [skipPrefill, usable).
        let span = usable - skipPrefill
        guard span >= snapshotBlock else {
            let b = align(usable); return valid(b) ? [b] : []
        }
        var set = Set<Int>()
        for i in 1 ... Self.coldCaptureCount {
            let pos = skipPrefill + span * i / (Self.coldCaptureCount + 1)
            let b = align(pos)
            if valid(b) { set.insert(b) }
        }
        // Always include the largest aligned boundary too (so the next IDENTICAL prompt reuses fully).
        let tail = align(usable)
        if valid(tail) { set.insert(tail) }
        return set
    }()

    /// Cached host copy of the prompt tokens (avoids re-reading the MLXArray each boundary check).
    private lazy var promptTokensArray: [Int32] = promptTokens.asArray(Int32.self)

    private var capturedAt: Set<Int> = []

    /// Force the model out of the `<think>` block by feeding a graceful transition string into its
    /// context (Qwen's reference approach: a sentence that wraps up + `</think>`), so the model's next
    /// generated tokens are the answer rather than more reasoning. Runs the transition tokens through
    /// the backbone to advance the KV/conv caches, emits the transition text to the client, resets the
    /// pending MTP draft, and points `y` at the last transition token. Called at most once.
    private func forceCloseThink() {
        thinkForceClosed = true
        inThink = false
        if let s = decodeStart { reasoningSeconds = Date().timeIntervalSince(s) }
        let transition = "\nConsidering the limited time, I'll answer based on the above.\n</think>\n\n"
        let ids = context.tokenizer.encode(text: transition).map { Int32($0) }
        guard !ids.isEmpty else { return }
        let arr = MLXArray(ids)
        // Advance the backbone over the transition (nConfirmed=0: no draft split). Discard logits.
        _ = stepBackbone(arr, nPredict: 1, nConfirmed: 0)
        clearRollback()
        // Emit the transition text so the SSE stream sees the think block close (ThinkParser routes
        // the part up to `</think>` as reasoning, the rest as the answer's lead-in).
        for id in ids { detokenizer.append(token: Int(id)) }
        if let chunk = detokenizer.next() { continuation.yield(.chunk(chunk)) }
        // Resume normal decode from the last transition token; rebuild the draft next step.
        draftTok = nil
        y = MLXArray([ids[ids.count - 1]]).asType(.uint32)
    }

    private func finishPrefill() {
        result?.prefillSeconds = Date().timeIntervalSince(prefillStart ?? start)
        // Decode begins from the final prompt token (the standard loop sets y to the last token via
        // the prompt; the original fed the whole prompt then advanced — here the cache is warmed by
        // prefill of all but the last token, and decode's first backbone step consumes the last one).
        y = promptTokens[(promptCount - 1)...].asType(.uint32)
        phase = .decoding
        if Self.decodeDiag || reasoningBudget > 0 { decodeStart = Date() }
    }

    /// Perform ONE decode iteration (the body of mtpGenerate's `while`). Returns true while more
    /// decode remains; false when finished (stop/EOS/maxTokens) — at which point the stream is
    /// finished and `result` populated. Must be called inside the container.
    public func decodeStepOnce() -> Bool {
        if stopped || ntoks >= maxTokens { finishDecode(); return false }
        if Self.decodeDiag { diagSteps += 1 }
        let stepT0 = Self.decodeDiag ? Date() : nil
        defer { if let stepT0 { diagStepWallS += Date().timeIntervalSince(stepT0) } }

        // HARD reasoning-token cap: if the model is still reasoning past the budget, force-close the
        // `<think>` block by feeding a graceful transition into its context, so its next tokens are
        // the ANSWER. We don't trust the model to self-limit (it usually won't). Done once.
        if reasoningBudget > 0, inThink, !thinkForceClosed, reasoningTokens >= reasoningBudget {
            forceCloseThink()
        }

        if draftTok == nil {
            let (toks, lps, _, hidden) = stepBackbone(y, nPredict: 1, nConfirmed: 0)
            let mainTok = toks[0]
            let hiddenAtMain = hidden[0..., (hidden.dim(1) - 1)..., 0...]
            let d = stepMTP(hiddenLast: hiddenAtMain, mainTok: mainTok)
            asyncEval(mainTok, d.token)
            if emit(mainTok, lps[0]) { finishDecode(); return false }
            if ntoks >= maxTokens { finishDecode(); return false }
            draftTok = d.token; draftLp = d.logprobs; draftAccept = d.accept
            y = mainTok.reshaped(1).asType(.uint32)
        } else {
            let dtok = draftTok!
            let yWithDraft = concatenated([y, dtok.reshaped(1).asType(.uint32)], axis: 0)
            let (toks, lps, accept, hidden) = stepBackbone(yWithDraft, nPredict: 2, nConfirmed: 1)
            let u = uniform(Float(0) ..< Float(1), [1])
            eval(toks[0], toks[1], dtok, u)

            let verifyPred = toks[0], bonusTok = toks[1]
            let verifyLp = lps[0], bonusLp = lps[1]
            let verifyAccept = accept[0]
            let draftId = dtok.item(Int.self)

            let accepted: Bool
            if isGreedy {
                accepted = verifyPred.item(Int.self) == draftId
            } else {
                let logAccept = (verifyAccept[draftId] - draftAccept[draftId]).item(Float.self)
                accepted = logAccept >= 0 || u.item(Float.self) < exp(logAccept)
            }

            let hiddenAtConfirmed = hidden[0..., 0 ..< 1, 0...]
            let hiddenAtDraft = hidden[0..., 1 ..< 2, 0...]

            if accepted {
                clearRollback()
                let d = stepMTP(
                    hiddenLast: hiddenAtDraft, mainTok: bonusTok,
                    cacheCommit: (hiddenAtConfirmed, dtok))
                asyncEval(d.token)
                if emit(dtok, draftLp) { finishDecode(); return false }
                if ntoks >= maxTokens { finishDecode(); return false }
                if emit(bonusTok, bonusLp) { finishDecode(); return false }
                if ntoks >= maxTokens { finishDecode(); return false }
                draftTok = d.token; draftLp = d.logprobs; draftAccept = d.accept
                y = bonusTok.reshaped(1).asType(.uint32)
            } else {
                rollbackDraft()
                var verifyId = verifyPred.item(Int.self)
                if !isGreedy {
                    let pTarget = exp(verifyAccept)
                    let pDraft = exp(draftAccept)
                    let residual = maximum(pTarget - pDraft, MLXArray(Float(0)))
                    let z = residual.sum(keepDims: true)
                    let dist = MLX.where(z .> 0, residual, pTarget)
                    verifyId = categorical(MLX.log(dist).reshaped(1, -1)).item(Int.self)
                }
                let vtok = MLXArray(UInt32(verifyId))
                let d = stepMTP(hiddenLast: hiddenAtConfirmed, mainTok: vtok)
                asyncEval(d.token)
                if emit(
                    verifyPred.dtype == .uint32 ? MLXArray(UInt32(verifyId)) : MLXArray(verifyId),
                    verifyLp)
                {
                    finishDecode(); return false
                }
                if ntoks >= maxTokens { finishDecode(); return false }
                draftTok = d.token; draftLp = d.logprobs; draftAccept = d.accept
                y = vtok.reshaped(1)
            }
        }
        return true
    }

    private func finishDecode() {
        guard phase != .finished else { return }
        phase = .finished
        if let tail = detokenizer.next(), !tail.isEmpty { continuation.yield(.chunk(tail)) }
        // Prefix snapshots are surfaced via `pendingSnapshots`/`takeCapturedSnapshot` and inserted
        // into the LRU by the scheduler — not via `result`. `result` carries only timing/prompt info.
        result?.promptTokens = promptTokens.asArray(Int32.self)
        let elapsed = Date().timeIntervalSince(start)
        let info = GenerateCompletionInfo(
            promptTokenCount: promptCount, generationTokenCount: ntoks,
            promptTime: 0, generationTime: elapsed,
            stopReason: (!stopped && ntoks >= maxTokens) ? .length : .stop)
        continuation.yield(.info(info))
        continuation.finish()
        if Self.decodeDiag, diagSteps > 0 {
            let steps = Double(diagSteps)
            let bb = String(format: "%.1f", diagBackboneS / steps * 1000)
            let mtp = String(format: "%.1f", diagMTPS / steps * 1000)
            let wall = String(format: "%.1f", diagStepWallS / steps * 1000)
            FileHandle.standardError.write(Data(
                "[DECODE] ctx=\(promptCount) steps=\(diagSteps) gen=\(ntoks) backbone=\(bb)ms mtp=\(mtp)ms STEPWALL=\(wall)ms (unaccounted=\(String(format: "%.1f", (diagStepWallS-diagBackboneS-diagMTPS)/steps*1000))ms) total=\(String(format: "%.1f", elapsed))s\n".utf8))
            let rt = reasoningSeconds.map { String(format: "%.1f", $0) } ?? "n/a"
            let close = thinkForceClosed ? "forced" : (inThink ? "open" : "self")
            FileHandle.standardError.write(Data(
                "[REASONING] tokens=\(reasoningTokens) budget=\(reasoningBudget) close=\(close) time=\(rt)s\n".utf8))
        }
    }

    /// Abort (client disconnect): finish the stream without populating `result` (caches are
    /// indeterminate, so no snapshot is stored).
    public func cancel() {
        guard phase != .finished else { return }
        phase = .finished
        continuation.finish()
    }
}
