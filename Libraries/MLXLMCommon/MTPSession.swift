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
    private let modelCache: [KVCache]
    private let mtpCache: [KVCache]

    // Prompt + prefill bookkeeping.
    private let promptTokens: MLXArray
    private let promptCount: Int
    private let skipPrefill: Int
    private let snapshotAt: Int
    private let result: MTPCacheResult?
    private var prefillY: MLXArray
    private var prefillTotal: Int
    private var prefilled: Int
    private var snapModel: [KVCache]? = nil
    private var snapMtp: [KVCache]? = nil
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
    public private(set) var capturedSnapshot: (tokens: [Int32], model: [KVCache], mtp: [KVCache])?
    public func takeCapturedSnapshot() -> (tokens: [Int32], model: [KVCache], mtp: [KVCache])? {
        defer { capturedSnapshot = nil }
        return capturedSnapshot
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

    private func stepBackbone(_ y: MLXArray, nPredict: Int, nConfirmed: Int)
        -> (toks: [MLXArray], lps: [MLXArray], accept: [MLXArray], hidden: MLXArray)
    {
        let (logitsAll, hidden) = model.backboneWithHidden(
            y.expandedDimensions(axis: 0), cache: modelCache, nConfirmed: nConfirmed)
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
        let mtpLogitsAll = model.mtpForward(
            hiddenIn, nextTokenIds: nextIds, cache: mtpCache.map { $0 as KVCache? })
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
        if let chunk = detokenizer.next() { continuation.yield(.chunk(chunk)) }
        _ = lp
        ntoks += 1
        return false
    }

    // MARK: - Steppable prefill + decode

    /// Run ONE prefill chunk. Returns true while more prefill remains. Must be called inside the
    /// container. When prefill completes, transitions to `.decoding`.
    public func prefillStepOnce() -> Bool {
        if prefillStart == nil { prefillStart = Date() }
        guard prefillTotal > 1 else { finishPrefill(); return false }
        var n = min(prefillStep, prefillTotal - 1)
        if snapshotAt > prefilled && snapshotAt < prefilled + n { n = snapshotAt - prefilled }
        let chunk = prefillY[0 ..< n]
        let (_, _) = model.backboneWithHidden(
            chunk.expandedDimensions(axis: 0), cache: modelCache, nConfirmed: 0)
        // MTP head intentionally NOT warmed during prefill (cold MTP cache self-warms in decode;
        // output is backbone-only — see MTPGenerate.swift).
        eval(modelCache.map { $0.state }.flatMap { $0 })
        prefillY = prefillY[n...]
        prefillTotal -= n
        prefilled += n
        if snapshotAt > 0, prefilled == snapshotAt, snapModel == nil {
            snapModel = modelCache.map { $0.copy() }
            snapMtp = mtpCache.map { $0.copy() }
            // Surface the snapshot so the scheduler can store it for reuse (taken once).
            capturedSnapshot = (
                Array(promptTokens.asArray(Int32.self).prefix(snapshotAt)),
                snapModel!, snapMtp!)
        }
        if prefillTotal <= 1 { finishPrefill(); return false }
        return true
    }

    private func finishPrefill() {
        result?.prefillSeconds = Date().timeIntervalSince(prefillStart ?? start)
        // Decode begins from the final prompt token (the standard loop sets y to the last token via
        // the prompt; the original fed the whole prompt then advanced — here the cache is warmed by
        // prefill of all but the last token, and decode's first backbone step consumes the last one).
        y = promptTokens[(promptCount - 1)...].asType(.uint32)
        phase = .decoding
    }

    /// Perform ONE decode iteration (the body of mtpGenerate's `while`). Returns true while more
    /// decode remains; false when finished (stop/EOS/maxTokens) — at which point the stream is
    /// finished and `result` populated. Must be called inside the container.
    public func decodeStepOnce() -> Bool {
        if stopped || ntoks >= maxTokens { finishDecode(); return false }

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
        if let result {
            result.promptTokens = promptTokens.asArray(Int32.self)
            if let snapModel, let snapMtp, snapshotAt > 0 {
                result.snapshotModelCache = snapModel
                result.snapshotMtpCache = snapMtp
                result.snapshotTokens = Array(promptTokens.asArray(Int32.self).prefix(snapshotAt))
            }
        }
        let elapsed = Date().timeIntervalSince(start)
        let info = GenerateCompletionInfo(
            promptTokenCount: promptCount, generationTokenCount: ntoks,
            promptTime: 0, generationTime: elapsed,
            stopReason: (!stopped && ntoks >= maxTokens) ? .length : .stop)
        continuation.yield(.info(info))
        continuation.finish()
    }

    /// Abort (client disconnect): finish the stream without populating `result` (caches are
    /// indeterminate, so no snapshot is stored).
    public func cancel() {
        guard phase != .finished else { return }
        phase = .finished
        continuation.finish()
    }
}
