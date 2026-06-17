// Native MTP self-speculative decode loop. Faithful Swift port of mlx-lm's
// `mtp_generate_step` (PR #990), supporting greedy (temp == 0) and plain-temperature sampling.
// Filter chains (top-p/top-k/min-p), XTC, and logits processors are intentionally not ported —
// the server uses temperature/top-p at the request layer, and the acceptance/residual math below
// is exact for greedy and plain-temperature, which is what guarantees output parity.

import Foundation
import MLX

/// Generate tokens using a model's native MTP head for self-speculative decoding.
///
/// One backbone forward can emit up to two tokens: a verify token (always correct) and a draft
/// token accepted via `min(1, p_target/p_draft)` rejection sampling, so the output distribution is
/// identical to non-speculative decoding (greedy ⇒ token-identical; sampling ⇒ same marginal).
///
/// - Returns: an `AsyncStream<Generation>` of `.chunk`/`.info`, matching `generate(...)`.
public func mtpGenerate(
    input: LMInput,
    parameters: GenerateParameters,
    context: ModelContext,
    // Restore a previously-taken prefix snapshot (copies of model+MTP caches that encode exactly
    // `restoreCount` leading prompt tokens). When set, prefill skips those tokens.
    restore: (model: [KVCache], mtp: [KVCache])? = nil,
    restoreCount: Int = 0,
    // If > 0, take a `.copy()` snapshot of both caches once exactly `snapshotAt` tokens have been
    // prefilled, and report it via `result.snapshot*`/`result.snapshotTokens` on clean completion.
    // Lets a future request whose prompt shares that prefix restore it and skip re-prefilling.
    snapshotAt: Int = 0,
    // Filled once, only on clean completion. Carries the prompt tokens (for the next snapshot-point
    // decision) and, if `snapshotAt > 0`, the snapshot caches + the tokens they encode. Left empty
    // on cancellation/error. A reference box (not a closure) so the non-Sendable caches never cross
    // the Task boundary as `sending` values.
    result: MTPCacheResult? = nil
) throws -> AsyncStream<Generation> {
    guard (context.model as? any MTPSpeculativeModel)?.hasMTP == true else {
        throw MTPError.notSupported
    }

    let temp = parameters.temperature
    let isGreedy = temp == 0
    let maxTokens = parameters.maxTokens ?? Int.max

    let (stream, continuation) = AsyncStream<Generation>.makeStream()
    // Box the non-Sendable inputs so they can cross into the Task without tripping Swift 6
    // sending checks; everything is consumed once inside the (serialized) generation task.
    let boxedInput = SendableBox(input)
    let boxedContext = SendableBox(context)
    let boxedRestore = SendableBox(restore)

    let task = Task {
        let context = boxedContext.consume()
        let input = boxedInput.consume()
        let restore = boxedRestore.consume()
        // safe: guarded above
        let model = context.model as! any MTPSpeculativeModel
        // The processor yields batched tokens (shape [1, seqLen]); flatten to a 1-D [seqLen]
        // sequence so the prefill/decode slicing + `expandedDimensions(axis: 0)` produce [1, n].
        let promptTokens = input.text.tokens.reshaped([-1])
        let start = Date()
        let promptCount = promptTokens.dim(-1)

        // Restore a prefix snapshot (its caches encode exactly `restoreCount` leading prompt
        // tokens) or build fresh. We work on COPIES of the restored caches so the stored snapshot
        // stays immutable and reusable for later requests. With a restore, prefill skips the first
        // `restoreCount` tokens — this is what avoids re-prefilling a constant system prompt.
        let modelCache: [KVCache]
        let mtpCache: [KVCache]
        if let restore {
            modelCache = restore.model.map { $0.copy() }
            mtpCache = restore.mtp.map { $0.copy() }
        } else {
            modelCache = context.model.newCache(parameters: parameters)
            mtpCache = model.makeMTPCache()
        }
        let skipPrefill = restore != nil ? max(0, restoreCount) : 0

        // Stop-token set: model EOS ids + tokenizer EOS + any extra EOS strings (e.g. <|im_end|>).
        // Without this the loop runs to maxTokens, sailing past <|im_end|>/<|endoftext|> and making
        // the model role-play both sides of the chat (the "infinite waffle").
        let stopTokenIds = MTPStopTokens.build(
            eosTokenIds: context.configuration.eosTokenIds,
            tokenizerEOSTokenId: context.tokenizer.eosTokenId,
            extraEOSTokens: context.configuration.extraEOSTokens,
            tokenToId: { context.tokenizer.convertTokenToId($0) })

            // Sample a token + return the acceptance log-probs that produced it.
            // `lpAccept` is the temperature-adjusted, normalized log-prob distribution used by
            // the acceptance test and residual sampling; `logprobs` is the raw (unscaled) dist.
            func processAndSample(_ logits: MLXArray) -> (token: MLXArray, logprobs: MLXArray, lpAccept: MLXArray) {
                let logprobs = logits - logSumExp(logits, axis: -1, keepDims: true)
                if isGreedy {
                    return (argMax(logprobs, axis: -1), logprobs, logprobs)
                }
                let scaled = logprobs * (1 / temp)
                let token = categorical(scaled)
                let lpAccept = scaled - logSumExp(scaled, axis: -1, keepDims: true)
                return (token, logprobs, lpAccept)
            }

            // Backbone step → (tokens, logprobs, acceptLps, hidden) for the last nPredict positions.
            func stepBackbone(_ y: MLXArray, nPredict: Int, nConfirmed: Int)
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

            // MTP draft step. `cacheCommit` prepends (hidden, tok) so an accepted draft token is
            // committed into mtpCache in the same forward that produces the next draft.
            func stepMTP(
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

            func clearRollback() {
                for c in modelCache {
                    if let m = c as? MambaCache { m.rollbackState = nil }
                }
            }

            func rollbackDraft() {
                for c in modelCache {
                    if let m = c as? MambaCache, let snap = m.rollbackState {
                        m.state = [snap.0 ?? m.state[0], snap.1 ?? m.state[1]]
                        m.rollbackState = nil
                    } else if c.isTrimmable {
                        _ = c.trim(1)
                    }
                }
            }

            // Snapshot of the caches taken mid-prefill at `snapshotAt` tokens (for future reuse).
            var snapModel: [KVCache]? = nil
            var snapMtp: [KVCache]? = nil

            // --- Prefill: process all but the last prompt token, warming both caches. ---
            // When restoring, the leading `skipPrefill` tokens are already encoded, so drop them and
            // prefill only the new suffix. `prefilled` tracks the absolute position in the prompt
            // (counting restored tokens) so we can snapshot exactly at `snapshotAt`.
            do {
                var y = skipPrefill > 0 ? promptTokens[skipPrefill...] : promptTokens
                var total = y.dim(-1)
                var prefilled = skipPrefill
                let prefillStep = parameters.prefillStepSize
                while total > 1 {
                    // Cap the chunk so we land exactly on `snapshotAt` when one is requested ahead.
                    var n = min(prefillStep, total - 1)
                    if snapshotAt > prefilled && snapshotAt < prefilled + n {
                        n = snapshotAt - prefilled
                    }
                    let chunk = y[0 ..< n]
                    let (_, hidden) = model.backboneWithHidden(
                        chunk.expandedDimensions(axis: 0), cache: modelCache, nConfirmed: 0)
                    // Warm the MTP cache: feed the backbone hidden states + the *next* token ids
                    // (shift +1), matching how the head is used during decode.
                    _ = model.mtpForward(
                        hidden,
                        nextTokenIds: y[1 ..< (n + 1)].expandedDimensions(axis: 0),
                        cache: mtpCache.map { $0 as KVCache? })
                    eval(modelCache.map { $0.state }.flatMap { $0 }
                        + mtpCache.map { $0.state }.flatMap { $0 })
                    y = y[n...]
                    total -= n
                    prefilled += n
                    // Take the snapshot the moment the caches encode exactly `snapshotAt` tokens.
                    if snapshotAt > 0, prefilled == snapshotAt, snapModel == nil {
                        snapModel = modelCache.map { $0.copy() }
                        snapMtp = mtpCache.map { $0.copy() }
                    }
                }

                var ntoks = 0
                var stopped = false
                var draftTok: MLXArray? = nil
                var draftLp = MLXArray(0)
                var draftAccept = MLXArray(0)
                var detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)

                /// Emit a token's text. Returns true if it was a stop token (EOS): the caller must
                /// stop and the stop token is NOT detokenized/yielded (matches the standard loop's
                /// includeStopToken:false default).
                @discardableResult
                func emit(_ tokenArray: MLXArray, _ lp: MLXArray) -> Bool {
                    let id = tokenArray.item(Int.self)
                    if stopTokenIds.contains(id) {
                        stopped = true
                        return true
                    }
                    detokenizer.append(token: id)
                    if let chunk = detokenizer.next() {
                        continuation.yield(.chunk(chunk))
                    }
                    _ = lp
                    ntoks += 1
                    return false
                }

                while ntoks < maxTokens {
                    if stopped { break }
                    if Task.isCancelled { break }

                    if draftTok == nil {
                        // No pending draft: backbone only, emit, then make the first draft.
                        let (toks, lps, _, hidden) = stepBackbone(y, nPredict: 1, nConfirmed: 0)
                        let mainTok = toks[0]
                        let hiddenAtMain = hidden[0..., (hidden.dim(1) - 1)..., 0...]
                        // Dispatch the MTP draft in the same graph as the backbone so both evaluate
                        // together; `asyncEval` schedules without blocking the CPU.
                        let d = stepMTP(hiddenLast: hiddenAtMain, mainTok: mainTok)
                        asyncEval(mainTok, d.token)
                        if emit(mainTok, lps[0]) { break }
                        if ntoks >= maxTokens { break }
                        draftTok = d.token; draftLp = d.logprobs; draftAccept = d.accept
                        y = mainTok.reshaped(1).asType(.uint32)
                    } else {
                        // Verify the pending draft over [y, draft]; nConfirmed=1 snapshots SSM state.
                        let dtok = draftTok!
                        let yWithDraft = concatenated([y, dtok.reshaped(1).asType(.uint32)], axis: 0)
                        let (toks, lps, accept, hidden) = stepBackbone(
                            yWithDraft, nPredict: 2, nConfirmed: 1)
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
                            let logAccept =
                                (verifyAccept[draftId] - draftAccept[draftId]).item(Float.self)
                            accepted = logAccept >= 0 || u.item(Float.self) < exp(logAccept)
                        }

                        let hiddenAtConfirmed = hidden[0..., 0 ..< 1, 0...]
                        let hiddenAtDraft = hidden[0..., 1 ..< 2, 0...]

                        if accepted {
                            clearRollback()
                            // Compute the next draft and schedule it (asyncEval) so its forward
                            // overlaps the CPU-side detokenize/emit work below.
                            let d = stepMTP(
                                hiddenLast: hiddenAtDraft, mainTok: bonusTok,
                                cacheCommit: (hiddenAtConfirmed, dtok))
                            asyncEval(d.token)
                            if emit(dtok, draftLp) { break }
                            if ntoks >= maxTokens { break }
                            if emit(bonusTok, bonusLp) { break }
                            if ntoks >= maxTokens { break }
                            draftTok = d.token; draftLp = d.logprobs; draftAccept = d.accept
                            y = bonusTok.reshaped(1).asType(.uint32)
                        } else {
                            rollbackDraft()
                            var verifyId = verifyPred.item(Int.self)
                            if !isGreedy {
                                // Residual sample: max(p_target - p_draft, 0) / Z, fallback p_target.
                                let pTarget = exp(verifyAccept)
                                let pDraft = exp(draftAccept)
                                let residual = maximum(pTarget - pDraft, MLXArray(Float(0)))
                                let z = residual.sum(keepDims: true)
                                let dist = MLX.where(z .> 0, residual, pTarget)
                                verifyId = categorical(
                                    MLX.log(dist).reshaped(1, -1)).item(Int.self)
                            }
                            let vtok = MLXArray(UInt32(verifyId))
                            // Schedule the next draft so it overlaps the emit below.
                            let d = stepMTP(hiddenLast: hiddenAtConfirmed, mainTok: vtok)
                            asyncEval(d.token)
                            if emit(verifyPred.dtype == .uint32 ? MLXArray(UInt32(verifyId)) : MLXArray(verifyId), verifyLp) { break }
                            if ntoks >= maxTokens { break }
                            draftTok = d.token; draftLp = d.logprobs; draftAccept = d.accept
                            y = vtok.reshaped(1)
                        }
                    }
                }

                // Flush any buffered detokenizer text.
                if let tail = detokenizer.next(), !tail.isEmpty {
                    continuation.yield(.chunk(tail))
                }

                // Report caches + the exact token sequence they encode for whole-prefix reuse on
                // the next request — but ONLY on a clean finish. If the run was cancelled mid-stream
                // the caches are in an indeterminate state, so we skip reporting and the engine will
                // discard them (rebuild next time). (Pure derivation; see committedSequence.)
                if !Task.isCancelled, let result {
                    result.promptTokens = promptTokens.asArray(Int32.self)
                    if let snapModel, let snapMtp, snapshotAt > 0 {
                        result.snapshotModelCache = snapModel
                        result.snapshotMtpCache = snapMtp
                        result.snapshotTokens = Array(
                            promptTokens.asArray(Int32.self).prefix(snapshotAt))
                    }
                }

                // Completion info.
                let elapsed = Date().timeIntervalSince(start)
                let info = GenerateCompletionInfo(
                    promptTokenCount: promptCount,
                    generationTokenCount: ntoks,
                    promptTime: 0,
                    generationTime: elapsed,
                    stopReason: (!stopped && ntoks >= maxTokens) ? .length : .stop)
                continuation.yield(.info(info))
                continuation.finish()
            } catch {
                continuation.finish()
            }
    }
    continuation.onTermination = { _ in task.cancel() }
    return stream
}

public enum MTPError: Error {
    case notSupported
}
