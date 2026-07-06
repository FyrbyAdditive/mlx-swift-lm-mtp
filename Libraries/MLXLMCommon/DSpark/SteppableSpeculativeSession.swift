// A drafter-agnostic surface over steppable speculative sessions (MTP, DSpark) so one
// fair scheduler can interleave either kind. `aux` is the drafter-side cache carried in
// prefix snapshots next to the target's KV: the MTP head's caches for MTPSession, the
// DSpark drafter's context caches for DSparkSession.

import Foundation
import MLX

public protocol SteppableSpeculativeSession: AnyObject {
    var isFinished: Bool { get }
    var isPrefilling: Bool { get }
    func prefillStepOnce() -> Bool
    func decodeStepOnce() -> Bool
    func takeSnapshot() -> (tokens: [Int32], model: [KVCache], aux: [KVCache])?
    func cancel()
}

extension MTPSession: SteppableSpeculativeSession {
    public var isFinished: Bool { phase == .finished }
    public var isPrefilling: Bool { phase == .prefilling }
    public func takeSnapshot() -> (tokens: [Int32], model: [KVCache], aux: [KVCache])? {
        guard let s = takeCapturedSnapshot() else { return nil }
        return (s.tokens, s.model, s.mtp)
    }
}

extension DSparkSession: SteppableSpeculativeSession {
    public var isFinished: Bool { phase == .finished }
    public var isPrefilling: Bool { phase == .prefilling }
    public func takeSnapshot() -> (tokens: [Int32], model: [KVCache], aux: [KVCache])? {
        takeCapturedSnapshot()
    }
}
