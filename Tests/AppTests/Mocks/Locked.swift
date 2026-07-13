import Foundation

/// A tiny thread-safe box for capturing values inside `@Sendable` mock-network closures in tests.
final class Locked<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: Value

    init(_ value: Value) {
        self._value = value
    }

    var value: Value {
        lock.lock(); defer { lock.unlock() }
        return _value
    }

    func mutate(_ body: (inout Value) -> Void) {
        lock.lock(); defer { lock.unlock() }
        body(&_value)
    }

    /// Atomically applies `transform` and returns the resulting value. Handy for call counters
    /// shared between a mocked network handler and the test body.
    @discardableResult
    func update(_ transform: (Value) -> Value) -> Value {
        lock.lock(); defer { lock.unlock() }
        _value = transform(_value)
        return _value
    }
}
