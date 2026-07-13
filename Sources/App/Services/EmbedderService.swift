import Foundation

/// Implemented by any embedding backend — on-device (NLEmbedding) or HTTP (Ollama, OpenAI-compatible).
protocol EmbedderService: Sendable {
    /// Dimensionality of the vectors this service produces.
    var dimension: Int { get }

    /// Embed one or more texts, returning one vector per input in the same order.
    func embed(texts: [String]) async throws -> [[Float]]
}
