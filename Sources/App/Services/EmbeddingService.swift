import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

enum EmbeddingError: Error, LocalizedError {
    case emptyInput
    case httpError(Int, String)
    case decodingFailed(Error)
    case emptyResponse

    var errorDescription: String? {
        switch self {
        case .emptyInput:              return "No texts provided for embedding"
        case .httpError(let code, let body): return "Embedding API error \(code): \(body)"
        case .decodingFailed(let err): return "Failed to decode embedding response: \(err)"
        case .emptyResponse:           return "Embedding API returned no data"
        }
    }
}

/// HTTP-based embedder for OpenAI-compatible endpoints (Ollama, LM Studio, OpenAI, etc.).
/// Use this on Linux or when you want a specific model not available via NLEmbedding.
final class OpenAIEmbeddingService: EmbedderService, @unchecked Sendable {

    let baseURL: String
    let apiKey: String
    let model: String
    let dimension: Int
    let batchSize: Int

    private let session: any HTTPClient

    init(baseURL: String, apiKey: String, model: String, dimension: Int, batchSize: Int = 100, httpClient: any HTTPClient) {
        self.baseURL   = baseURL
        self.apiKey    = apiKey
        self.model     = model
        self.dimension = dimension
        self.batchSize = batchSize
        self.session   = httpClient
    }

    /// Embed an arbitrary number of texts, batching to stay within API limits.
    func embed(texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { throw EmbeddingError.emptyInput }

        let batches = stride(from: 0, to: texts.count, by: batchSize).map { start -> (Int, [String]) in
            let end = min(start + batchSize, texts.count)
            return (start, Array(texts[start..<end]))
        }

        var results: [[Float]] = Array(repeating: [], count: texts.count)
        let maxConcurrent = 4

        // Process batches in windows of maxConcurrent to avoid overwhelming the embedding server
        for windowStart in stride(from: 0, to: batches.count, by: maxConcurrent) {
            let window = batches[windowStart..<min(windowStart + maxConcurrent, batches.count)]
            try await withThrowingTaskGroup(of: (Int, [[Float]]).self) { group in
                for (startIndex, batch) in window {
                    group.addTask { (startIndex, try await self.embedBatch(batch)) }
                }
                for try await (startIndex, embeddings) in group {
                    for (i, vec) in embeddings.enumerated() {
                        results[startIndex + i] = vec
                    }
                }
            }
        }
        return results
    }

    // MARK: - Private

    private func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        guard let url = URL(string: "\(baseURL)/embeddings") else {
            throw EmbeddingError.httpError(0, "Invalid base URL: \(baseURL)")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        // Only text-embedding-3-* supports explicit `dimensions`
        let explicitDimensions = model.contains("text-embedding-3") ? dimension : nil
        let body = OpenAIEmbeddingRequest(input: texts, model: model, dimensions: explicitDimensions)
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await session.data(for: request)

        guard let http = response as? HTTPURLResponse else {
            throw EmbeddingError.httpError(0, "Non-HTTP response")
        }
        guard http.statusCode == 200 else {
            let body = String(data: data, encoding: .utf8) ?? "unknown"
            throw EmbeddingError.httpError(http.statusCode, body)
        }

        do {
            let decoded = try JSONDecoder().decode(OpenAIEmbeddingResponse.self, from: data)
            guard !decoded.data.isEmpty else { throw EmbeddingError.emptyResponse }
            return decoded.data.sorted { $0.index < $1.index }.map { $0.embedding }
        } catch {
            throw EmbeddingError.decodingFailed(error)
        }
    }
}
