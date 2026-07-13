import Foundation
import Testing
@testable import App

@Suite("OpenAIEmbeddingService")
struct OpenAIEmbeddingServiceTests {

    private static func embeddingResponseBody(indices: [Int], dimension: Int) -> Data {
        let objects = indices.map { i in
            """
            {"object":"embedding","embedding":[\((0..<dimension).map { "\(Float($0 + i))" }.joined(separator: ","))],"index":\(i)}
            """
        }.joined(separator: ",")
        let json = """
        {"object":"list","data":[\(objects)],"model":"test-model","usage":{"prompt_tokens":1,"total_tokens":1}}
        """
        return Data(json.utf8)
    }

    @Test("empty input throws without making a network request")
    func emptyInputThrowsWithoutRequest() async throws {
        let requestMade = Locked<Bool>(false)
        let session = MockHTTPClient { request in
            requestMade.mutate { $0 = true }
            return (200, Data())
        }
        let service = OpenAIEmbeddingService(
            baseURL: "https://embed.test", apiKey: "key", model: "nomic-embed-text",
            dimension: 4, httpClient: session
        )
        await #expect(throws: EmbeddingError.self) {
            _ = try await service.embed(texts: [])
        }
        #expect(!requestMade.value)
    }

    @Test("embeds a single batch and returns vectors in original order")
    func embedsSingleBatchInOrder() async throws {
        let session = MockHTTPClient { request in
            // Respond with embeddings out of order to verify the service re-sorts by index.
            (200, Self.embeddingResponseBody(indices: [1, 0, 2], dimension: 3))
        }
        let service = OpenAIEmbeddingService(
            baseURL: "https://embed.test", apiKey: "key", model: "nomic-embed-text",
            dimension: 3, httpClient: session
        )
        let vectors = try await service.embed(texts: ["a", "b", "c"])
        #expect(vectors.count == 3)
        #expect(vectors[0] == [0, 1, 2])
        #expect(vectors[1] == [1, 2, 3])
        #expect(vectors[2] == [2, 3, 4])
    }

    @Test("batches large inputs and preserves overall order across batches")
    func embedsMultipleBatchesPreservingOrder() async throws {
        let recordedBatchSizes = Locked<[Int]>([])
        let session = MockHTTPClient { request in
            let body = request.capturedBody() ?? Data()
            let decoded = try JSONDecoder().decode(OpenAIEmbeddingRequest.self, from: body)
            recordedBatchSizes.mutate { $0.append(decoded.input.count) }
            // index embeddings by their position *within this batch* using the input text itself.
            let indices = decoded.input.map { Int($0)! }
            return (200, Self.embeddingResponseBody(indices: indices, dimension: 2))
        }
        let service = OpenAIEmbeddingService(
            baseURL: "https://embed.test", apiKey: "key", model: "nomic-embed-text",
            dimension: 2, batchSize: 2, httpClient: session
        )
        // 5 texts, batch size 2 -> batches of [0,1], [2,3], [4]. Use the text itself
        // to encode "which original index is this" so the handler above can echo it back.
        let texts = (0..<5).map { String($0) }
        let vectors = try await service.embed(texts: texts)

        #expect(vectors.count == 5)
        for i in 0..<5 {
            #expect(vectors[i] == [Float(i), Float(i) + 1])
        }
        #expect(recordedBatchSizes.value.sorted() == [1, 2, 2])
    }

    @Test("non-200 response throws httpError with status code and body")
    func non200ThrowsHTTPError() async throws {
        let session = MockHTTPClient { _ in
            (500, Data("internal error".utf8))
        }
        let service = OpenAIEmbeddingService(
            baseURL: "https://embed.test", apiKey: "key", model: "nomic-embed-text",
            dimension: 3, httpClient: session
        )
        do {
            _ = try await service.embed(texts: ["hello"])
            Issue.record("expected embed to throw")
        } catch EmbeddingError.httpError(let code, let body) {
            #expect(code == 500)
            #expect(body == "internal error")
        }
    }

    @Test("empty data array in a 200 response throws emptyResponse")
    func emptyDataArrayThrowsEmptyResponse() async throws {
        let session = MockHTTPClient { _ in
            (200, Data(#"{"object":"list","data":[],"model":"m","usage":{"prompt_tokens":0,"total_tokens":0}}"#.utf8))
        }
        let service = OpenAIEmbeddingService(
            baseURL: "https://embed.test", apiKey: "key", model: "nomic-embed-text",
            dimension: 3, httpClient: session
        )
        await #expect(throws: EmbeddingError.self) {
            _ = try await service.embed(texts: ["hello"])
        }
    }

    @Test("malformed JSON throws decodingFailed")
    func malformedJSONThrowsDecodingFailed() async throws {
        let session = MockHTTPClient { _ in
            (200, Data("not json".utf8))
        }
        let service = OpenAIEmbeddingService(
            baseURL: "https://embed.test", apiKey: "key", model: "nomic-embed-text",
            dimension: 3, httpClient: session
        )
        await #expect(throws: EmbeddingError.self) {
            _ = try await service.embed(texts: ["hello"])
        }
    }

    @Test("text-embedding-3 models include an explicit dimensions field in the request")
    func explicitDimensionsForTextEmbedding3() async throws {
        let capturedDimensions = Locked<Int?>(nil)
        let session = MockHTTPClient { request in
            let body = request.capturedBody() ?? Data()
            let decoded = try JSONDecoder().decode(OpenAIEmbeddingRequest.self, from: body)
            capturedDimensions.mutate { $0 = decoded.dimensions }
            return (200, Self.embeddingResponseBody(indices: [0], dimension: 5))
        }
        let service = OpenAIEmbeddingService(
            baseURL: "https://embed.test", apiKey: "key", model: "text-embedding-3-small",
            dimension: 5, httpClient: session
        )
        _ = try await service.embed(texts: ["hello"])
        #expect(capturedDimensions.value == 5)
    }

    @Test("non text-embedding-3 models omit the dimensions field")
    func omitsDimensionsForOtherModels() async throws {
        let capturedDimensions = Locked<Int?>(999)
        let session = MockHTTPClient { request in
            let body = request.capturedBody() ?? Data()
            let decoded = try JSONDecoder().decode(OpenAIEmbeddingRequest.self, from: body)
            capturedDimensions.mutate { $0 = decoded.dimensions }
            return (200, Self.embeddingResponseBody(indices: [0], dimension: 5))
        }
        let service = OpenAIEmbeddingService(
            baseURL: "https://embed.test", apiKey: "key", model: "nomic-embed-text",
            dimension: 5, httpClient: session
        )
        _ = try await service.embed(texts: ["hello"])
        #expect(capturedDimensions.value == nil)
    }
}
