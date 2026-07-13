import Foundation
import Testing
@testable import App

private struct FakeEmbedderService: EmbedderService {
    let dimension: Int = 4
    let vector: [Float]

    func embed(texts: [String]) async throws -> [[Float]] {
        texts.map { _ in vector }
    }
}

@Suite("OpenAILLMService")
struct OpenAILLMServiceTests {

    private func makeService(httpClient: any HTTPClient) -> OpenAILLMService {
        OpenAILLMService(baseURL: "https://llm.test", apiKey: "key", model: "test-model", httpClient: httpClient)
    }

    private static func chatCompletionBody(content: String) -> Data {
        let json = """
        {"choices":[{"message":{"role":"assistant","content":"\(content)"}}]}
        """
        return Data(json.utf8)
    }

    // MARK: - summarize

    @Test("summarize returns the trimmed assistant content")
    func summarizeReturnsContent() async throws {
        let session = MockHTTPClient { _ in
            (200, Self.chatCompletionBody(content: "A concise summary."))
        }
        let service = makeService(httpClient: session)
        let summary = try await service.summarize(documentID: "doc-1", orderedChunks: ["chunk one", "chunk two"])
        #expect(summary == "A concise summary.")
    }

    // MARK: - chat

    @Test("chat sends context chunks with their source document IDs")
    func chatIncludesContextAndSources() async throws {
        let capturedBody = Locked<Data?>(nil)
        let session = MockHTTPClient { request in
            capturedBody.mutate { $0 = request.capturedBody() }
            return (200, Self.chatCompletionBody(content: "The answer."))
        }
        let service = makeService(httpClient: session)
        let answer = try await service.chat(
            query: "What does the regulation require?",
            contextChunks: ["Widgets must be labeled."],
            documentIDs: ["WIDGET-2021-N-001"],
            conversationHistory: []
        )
        #expect(answer == "The answer.")

        let body = try #require(capturedBody.value)
        let json = try JSONSerialization.jsonObject(with: body) as? [String: Any]
        let messages = try #require(json?["messages"] as? [[String: Any]])
        let userMessage = try #require(messages.last?["content"] as? String)
        #expect(userMessage.contains("WIDGET-2021-N-001"))
        #expect(userMessage.contains("Widgets must be labeled."))
        #expect(userMessage.contains("What does the regulation require?"))
    }

    @Test("chat throws modelUnavailable when the model is not loaded")
    func chatThrowsModelUnavailable() async throws {
        let session = MockHTTPClient { _ in
            (404, Data(#"{"error":"model not found"}"#.utf8))
        }
        let service = makeService(httpClient: session)
        await #expect(throws: LLMError.self) {
            _ = try await service.chat(query: "q", contextChunks: [], documentIDs: [], conversationHistory: [])
        }
    }

    @Test("chat throws httpError for other failure status codes")
    func chatThrowsHTTPError() async throws {
        let session = MockHTTPClient { _ in (500, Data("boom".utf8)) }
        let service = makeService(httpClient: session)
        await #expect(throws: LLMError.self) {
            _ = try await service.chat(query: "q", contextChunks: [], documentIDs: [], conversationHistory: [])
        }
    }

    @Test("chat throws emptyResponse when there are no choices")
    func chatThrowsEmptyResponse() async throws {
        let session = MockHTTPClient { _ in
            (200, Data(#"{"choices":[]}"#.utf8))
        }
        let service = makeService(httpClient: session)
        await #expect(throws: LLMError.self) {
            _ = try await service.chat(query: "q", contextChunks: [], documentIDs: [], conversationHistory: [])
        }
    }

    // MARK: - chatWithTools (agentic search loop)

    @Test("chatWithTools calls search_documents, feeds results back, and returns the final answer with sources")
    func chatWithToolsCompletesSearchRoundTrip() async throws {
        let callCount = Locked<Int>(0)
        let llmSession = MockHTTPClient { request in
            let count = callCount.update { $0 + 1 }
            if count == 1 {
                let json = """
                {"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"search_documents","arguments":"{\\"query\\":\\"widget safety rules\\"}"}}]},"finish_reason":"tool_calls"}]}
                """
                return (200, Data(json.utf8))
            } else {
                let json = """
                {"choices":[{"message":{"role":"assistant","content":"Widgets must meet ASTM F963 per [WIDGET-2021-N-001]."},"finish_reason":"stop"}]}
                """
                return (200, Data(json.utf8))
            }
        }

        let qdrantSession = MockHTTPClient { _ in
            let json = """
            {"result":{"points":[{"id":"chunk-1","score":0.87,"payload":{"text":"Widgets must comply with ASTM F963.","documentID":"WIDGET-2021-N-001","chunkIndex":0,"metadata":{},"summaryMetadata":null}}]},"status":"ok"}
            """
            return (200, Data(json.utf8))
        }

        let llmService = makeService(httpClient: llmSession)
        let qdrant = QdrantService(baseURL: "https://qdrant.test", apiKey: nil, collectionName: "regs", vectorDimension: 4, httpClient: qdrantSession)
        let embedder = FakeEmbedderService(vector: [0, 0, 0, 0])
        let bm25 = BM25Service()

        let result = try await llmService.chatWithTools(
            query: "What are the safety rules for widgets?",
            conversationHistory: [],
            embedder: embedder,
            qdrant: qdrant,
            bm25: bm25,
            limit: 5,
            threshold: 0
        )

        #expect(result.answer == "Widgets must meet ASTM F963 per [WIDGET-2021-N-001].")
        #expect(result.sources.count == 1)
        #expect(result.sources.first?.documentID == "WIDGET-2021-N-001")
        #expect(callCount.value == 2)
    }

    @Test("chatWithTools sanitizes malformed escape sequences in tool arguments before decoding")
    func chatWithToolsSanitizesToolArguments() async throws {
        let callCount = Locked<Int>(0)
        let llmSession = MockHTTPClient { request in
            let count = callCount.update { $0 + 1 }
            if count == 1 {
                // The outer envelope is valid JSON (only \" and \\ escapes). Once decoded, the
                // "arguments" string itself is `{"query":"widget\_safety"}` — an embedded
                // backslash-underscore that is invalid JSON on its own and must go through
                // sanitizeToolArgs before it can be re-parsed as SearchDocumentsArgs.
                let json = #"""
                {"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"search_documents","arguments":"{\"query\":\"widget\\_safety\"}"}}]},"finish_reason":"tool_calls"}]}
                """#
                return (200, Data(json.utf8))
            } else {
                let json = """
                {"choices":[{"message":{"role":"assistant","content":"Final answer."},"finish_reason":"stop"}]}
                """
                return (200, Data(json.utf8))
            }
        }
        let qdrantSession = MockHTTPClient { _ in
            (200, Data(#"{"result":{"points":[]},"status":"ok"}"#.utf8))
        }

        let llmService = makeService(httpClient: llmSession)
        let qdrant = QdrantService(baseURL: "https://qdrant.test", apiKey: nil, collectionName: "regs", vectorDimension: 4, httpClient: qdrantSession)
        let embedder = FakeEmbedderService(vector: [0, 0, 0, 0])

        let result = try await llmService.chatWithTools(
            query: "widget safety",
            conversationHistory: [],
            embedder: embedder,
            qdrant: qdrant,
            bm25: BM25Service(),
            limit: 5,
            threshold: 0
        )

        // Sanitization succeeded (arguments parsed) and the loop completed normally.
        #expect(result.answer == "Final answer.")
    }

    @Test("chatWithTools resolves the list_documents tool")
    func chatWithToolsListsDocuments() async throws {
        let callCount = Locked<Int>(0)
        let llmSession = MockHTTPClient { _ in
            let count = callCount.update { $0 + 1 }
            if count == 1 {
                let json = """
                {"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"list_documents","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}
                """
                return (200, Data(json.utf8))
            } else {
                let json = """
                {"choices":[{"message":{"role":"assistant","content":"There are 2 documents available."},"finish_reason":"stop"}]}
                """
                return (200, Data(json.utf8))
            }
        }
        let qdrantSession = MockHTTPClient { _ in
            let json = """
            {"result":{"points":[
                {"id":"1","payload":{"text":"a","documentID":"doc-a","chunkIndex":0,"metadata":{},"summaryMetadata":null}},
                {"id":"2","payload":{"text":"b","documentID":"doc-b","chunkIndex":0,"metadata":{},"summaryMetadata":null}}
            ],"next_page_offset":null},"status":"ok"}
            """
            return (200, Data(json.utf8))
        }

        let llmService = makeService(httpClient: llmSession)
        let qdrant = QdrantService(baseURL: "https://qdrant.test", apiKey: nil, collectionName: "regs", vectorDimension: 4, httpClient: qdrantSession)
        let embedder = FakeEmbedderService(vector: [0, 0, 0, 0])

        let result = try await llmService.chatWithTools(
            query: "What documents do you have?",
            conversationHistory: [],
            embedder: embedder,
            qdrant: qdrant,
            bm25: BM25Service(),
            limit: 5,
            threshold: 0
        )

        #expect(result.answer == "There are 2 documents available.")
        #expect(result.sources.isEmpty)
    }
}
