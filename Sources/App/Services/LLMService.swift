import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

// MARK: - Agentic result type

struct AgenticChatResult: Sendable {
    let answer: String
    let sources: [SearchResultDTO]
}

// MARK: - Protocol

protocol LLMService: Sendable {
    func summarize(documentID: String, orderedChunks: [String]) async throws -> String
    func chat(query: String, contextChunks: [String], documentIDs: [String], conversationHistory: [ChatMessageDTO]) async throws -> String
    func chatWithTools(
        query: String,
        conversationHistory: [ChatMessageDTO],
        embedder: any EmbedderService,
        qdrant: QdrantService,
        bm25: BM25Service,
        limit: Int,
        threshold: Float
    ) async throws -> AgenticChatResult
}

// MARK: - Default implementation (single search + existing chat, used by any service that doesn't override)

extension LLMService {
    func chatWithTools(
        query: String,
        conversationHistory: [ChatMessageDTO],
        embedder: any EmbedderService,
        qdrant: QdrantService,
        bm25: BM25Service,
        limit: Int,
        threshold: Float
    ) async throws -> AgenticChatResult {
        let vectors = try await embedder.embed(texts: [query])
        guard let queryVector = vectors.first else {
            return AgenticChatResult(answer: "Failed to embed query.", sources: [])
        }
        let sparseQuery = bm25.encode(query)
        let hits = try await qdrant.hybridSearch(
            denseVector: queryVector, sparseVector: sparseQuery, limit: limit, threshold: threshold
        )
        let sources: [SearchResultDTO] = hits.compactMap { hit in
            guard let payload = hit.payload else { return nil }
            return SearchResultDTO(
                id: hit.id, text: payload.text, score: hit.score,
                documentID: payload.documentID, chunkIndex: payload.chunkIndex,
                metadata: payload.metadata, summaryMetadata: payload.summaryMetadata
            )
        }
        if sources.isEmpty {
            return AgenticChatResult(
                answer: "No relevant documents were found for your query. Try rephrasing or lowering the similarity threshold.",
                sources: []
            )
        }
        let answer = try await chat(
            query: query,
            contextChunks: sources.map { $0.text },
            documentIDs: sources.map { $0.documentID },
            conversationHistory: conversationHistory
        )
        return AgenticChatResult(answer: answer, sources: sources)
    }
}

// MARK: - Summary cache

actor SummaryCache: Sendable {
    private var cache: [String: String] = [:]
    func get(_ key: String) -> String? { cache[key] }
    func set(_ key: String, value: String) { cache[key] = value }
}

// MARK: - Errors

enum LLMError: Error, LocalizedError {
    case invalidURL
    case httpError(Int, String)
    case emptyResponse
    case modelUnavailable(String)
    case contentTooLarge

    var errorDescription: String? {
        switch self {
        case .invalidURL:                      return "Invalid LLM base URL"
        case .httpError(let code, let body):   return "LLM API error \(code): \(body)"
        case .emptyResponse:                   return "LLM returned an empty response"
        case .modelUnavailable(let reason):    return "LLM model unavailable: \(reason)"
        case .contentTooLarge:                 return "Content could not be reduced to fit the model's context window"
        }
    }
}

// MARK: - OpenAI wire types (basic chat)

private struct LLMChatRequest: Encodable, Sendable {
    let model: String
    let messages: [LLMChatMessage]
    let stream: Bool
}

private struct LLMChatMessage: Codable, Sendable {
    let role: String
    let content: String
}

private struct LLMChatResponse: Decodable, Sendable {
    let choices: [LLMChatChoice]
}

private struct LLMChatChoice: Decodable, Sendable {
    let message: LLMChatMessage
}

// MARK: - OpenAI tool-calling wire types

private struct LLMToolChatRequest: Encodable, Sendable {
    let model: String
    let messages: [LLMToolMessage]
    let tools: [LLMToolDefinition]
    let toolChoice: String
    let stream: Bool

    enum CodingKeys: String, CodingKey {
        case model, messages, tools, stream
        case toolChoice = "tool_choice"
    }
}

private struct LLMToolDefinition: Encodable, Sendable {
    let type: String
    let function: LLMToolFunction
}

private struct LLMToolFunction: Encodable, Sendable {
    let name: String
    let description: String
    let parameters: LLMToolParameters
}

private struct LLMToolParameters: Encodable, Sendable {
    let type: String
    let properties: [String: LLMToolProperty]
    let required: [String]
}

private struct LLMToolProperty: Encodable, Sendable {
    let type: String
    let description: String
}

// Flexible message type that handles assistant tool_calls and tool results.
// Uses a custom encoder so nil optional fields are OMITTED (not "field": null),
// which is required by OpenAI-compatible APIs.
private struct LLMToolMessage: Codable, Sendable {
    let role: String
    var content: String?
    var toolCallId: String?
    var toolCalls: [LLMToolCall]?

    enum CodingKeys: String, CodingKey {
        case role, content
        case toolCallId = "tool_call_id"
        case toolCalls  = "tool_calls"
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(role, forKey: .role)
        try container.encodeIfPresent(content, forKey: .content)
        try container.encodeIfPresent(toolCallId, forKey: .toolCallId)
        try container.encodeIfPresent(toolCalls, forKey: .toolCalls)
    }

    static func system(_ content: String) -> Self {
        .init(role: "system", content: content)
    }
    static func user(_ content: String) -> Self {
        .init(role: "user", content: content)
    }
    static func history(role: String, content: String) -> Self {
        .init(role: role, content: content)
    }
    static func toolResult(id: String, content: String) -> Self {
        .init(role: "tool", content: content, toolCallId: id)
    }
}

private struct LLMToolCall: Codable, Sendable {
    let id: String
    let type: String
    let function: LLMToolCallFunction
}

private struct LLMToolCallFunction: Codable, Sendable {
    let name: String
    let arguments: String   // JSON-encoded string from the LLM
}

private struct LLMToolResponse: Decodable, Sendable {
    let choices: [LLMToolChoice]
}

private struct LLMToolChoice: Decodable, Sendable {
    let message: LLMToolMessage
    let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case message
        case finishReason = "finish_reason"
    }
}

// Argument schemas for JSON decoding tool call payloads
private struct SearchDocumentsArgs: Decodable, Sendable {
    let query: String
    let agency: String?
    let limit: Int?
}

// MARK: - OpenAI-compatible HTTP implementation (Ollama / OpenAI / etc.)

final class OpenAILLMService: LLMService, @unchecked Sendable {

    let baseURL: String
    let apiKey: String
    let model: String

    private let session: any HTTPClient
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    private let maxInputLength = 4_000

    init(baseURL: String, apiKey: String, model: String, httpClient: any HTTPClient) {
        self.baseURL = baseURL
        self.apiKey  = apiKey
        self.model   = model
        self.session = httpClient
    }

    func summarize(documentID: String, orderedChunks: [String]) async throws -> String {
        var combined = orderedChunks.joined(separator: "\n\n")
        if combined.count > maxInputLength {
            combined = String(combined.prefix(maxInputLength)) + "…"
        }

        let user = """
        Summarize the following regulation document in 2–3 sentences. \
        Cover what it regulates, who it affects, and any key requirements or dates.

        Document: \(documentID)

        \(combined)
        """

        return try await chatComplete(messages: [
            LLMChatMessage(role: "system", content: "You are a concise regulatory document assistant. Respond only with the summary — no preamble."),
            LLMChatMessage(role: "user",   content: user)
        ])
    }

    func chat(query: String, contextChunks: [String], documentIDs: [String], conversationHistory: [ChatMessageDTO]) async throws -> String {
        let systemPrompt = """
        You are a knowledgeable regulatory document assistant. Answer questions using ONLY the provided document excerpts.
        Cite sources by referring to their document ID (e.g. "According to [AGENCY-2021-N-0270], ...").
        If the excerpts do not contain enough information to answer confidently, say so clearly rather than guessing.
        Be thorough and comprehensive: provide as much relevant detail as the question warrants. Never truncate an answer just to be brief.
        Do not repeat the full text of the excerpts — synthesize the key information.
        """

        var contextParts: [String] = []
        var totalLen = 0
        let maxContextLen = 8_000
        for (i, (text, docID)) in zip(contextChunks, documentIDs).enumerated() {
            let part = "Source \(i + 1) (\(docID)): \(text)"
            guard totalLen + part.count <= maxContextLen else { break }
            contextParts.append(part)
            totalLen += part.count
        }

        let userContent = """
        [Context]
        \(contextParts.joined(separator: "\n\n"))

        [Question]
        \(query)
        """

        var messages: [LLMChatMessage] = [LLMChatMessage(role: "system", content: systemPrompt)]
        for msg in conversationHistory {
            messages.append(LLMChatMessage(role: msg.role, content: msg.content))
        }
        messages.append(LLMChatMessage(role: "user", content: userContent))

        return try await chatComplete(messages: messages)
    }

    // MARK: - Tool-calling chat (overrides protocol extension default)

    func chatWithTools(
        query: String,
        conversationHistory: [ChatMessageDTO],
        embedder: any EmbedderService,
        qdrant: QdrantService,
        bm25: BM25Service,
        limit: Int,
        threshold: Float
    ) async throws -> AgenticChatResult {
        let systemPrompt = """
        You are a knowledgeable regulatory document assistant with access to document search tools.
        ALWAYS use the search_documents tool to find relevant information before answering questions.
        You may call search_documents multiple times with different queries to gather comprehensive information.
        Cite sources by document ID (e.g. "According to [AGENCY-2021-N-0270], ...").
        If searches return no relevant results, say so clearly rather than guessing.
        Be thorough and comprehensive: provide as much relevant detail as the question warrants. Never truncate an answer just to be brief.
        """

        var messages: [LLMToolMessage] = [.system(systemPrompt)]
        for msg in conversationHistory {
            messages.append(.history(role: msg.role, content: msg.content))
        }
        messages.append(.user(query))

        let tools = makeToolDefinitions(defaultLimit: limit)
        var collectedSources: [SearchResultDTO] = []
        var seenSourceIDs: Set<String> = []
        let maxIterations = 5

        for _ in 0..<maxIterations {
            let response = try await toolChatComplete(messages: messages, tools: tools)
            guard let choice = response.choices.first else { throw LLMError.emptyResponse }

            messages.append(choice.message)

            if choice.finishReason == "tool_calls", let toolCalls = choice.message.toolCalls {
                for toolCall in toolCalls {
                    let (resultText, newSources) = try await executeTool(
                        toolCall: toolCall,
                        embedder: embedder, qdrant: qdrant, bm25: bm25,
                        limit: limit, threshold: threshold
                    )
                    for source in newSources where !seenSourceIDs.contains(source.id) {
                        seenSourceIDs.insert(source.id)
                        collectedSources.append(source)
                    }
                    messages.append(.toolResult(id: toolCall.id, content: resultText))
                }
            } else {
                let answer = (choice.message.content ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                guard !answer.isEmpty else { throw LLMError.emptyResponse }
                return AgenticChatResult(answer: answer, sources: collectedSources)
            }
        }

        throw LLMError.emptyResponse
    }

    // MARK: - Private HTTP helpers

    private func chatComplete(messages: [LLMChatMessage]) async throws -> String {
        guard let url = URL(string: "\(baseURL)/chat/completions") else {
            throw LLMError.invalidURL
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.httpBody = try encoder.encode(LLMChatRequest(model: model, messages: messages, stream: false))

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw LLMError.httpError(0, "Non-HTTP response")
        }
        guard http.statusCode == 200 else {
            let body = String(data: data, encoding: .utf8) ?? "unknown"
            if http.statusCode == 404, body.contains("not found") {
                throw LLMError.modelUnavailable("'\(model)' is not loaded on the vLLM server")
            }
            throw LLMError.httpError(http.statusCode, body)
        }
        let decoded = try decoder.decode(LLMChatResponse.self, from: data)
        guard let content = decoded.choices.first?.message.content else {
            throw LLMError.emptyResponse
        }
        return content.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func toolChatComplete(messages: [LLMToolMessage], tools: [LLMToolDefinition]) async throws -> LLMToolResponse {
        guard let url = URL(string: "\(baseURL)/chat/completions") else {
            throw LLMError.invalidURL
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.httpBody = try encoder.encode(LLMToolChatRequest(
            model: model, messages: messages, tools: tools, toolChoice: "auto", stream: false
        ))

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw LLMError.httpError(0, "Non-HTTP response")
        }
        guard http.statusCode == 200 else {
            let body = String(data: data, encoding: .utf8) ?? "unknown"
            if http.statusCode == 404, body.contains("not found") {
                throw LLMError.modelUnavailable("'\(model)' is not loaded on the vLLM server")
            }
            throw LLMError.httpError(http.statusCode, body)
        }
        return try decoder.decode(LLMToolResponse.self, from: data)
    }

    private func executeTool(
        toolCall: LLMToolCall,
        embedder: any EmbedderService,
        qdrant: QdrantService,
        bm25: BM25Service,
        limit: Int,
        threshold: Float
    ) async throws -> (resultText: String, sources: [SearchResultDTO]) {
        switch toolCall.function.name {

        case "search_documents":
            let sanitizedArgs = sanitizeToolArgs(toolCall.function.arguments)
            guard let argsData = sanitizedArgs.data(using: .utf8),
                  let args = try? decoder.decode(SearchDocumentsArgs.self, from: argsData) else {
                return ("Error: could not parse search_documents arguments. Use a plain-text query without backslashes, boolean operators (AND/OR), or special characters.", [])
            }
            let effectiveLimit = min(args.limit ?? limit, 10)
            let vectors = try await embedder.embed(texts: [args.query])
            guard let queryVector = vectors.first else {
                return ("No embedding generated for query: \(args.query)", [])
            }
            let sparseQuery = bm25.encode(args.query)
            let hits = try await qdrant.hybridSearch(
                denseVector: queryVector, sparseVector: sparseQuery,
                limit: effectiveLimit, threshold: threshold
            )
            var sources: [SearchResultDTO] = []
            var resultLines: [String] = []
            for hit in hits {
                guard let payload = hit.payload else { continue }
                if let agency = args.agency, !agency.isEmpty {
                    let docAgency = payload.summaryMetadata?.agencyId
                        ?? payload.metadata["agency"] ?? ""
                    guard docAgency.localizedCaseInsensitiveContains(agency) else { continue }
                }
                sources.append(SearchResultDTO(
                    id: hit.id, text: payload.text, score: hit.score,
                    documentID: payload.documentID, chunkIndex: payload.chunkIndex,
                    metadata: payload.metadata, summaryMetadata: payload.summaryMetadata
                ))
                resultLines.append("[\(payload.documentID)]: \(payload.text)")
            }
            if resultLines.isEmpty {
                return ("No relevant results found for: \(args.query)", [])
            }
            return (resultLines.joined(separator: "\n\n"), sources)

        case "list_documents":
            let docs = try await qdrant.fetchAllUniqueDocuments()
            if docs.isEmpty {
                return ("No documents are currently indexed.", [])
            }
            let list = docs.map { "- \($0.documentID) (\($0.chunkCount) chunks)" }.joined(separator: "\n")
            return ("Available documents:\n\(list)", [])

        default:
            return ("Unknown tool: \(toolCall.function.name)", [])
        }
    }

    // Strips invalid JSON escape sequences that LLMs sometimes generate (e.g. `\_`, `\(`, `\)`).
    // JSON only allows: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    private func sanitizeToolArgs(_ json: String) -> String {
        var result = ""
        result.reserveCapacity(json.count)
        var i = json.startIndex
        while i < json.endIndex {
            let c = json[i]
            if c == "\\" {
                let next = json.index(after: i)
                guard next < json.endIndex else { result.append(c); i = next; continue }
                let nc = json[next]
                switch nc {
                case "\"", "\\", "/", "b", "f", "n", "r", "t", "u":
                    result.append(c)
                    result.append(nc)
                    i = json.index(after: next)
                default:
                    // Drop the backslash — keep only the following character
                    result.append(nc)
                    i = json.index(after: next)
                }
            } else {
                result.append(c)
                i = json.index(after: i)
            }
        }
        return result
    }

    private func makeToolDefinitions(defaultLimit: Int) -> [LLMToolDefinition] {
        [
            LLMToolDefinition(
                type: "function",
                function: LLMToolFunction(
                    name: "search_documents",
                    description: "Search the regulatory document database for relevant excerpts. Use this to find information before answering questions.",
                    parameters: LLMToolParameters(
                        type: "object",
                        properties: [
                            "query": LLMToolProperty(
                                type: "string",
                                description: "The search query to find relevant regulatory document excerpts. Use plain natural language — do not use boolean operators (AND/OR/NOT), backslashes, or special syntax."
                            ),
                            "agency": LLMToolProperty(
                                type: "string",
                                description: "Optional: filter results by regulatory agency name (e.g. FDA, EPA, USDA)"
                            ),
                            "limit": LLMToolProperty(
                                type: "integer",
                                description: "Number of results to return (1–10, default \(defaultLimit))"
                            )
                        ],
                        required: ["query"]
                    )
                )
            ),
            LLMToolDefinition(
                type: "function",
                function: LLMToolFunction(
                    name: "list_documents",
                    description: "List all documents currently available in the database. Use when the user asks what topics or documents are available.",
                    parameters: LLMToolParameters(
                        type: "object",
                        properties: [:],
                        required: []
                    )
                )
            )
        ]
    }
}

