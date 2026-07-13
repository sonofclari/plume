import Vapor

// MARK: - Request DTOs

struct IndexTextRequest: Content, Sendable {
    let text: String
    let documentID: String?
    let strategy: String?
    let chunkSize: Int?
    let overlapPercentage: Double?
    let metadata: [String: String]?
    let summaryMetadata: SummaryMetadata?
}

struct IndexFileRequest: Content, Sendable {
    let file: File
    let documentID: String?
    let strategy: String?
    let chunkSize: Int?
    let overlapPercentage: Double?
    let summaryMetadataJson: String?   // JSON-encoded SummaryMetadata (multipart safe)
}

struct IndexRegulationRequest: Content, Sendable {
    let data: RegulationDocument
    let text: String
    let strategy: String?
    let chunkSize: Int?
    let overlapPercentage: Double?
}

struct SearchRequest: Content, Sendable {
    let query: String
    let limit: Int?
    let threshold: Float?
}

// MARK: - Response DTOs

struct IndexResponse: Content, Sendable {
    let documentID: String
    let chunksIndexed: Int
    let message: String
}

struct SearchResultDTO: Content, Sendable {
    let id: String
    let text: String
    let score: Float
    let documentID: String
    let chunkIndex: Int
    let metadata: [String: String]
    let summaryMetadata: SummaryMetadata?
}

struct SearchResponse: Content, Sendable {
    let query: String
    let results: [SearchResultDTO]
    let count: Int
}

struct DeleteResponse: Content, Sendable {
    let documentID: String
    let message: String
}

struct SummaryResponse: Content, Sendable {
    let documentID: String
    let summary: String
}

struct ChatMessageDTO: Content, Sendable {
    let role: String
    let content: String
}

struct ChatRequest: Content, Sendable {
    let query: String
    let conversationHistory: [ChatMessageDTO]?
}

struct ChatResponse: Content, Sendable {
    let answer: String
    let sources: [SearchResultDTO]
    let query: String
}

struct DocumentInfo: Content, Sendable {
    let documentID: String
    let chunkCount: Int
    let metadata: [String: String]
    let summaryMetadata: SummaryMetadata?
}

struct DocumentsListResponse: Content, Sendable {
    let documents: [DocumentInfo]
    let count: Int
}

