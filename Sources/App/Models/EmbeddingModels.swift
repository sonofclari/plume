import Foundation

struct OpenAIEmbeddingRequest: Codable, Sendable {
    let input: [String]
    let model: String
    let dimensions: Int?
}

struct OpenAIEmbeddingResponse: Codable, Sendable {
    let object: String
    let data: [EmbeddingObject]
    let model: String
    let usage: EmbeddingUsage
}

struct EmbeddingObject: Codable, Sendable {
    let object: String
    let embedding: [Float]
    let index: Int
}

struct EmbeddingUsage: Codable, Sendable {
    let promptTokens: Int
    let totalTokens: Int

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case totalTokens = "total_tokens"
    }
}
