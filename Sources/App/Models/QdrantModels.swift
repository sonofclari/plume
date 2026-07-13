import Foundation

// MARK: - Collection

struct QdrantCollectionConfig: Codable, Sendable {
    let vectors: [String: DenseVectorParams]
    let sparseVectors: [String: SparseVectorConfig]

    enum CodingKeys: String, CodingKey {
        case vectors
        case sparseVectors = "sparse_vectors"
    }

    struct DenseVectorParams: Codable, Sendable {
        let size: Int
        let distance: String
    }

    struct SparseVectorConfig: Codable, Sendable {
        let index: SparseIndexParams

        struct SparseIndexParams: Codable, Sendable {
            let onDisk: Bool

            enum CodingKeys: String, CodingKey {
                case onDisk = "on_disk"
            }
        }
    }
}

// MARK: - Sparse + Named Vectors

struct SparseVector: Codable, Sendable {
    let indices: [Int]
    let values: [Float]
}

struct QdrantNamedVectors: Codable, Sendable {
    let dense: [Float]
    let sparse: SparseVector
}

// MARK: - Upsert

struct QdrantUpsertRequest: Codable, Sendable {
    let points: [QdrantPoint]
}

struct QdrantPoint: Codable, Sendable {
    let id: String
    let vector: QdrantNamedVectors
    let payload: QdrantPayload
}

struct QdrantPayload: Codable, Sendable {
    let text: String
    let documentID: String
    let chunkIndex: Int
    let metadata: [String: String]
    let summaryMetadata: SummaryMetadata?
}

// MARK: - Search (dense-only, kept for reference)

struct QdrantSearchRequest: Codable, Sendable {
    let vector: [Float]
    let limit: Int
    let scoreThreshold: Float?
    let withPayload: Bool

    enum CodingKeys: String, CodingKey {
        case vector
        case limit
        case scoreThreshold = "score_threshold"
        case withPayload = "with_payload"
    }
}

struct QdrantSearchResponse: Codable, Sendable {
    let result: [QdrantSearchHit]
    let status: String
}

struct QdrantSearchHit: Codable, Sendable {
    let id: String
    let score: Float
    let payload: QdrantPayload?
}

// MARK: - Hybrid Search

enum QdrantPrefetchQuery: Encodable, Sendable {
    case dense(vector: [Float], limit: Int)
    case sparse(vector: SparseVector, limit: Int)

    private enum CodingKeys: String, CodingKey {
        case query
        case vectorName = "using"
        case limit
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .dense(let vector, let limit):
            try container.encode(vector, forKey: .query)
            try container.encode("dense", forKey: .vectorName)
            try container.encode(limit, forKey: .limit)
        case .sparse(let vector, let limit):
            try container.encode(vector, forKey: .query)
            try container.encode("sparse", forKey: .vectorName)
            try container.encode(limit, forKey: .limit)
        }
    }
}

struct QdrantHybridSearchRequest: Encodable, Sendable {
    let prefetch: [QdrantPrefetchQuery]
    let query: FusionQuery
    let limit: Int
    let withPayload: Bool

    enum CodingKeys: String, CodingKey {
        case prefetch, query, limit
        case withPayload = "with_payload"
    }

    struct FusionQuery: Encodable, Sendable {
        let fusion: String
    }
}

struct QdrantQueryResponse: Codable, Sendable {
    let result: QdrantQueryResult
    let status: String
}

struct QdrantQueryResult: Codable, Sendable {
    let points: [QdrantSearchHit]
}

// MARK: - Scroll

struct QdrantScrollRequest: Codable, Sendable {
    let filter: QdrantFilter
    let limit: Int
    let withPayload: Bool
    let withVectors: Bool

    enum CodingKeys: String, CodingKey {
        case filter
        case limit
        case withPayload  = "with_payload"
        case withVectors  = "with_vectors"
    }
}

struct QdrantScrollResponse: Codable, Sendable {
    let result: QdrantScrollResult
    let status: String
}

struct QdrantScrollResult: Codable, Sendable {
    let points: [QdrantScrollHit]
    let nextPageOffset: String?

    enum CodingKeys: String, CodingKey {
        case points
        case nextPageOffset = "next_page_offset"
    }
}

struct QdrantScrollHit: Codable, Sendable {
    let id: String
    let payload: QdrantPayload?
}

// MARK: - Delete

struct QdrantDeleteRequest: Codable, Sendable {
    let filter: QdrantFilter
}

// MARK: - Document Listing

struct UniqueDocument: Sendable {
    let documentID: String
    let chunkCount: Int
    let metadata: [String: String]
    let summaryMetadata: SummaryMetadata?
}

struct QdrantFilter: Codable, Sendable {
    let must: [QdrantCondition]
}

struct QdrantCondition: Codable, Sendable {
    let key: String
    let match: QdrantMatch
}

struct QdrantMatch: Codable, Sendable {
    let value: String
}
