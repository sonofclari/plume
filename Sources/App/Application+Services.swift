import Vapor

extension Application {

    struct EmbedderKey: StorageKey {
        typealias Value = any EmbedderService
    }

    struct QdrantServiceKey: StorageKey {
        typealias Value = QdrantService
    }

    struct LLMServiceKey: StorageKey {
        typealias Value = any LLMService
    }

    struct SummaryCacheKey: StorageKey {
        typealias Value = SummaryCache
    }

    var embedder: any EmbedderService {
        get { storage[EmbedderKey.self]! }
        set { storage[EmbedderKey.self] = newValue }
    }

    var qdrantService: QdrantService {
        get { storage[QdrantServiceKey.self]! }
        set { storage[QdrantServiceKey.self] = newValue }
    }

    var llmService: any LLMService {
        get { storage[LLMServiceKey.self]! }
        set { storage[LLMServiceKey.self] = newValue }
    }

    var summaryCache: SummaryCache {
        get { storage[SummaryCacheKey.self]! }
        set { storage[SummaryCacheKey.self] = newValue }
    }
}
