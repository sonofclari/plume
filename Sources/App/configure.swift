import Vapor

public func configure(_ app: Application) async throws {
    // Allow large file uploads (100 MB)
    app.routes.defaultMaxBodySize = "100mb"

    // Serve static files from Public/
    app.middleware.use(FileMiddleware(publicDirectory: app.directory.publicDirectory))

    // Port
    app.http.server.configuration.port =
        Int(Environment.get("PORT") ?? "8080") ?? 8080

    // MARK: - Choose Embedder

    let embedder: any EmbedderService = makeHTTPEmbedder(app: app)
    app.logger.info("🌐 HTTP embeddings via: \(Environment.get("EMBEDDER_URL") ?? "http://localhost:8001/v1")")

    let qdrantSessionConfig = URLSessionConfiguration.ephemeral
    qdrantSessionConfig.timeoutIntervalForRequest = 30

    let qdrantService = QdrantService(
        baseURL:         Environment.get("QDRANT_URL")        ?? "http://localhost:6333",
        apiKey:          Environment.get("QDRANT_API_KEY"),
        collectionName:  Environment.get("QDRANT_COLLECTION") ?? "rag-documents",
        vectorDimension: embedder.dimension,
        httpClient:      URLSession(configuration: qdrantSessionConfig)
    )

    app.embedder      = embedder
    app.qdrantService = qdrantService

    // MARK: - LLM Service (chat completions for document summarisation)

    let llmService: any LLMService = makeHTTPLLM(app: app)
    app.logger.info("🌐 LLM via HTTP: \(Environment.get("LLM_URL") ?? "http://localhost:8000/v1")")

    app.llmService   = llmService
    app.summaryCache = SummaryCache()

    // Ensure the Qdrant collection is ready before accepting traffic
    try await qdrantService.ensureCollection()
    app.logger.info("✅ Qdrant collection '\(qdrantService.collectionName)' ready (dim: \(embedder.dimension))")

    try routes(app)
}

// MARK: - Helper

private func makeHTTPEmbedder(app: Application) -> OpenAIEmbeddingService {
    let dimension = Int(Environment.get("EMBEDDING_DIMENSION") ?? "768") ?? 768

    let sessionConfig = URLSessionConfiguration.ephemeral
    sessionConfig.timeoutIntervalForRequest  = 300
    sessionConfig.timeoutIntervalForResource = 900

    return OpenAIEmbeddingService(
        baseURL:    Environment.get("EMBEDDER_URL")   ?? "http://localhost:8001/v1",
        apiKey:     Environment.get("EMBEDDER_KEY")   ?? "vllm",
        model:      Environment.get("EMBEDDER_MODEL") ?? "nomic-ai/nomic-embed-text-v1.5",
        dimension:  dimension,
        batchSize:  Int(Environment.get("EMBEDDING_BATCH_SIZE") ?? "50") ?? 50,
        httpClient: URLSession(configuration: sessionConfig)
    )
}

private func makeHTTPLLM(app: Application) -> OpenAILLMService {
    let sessionConfig = URLSessionConfiguration.ephemeral
    sessionConfig.timeoutIntervalForRequest  = 300
    sessionConfig.timeoutIntervalForResource = 600

    return OpenAILLMService(
        baseURL:    Environment.get("LLM_URL")     ?? Environment.get("EMBEDDER_URL") ?? "http://localhost:8000/v1",
        apiKey:     Environment.get("LLM_API_KEY") ?? Environment.get("EMBEDDER_KEY") ?? "vllm",
        model:      Environment.get("LLM_MODEL")   ?? "Qwen/Qwen2.5-7B-Instruct",
        httpClient: URLSession(configuration: sessionConfig)
    )
}
