import Vapor

struct RAGController: Sendable {

    let embedder: any EmbedderService
    let qdrantService: QdrantService
    let chunkingService = ChunkingService()
    let parserService   = DocumentParserService()
    let bm25            = BM25Service()

    // MARK: - POST /api/index/text

    func indexText(_ req: Request) async throws -> IndexResponse {
        let body = try req.content.decode(IndexTextRequest.self)

        let text = body.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            throw Abort(.badRequest, reason: "Text cannot be empty")
        }

        let documentID = body.documentID ?? UUID().uuidString
        let config     = makeChunkingConfig(strategy: body.strategy,
                                            chunkSize: body.chunkSize,
                                            overlap: body.overlapPercentage)
        let chunks = chunkingService.chunk(text: text, documentID: documentID, config: config)
        guard !chunks.isEmpty else {
            throw Abort(.badRequest, reason: "No chunks generated from the provided text")
        }

        let points = try await buildPoints(from: chunks, metadata: body.metadata ?? [:], summaryMetadata: body.summaryMetadata)
        try await qdrantService.upsertPoints(points)

        req.logger.info("Indexed \(chunks.count) chunks for document '\(documentID)'")
        return IndexResponse(
            documentID: documentID,
            chunksIndexed: chunks.count,
            message: "Successfully indexed \(chunks.count) chunk(s) for document '\(documentID)'"
        )
    }

    // MARK: - POST /api/index/file

    func indexFile(_ req: Request) async throws -> IndexResponse {
        let body = try req.content.decode(IndexFileRequest.self)

        let data     = Data(buffer: body.file.data)
        let filename = body.file.filename

        guard !filename.isEmpty else {
            throw Abort(.badRequest, reason: "File must have a filename")
        }

        let text = try await parserService.parse(data: data, filename: filename)
        let baseName   = URL(fileURLWithPath: filename).deletingPathExtension().lastPathComponent
        let documentID = body.documentID ?? "\(baseName)_\(UUID().uuidString)"

        let config = makeChunkingConfig(strategy: body.strategy,
                                        chunkSize: body.chunkSize,
                                        overlap: body.overlapPercentage)
        var chunks = chunkingService.chunk(text: text, documentID: documentID, config: config)
        guard !chunks.isEmpty else {
            throw Abort(.unprocessableEntity, reason: "No content extracted from '\(filename)'")
        }

        let maxChunks = Int(Environment.get("MAX_DOCUMENT_CHUNKS") ?? "300") ?? 300
        if chunks.count > maxChunks {
            req.logger.warning("'\(filename)' produced \(chunks.count) chunks — capping at \(maxChunks)")
            chunks = Array(chunks.prefix(maxChunks))
        }

        let summaryMetadata: SummaryMetadata? = body.summaryMetadataJson
            .flatMap { $0.data(using: .utf8) }
            .flatMap { try? JSONDecoder().decode(SummaryMetadata.self, from: $0) }

        let points = try await buildPoints(from: chunks, metadata: ["source_file": filename], summaryMetadata: summaryMetadata)
        try await qdrantService.upsertPoints(points)

        req.logger.info("Indexed \(chunks.count) chunks from '\(filename)' as '\(documentID)'")
        return IndexResponse(
            documentID: documentID,
            chunksIndexed: chunks.count,
            message: "Successfully indexed \(chunks.count) chunk(s) from '\(filename)'"
        )
    }

    // MARK: - POST /api/index/regulation

    func indexRegulation(_ req: Request) async throws -> IndexResponse {
        let body = try req.content.decode(IndexRegulationRequest.self)

        let text = body.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            throw Abort(.badRequest, reason: "Text cannot be empty")
        }

        let documentID = body.data.id
        let config = makeChunkingConfig(strategy: body.strategy,
                                        chunkSize: body.chunkSize,
                                        overlap: body.overlapPercentage)
        let chunks = chunkingService.chunk(text: text, documentID: documentID, config: config)
        guard !chunks.isEmpty else {
            throw Abort(.badRequest, reason: "No chunks generated from the provided text")
        }

        let points = try await buildPoints(from: chunks, metadata: [:], summaryMetadata: body.data.attributes)
        try await qdrantService.upsertPoints(points)

        req.logger.info("Indexed \(chunks.count) chunks for regulation '\(documentID)'")
        return IndexResponse(
            documentID: documentID,
            chunksIndexed: chunks.count,
            message: "Successfully indexed \(chunks.count) chunk(s) for regulation '\(documentID)'"
        )
    }


    func search(_ req: Request) async throws -> SearchResponse {
        let body = try req.content.decode(SearchRequest.self)

        let query = body.query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else {
            throw Abort(.badRequest, reason: "Query cannot be empty")
        }

        let limit     = min(max(body.limit ?? 20, 1), 100)
        let threshold = body.threshold ?? 0.0

        let vectors = try await embedder.embed(texts: [query])
        guard let queryVector = vectors.first else {
            throw Abort(.internalServerError, reason: "Failed to embed query")
        }
        let sparseQuery = bm25.encode(query)

        let hits = try await qdrantService.hybridSearch(
            denseVector: queryVector, sparseVector: sparseQuery, limit: limit, threshold: threshold)

        req.logger.info("Qdrant returned \(hits.count) hit(s) for query '\(query.prefix(60))'")

        let results: [SearchResultDTO] = hits.compactMap { hit in
            guard let payload = hit.payload else {
                req.logger.warning("Hit \(hit.id) (score=\(hit.score)) has no payload — skipping")
                return nil
            }
            return SearchResultDTO(
                id: hit.id,
                text: payload.text,
                score: hit.score,
                documentID: payload.documentID,
                chunkIndex: payload.chunkIndex,
                metadata: payload.metadata,
                summaryMetadata: payload.summaryMetadata
            )
        }

        req.logger.info("Returning \(results.count) result(s) after payload filtering")
        return SearchResponse(query: query, results: results, count: results.count)
    }

    // MARK: - POST /api/chat

    func chat(_ req: Request) async throws -> ChatResponse {
        let body = try req.content.decode(ChatRequest.self)

        let query = body.query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else {
            throw Abort(.badRequest, reason: "Query cannot be empty")
        }

        let limit     = 8
        let threshold: Float = 0.0
        let history   = body.conversationHistory ?? []

        let result = try await req.application.llmService.chatWithTools(
            query: query,
            conversationHistory: history,
            embedder: embedder,
            qdrant: qdrantService,
            bm25: bm25,
            limit: limit,
            threshold: threshold
        )

        req.logger.info("Chat: Answer generated for '\(query.prefix(60))' with \(result.sources.count) source(s)")
        return ChatResponse(answer: result.answer, sources: result.sources, query: query)
    }

    // MARK: - GET /api/documents

    func listDocuments(_ req: Request) async throws -> DocumentsListResponse {
        let entries = try await qdrantService.fetchAllUniqueDocuments()
        let documents = entries.map { entry in
            DocumentInfo(
                documentID: entry.documentID,
                chunkCount: entry.chunkCount,
                metadata: entry.metadata,
                summaryMetadata: entry.summaryMetadata
            )
        }
        req.logger.info("Listed \(documents.count) unique document(s)")
        return DocumentsListResponse(documents: documents, count: documents.count)
    }

    // MARK: - GET /api/documents/:documentID/summary

    func summarizeDocument(_ req: Request) async throws -> SummaryResponse {
        guard let documentID = req.parameters.get("documentID") else {
            throw Abort(.badRequest, reason: "Missing documentID parameter")
        }

        // Return cached summary if available
        if let cached = await req.application.summaryCache.get(documentID) {
            return SummaryResponse(documentID: documentID, summary: cached)
        }

        let payloads = try await qdrantService.fetchChunksByDocumentID(documentID)
        guard !payloads.isEmpty else {
            throw Abort(.notFound, reason: "No chunks found for document '\(documentID)'")
        }

        let orderedTexts = payloads
            .sorted { $0.chunkIndex < $1.chunkIndex }
            .map { $0.text }

        let summary = try await req.application.llmService.summarize(
            documentID: documentID,
            orderedChunks: orderedTexts
        )

        await req.application.summaryCache.set(documentID, value: summary)
        req.logger.info("📝 Summary generated for document '\(documentID)'")

        return SummaryResponse(documentID: documentID, summary: summary)
    }

    // MARK: - DELETE /api/documents/:documentID

    func deleteDocument(_ req: Request) async throws -> DeleteResponse {
        guard let documentID = req.parameters.get("documentID") else {
            throw Abort(.badRequest, reason: "Missing documentID parameter")
        }

        try await qdrantService.deleteByDocumentID(documentID)
        req.logger.info("Deleted all chunks for document '\(documentID)'")

        return DeleteResponse(
            documentID: documentID,
            message: "All chunks for document '\(documentID)' have been deleted"
        )
    }

    // MARK: - Private Helpers

    private func buildPoints(from chunks: [Chunk], metadata: [String: String], summaryMetadata: SummaryMetadata? = nil) async throws -> [QdrantPoint] {
        let texts      = chunks.map { $0.text }
        let embeddings = try await embedder.embed(texts: texts)

        return zip(chunks, embeddings).map { chunk, denseVector in
            let sparseVector = bm25.encode(chunk.text)
            return QdrantPoint(
                id: chunk.id.uuidString,
                vector: QdrantNamedVectors(dense: denseVector, sparse: sparseVector),
                payload: QdrantPayload(
                    text: chunk.text,
                    documentID: chunk.documentID,
                    chunkIndex: chunk.chunkIndex,
                    metadata: metadata,
                    summaryMetadata: summaryMetadata
                )
            )
        }
    }

    private func makeChunkingConfig(strategy: String?, chunkSize: Int?, overlap: Double?) -> ChunkingConfig {
        ChunkingConfig(
            strategy: strategy.flatMap { ChunkingStrategy(rawValue: $0) } ?? .paragraph,
            chunkSize: chunkSize ?? 1_000,
            overlapPercentage: overlap ?? 0.15
        )
    }
}
