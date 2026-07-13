import Vapor

func routes(_ app: Application) throws {
    let rag = RAGController(
        embedder:      app.embedder,
        qdrantService: app.qdrantService
    )

    // Root → search UI
    app.get { req async throws -> Response in
        return req.redirect(to: "/index.html")
    }

    let api = app.grouped("api")

    // Health
    api.get("health") { req async throws -> [String: String] in
        let count = try await app.qdrantService.pointCount()
        return [
            "status": "ok",
            "service": "Plume",
            "collection": app.qdrantService.collectionName,
            "points_indexed": count.map(String.init) ?? "unknown",
        ]
    }

    // Indexing
    let index = api.grouped("index")
    index.post("text", use: rag.indexText)
    index.post("file", use: rag.indexFile)
    index.post("regulation", use: rag.indexRegulation)

    // Search
    api.post("search", use: rag.search)

    // Conversational chat (RAG + LLM answer)
    api.post("chat", use: rag.chat)

    // Document management
    api.get("documents", use: rag.listDocuments)
    api.get("documents", ":documentID", "summary", use: rag.summarizeDocument)
    api.delete("documents", ":documentID", use: rag.deleteDocument)
}
