import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

enum QdrantError: Error, LocalizedError {
    case httpError(Int, String)
    case decodingFailed(Error)
    case collectionSetupFailed(String)

    var errorDescription: String? {
        switch self {
        case .httpError(let code, let body):     return "Qdrant error \(code): \(body)"
        case .decodingFailed(let err):           return "Failed to decode Qdrant response: \(err)"
        case .collectionSetupFailed(let reason): return "Collection setup failed: \(reason)"
        }
    }
}

final class QdrantService: @unchecked Sendable {

    let baseURL: String
    let apiKey: String?
    let collectionName: String
    let vectorDimension: Int

    private let session: any HTTPClient
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init(baseURL: String, apiKey: String?, collectionName: String, vectorDimension: Int, httpClient: any HTTPClient) {
        self.baseURL         = baseURL
        self.apiKey          = apiKey
        self.collectionName  = collectionName
        self.vectorDimension = vectorDimension
        self.session         = httpClient
    }

    // MARK: - Collection

    /// Returns the number of points currently stored in the collection, or nil if the collection doesn't exist.
    func pointCount() async throws -> Int? {
        var req = urlRequest(method: "GET", path: "/collections/\(collectionName)")
        addAuthHeader(to: &req)
        let (data, res) = try await session.data(for: req)
        guard (res as? HTTPURLResponse)?.statusCode == 200 else { return nil }
        struct CollectionInfo: Decodable {
            struct Result: Decodable {
                let pointsCount: Int?
                enum CodingKeys: String, CodingKey { case pointsCount = "points_count" }
            }
            let result: Result
        }
        return (try? decoder.decode(CollectionInfo.self, from: data))?.result.pointsCount
    }

    /// Creates the collection if it does not already exist.
    func ensureCollection() async throws {
        guard !(try await collectionExists()) else { return }
        try await createCollection()
    }

    private func collectionExists() async throws -> Bool {
        var req = urlRequest(method: "GET", path: "/collections/\(collectionName)")
        addAuthHeader(to: &req)
        let (_, res) = try await session.data(for: req)
        return (res as? HTTPURLResponse)?.statusCode == 200
    }

    private func createCollection() async throws {
        var req = urlRequest(method: "PUT", path: "/collections/\(collectionName)")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        addAuthHeader(to: &req)

        let config = QdrantCollectionConfig(
            vectors: ["dense": .init(size: vectorDimension, distance: "Cosine")],
            sparseVectors: ["sparse": .init(index: .init(onDisk: false))]
        )
        req.httpBody = try encoder.encode(config)

        let (data, res) = try await session.data(for: req)
        guard let http = res as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw QdrantError.collectionSetupFailed(body)
        }
    }

    // MARK: - Upsert

    /// Upsert vector points in batches of 500 to stay within Qdrant payload limits.
    /// Automatically recreates the collection if it was deleted after startup.
    func upsertPoints(_ points: [QdrantPoint]) async throws {
        guard !points.isEmpty else { return }
        if !(try await collectionExists()) {
            try await createCollection()
        }
        let batchSize = 500
        for start in stride(from: 0, to: points.count, by: batchSize) {
            let end = min(start + batchSize, points.count)
            try await upsertBatch(Array(points[start..<end]))
        }
    }

    private func upsertBatch(_ points: [QdrantPoint]) async throws {
        var req = urlRequest(method: "PUT", path: "/collections/\(collectionName)/points")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        addAuthHeader(to: &req)
        req.httpBody = try encoder.encode(QdrantUpsertRequest(points: points))

        let (data, res) = try await session.data(for: req)
        try assertSuccess(data: data, response: res, context: "upsert")
    }

    // MARK: - Search

    func search(vector: [Float], limit: Int, threshold: Float) async throws -> [QdrantSearchHit] {
        var req = urlRequest(method: "POST", path: "/collections/\(collectionName)/points/search")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        addAuthHeader(to: &req)

        let body = QdrantSearchRequest(
            vector: vector,
            limit: limit,
            scoreThreshold: threshold,
            withPayload: true)
        req.httpBody = try encoder.encode(body)

        let (data, res) = try await session.data(for: req)
        try assertSuccess(data: data, response: res, context: "search")

        do {
            return try decoder.decode(QdrantSearchResponse.self, from: data).result
        } catch {
            throw QdrantError.decodingFailed(error)
        }
    }

    // MARK: - Hybrid Search

    func hybridSearch(
        denseVector: [Float],
        sparseVector: SparseVector,
        limit: Int,
        threshold: Float
    ) async throws -> [QdrantSearchHit] {
        var req = urlRequest(method: "POST", path: "/collections/\(collectionName)/points/query")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        addAuthHeader(to: &req)

        let prefetchLimit = min(limit * 5, 200)
        let body = QdrantHybridSearchRequest(
            prefetch: [
                .dense(vector: denseVector, limit: prefetchLimit),
                .sparse(vector: sparseVector, limit: prefetchLimit)
            ],
            query: .init(fusion: "rrf"),
            limit: limit,
            withPayload: true
        )
        req.httpBody = try encoder.encode(body)

        let (data, res) = try await session.data(for: req)
        try assertSuccess(data: data, response: res, context: "hybridSearch")

        do {
            let response = try decoder.decode(QdrantQueryResponse.self, from: data)
            let points = response.result.points
            return threshold > 0 ? points.filter { $0.score > threshold } : points
        } catch {
            throw QdrantError.decodingFailed(error)
        }
    }

    // MARK: - Scroll

    /// Fetch all payloads for a given documentID using Qdrant's scroll API (no vector needed).
    func fetchChunksByDocumentID(_ documentID: String) async throws -> [QdrantPayload] {
        var allPayloads: [QdrantPayload] = []
        var offset: String? = nil

        repeat {
            var req = urlRequest(method: "POST", path: "/collections/\(collectionName)/points/scroll")
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            addAuthHeader(to: &req)

            let body = QdrantScrollRequest(
                filter: .init(must: [.init(key: "documentID", match: .init(value: documentID))]),
                limit: 500,
                withPayload: true,
                withVectors: false
            )

            // Append offset for pagination if present
            if let off = offset {
                var bodyDict = try encoder.encode(body)
                var dict = try JSONSerialization.jsonObject(with: bodyDict) as? [String: Any] ?? [:]
                dict["offset"] = off
                bodyDict = try JSONSerialization.data(withJSONObject: dict)
                req.httpBody = bodyDict
            } else {
                req.httpBody = try encoder.encode(body)
            }

            let (data, res) = try await session.data(for: req)
            try assertSuccess(data: data, response: res, context: "scroll")

            let decoded = try decoder.decode(QdrantScrollResponse.self, from: data)
            let payloads = decoded.result.points.compactMap { $0.payload }
            allPayloads.append(contentsOf: payloads)
            offset = decoded.result.nextPageOffset
        } while offset != nil

        return allPayloads
    }

    // MARK: - List All Documents

    /// Scroll through all points and return one entry per unique documentID with chunk count and metadata.
    func fetchAllUniqueDocuments() async throws -> [UniqueDocument] {
        var documentMap: [String: (count: Int, metadata: [String: String], summaryMetadata: SummaryMetadata?)] = [:]
        var offset: String? = nil

        repeat {
            var req = urlRequest(method: "POST", path: "/collections/\(collectionName)/points/scroll")
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            addAuthHeader(to: &req)

            var dict: [String: Any] = [
                "limit": 500,
                "with_payload": true,
                "with_vectors": false
            ]
            if let off = offset { dict["offset"] = off }
            req.httpBody = try JSONSerialization.data(withJSONObject: dict)

            let (data, res) = try await session.data(for: req)
            try assertSuccess(data: data, response: res, context: "scrollAll")

            let decoded = try decoder.decode(QdrantScrollResponse.self, from: data)
            for point in decoded.result.points {
                guard let payload = point.payload else { continue }
                let docID = payload.documentID
                if let existing = documentMap[docID] {
                    documentMap[docID] = (count: existing.count + 1, metadata: existing.metadata, summaryMetadata: existing.summaryMetadata)
                } else {
                    documentMap[docID] = (count: 1, metadata: payload.metadata, summaryMetadata: payload.summaryMetadata)
                }
            }
            offset = decoded.result.nextPageOffset
        } while offset != nil

        return documentMap
            .map { UniqueDocument(documentID: $0.key, chunkCount: $0.value.count, metadata: $0.value.metadata, summaryMetadata: $0.value.summaryMetadata) }
            .sorted { $0.documentID < $1.documentID }
    }

    // MARK: - Delete

    func deleteByDocumentID(_ documentID: String) async throws {
        var req = urlRequest(method: "POST", path: "/collections/\(collectionName)/points/delete")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        addAuthHeader(to: &req)

        let body = QdrantDeleteRequest(
            filter: .init(must: [
                .init(key: "documentID", match: .init(value: documentID))
            ]))
        req.httpBody = try encoder.encode(body)

        let (data, res) = try await session.data(for: req)
        try assertSuccess(data: data, response: res, context: "delete")
    }

    // MARK: - Helpers

    private func urlRequest(method: String, path: String) -> URLRequest {
        var req = URLRequest(url: URL(string: "\(baseURL)\(path)")!)
        req.httpMethod = method
        return req
    }

    private func addAuthHeader(to request: inout URLRequest) {
        if let key = apiKey, !key.isEmpty {
            request.setValue(key, forHTTPHeaderField: "api-key")
        }
    }

    private func assertSuccess(data: Data, response: URLResponse, context: String) throws {
        guard let http = response as? HTTPURLResponse,
              (200...299).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            let body = String(data: data, encoding: .utf8) ?? ""
            throw QdrantError.httpError(code, "\(context): \(body)")
        }
    }
}
