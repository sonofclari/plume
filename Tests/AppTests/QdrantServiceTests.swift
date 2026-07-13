import Foundation
import Testing
@testable import App

@Suite("QdrantService")
struct QdrantServiceTests {

    private func makeService(httpClient: any HTTPClient) -> QdrantService {
        QdrantService(
            baseURL: "https://qdrant.test", apiKey: "secret",
            collectionName: "regs", vectorDimension: 4, httpClient: httpClient
        )
    }

    // MARK: - pointCount

    @Test("pointCount returns nil when the collection does not exist")
    func pointCountReturnsNilWhenMissing() async throws {
        let session = MockHTTPClient { _ in (404, Data()) }
        let service = makeService(httpClient: session)
        let count = try await service.pointCount()
        #expect(count == nil)
    }

    @Test("pointCount parses points_count from a successful response")
    func pointCountParsesResponse() async throws {
        let session = MockHTTPClient { _ in
            (200, Data(#"{"result":{"points_count":42},"status":"ok"}"#.utf8))
        }
        let service = makeService(httpClient: session)
        let count = try await service.pointCount()
        #expect(count == 42)
    }

    @Test("pointCount sends an authenticated GET request")
    func pointCountSendsAuthHeader() async throws {
        let capturedRequest = Locked<(method: String?, apiKey: String?, path: String?)?>(nil)
        let session = MockHTTPClient { request in
            capturedRequest.mutate {
                $0 = (request.httpMethod, request.value(forHTTPHeaderField: "api-key"), request.url?.path)
            }
            return (200, Data(#"{"result":{"points_count":0},"status":"ok"}"#.utf8))
        }
        let service = makeService(httpClient: session)
        _ = try await service.pointCount()
        #expect(capturedRequest.value?.method == "GET")
        #expect(capturedRequest.value?.apiKey == "secret")
        #expect(capturedRequest.value?.path == "/collections/regs")
    }

    // MARK: - ensureCollection

    @Test("ensureCollection does nothing when the collection already exists")
    func ensureCollectionSkipsCreationWhenExists() async throws {
        let requestCount = Locked<Int>(0)
        let session = MockHTTPClient { _ in
            requestCount.mutate { $0 += 1 }
            return (200, Data())
        }
        let service = makeService(httpClient: session)
        try await service.ensureCollection()
        #expect(requestCount.value == 1) // only the existence check, no create
    }

    @Test("ensureCollection creates the collection with dense + sparse vector config when missing")
    func ensureCollectionCreatesWhenMissing() async throws {
        let createRequestBody = Locked<Data?>(nil)
        let createRequestMethod = Locked<String?>(nil)
        let session = MockHTTPClient { request in
            if request.httpMethod == "GET" {
                return (404, Data())
            }
            createRequestMethod.mutate { $0 = request.httpMethod }
            createRequestBody.mutate { $0 = request.capturedBody() }
            return (200, Data())
        }
        let service = makeService(httpClient: session)
        try await service.ensureCollection()

        #expect(createRequestMethod.value == "PUT")
        let body = try #require(createRequestBody.value)
        let config = try JSONDecoder().decode(QdrantCollectionConfig.self, from: body)
        #expect(config.vectors["dense"]?.size == 4)
        #expect(config.vectors["dense"]?.distance == "Cosine")
        #expect(config.sparseVectors["sparse"] != nil)
    }

    // MARK: - upsertPoints

    @Test("upsertPoints does nothing for an empty array")
    func upsertEmptyArrayIsNoOp() async throws {
        let requestMade = Locked<Bool>(false)
        let session = MockHTTPClient { _ in
            requestMade.mutate { $0 = true }
            return (200, Data())
        }
        let service = makeService(httpClient: session)
        try await service.upsertPoints([])
        #expect(!requestMade.value)
    }

    @Test("upsertPoints sends all points to the points endpoint")
    func upsertSendsAllPoints() async throws {
        let upsertedCount = Locked<Int?>(nil)
        let session = MockHTTPClient { request in
            if request.httpMethod == "GET" { return (200, Data()) } // collection exists
            let body = request.capturedBody() ?? Data()
            let decoded = try JSONDecoder().decode(QdrantUpsertRequest.self, from: body)
            upsertedCount.mutate { $0 = decoded.points.count }
            return (200, Data(#"{"result":{},"status":"ok"}"#.utf8))
        }
        let service = makeService(httpClient: session)
        let points = (0..<3).map { i in
            QdrantPoint(
                id: "\(i)",
                vector: QdrantNamedVectors(dense: [0, 0, 0, 0], sparse: SparseVector(indices: [], values: [])),
                payload: QdrantPayload(text: "t\(i)", documentID: "doc", chunkIndex: i, metadata: [:], summaryMetadata: nil)
            )
        }
        try await service.upsertPoints(points)
        #expect(upsertedCount.value == 3)
    }

    // MARK: - search

    @Test("search decodes hits from a successful response")
    func searchDecodesHits() async throws {
        let session = MockHTTPClient { _ in
            let json = """
            {"result":[{"id":"a","score":0.9,"payload":{"text":"hello","documentID":"doc-1","chunkIndex":0,"metadata":{},"summaryMetadata":null}}],"status":"ok"}
            """
            return (200, Data(json.utf8))
        }
        let service = makeService(httpClient: session)
        let hits = try await service.search(vector: [0, 0, 0, 0], limit: 5, threshold: 0)
        #expect(hits.count == 1)
        #expect(hits[0].id == "a")
        #expect(hits[0].payload?.text == "hello")
    }

    @Test("search throws httpError for a non-2xx response")
    func searchThrowsOnHTTPError() async throws {
        let session = MockHTTPClient { _ in (503, Data("unavailable".utf8)) }
        let service = makeService(httpClient: session)
        await #expect(throws: QdrantError.self) {
            _ = try await service.search(vector: [0, 0, 0, 0], limit: 5, threshold: 0)
        }
    }

    @Test("search throws decodingFailed for malformed JSON")
    func searchThrowsOnMalformedJSON() async throws {
        let session = MockHTTPClient { _ in (200, Data("not json".utf8)) }
        let service = makeService(httpClient: session)
        await #expect(throws: QdrantError.self) {
            _ = try await service.search(vector: [0, 0, 0, 0], limit: 5, threshold: 0)
        }
    }

    // MARK: - hybridSearch

    @Test("hybridSearch filters out hits at or below the threshold")
    func hybridSearchFiltersLowScores() async throws {
        let session = MockHTTPClient { _ in
            let json = """
            {"result":{"points":[
                {"id":"low","score":0.1,"payload":null},
                {"id":"high","score":0.9,"payload":null}
            ]},"status":"ok"}
            """
            return (200, Data(json.utf8))
        }
        let service = makeService(httpClient: session)
        let hits = try await service.hybridSearch(
            denseVector: [0, 0, 0, 0], sparseVector: SparseVector(indices: [], values: []),
            limit: 5, threshold: 0.5
        )
        #expect(hits.map(\.id) == ["high"])
    }

    @Test("hybridSearch keeps all hits when threshold is zero")
    func hybridSearchKeepsAllWhenThresholdZero() async throws {
        let session = MockHTTPClient { _ in
            let json = """
            {"result":{"points":[
                {"id":"low","score":0.0,"payload":null},
                {"id":"high","score":0.9,"payload":null}
            ]},"status":"ok"}
            """
            return (200, Data(json.utf8))
        }
        let service = makeService(httpClient: session)
        let hits = try await service.hybridSearch(
            denseVector: [0, 0, 0, 0], sparseVector: SparseVector(indices: [], values: []),
            limit: 5, threshold: 0
        )
        #expect(hits.count == 2)
    }

    @Test("hybridSearch caps prefetch limit at 200 even for large limits")
    func hybridSearchCapsPrefetchLimit() async throws {
        let capturedBody = Locked<Data?>(nil)
        let session = MockHTTPClient { request in
            capturedBody.mutate { $0 = request.capturedBody() }
            return (200, Data(#"{"result":{"points":[]},"status":"ok"}"#.utf8))
        }
        let service = makeService(httpClient: session)
        _ = try await service.hybridSearch(
            denseVector: [0, 0, 0, 0], sparseVector: SparseVector(indices: [], values: []),
            limit: 100, threshold: 0
        )
        let body = try #require(capturedBody.value)
        let json = try JSONSerialization.jsonObject(with: body) as? [String: Any]
        let prefetch = try #require(json?["prefetch"] as? [[String: Any]])
        for entry in prefetch {
            #expect(entry["limit"] as? Int == 200)
        }
    }

    // MARK: - fetchChunksByDocumentID (pagination)

    @Test("fetchChunksByDocumentID follows next_page_offset until exhausted")
    func fetchChunksByDocumentIDPaginates() async throws {
        let session = MockHTTPClient { request in
            let bodyData = request.capturedBody() ?? Data()
            let dict = try JSONSerialization.jsonObject(with: bodyData) as? [String: Any]
            let isSecondPage = dict?["offset"] != nil
            if isSecondPage {
                let json = """
                {"result":{"points":[{"id":"2","payload":{"text":"page2","documentID":"doc","chunkIndex":1,"metadata":{},"summaryMetadata":null}}],"next_page_offset":null},"status":"ok"}
                """
                return (200, Data(json.utf8))
            } else {
                let json = """
                {"result":{"points":[{"id":"1","payload":{"text":"page1","documentID":"doc","chunkIndex":0,"metadata":{},"summaryMetadata":null}}],"next_page_offset":"cursor-1"},"status":"ok"}
                """
                return (200, Data(json.utf8))
            }
        }
        let service = makeService(httpClient: session)
        let payloads = try await service.fetchChunksByDocumentID("doc")
        #expect(payloads.map(\.text).sorted() == ["page1", "page2"])
    }

    // MARK: - fetchAllUniqueDocuments

    @Test("fetchAllUniqueDocuments aggregates chunk counts per document and sorts by documentID")
    func fetchAllUniqueDocumentsAggregates() async throws {
        let session = MockHTTPClient { _ in
            let json = """
            {"result":{"points":[
                {"id":"1","payload":{"text":"a","documentID":"zzz-doc","chunkIndex":0,"metadata":{},"summaryMetadata":null}},
                {"id":"2","payload":{"text":"b","documentID":"aaa-doc","chunkIndex":0,"metadata":{},"summaryMetadata":null}},
                {"id":"3","payload":{"text":"c","documentID":"aaa-doc","chunkIndex":1,"metadata":{},"summaryMetadata":null}}
            ],"next_page_offset":null},"status":"ok"}
            """
            return (200, Data(json.utf8))
        }
        let service = makeService(httpClient: session)
        let docs = try await service.fetchAllUniqueDocuments()
        #expect(docs.map(\.documentID) == ["aaa-doc", "zzz-doc"])
        #expect(docs.first(where: { $0.documentID == "aaa-doc" })?.chunkCount == 2)
        #expect(docs.first(where: { $0.documentID == "zzz-doc" })?.chunkCount == 1)
    }

    // MARK: - deleteByDocumentID

    @Test("deleteByDocumentID sends a filtered delete request")
    func deleteByDocumentIDSendsFilter() async throws {
        let capturedBody = Locked<Data?>(nil)
        let session = MockHTTPClient { request in
            capturedBody.mutate { $0 = request.capturedBody() }
            return (200, Data())
        }
        let service = makeService(httpClient: session)
        try await service.deleteByDocumentID("doc-99")
        let body = try #require(capturedBody.value)
        let decoded = try JSONDecoder().decode(QdrantDeleteRequest.self, from: body)
        #expect(decoded.filter.must.first?.key == "documentID")
        #expect(decoded.filter.must.first?.match.value == "doc-99")
    }

    @Test("deleteByDocumentID throws httpError on failure")
    func deleteByDocumentIDThrowsOnFailure() async throws {
        let session = MockHTTPClient { _ in (500, Data("boom".utf8)) }
        let service = makeService(httpClient: session)
        await #expect(throws: QdrantError.self) {
            try await service.deleteByDocumentID("doc-99")
        }
    }
}
