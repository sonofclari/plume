import Foundation
import Testing
@testable import App

@Suite("Chunk")
struct ChunkTests {

    @Test("defaults to empty metadata")
    func defaultsToEmptyMetadata() {
        let chunk = Chunk(text: "body", documentID: "doc-1", chunkIndex: 0)
        #expect(chunk.metadata.isEmpty)
    }

    @Test("stores the fields it was given")
    func storesGivenFields() {
        let chunk = Chunk(text: "body", documentID: "doc-1", chunkIndex: 3, metadata: ["agency": "FDA"])
        #expect(chunk.text == "body")
        #expect(chunk.documentID == "doc-1")
        #expect(chunk.chunkIndex == 3)
        #expect(chunk.metadata["agency"] == "FDA")
    }

    @Test("each instance gets a unique id")
    func eachInstanceGetsUniqueID() {
        let first = Chunk(text: "a", documentID: "doc", chunkIndex: 0)
        let second = Chunk(text: "a", documentID: "doc", chunkIndex: 0)
        #expect(first.id != second.id)
    }
}
