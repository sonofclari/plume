import Foundation

struct Chunk: Sendable {
    let id: UUID
    let text: String
    let documentID: String
    let chunkIndex: Int
    let metadata: [String: String]

    init(text: String, documentID: String, chunkIndex: Int, metadata: [String: String] = [:]) {
        self.id = UUID()
        self.text = text
        self.documentID = documentID
        self.chunkIndex = chunkIndex
        self.metadata = metadata
    }
}
