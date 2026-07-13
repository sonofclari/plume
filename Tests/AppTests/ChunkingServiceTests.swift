import Foundation
import Testing
@testable import App

@Suite("ChunkingService")
struct ChunkingServiceTests {

    @Test("empty text produces no chunks")
    func emptyTextProducesNoChunks() {
        let service = ChunkingService()
        let chunks = service.chunk(text: "", documentID: "doc-1", config: .default)
        #expect(chunks.isEmpty)
    }

    @Test("chunk indices are sequential starting at zero")
    func chunkIndicesAreSequential() {
        let service = ChunkingService()
        let text = Array(repeating: "This is a paragraph of text.", count: 50).joined(separator: "\n\n")
        let config = ChunkingConfig(strategy: .paragraph, chunkSize: 200, overlapPercentage: 0.1)
        let chunks = service.chunk(text: text, documentID: "doc-1", config: config)
        #expect(chunks.count > 1)
        #expect(chunks.map(\.chunkIndex) == Array(0..<chunks.count))
    }

    @Test("every produced chunk carries the given documentID")
    func chunksCarryDocumentID() {
        let service = ChunkingService()
        let text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        let chunks = service.chunk(text: text, documentID: "reg-42", config: .default)
        #expect(!chunks.isEmpty)
        #expect(chunks.allSatisfy { $0.documentID == "reg-42" })
    }

    @Test("paragraph strategy keeps chunks under roughly chunkSize when paragraphs are small")
    func paragraphStrategyRespectsChunkSize() {
        let service = ChunkingService()
        let paragraphs = (1...20).map { "Paragraph number \($0) with some extra padding text to add length." }
        let text = paragraphs.joined(separator: "\n\n")
        let config = ChunkingConfig(strategy: .paragraph, chunkSize: 150, overlapPercentage: 0.0)
        let chunks = service.chunk(text: text, documentID: "doc", config: config)
        #expect(chunks.count > 1)
        // A single paragraph is never split further, so allow one paragraph's worth of slack.
        for chunk in chunks.dropLast() {
            #expect(chunk.text.count <= 150 + 80)
        }
    }

    @Test("sentence strategy splits on sentence terminators")
    func sentenceStrategySplitsOnTerminators() {
        let service = ChunkingService()
        let text = "First sentence here. Second sentence follows! Third one asks something? Fourth wraps up."
        let config = ChunkingConfig(strategy: .sentence, chunkSize: 30, overlapPercentage: 0.0)
        let chunks = service.chunk(text: text, documentID: "doc", config: config)
        #expect(chunks.count > 1)
    }

    @Test("sentence strategy ignores abbreviation-like short fragments under 10 characters")
    func sentenceStrategyIgnoresTinyFragments() {
        let service = ChunkingService()
        // "Ok." is 3 characters and should not become its own sentence chunk.
        let text = "Ok. This is a much longer sentence that should be kept as a real sentence."
        let config = ChunkingConfig(strategy: .sentence, chunkSize: 1000, overlapPercentage: 0.0)
        let chunks = service.chunk(text: text, documentID: "doc", config: config)
        #expect(chunks.count == 1)
        #expect(chunks[0].text.contains("This is a much longer sentence"))
    }

    @Test("fixed strategy produces windows of at most chunkSize characters")
    func fixedStrategyWindowSize() {
        let service = ChunkingService()
        let text = String(repeating: "abcdefghij", count: 30) // 300 chars
        let config = ChunkingConfig(strategy: .fixed, chunkSize: 100, overlapPercentage: 0.0)
        let chunks = service.chunk(text: text, documentID: "doc", config: config)
        for chunk in chunks {
            #expect(chunk.text.count <= 100)
        }
        // Non-overlapping fixed windows over exact multiples should reconstruct the length.
        #expect(chunks.reduce(0) { $0 + $1.text.count } == text.count)
    }

    @Test("fixed strategy with overlap produces overlapping windows")
    func fixedStrategyWithOverlap() {
        let service = ChunkingService()
        let text = String(repeating: "0123456789", count: 10) // 100 chars
        let config = ChunkingConfig(strategy: .fixed, chunkSize: 40, overlapPercentage: 0.25) // overlap 10
        let chunks = service.chunk(text: text, documentID: "doc", config: config)
        #expect(chunks.count > 1)
        // With overlap, total chunked characters exceed the source length.
        let totalChars = chunks.reduce(0) { $0 + $1.text.count }
        #expect(totalChars > text.count)
    }

    @Test("whitespace-only paragraphs are filtered out of the result")
    func whitespaceOnlyParagraphsAreDropped() {
        let service = ChunkingService()
        let text = "Real paragraph one.\n\n   \n\nReal paragraph two."
        let chunks = service.chunk(text: text, documentID: "doc", config: .default)
        #expect(chunks.allSatisfy { !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty })
    }

    @Test("overlapSize is computed from chunkSize and overlapPercentage")
    func overlapSizeComputation() {
        let config = ChunkingConfig(strategy: .fixed, chunkSize: 1000, overlapPercentage: 0.15)
        #expect(config.overlapSize == 150)
    }
}
