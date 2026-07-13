import Foundation
import Testing
@testable import App

@Suite("DocumentParserService")
struct DocumentParserServiceTests {

    @Test("parses plain .txt files")
    func parsesPlainText() async throws {
        let service = DocumentParserService()
        let data = Data("Hello, regulatory world.".utf8)
        let text = try await service.parse(data: data, filename: "notes.txt")
        #expect(text == "Hello, regulatory world.")
    }

    @Test("parses .md files as plain text")
    func parsesMarkdown() async throws {
        let service = DocumentParserService()
        let data = Data("# Title\n\nSome body text.".utf8)
        let text = try await service.parse(data: data, filename: "readme.md")
        #expect(text == "# Title\n\nSome body text.")
    }

    @Test("strips HTML tags and decodes entities")
    func parsesHTML() async throws {
        let service = DocumentParserService()
        let html = "<html><body><p>Hello &amp; welcome</p><script>evil()</script></body></html>"
        let data = Data(html.utf8)
        let text = try await service.parse(data: data, filename: "page.html")
        #expect(text.contains("Hello & welcome"))
        #expect(!text.contains("evil()"))
        #expect(!text.contains("<p>"))
    }

    @Test("block-level tags in HTML become newlines between entries")
    func htmlBlockTagsBecomeNewlines() async throws {
        let service = DocumentParserService()
        let html = "<div>First</div><div>Second</div>"
        let data = Data(html.utf8)
        let text = try await service.parse(data: data, filename: "page.html")
        #expect(text == "First\nSecond")
    }

    @Test("empty text file throws emptyDocument")
    func emptyTextFileThrows() async throws {
        let service = DocumentParserService()
        let data = Data("   \n\n  ".utf8)
        await #expect(throws: DocumentParserError.self) {
            _ = try await service.parse(data: data, filename: "empty.txt")
        }
    }

    @Test("unsupported binary extension with non-UTF8 content throws unsupportedFormat")
    func unsupportedBinaryThrows() async throws {
        let service = DocumentParserService()
        let data = Data([0xFF, 0xFE, 0x00, 0xD8, 0x00, 0xFF])
        await #expect(throws: DocumentParserError.self) {
            _ = try await service.parse(data: data, filename: "image.png")
        }
    }

    @Test("unknown extension with valid UTF-8 text is parsed as plain text")
    func unknownExtensionWithValidTextIsParsed() async throws {
        let service = DocumentParserService()
        let data = Data("Just some text content".utf8)
        let text = try await service.parse(data: data, filename: "file.unknownext")
        #expect(text == "Just some text content")
    }
}
