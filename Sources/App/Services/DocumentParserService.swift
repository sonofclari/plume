import Foundation

#if canImport(PDFKit)
import PDFKit
#endif

#if canImport(Vision)
import Vision
#endif

enum DocumentParserError: Error, LocalizedError {
    case unsupportedFormat(String)
    case parseFailed(String)
    case emptyDocument
    case missingDependency(String)

    var errorDescription: String? {
        switch self {
        case .unsupportedFormat(let ext):  return "Unsupported file format: .\(ext)"
        case .parseFailed(let reason):     return "Failed to parse document: \(reason)"
        case .emptyDocument:               return "Document produced no extractable text"
        case .missingDependency(let msg):  return msg
        }
    }
}

struct DocumentParserService: Sendable {

    /// Parses a document into plain text. Supports PDF, TXT, MD, HTML on macOS and Linux.
    /// Scanned (image-only) PDFs are handled via OCR: Vision framework on macOS,
    /// Tesseract + pdftoppm on Linux.
    func parse(data: Data, filename: String) async throws -> String {
        let ext = URL(fileURLWithPath: filename).pathExtension.lowercased()

        switch ext {
        case "pdf":
            return try await parsePDF(data: data)
        case "txt", "md", "markdown":
            return try parseText(data: data)
        case "html", "htm":
            return try parseHTML(data: data)
        default:
            guard let text = String(data: data, encoding: .utf8) else {
                throw DocumentParserError.unsupportedFormat(ext)
            }
            return try nonEmpty(text)
        }
    }

    // MARK: - PDF (platform-specific)

    private func parsePDF(data: Data) async throws -> String {
        #if canImport(PDFKit)
        return try parsePDFWithPDFKit(data: data)        // macOS
        #else
        return try await parsePDFWithPoppler(data: data) // Linux
        #endif
    }

    // MARK: - macOS: PDFKit text extraction → Vision OCR fallback

    #if canImport(PDFKit)
    private func parsePDFWithPDFKit(data: Data) throws -> String {
        guard let document = PDFDocument(data: data) else {
            throw DocumentParserError.parseFailed("Could not open PDF")
        }
        let pages = (0..<document.pageCount).compactMap { document.page(at: $0)?.string }
        let extracted = pages.joined(separator: "\n\n").trimmingCharacters(in: .whitespacesAndNewlines)

        if !extracted.isEmpty {
            return extracted
        }

        // Scanned/image-only PDF — fall back to Vision OCR
        #if canImport(Vision)
        return try ocrPDFWithVision(document: document)
        #else
        throw DocumentParserError.emptyDocument
        #endif
    }

    #if canImport(Vision)
    /// OCR each page using Vision's VNRecognizeTextRequest (macOS 10.15+).
    private func ocrPDFWithVision(document: PDFDocument) throws -> String {
        var pageTexts: [String] = []

        for i in 0..<document.pageCount {
            guard let page = document.page(at: i),
                  let cgImage = renderPageToCGImage(page: page) else { continue }

            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = true

            try VNImageRequestHandler(cgImage: cgImage, options: [:]).perform([request])

            let pageText = (request.results ?? [])
                .compactMap { $0.topCandidates(1).first?.string }
                .joined(separator: "\n")

            if !pageText.isEmpty {
                pageTexts.append(pageText)
            }
        }

        return try nonEmpty(pageTexts.joined(separator: "\n\n"))
    }

    /// Render a PDFPage to a CGImage at 2× scale for better OCR accuracy.
    private func renderPageToCGImage(page: PDFPage, scale: CGFloat = 2.0) -> CGImage? {
        let pageRect = page.bounds(for: .mediaBox)
        let width    = Int(pageRect.width  * scale)
        let height   = Int(pageRect.height * scale)

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(
                  data: nil, width: width, height: height,
                  bitsPerComponent: 8, bytesPerRow: 0,
                  space: colorSpace,
                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
              ) else { return nil }

        // White background
        ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        ctx.scaleBy(x: scale, y: scale)
        page.draw(with: .mediaBox, to: ctx)

        return ctx.makeImage()
    }
    #endif // canImport(Vision)
    #endif // canImport(PDFKit)

    // MARK: - Linux: pdftotext extraction → Tesseract OCR fallback

    #if !canImport(PDFKit)
    private func parsePDFWithPoppler(data: Data) async throws -> String {
        // Try embedded text extraction first; fall back to OCR if empty or unavailable
        if let text = try? await extractTextWithPdfToText(data: data), !text.isEmpty {
            return text
        }
        // Scanned/image-only PDF — fall back to Tesseract OCR
        return try await extractTextWithTesseract(data: data)
    }

    /// Extract embedded text from a PDF using pdftotext (poppler-utils).
    private func extractTextWithPdfToText(data: Data) async throws -> String {
        guard let pdfToTextPath = Self.findExecutable("pdftotext") else {
            throw DocumentParserError.missingDependency(
                "pdftotext not found. Install with: sudo apt-get install poppler-utils")
        }

        let tmp     = FileManager.default.temporaryDirectory
        let inPath  = tmp.appendingPathComponent("\(UUID().uuidString).pdf").path
        let outPath = tmp.appendingPathComponent("\(UUID().uuidString).txt").path

        defer {
            try? FileManager.default.removeItem(atPath: inPath)
            try? FileManager.default.removeItem(atPath: outPath)
        }

        guard FileManager.default.createFile(atPath: inPath, contents: data) else {
            throw DocumentParserError.parseFailed("Could not write temporary PDF file")
        }

        try await runProcess(executable: pdfToTextPath, arguments: [inPath, outPath])

        return (try? String(contentsOfFile: outPath, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    /// OCR a scanned PDF using pdftoppm (render pages) + Tesseract (recognize text).
    /// Requires: sudo apt-get install poppler-utils tesseract-ocr
    private func extractTextWithTesseract(data: Data) async throws -> String {
        guard let pdftoppmPath = Self.findExecutable("pdftoppm") else {
            throw DocumentParserError.missingDependency(
                "pdftoppm not found. Install with: sudo apt-get install poppler-utils")
        }
        guard let tesseractPath = Self.findExecutable("tesseract") else {
            throw DocumentParserError.missingDependency(
                "tesseract not found. Install with: sudo apt-get install tesseract-ocr")
        }

        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)

        let inPath    = tmp.appendingPathComponent("input.pdf").path
        let imgPrefix = tmp.appendingPathComponent("page").path

        defer { try? FileManager.default.removeItem(at: tmp) }

        guard FileManager.default.createFile(atPath: inPath, contents: data) else {
            throw DocumentParserError.parseFailed("Could not write temporary PDF file")
        }

        // Render all pages to 300 DPI PNG images: page-001.png, page-002.png, …
        try await runProcess(executable: pdftoppmPath,
                             arguments: ["-r", "300", "-png", inPath, imgPrefix])

        let pageImages = (try? FileManager.default.contentsOfDirectory(atPath: tmp.path)) ?? []
        let sortedImages = pageImages.filter { $0.hasSuffix(".png") }.sorted()

        guard !sortedImages.isEmpty else {
            throw DocumentParserError.parseFailed("pdftoppm produced no page images")
        }

        var pageTexts: [String] = []
        for imageName in sortedImages {
            let imagePath = tmp.appendingPathComponent(imageName).path
            let outBase   = tmp.appendingPathComponent(
                imageName.replacingOccurrences(of: ".png", with: "_ocr")).path

            // tesseract writes recognized text to <outBase>.txt
            try await runProcess(executable: tesseractPath, arguments: [imagePath, outBase])

            if let text = try? String(contentsOfFile: outBase + ".txt", encoding: .utf8),
               !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                pageTexts.append(text)
            }
        }

        return try nonEmpty(pageTexts.joined(separator: "\n\n"))
    }

    private func runProcess(executable: String, arguments: [String]) async throws {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: executable)
        proc.arguments     = arguments

        let stderrPipe = Pipe()
        proc.standardError = stderrPipe

        let execName = URL(fileURLWithPath: executable).lastPathComponent

        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, any Error>) in
            DispatchQueue.global(qos: .utility).async {
                do {
                    try proc.run()
                    proc.waitUntilExit()
                    if proc.terminationStatus == 0 {
                        cont.resume()
                    } else {
                        let errData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
                        let errMsg  = String(data: errData, encoding: .utf8)?
                            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                        cont.resume(throwing: DocumentParserError.parseFailed(
                            "\(execName) exited with status \(proc.terminationStatus)" +
                            (errMsg.isEmpty ? "" : ": \(errMsg)")))
                    }
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    private static func findExecutable(_ name: String) -> String? {
        ["/usr/bin/\(name)", "/usr/local/bin/\(name)",
         "/opt/homebrew/bin/\(name)", "/opt/homebrew/opt/poppler/bin/\(name)"]
            .first { FileManager.default.isExecutableFile(atPath: $0) }
    }
    #endif

    // MARK: - Plain Text / Markdown

    private func parseText(data: Data) throws -> String {
        guard let text = String(data: data, encoding: .utf8)
                      ?? String(data: data, encoding: .isoLatin1) else {
            throw DocumentParserError.parseFailed("Could not decode text file")
        }
        return try nonEmpty(text)
    }

    // MARK: - HTML

    private func parseHTML(data: Data) throws -> String {
        guard let html = String(data: data, encoding: .utf8) else {
            throw DocumentParserError.parseFailed("Could not decode HTML")
        }
        return try nonEmpty(stripHTMLTags(html))
    }

    private func stripHTMLTags(_ html: String) -> String {
        var result = html

        // Remove script/style blocks
        for pattern in ["<script[^>]*>[\\s\\S]*?</script>", "<style[^>]*>[\\s\\S]*?</style>"] {
            if let re = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) {
                result = re.stringByReplacingMatches(
                    in: result, range: NSRange(result.startIndex..., in: result), withTemplate: " ")
            }
        }
        // Block tags → newlines
        for pattern in ["</p>", "</div>", "<br\\s*/?>", "</h[1-6]>", "</li>", "</tr>"] {
            if let re = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) {
                result = re.stringByReplacingMatches(
                    in: result, range: NSRange(result.startIndex..., in: result), withTemplate: "\n")
            }
        }
        // Strip remaining tags
        if let re = try? NSRegularExpression(pattern: "<[^>]+>") {
            result = re.stringByReplacingMatches(
                in: result, range: NSRange(result.startIndex..., in: result), withTemplate: "")
        }
        // Decode common HTML entities
        result = result
            .replacingOccurrences(of: "&amp;",  with: "&")
            .replacingOccurrences(of: "&lt;",   with: "<")
            .replacingOccurrences(of: "&gt;",   with: ">")
            .replacingOccurrences(of: "&nbsp;", with: " ")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&#39;",  with: "'")

        return result
            .components(separatedBy: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    // MARK: - Helper

    private func nonEmpty(_ text: String) throws -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { throw DocumentParserError.emptyDocument }
        return trimmed
    }
}
