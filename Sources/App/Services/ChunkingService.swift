import Foundation

// MARK: - Strategy + Config

enum ChunkingStrategy: String, Codable, Sendable {
    /// Split on blank lines, merge until chunkSize is reached.
    case paragraph
    /// Split on sentence boundaries, merge until chunkSize is reached.
    case sentence
    /// Fixed character window with configurable overlap.
    case fixed
}

struct ChunkingConfig: Sendable {
    let strategy: ChunkingStrategy
    /// Target chunk size in characters.
    let chunkSize: Int
    /// Overlap as a fraction of chunkSize (0.0 – 0.5).
    let overlapPercentage: Double

    static let `default` = ChunkingConfig(
        strategy: .paragraph,
        chunkSize: 1_000,
        overlapPercentage: 0.15
    )

    var overlapSize: Int { Int(Double(chunkSize) * overlapPercentage) }
}

// MARK: - Service

struct ChunkingService: Sendable {

    func chunk(text: String, documentID: String, config: ChunkingConfig) -> [Chunk] {
        let raw: [String]
        switch config.strategy {
        case .paragraph: raw = chunkByParagraph(text: text, config: config)
        case .sentence:  raw = chunkBySentence(text: text, config: config)
        case .fixed:     raw = chunkFixed(text: text, config: config)
        }
        return raw
            .enumerated()
            .compactMap { index, body in
                let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { return nil }
                return Chunk(text: trimmed, documentID: documentID, chunkIndex: index)
            }
    }

    // MARK: Paragraph

    private func chunkByParagraph(text: String, config: ChunkingConfig) -> [String] {
        let paragraphs = text
            .components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        var chunks: [String] = []
        var current = ""

        for paragraph in paragraphs {
            let candidate = current.isEmpty ? paragraph : "\(current)\n\n\(paragraph)"
            if candidate.count > config.chunkSize && !current.isEmpty {
                chunks.append(current)
                let tail = tailOverlap(of: current, size: config.overlapSize)
                current = tail.isEmpty ? paragraph : "\(tail)\n\n\(paragraph)"
            } else {
                current = candidate
            }
        }
        if !current.isEmpty { chunks.append(current) }
        return chunks
    }

    // MARK: Sentence

    private func chunkBySentence(text: String, config: ChunkingConfig) -> [String] {
        let sentences = splitIntoSentences(text)
        var chunks: [String] = []
        var current = ""

        for sentence in sentences {
            let candidate = current.isEmpty ? sentence : "\(current) \(sentence)"
            if candidate.count > config.chunkSize && !current.isEmpty {
                chunks.append(current)
                let tail = tailOverlap(of: current, size: config.overlapSize)
                current = tail.isEmpty ? sentence : "\(tail) \(sentence)"
            } else {
                current = candidate
            }
        }
        if !current.isEmpty { chunks.append(current) }
        return chunks
    }

    private func splitIntoSentences(_ text: String) -> [String] {
        var sentences: [String] = []
        var current = ""
        let terminators: Set<Character> = [".", "!", "?"]
        let chars = Array(text)

        for (i, char) in chars.enumerated() {
            current.append(char)
            if terminators.contains(char) {
                let next = i + 1
                if next >= chars.count || chars[next].isWhitespace || chars[next].isNewline {
                    let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
                    if trimmed.count > 10 { sentences.append(trimmed) }
                    current = ""
                }
            }
        }
        let last = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !last.isEmpty { sentences.append(last) }
        return sentences
    }

    // MARK: Fixed

    private func chunkFixed(text: String, config: ChunkingConfig) -> [String] {
        let chars = Array(text)
        guard !chars.isEmpty else { return [] }

        var chunks: [String] = []
        let step = max(config.chunkSize - config.overlapSize, 1)
        var start = 0

        while start < chars.count {
            let end = min(start + config.chunkSize, chars.count)
            chunks.append(String(chars[start..<end]))
            start += step
        }
        return chunks
    }

    // MARK: Helpers

    private func tailOverlap(of text: String, size: Int) -> String {
        guard size > 0, text.count > size else { return "" }
        return String(text.suffix(size))
    }
}
