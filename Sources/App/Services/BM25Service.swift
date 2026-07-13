import Foundation

struct BM25Service: Sendable {

    let vocabSize: Int

    init(vocabSize: Int = 30_000) {
        self.vocabSize = vocabSize
    }

    func encode(_ text: String) -> SparseVector {
        let tokens = tokenize(text)
        guard !tokens.isEmpty else {
            return SparseVector(indices: [], values: [])
        }

        var termFrequencies: [Int: Float] = [:]
        for token in tokens {
            let idx = Int(fnv1a32(token) % UInt32(vocabSize))
            termFrequencies[idx, default: 0] += 1
        }

        var indices: [Int] = []
        var values: [Float] = []
        for (idx, freq) in termFrequencies {
            indices.append(idx)
            values.append(log(1 + freq))
        }

        // L2 normalize so magnitudes are comparable across chunks of different lengths
        let magnitude = sqrt(values.reduce(0) { $0 + $1 * $1 })
        if magnitude > 0 {
            values = values.map { $0 / magnitude }
        }

        // Qdrant requires sparse vector indices in ascending order
        let paired = zip(indices, values).sorted { $0.0 < $1.0 }
        return SparseVector(
            indices: paired.map { $0.0 },
            values: paired.map { $0.1 }
        )
    }

    private func tokenize(_ text: String) -> [String] {
        text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count >= 2 && !Self.stopWords.contains($0) }
    }

    // FNV-1a 32-bit hash — fast, well-distributed, branch-free for short strings
    private func fnv1a32(_ string: String) -> UInt32 {
        var hash: UInt32 = 2_166_136_261
        for byte in string.utf8 {
            hash ^= UInt32(byte)
            hash &*= 16_777_619
        }
        return hash
    }

    private static let stopWords: Set<String> = [
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "as", "is", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "could", "should", "may",
        "might", "shall", "can", "it", "its", "this", "that", "these",
        "those", "he", "she", "they", "we", "you", "me", "him", "her",
        "us", "them", "my", "your", "his", "our", "their", "which",
        "who", "what", "when", "where", "how", "if", "then", "than",
        "so", "not", "no", "nor", "any", "all", "both", "each", "few",
        "more", "most", "other", "such", "into", "through", "during",
        "before", "after", "above", "below", "up", "down", "out", "off",
        "over", "under", "again", "further", "also", "only", "same",
        "very", "just", "about", "need", "use", "used", "using"
    ]
}
