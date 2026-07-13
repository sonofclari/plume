import Foundation
import Testing
@testable import App

@Suite("BM25Service")
struct BM25ServiceTests {

    @Test("empty text produces an empty sparse vector")
    func emptyText() {
        let service = BM25Service()
        let vector = service.encode("")
        #expect(vector.indices.isEmpty)
        #expect(vector.values.isEmpty)
    }

    @Test("text made up only of stop words and short tokens produces an empty vector")
    func onlyStopWords() {
        let service = BM25Service()
        let vector = service.encode("the a an it is")
        #expect(vector.indices.isEmpty)
        #expect(vector.values.isEmpty)
    }

    @Test("encoding is deterministic for the same input")
    func deterministic() {
        let service = BM25Service()
        let first = service.encode("regulatory compliance requirements for agencies")
        let second = service.encode("regulatory compliance requirements for agencies")
        #expect(first.indices == second.indices)
        #expect(first.values == second.values)
    }

    @Test("indices are returned in ascending order")
    func indicesAreSorted() {
        let service = BM25Service()
        let vector = service.encode("apple banana cherry date elderberry fig grape honeydew")
        #expect(vector.indices == vector.indices.sorted())
    }

    @Test("values are L2 normalized")
    func valuesAreL2Normalized() {
        let service = BM25Service()
        let vector = service.encode("apple banana cherry date elderberry fig grape honeydew")
        let magnitude = sqrt(vector.values.reduce(Float(0)) { $0 + $1 * $1 })
        #expect(abs(magnitude - 1.0) < 0.0001)
    }

    @Test("repeated terms increase term frequency but not the number of indices")
    func repeatedTermsCollapseToOneIndex() {
        let service = BM25Service()
        let single = service.encode("regulation")
        let repeated = service.encode("regulation regulation regulation")
        #expect(single.indices.count == 1)
        #expect(repeated.indices.count == 1)
        #expect(single.indices == repeated.indices)
    }

    @Test("vocabSize bounds the hashed index range")
    func vocabSizeBoundsIndices() {
        let service = BM25Service(vocabSize: 100)
        let vector = service.encode("apple banana cherry date elderberry fig grape honeydew")
        for index in vector.indices {
            #expect(index >= 0 && index < 100)
        }
    }
}
