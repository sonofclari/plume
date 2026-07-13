import Foundation

// MARK: - Regulation Metadata

/// Typed metadata for a regulation document sourced from regulations.gov.
struct SummaryMetadata: Codable, Sendable {
    let agencyId: String?
    let objectId: String?
    let frDocNum: String?
    let documentType: String?
    let withdrawn: Bool?
    let title: String?
    let docketId: String?
    let subtype: String?
    let postedDate: String?
    let lastModifiedDate: String?
    let openForComment: Bool?
    let allowLateComments: Bool?
    let commentStartDate: String?
    let commentEndDate: String?
}

// MARK: - regulations.gov API Shape (for /api/index/regulation)

struct RegulationDocument: Codable, Sendable {
    let type: String?
    let id: String
    let attributes: SummaryMetadata
}
