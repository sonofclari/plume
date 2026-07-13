import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Networking dependency for HTTP-backed services. Injecting this (rather than depending on
/// `URLSession` directly) lets tests supply a fake implementation instead of routing fake
/// responses through a real `URLSession` + `URLProtocol`.
protocol HTTPClient: Sendable {
    func data(for request: URLRequest) async throws -> (Data, URLResponse)
}

extension URLSession: HTTPClient {}
