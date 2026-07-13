import Foundation
@testable import App

/// A fake `HTTPClient` that replays a canned response instead of touching the network.
/// Injected directly into services under test in place of a real `URLSession`.
struct MockHTTPClient: HTTPClient {
    typealias Handler = @Sendable (URLRequest) throws -> (statusCode: Int, body: Data)

    let handler: Handler

    init(_ handler: @escaping Handler) {
        self.handler = handler
    }

    func data(for request: URLRequest) async throws -> (Data, URLResponse) {
        let (statusCode, body) = try handler(request)
        let response = HTTPURLResponse(
            url: request.url ?? URL(string: "https://mock.invalid")!,
            statusCode: statusCode,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"]
        )!
        return (body, response)
    }
}

extension URLRequest {
    /// Convenience accessor for inspecting the request payload in tests. `MockHTTPClient` never
    /// runs requests through a real `URLSession`, so `httpBody` is always the value set on the
    /// request — no `httpBodyStream` fallback is needed.
    func capturedBody() -> Data? {
        httpBody
    }
}
