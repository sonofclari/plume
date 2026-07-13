// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Plume",
    platforms: [
        .macOS(.v13)  // minimum for Vapor; Linux has no platform restriction
    ],
    dependencies: [
        .package(url: "https://github.com/vapor/vapor.git", from: "4.99.3"),
    ],
    targets: [
        .executableTarget(
            name: "App",
            dependencies: [
                .product(name: "Vapor", package: "vapor"),
            ],
            swiftSettings: [
                .unsafeFlags(["-cross-module-optimization"], .when(configuration: .release)),
            ]
        ),
        .testTarget(
            name: "AppTests",
            dependencies: [
                .target(name: "App"),
            ]
        ),
    ]
)
