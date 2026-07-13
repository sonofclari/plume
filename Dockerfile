# ---- Build Stage ----
FROM swift:6.2-jammy AS builder

WORKDIR /app

# Install pdftotext for PDF parsing on Linux
RUN apt-get update && \
    apt-get install -y poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Resolve dependencies first (cached layer unless Package.swift changes)
COPY Package.swift Package.resolved* ./
RUN swift package resolve

# Build app
COPY Sources ./Sources
RUN swift build -c release

# ---- Runtime Stage ----
FROM swift:6.2-jammy

WORKDIR /app

# Only install poppler-utils runtime (no compiler)
RUN apt-get update && \
    apt-get install -y poppler-utils && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.build/release/App .
COPY Public ./Public

EXPOSE 8080

CMD ["./App", "serve", "--env", "production", "--hostname", "0.0.0.0"]
