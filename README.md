# Plume (powered by Vapor Swift AI Chat using RAG)

A Swift Vapor server that provides conversational AI search over your documents — document parsing → chunking → embedding → hybrid search → LLM-generated answers. Backed by **Qdrant** for large-scale vector storage and any **OpenAI-compatible inference endpoint** for on-premises AI.

Two deployment targets are supported out of the box:

| Environment | Inference backend | Compose file |
|---|---|---|
| Google Cloud VM (NVIDIA L4 / g2-standard-4) | **vLLM** — PagedAttention, fp16, full GPU utilization | `docker-compose.yml` |
| Mac / Apple Silicon | **Ollama** — runs natively with Metal acceleration | `docker-compose.mac.yml` |

## Features

- **Agentic tool calling** — the LLM drives its own retrieval: it decides what to search, searches multiple times if needed, and synthesizes a final answer from everything it found
- **Conversational search** — ask questions in natural language; answers are grounded in your indexed documents with source citations
- **Multi-turn conversations** — follow-up questions carry context from previous turns
- **Hybrid search** — combines dense semantic vectors and sparse BM25 keyword vectors via Reciprocal Rank Fusion for best-of-both recall
- **Source transparency** — every AI answer shows the matched document chunks, summaries, and links to the original source
- **Fully on-premises** — runs entirely on your own hardware; no data leaves your server

## Architecture

```
Client Request
     │
     ▼
Vapor HTTP Server
     │
     ├── POST /api/index/text ──► DocumentChunker ──► EmbeddingService + BM25Service ──► QdrantService
     ├── POST /api/index/file ──► DocumentParser ──► DocumentChunker ──► EmbeddingService + BM25Service ──► QdrantService
     ├── POST /api/search ───────► EmbeddingService + BM25Service ──► QdrantService (Hybrid RRF) ──► Results
     └── DELETE /api/documents/:id ──────────────────► QdrantService (filter delete)
```

### Chat (Agentic Tool Calling)

```
POST /api/chat
     │
     ▼
LLMService.chatWithTools()
     │
     ├── [LLM decides what to search]
     │        │
     │        ├── tool: search_documents("Labeling requirements")
     │        │         ──► EmbeddingService + BM25Service ──► QdrantService (Hybrid RRF) ──► chunks
     │        │
     │        ├── tool: search_documents("Nbiosimilar naming rules", agency: "Agency")   ← LLM may search again
     │        │         ──► EmbeddingService + BM25Service ──► QdrantService (Hybrid RRF) ──► chunks
     │        │
     │        └── tool: list_documents()   ← optional: LLM can ask what's available
     │                  ──► QdrantService (scroll all)
     │
     └── [LLM generates final answer from accumulated chunks]
              │
              ▼
         Answer + deduplicated Sources
```

The LLM can search up to 5 times per request with different queries. Sources are deduplicated across all tool calls so the response never contains duplicate chunks.

| Component | Role |
|---|---|
| `DocumentParserService` | PDF / HTML / MD / TXT → plain text |
| `ChunkingService` | paragraph / sentence / fixed chunking with overlap |
| `EmbeddingService` | OpenAI-compatible `/v1/embeddings` (vLLM, Ollama, OpenAI, LM Studio) |
| `BM25Service` | Sparse TF keyword encoding (FNV-1a hash trick) |
| `QdrantService` | REST client for Qdrant vector DB — upsert, hybrid search, scroll, delete |
| `LLMService` | Tool-calling loop + answer synthesis via any OpenAI-compatible endpoint (vLLM, Ollama, OpenAI, Groq) |

## Agentic Tool Calling

Tool calling (also called function calling) lets the LLM decide what to retrieve rather than having the server do a single fixed search. The LLM is given tool definitions as JSON schemas, returns a structured tool call when it wants to search, and your code executes the search and feeds results back — repeating until the LLM is ready to answer.

**Previous behaviour (fixed pipeline):**
```
query → 1 hybrid search → fixed top-N chunks → LLM → answer
```

**New behaviour (agentic):**
```
query → LLM → search("Agency labeling") → results
             → search("similar naming", agency: "Agency") → more results
             → final answer citing both sets of sources
```

### Available tools

| Tool | Arguments | Description |
|---|---|---|
| `search_documents` | `query` (required), `agency` (optional), `limit` (optional, 1–10) | Hybrid vector+BM25 search across all indexed documents. The `agency` filter matches against `SummaryMetadata.agencyId` or the `agency` metadata field. |
| `list_documents` | _(none)_ | Returns a list of all indexed document IDs with chunk counts. Useful when the user asks "what do you have?" or "what topics are covered?" |

### Implementation

`OpenAILLMService` sends `tools` + `tool_choice: "auto"` to any OpenAI-compatible `/v1/chat/completions` endpoint. It executes the tool call loop in Swift (up to 5 iterations). **Requires a model that supports function/tool calling.**

### Model requirements

**vLLM (Google Cloud VM):**

```bash
# Recommended — strong tool calling, fits in ~14 GB fp16 on the L4's 24 GB VRAM
Qwen/Qwen2.5-7B-Instruct        # default in docker-compose.yml

# Alternatives (update --model in docker-compose.yml and LLM_MODEL in .env)
meta-llama/Llama-3.1-8B-Instruct  # requires HF_TOKEN
mistralai/Mistral-7B-Instruct-v0.3 # requires HF_TOKEN
```

**Ollama (Mac):**

```bash
ollama pull gemma4:12b           # recommended: strong tool calling, ~8 GB
ollama pull llama3.1:8b          # alternative: solid tool calling, ~5 GB
ollama pull mistral-nemo         # good quality, smaller
ollama pull qwen2.5:7b           # strong tool calling support

# OpenAI / Groq — all current models support tool calling
LLM_MODEL=gpt-4o-mini
LLM_MODEL=llama-3.1-8b-instant  # Groq
```

## Hybrid Search

Each document chunk is stored with **two vector representations**:

- **Dense vector** — from your embedding model (semantic meaning)
- **Sparse BM25 vector** — term-frequency encoding computed in-app, no external service needed

At query time, both vectors are sent to Qdrant's `/query` endpoint, which retrieves candidates from each arm independently and merges them using **Reciprocal Rank Fusion (RRF)**. This gives you the best of both worlds:

| Query type | Example | Winner |
|---|---|---|
| Semantic intent | "steps to submit a record" | Dense |
| Exact term | "market notification" | Sparse (BM25) |
| Both | "how to file a submission" | RRF fusion |

**Qdrant version requirement:** v1.7+ for sparse vectors, v1.10+ for the `/query` fusion endpoint.

> ⚠️ **Collection migration required**: The Qdrant collection schema changed to support hybrid search (named + sparse vectors). Before restarting the app, either:
> 1. Change `QDRANT_COLLECTION=rag-documents-v2` in `.env` (recommended), or
> 2. Delete the old collection: `DELETE http://localhost:6333/collections/rag-documents`
>
> All documents must be re-indexed after migration. The app creates the new collection schema automatically on startup.

---

## Running with Docker

### Google Cloud VM — vLLM (`docker-compose.yml`)

Runs three containers: **Vapor app + Qdrant + two vLLM servers** (one for chat, one for embeddings). Designed for the g2-standard-4 machine type (NVIDIA L4, 24 GB VRAM).

**VRAM allocation on the L4:**

| Container | Model | Port | VRAM |
|---|---|---|---|
| `vllm-llm` | `Qwen/Qwen2.5-7B-Instruct` | 8000 | ~18 GB (75%) |
| `vllm-embed` | `nomic-ai/nomic-embed-text-v1.5` | 8001 | ~4.3 GB (18%) |

Both containers share the L4's 24 GB (combined ~22.3 GB — comfortable headroom). Models are downloaded from Hugging Face on first start and cached in the `hf_cache` volume.

**Prerequisites:** NVIDIA Container Toolkit must be installed on the VM.

```bash
# Install NVIDIA Container Toolkit (if not already present)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Start the stack:**

```bash
cp .env.example .env
docker compose up -d
```

The app container waits for both vLLM servers to pass their healthchecks before starting. `vllm-llm` takes ~2–3 minutes to load. Check progress with:

```bash
docker compose logs -f vllm-llm
```

**Verify everything is running:**

```bash
curl http://localhost:8080/api/health
# {"status":"ok","service":"Plume"}
```

| Service | URL |
|---|---|
| Plume API | `http://localhost:8080` |
| Qdrant REST | `http://localhost:6333` |
| vLLM chat | `http://localhost:8000` |
| vLLM embed | `http://localhost:8001` |

---

### Mac / Apple Silicon — Ollama (`docker-compose.mac.yml`)

Runs two containers: **Vapor app + Qdrant**. Ollama runs **natively on the host** (not in Docker) so it can use Metal/MPS acceleration on Apple Silicon. The app container reaches the host Ollama via `host.docker.internal`.

**Step 1 — Install Ollama and pull models (one time):**

```bash
# Install from https://ollama.com
ollama pull nomic-embed-text    # embedding model
ollama pull gemma4:12b          # chat model (choose one that supports tool calling)
```

**Step 2 — Start Ollama:**

```bash
ollama serve
```

**Step 3 — Start the stack:**

```bash
cp .env.example .env
docker compose -f docker-compose.mac.yml up --build -d
```

**Verify everything is running:**

```bash
curl http://localhost:8080/api/health
# {"status":"ok","service":"Plume"}
```

| Service | URL |
|---|---|
| Plume API | `http://localhost:8080` |
| Qdrant REST | `http://localhost:6333` |
| Ollama (native) | `http://localhost:11434` |

---

## Running Locally (macOS, no Docker)

Useful for development — runs the Swift server directly without Docker.

### Requirements

- macOS 15+
- Swift 6.0+
- Docker (for Qdrant only)
- Ollama (for embeddings and LLM chat)

### 1. Start Qdrant

```bash
docker compose -f docker-compose.mac.yml up -d qdrant
```

> **Data is persistent.** The `qdrant_storage` named volume persists across restarts. If you run Qdrant directly with `docker run`, include `-v qdrant_storage:/qdrant/storage` — without it all indexed data is lost when the container stops.

### 2. Configure environment

```bash
cp .env.example .env
# Update EMBEDDER_URL and LLM_URL to http://localhost:11434/v1
# Update EMBEDDER_MODEL to nomic-embed-text
# Update LLM_MODEL to gemma4:12b (or whichever model you pulled)
```

### 3. Run the server

```bash
swift run App
```

Server starts on `http://localhost:8080`.

---

## Environment Variables

| Variable | GCE / vLLM default | Mac / Ollama default | Description |
|---|---|---|---|
| `PORT` | `8080` | `8080` | Vapor server port |
| `QDRANT_URL` | `http://localhost:6333` | `http://localhost:6333` | Qdrant REST endpoint |
| `QDRANT_COLLECTION` | `rag-documents` | `rag-documents` | Collection name |
| `QDRANT_API_KEY` | _(empty)_ | _(empty)_ | Qdrant API key (for Qdrant Cloud) |
| `EMBEDDER_URL` | `http://localhost:8001/v1` | `http://localhost:11434/v1` | OpenAI-compatible embeddings endpoint |
| `EMBEDDER_KEY` | _(empty)_ | _(empty)_ | API key for the embedder |
| `EMBEDDER_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | `nomic-embed-text` | Embedding model name |
| `EMBEDDING_DIMENSION` | `768` | `768` | Must match model output dimension |
| `EMBEDDING_BATCH_SIZE` | `50` | `20` | Texts per embedding API request |
| `LLM_URL` | `http://localhost:8000/v1` | `http://localhost:11434/v1` | OpenAI-compatible chat completions endpoint |
| `LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | `gemma4:12b` | Chat model (must support tool calling) |
| `LLM_API_KEY` | _(empty)_ | _(empty)_ | API key for the LLM |
| `HF_TOKEN` | _(empty)_ | — | Hugging Face token (only for gated models) |

> **Docker note:** The compose files override `QDRANT_URL`, `EMBEDDER_URL`, `LLM_URL`, `EMBEDDER_MODEL`, `LLM_MODEL`, and batch sizes automatically using internal service hostnames. You only need to customize these in `.env` when running the Swift server directly (without Docker).

### Using an online LLM instead of a local model

The chat endpoint works with any OpenAI-compatible provider — just set the URL, model, and key:

**OpenAI:**
```env
LLM_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-...
```

**Groq (fast inference, free tier):**
```env
LLM_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.1-8b-instant
LLM_API_KEY=gsk_...
```

---

## Qdrant Storage & Re-indexing

Qdrant stores all vector data on disk inside the container. When using Docker Compose, the `qdrant_storage` named volume persists this data across restarts — you index once and searches keep working.

**Switching inference backends does not require re-indexing** as long as you use the same embedding model. Both `nomic-embed-text` (Ollama) and `nomic-ai/nomic-embed-text-v1.5` (vLLM) are the same model weights and produce identical 768-dimensional vectors.

**If data is lost** (e.g. you ran Qdrant without a volume, or deleted the volume), you'll see `"points_indexed": "0"` on the health endpoint and searches return no results. Re-index your documents by POSTing them to the index endpoints again:

```bash
# Re-index a text document
curl -X POST http://localhost:8080/api/index/text \
  -H "Content-Type: application/json" \
  -d '{"text": "...", "documentID": "my-doc"}'

# Re-index a file
curl -X POST http://localhost:8080/api/index/file \
  -F "file=@/path/to/document.pdf"
```

The app is stateless with respect to documents — it does not keep a copy of your source text, so **you are responsible for re-submitting source documents** after a data loss event.

---

## API

### Health check

```bash
curl http://localhost:8080/api/health
```

Returns the service status and how many points are currently indexed in Qdrant. Use this to confirm documents have been ingested before searching.

```json
{
  "status": "ok",
  "service": "Plume",
  "collection": "rag-documents",
  "points_indexed": "42"
}
```

`points_indexed: "0"` means the collection is empty — you need to index documents before search will return results. This commonly happens after a Qdrant restart without a persistent volume.

### Index plain text

```bash
curl -X POST http://localhost:8080/api/index/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Qdrant is a vector similarity search engine...",
    "documentID": "qdrant-intro",
    "strategy": "paragraph",
    "chunkSize": 800,
    "overlapPercentage": 0.15
  }'
```

**Chunking strategies:** `paragraph` (default), `sentence`, `fixed`

### Index a file (PDF, MD, HTML, TXT)

```bash
curl -X POST http://localhost:8080/api/index/file \
  -F "file=@/path/to/document.pdf" \
  -F "strategy=paragraph" \
  -F "chunkSize=1000"
```

### Conversational chat (RAG + AI answer)

Ask a question and receive an AI-generated answer grounded in your indexed documents, along with the source chunks used to construct the answer.

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Describe information about adding new vitamins into my label",
    "conversationHistory": [],
    "limit": 5,
    "threshold": 0.3
  }'
```

**Response:**
```json
{
  "query": "Describe information about adding new vitamins into my label",
  "answer": "According to [AGENCY-2021-N-0270], manufacturers adding new vitamins to a dietary supplement label must notify the FDA at least 75 days before marketing. The Supplement Facts panel must list each vitamin by name and amount per serving in the required format...",
  "sources": [
    {
      "id": "550e8400-...",
      "text": "Manufacturers of dietary supplements must notify FDA...",
      "score": 0.032,
      "documentID": "AGENCY-2021-N-0270",
      "chunkIndex": 2,
      "metadata": {}
    }
  ]
}
```

**Multi-turn conversation:** pass previous turns in `conversationHistory` to enable follow-up questions:

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What about vitamin D specifically?",
    "conversationHistory": [
      {"role": "user",      "content": "Describe information about adding new vitamins into my label"},
      {"role": "assistant", "content": "According to [AGENCY-2021-N-0270], manufacturers adding new vitamins..."}
    ],
    "limit": 5,
    "threshold": 0.3
  }'
```

The web UI manages conversation history automatically — follow-up questions work out of the box.

| Parameter | Default | Description |
|---|---|---|
| `query` | _(required)_ | Natural language question |
| `conversationHistory` | `[]` | Previous turns for multi-turn context (capped at last 10) |
| `limit` | `5` | Maximum source chunks to retrieve (1–20) |
| `threshold` | `0.3` | Minimum similarity score to include a chunk |

### Hybrid search (BM25 + semantic)

```bash
curl -X POST http://localhost:8080/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is vector similarity search?",
    "limit": 5,
    "threshold": 0.0
  }'
```

> **Score note:** Results use RRF (Reciprocal Rank Fusion) scores, which are rank-based values (typically `0.01`–`0.05`) rather than cosine similarity scores (`0`–`1`). Set `threshold` to `0.0` to return all results, or raise it slightly (e.g. `0.01`) to filter out very weak matches.

**Response:**
```json
{
  "query": "What is vector similarity search?",
  "count": 3,
  "results": [
    {
      "id": "550e8400-...",
      "text": "Qdrant is a vector similarity search engine...",
      "score": 0.016,
      "documentID": "qdrant-intro",
      "chunkIndex": 0,
      "metadata": {}
    }
  ]
}
```

### Delete a document

```bash
curl -X DELETE http://localhost:8080/api/documents/qdrant-intro
```

---

## Using a different embedding model

Override the embedder in your `.env`. For example with **OpenAI embeddings:**

```env
EMBEDDER_URL=https://api.openai.com/v1
EMBEDDER_KEY=sk-...
EMBEDDER_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
```

Or with **Qdrant Cloud** + remote embedder:

```env
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
EMBEDDER_URL=https://api.openai.com/v1
EMBEDDER_KEY=sk-...
EMBEDDER_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
```

> ⚠️ Changing the embedding model requires re-indexing all documents — vectors from different models are incompatible.

---

## Why Qdrant over VecturaKit?

| | VecturaKit | Qdrant |
|---|---|---|
| Scale | Thousands of docs | Millions of docs |
| Deployment | On-device only | Self-hosted or cloud |
| Filtering | Basic | Rich payload filters |
| Horizontal scaling | ✗ | ✓ (distributed mode) |
| Persistence | Local disk | Distributed storage |
