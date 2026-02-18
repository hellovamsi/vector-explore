# Vector Explore

An educational, local-first CLI tool to learn how **chunking**, **embeddings**, **vector databases**, and **vector querying** work (and how they combine into RAG).

**Important:** Read this **Setup** section before running (recommended). The CLI also prints this reminder at startup.

## What this tool shows
- Download a free public-domain novel from Project Gutenberg (the downloaded text includes its own license/terms).
- Chunk it 3 ways (all with overlap on by default):
  - fixed-length
  - sentence-based
  - semantic (embedding-guided boundaries)
- Embed chunks (Ollama / SentenceTransformers / Cloud).
- Store vectors (LanceDB / Qdrant / Pinecone).
- Query with full transparency:
  - query vectors as numbers (paged)
  - vector DB request parameters
  - retrieved raw vectors (paged)
  - **Vector→Text approximation (lossy)**: nearest-neighbor reconstruction + extracted keywords
- Generate an answer using a local LLM (default: Ollama `mistral:7b`) with chunk-id citations.

## Setup (required)

### Python
Target Python is **3.12**.

Create a venv and install:
```bash
cd vector-explore
python3.12 -m venv .venv
source .venv/bin/activate

# lightweight path: SentenceTransformers + LanceDB + dev tools
pip install -e ".[st,lancedb,dev]"
```

### `.env`
Copy and edit:
```bash
cp .env.example .env
```

### Ollama (for fully-local embeddings and/or local answering)
You are responsible for installing and running Ollama, and pulling models ahead of time.
- Start Ollama
- Pull an embedding model (example): `nomic-embed-text`
- Pull a chat model (default): `mistral:7b`

This tool **will not** pull Ollama models automatically.

### Qdrant (optional)
You are responsible for setting up Qdrant locally (server) and configuring `.env`:
- `QDRANT_URL` (required if selecting Qdrant)
- `QDRANT_API_KEY` (optional)

The tool does not start Qdrant for you; it only reads `.env`.

### Pinecone (optional)
Configure `.env`:
- `PINECONE_API_KEY`
- `PINECONE_INDEX`

### Cloud embeddings (optional)
Set `.env`:
- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_EMBED_MODEL`

This uses an OpenAI-compatible embeddings endpoint (works with OpenRouter’s OpenAI-compatible API too).

## Quickstart
Run the interactive wizard:
```bash
cd vector-explore
pip install -e ".[st,lancedb]"
vector-explore
```

Or installed entrypoint:
```bash
vector-explore
```

## CLI commands
- `vector-explore wizard`
- `vector-explore download --novel frankenstein`
- `vector-explore chunk --novel frankenstein --method fixed`
- `vector-explore embed --novel frankenstein --method fixed --input runs/.../chunks.jsonl --backend st`
- `vector-explore index --novel frankenstein --method fixed --embed-backend st --embed-model sentence-transformers/all-MiniLM-L6-v2 --input runs/.../embeddings.jsonl --store lancedb`
- `vector-explore query --novel frankenstein --method fixed --embed-backend st --embed-model sentence-transformers/all-MiniLM-L6-v2 --store lancedb --question "What are the major themes?"`

## Outputs
Artifacts are written under `runs/` and are cached by input checksum + parameters:
- `chunks.jsonl`
- `embeddings.jsonl`
- `retrieval.jsonl`
- `vector_to_text.json`
- `answer.md`

See `docs/interfaces.md` for schemas.
