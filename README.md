# AI Code Generation Agent

Stack:
- FastAPI backend
- Ollama for chat + embeddings
- RAG from local PDF + tutorial files + persistent PostgreSQL pgvector
- React frontend (code panel left, chat right)

## 1) Run Ollama (Smaller Model + Parallelism + GPU)

Recommended default model profile:
- Chat model: `qwen2.5-coder:3b`
- Embeddings: `nomic-embed-text`

Host Ollama with parallel request tuning:

```bash
OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=15m ollama serve
ollama pull qwen2.5-coder:3b
ollama pull nomic-embed-text
```

If you run Ollama on GPU, make sure your Ollama runtime is using CUDA/Metal and drivers are installed.

## 2) Run App With Docker (CPU-safe default)

Put your documents into:
- `data/pdfs/` for PDF manuals
- `data/tutorials/` for tutorial folders (recursive):
  - PDFs (`.pdf`)
  - code/text/data files (`.md`, `.txt`, `.rst`, `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.json`, `.csv`, `.inc`, `.data`, `.dev`, `.xml`, `.log`, `.yml`, `.yaml`, `.toml`, `.ini`, `.cfg`)

Then run:

```bash
docker compose up --build
```

Services:
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- Ollama API (host by default): `http://localhost:11434`
- PostgreSQL pgvector: `localhost:5432`

Notes:
- Backend container calls host Ollama via `http://host.docker.internal:11434` by default.
- Vector index is stored in PostgreSQL pgvector and survives container restarts.
- Indexing runs as a background job in backend.

Recommended runtime overrides:

```bash
OLLAMA_CHAT_MODEL=qwen2.5-coder:3b \
OLLAMA_EMBEDDING_MODEL=nomic-embed-text \
OLLAMA_CHAT_NUM_CTX=4096 \
OLLAMA_CHAT_NUM_PREDICT=512 \
UVICORN_WORKERS=2 \
EMBED_WORKERS=4 \
INDEX_BATCH_SIZE=24 \
docker compose up --build
```

## 3) Optional: Enable GPU For Ollama Container

Base compose runs Ollama in CPU-safe mode by default.

To enable GPU, use override file:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

If Docker reports no adapters/runtime for NVIDIA, stay on CPU mode and fix GPU runtime first.

## 4) Optional Local (Non-Docker) Run

If you want to run without Docker:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

And in another terminal:

```bash
cd frontend
npm install
npm run dev
```

## API endpoints

- `GET /api/health`
- `POST /api/index` : start background indexing job
- `GET /api/index/{job_id}` : get index job status/result
- `POST /api/chat` : ask question and get response + extracted code block
- `WS /api/ws/chat` : websocket chat (frontend default, HTTP fallback)

## Typical flow

1. Start the stack (`docker compose up --build`) or run services locally.
2. Run indexing for `data/pdfs` and `data/tutorials`.
3. Use the **Use RAG** toggle:
   - ON: answers use indexed context
   - OFF: direct model response without retrieval
4. Ask for code generation in chat.

Indexing behavior:
- Incremental by file hash: only new/changed files are re-embedded.
- Removed files are deleted from the vector table.
- `indexed_chunks` is the number of chunks processed in the latest run.

