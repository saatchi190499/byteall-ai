# AI Code Generation Agent

Stack:
- FastAPI backend
- Ollama for chat + embeddings
- RAG from local PDF + tutorial files + persistent PostgreSQL pgvector
- React frontend (code panel left, chat right)

## 1) Run Ollama On Host

Start Ollama locally on your Mac:

```bash
ollama serve
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

## 2) Run App With Docker

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
- Ollama API (host): `http://localhost:11434`
- PostgreSQL pgvector: `localhost:5432`

Notes:
- Backend container calls host Ollama via `http://host.docker.internal:11434`.
- Vector index is stored in PostgreSQL pgvector and survives container restarts.
- Indexing runs as a background job in backend.
- You can override model names when starting compose:
```bash
OLLAMA_CHAT_MODEL=qwen3:8b OLLAMA_EMBEDDING_MODEL=nomic-embed-text docker compose up --build
```
- You can tune indexing speed:
```bash
EMBED_WORKERS=4 INDEX_BATCH_SIZE=24 docker compose up --build
```

## 3) Optional Local (Non-Docker) Run

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
2. Click **Index PDFs** in UI (starts background indexing for `data/pdfs` and `data/tutorials`, rebuilding the pgvector table).
3. Use the **Use RAG (PDF context)** toggle:
   - ON: answers use indexed PDF chunks
   - OFF: direct model response without PDF retrieval
4. Ask for code generation in chat.

Indexing behavior:
- Incremental by file hash: only new/changed files are re-embedded.
- Removed files are deleted from the vector table.
- `indexed_chunks` is the number of chunks processed in the latest run.
