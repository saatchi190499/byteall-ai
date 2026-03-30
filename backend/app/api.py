from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from .config import settings
from .indexing import indexing_manager
from .ollama_client import OllamaClient, extract_code, strip_code_blocks
from .rag import RagStore
from .schemas import (
    ChatRequest,
    ChatResponse,
    IndexResponse,
    IndexStartResponse,
    IndexStatusResponse,
    SourceChunk,
)

router = APIRouter(prefix="/api", tags=["agent"])

ollama = OllamaClient(
    base_url=settings.ollama_base_url,
    chat_model=settings.ollama_chat_model,
    embedding_model=settings.ollama_embedding_model,
)
store = RagStore(
    ollama,
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    database_url=settings.database_url,
    table_name=settings.vector_table_name,
    embed_workers=settings.embed_workers,
    add_batch_size=settings.index_batch_size,
)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/index", response_model=IndexStartResponse)
def start_index_documents() -> IndexStartResponse:
    job_id = indexing_manager.start(_run_index)
    return IndexStartResponse(job_id=job_id, status="queued")


@router.get("/index/{job_id}", response_model=IndexStatusResponse)
def get_index_status(job_id: str) -> IndexStatusResponse:
    job = indexing_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Index job not found")

    result_data = job.get("result")
    result = IndexResponse(**result_data) if result_data else None
    return IndexStatusResponse(
        job_id=job_id,
        status=job["status"],
        error=job.get("error"),
        result=result,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        return build_chat_response(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc


@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            req = ChatRequest.model_validate(payload)
            response = build_chat_response(req)
            await websocket.send_json(response.model_dump())
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"error": f"Chat failed: {exc}"})
        await websocket.close(code=1011)


def _normalize_rag_profile(raw: str) -> str | None:
    profile = str(raw or "").strip().lower()
    if profile in {"", "auto", "all", "none"}:
        return None
    if profile in {"petex", "gap"}:
        return "petex"
    if profile in {"tnav", "tnavigator", "t_nav", "t-navigator"}:
        return "tnav"
    if profile in {"pi", "pi-system", "pisystem"}:
        return "pi"
    return None


def build_chat_response(req: ChatRequest) -> ChatResponse:
    profile = _normalize_rag_profile(req.rag_profile)
    hits = store.search(req.message, settings.top_k, profile=profile) if req.use_rag else []
    prompt = build_prompt(req, hits, profile)
    answer = ollama.chat(prompt)
    return format_chat_response(answer, hits)


def build_prompt(req: ChatRequest, hits, profile: str | None) -> str:
    context = "\n\n".join(f"Source: {rec.source}\n{rec.text}" for rec, _score in hits)
    rag_state = "enabled" if req.use_rag else "disabled"
    rag_profile = profile or "auto"
    editor_code = req.editor_code.strip()
    editor_context = editor_code if editor_code else "No editor code provided."
    return (
        "User request:\n"
        f"{req.message}\n\n"
        "Current notebook editor code:\n"
        f"{editor_context}\n\n"
        f"RAG mode: {rag_state}\n"
        f"RAG profile: {rag_profile}\n"
        "Reference context from indexed docs:\n"
        f"{context if context else 'No context provided.'}\n\n"
        "If code is relevant, provide one markdown code block."
    )


def format_chat_response(answer: str, hits) -> ChatResponse:
    code = extract_code(answer)
    clean_answer = strip_code_blocks(answer)
    sources = [
        SourceChunk(source=rec.source, score=round(score, 4), excerpt=rec.text[:240])
        for rec, score in hits
    ]
    return ChatResponse(answer=clean_answer, code=code, sources=sources)


def _run_index() -> dict:
    files, chunks, skipped_files = store.build(settings.pdf_dir, settings.tutorials_dir)
    payload = IndexResponse(indexed_files=files, indexed_chunks=chunks, skipped_files=skipped_files)
    return payload.model_dump()
