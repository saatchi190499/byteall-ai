import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from .config import settings
from .indexing import indexing_manager
from .ollama_client import OllamaClient, extract_code, strip_code_blocks
from .rag import RagStore
from .schemas import (
    ChatRequest,
    ChatResponse,
    FileDeleteRequest,
    FileDeleteResponse,
    FileListResponse,
    FileUploadResponse,
    IndexResponse,
    IndexStartResponse,
    IndexStatusResponse,
    ManagedFile,
    SourceChunk,
)

router = APIRouter(prefix="/api", tags=["agent"])
logger = logging.getLogger(__name__)

MAX_EDITOR_CONTEXT_CHARS = 12_000
MAX_RAG_CONTEXT_CHARS = 18_000

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


@router.get("/files", response_model=FileListResponse)
def list_index_files(bucket: str | None = None) -> FileListResponse:
    requested_buckets = [_normalize_bucket(bucket)] if bucket else ["tutorials", "pdfs"]

    files: list[ManagedFile] = []
    for bucket_name in requested_buckets:
        root = _resolve_bucket_dir(bucket_name)
        files.extend(_scan_bucket_files(bucket_name, root))

    files.sort(key=lambda item: (item.bucket, item.path.lower()))
    return FileListResponse(files=files)


@router.post("/files/upload", response_model=FileUploadResponse)
async def upload_index_file(
    upload: UploadFile = File(...),
    bucket: str = Form("tutorials"),
    relative_dir: str = Form(""),
) -> FileUploadResponse:
    bucket_name = _normalize_bucket(bucket)
    root = _resolve_bucket_dir(bucket_name)

    filename = Path(str(upload.filename or "")).name.strip()
    if not filename:
        raise HTTPException(status_code=400, detail="File name is required")

    if bucket_name == "pdfs" and Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only .pdf files are allowed in pdfs bucket")

    rel_dir_path = _safe_relative_path(relative_dir, allow_empty=True)
    rel_file_path = (rel_dir_path / filename) if rel_dir_path != Path(".") else Path(filename)
    target_path = _resolve_under_root(root, rel_file_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    target_path.write_bytes(content)
    managed = _to_managed_file(bucket_name, root, target_path)
    return FileUploadResponse(message="uploaded", file=managed)


@router.delete("/files", response_model=FileDeleteResponse)
def delete_index_file(req: FileDeleteRequest) -> FileDeleteResponse:
    bucket_name = _normalize_bucket(req.bucket)
    root = _resolve_bucket_dir(bucket_name)
    rel_file_path = _safe_relative_path(req.path, allow_empty=False)
    target_path = _resolve_under_root(root, rel_file_path)

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    managed = _to_managed_file(bucket_name, root, target_path)
    target_path.unlink()
    _cleanup_empty_parent_dirs(target_path.parent, root)
    return FileDeleteResponse(message="deleted", file=managed)


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
    except HTTPException:
        raise
    except Exception as exc:
        detail = _format_chat_exception(exc)
        status_code = 503 if _is_upstream_error(exc) else 500
        raise HTTPException(status_code=status_code, detail=detail) from exc


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
        await websocket.send_json({"error": _format_chat_exception(exc)})
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


def _normalize_bucket(bucket: str | None) -> str:
    normalized = str(bucket or "").strip().lower()
    if normalized in {"tutorials", "tutorial", "docs", "doc"}:
        return "tutorials"
    if normalized in {"pdfs", "pdf"}:
        return "pdfs"
    raise HTTPException(status_code=400, detail="Invalid bucket. Use 'tutorials' or 'pdfs'.")


def _resolve_bucket_dir(bucket: str) -> Path:
    if bucket == "tutorials":
        root = settings.tutorials_dir
    elif bucket == "pdfs":
        root = settings.pdf_dir
    else:
        raise HTTPException(status_code=400, detail="Invalid bucket")

    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _safe_relative_path(path: str, allow_empty: bool) -> Path:
    raw = str(path or "").replace("\\", "/").strip().lstrip("/")
    if not raw:
        if allow_empty:
            return Path(".")
        raise HTTPException(status_code=400, detail="Path is required")

    candidate = Path(raw)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise HTTPException(status_code=400, detail="Invalid path")

    if any(":" in part for part in candidate.parts):
        raise HTTPException(status_code=400, detail="Invalid path")

    return candidate


def _resolve_under_root(root: Path, relative_path: Path) -> Path:
    target = (root / relative_path).resolve()
    if target != root and root not in target.parents:
        raise HTTPException(status_code=400, detail="Path traversal is not allowed")
    return target


def _to_managed_file(bucket: str, root: Path, file_path: Path) -> ManagedFile:
    stat = file_path.stat()
    return ManagedFile(
        bucket=bucket,
        path=file_path.relative_to(root).as_posix(),
        size=stat.st_size,
        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
    )


def _scan_bucket_files(bucket: str, root: Path) -> list[ManagedFile]:
    files: list[ManagedFile] = []
    if not root.exists():
        return files

    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(_to_managed_file(bucket, root, path))
    return files


def _cleanup_empty_parent_dirs(start_dir: Path, root: Path) -> None:
    current = start_dir
    while current != root:
        if not current.exists():
            current = current.parent
            continue
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _truncate_for_prompt(text: str, max_chars: int, label: str) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[{label} truncated: omitted {omitted} chars]"


def _flatten_exception_messages(exc: Exception) -> list[str]:
    parts: list[str] = []
    seen: set[int] = set()
    current: Exception | None = exc

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        msg = str(current).strip()
        if msg:
            parts.append(msg)
        current = current.__cause__ or current.__context__

    return parts or ["Unknown error"]


def _is_upstream_error(exc: Exception) -> bool:
    text = " | ".join(_flatten_exception_messages(exc)).lower()
    markers = [
        "ollama",
        "connection refused",
        "connecterror",
        "readtimeout",
        "timed out",
        "all connection attempts failed",
        "psycopg",
        "could not connect to server",
        "database",
    ]
    return any(marker in text for marker in markers)


def _format_chat_exception(exc: Exception) -> str:
    parts = _flatten_exception_messages(exc)
    joined = " | ".join(parts)
    return f"Chat failed: {joined}"


def build_chat_response(req: ChatRequest) -> ChatResponse:
    profile = _normalize_rag_profile(req.rag_profile)

    hits = []
    rag_notice = ""
    if req.use_rag:
        try:
            hits = store.search(req.message, settings.top_k, profile=profile)
        except Exception as rag_exc:
            logger.exception("RAG search failed; continuing without RAG")
            rag_notice = f"[System notice] RAG unavailable, answered without indexed context: {rag_exc}"
            hits = []

    prompt = build_prompt(req, hits, profile)
    answer = ollama.chat(prompt)
    response = format_chat_response(answer, hits)

    if rag_notice:
        response.answer = f"{rag_notice}\n\n{response.answer}".strip()

    return response


def build_prompt(req: ChatRequest, hits, profile: str | None) -> str:
    raw_context = "\n\n".join(f"Source: {rec.source}\n{rec.text}" for rec, _score in hits)
    context = _truncate_for_prompt(raw_context, MAX_RAG_CONTEXT_CHARS, "RAG context")

    rag_state = "enabled" if req.use_rag else "disabled"
    rag_profile = profile or "auto"

    editor_code_raw = req.editor_code.strip()
    editor_code = _truncate_for_prompt(editor_code_raw, MAX_EDITOR_CONTEXT_CHARS, "Editor code")
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