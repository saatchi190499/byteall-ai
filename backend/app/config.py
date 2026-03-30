from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen3:8b"
    ollama_chat_fallback_models: str = "llama3.1:latest,qwen2.5-coder:3b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_timeout_seconds: float = 180.0

    # Ollama generation tuning
    ollama_keep_alive: str = "15m"
    ollama_chat_num_ctx: int = 4096
    ollama_chat_num_predict: int = 512
    ollama_chat_temperature: float = 0.1
    ollama_chat_top_p: float = 0.9
    ollama_chat_repeat_penalty: float = 1.05
    ollama_chat_num_thread: int = 0  # 0 => Ollama auto
    ollama_chat_num_gpu: int = -1    # -1 => Ollama default behavior

    # Backend request parallelism
    uvicorn_workers: int = 2

    pdf_dir: Path = Path("data/pdfs")
    tutorials_dir: Path = Path("data/tutorials")
    database_url: str = "postgresql://postgres:postgres@localhost:5432/ai_agent"
    vector_table_name: str = "rag_chunks"
    embed_workers: int = 4
    index_batch_size: int = 24
    chunk_size: int = 900
    chunk_overlap: int = 120
    top_k: int = 4
    cors_allow_origins: str = (
        "http://localhost:5173,"
        "http://127.0.0.1:5173,"
        "http://localhost:3000,"
        "http://127.0.0.1:3000,"
        "https://btlweb"
    )
    cors_allow_origin_regex: str = ""

    model_config = SettingsConfigDict(
        env_file=("backend/.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
