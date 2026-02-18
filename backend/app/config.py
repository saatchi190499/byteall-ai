from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen3:8b"
    ollama_embedding_model: str = "nomic-embed-text"
    pdf_dir: Path = Path("data/pdfs")
    tutorials_dir: Path = Path("data/tutorials")
    database_url: str = "postgresql://postgres:postgres@localhost:5432/ai_agent"
    vector_table_name: str = "rag_chunks"
    embed_workers: int = 4
    index_batch_size: int = 24
    chunk_size: int = 900
    chunk_overlap: int = 120
    top_k: int = 4

    model_config = SettingsConfigDict(
        env_file=("backend/.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
