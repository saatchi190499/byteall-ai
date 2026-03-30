from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re

import psycopg
from psycopg import sql

from .ollama_client import OllamaClient
from .pdf_loader import chunk_text, load_documents


@dataclass
class ChunkRecord:
    source: str
    text: str


class RagStore:
    def __init__(
        self,
        ollama: OllamaClient,
        chunk_size: int,
        chunk_overlap: int,
        database_url: str,
        table_name: str,
        embed_workers: int,
        add_batch_size: int,
    ) -> None:
        self.ollama = ollama
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.database_url = database_url
        self.table_name = self._validate_table_name(table_name)
        self.embed_workers = max(int(embed_workers), 1)
        self.add_batch_size = max(int(add_batch_size), 1)

        self._ensure_schema()

    @staticmethod
    def _validate_table_name(name: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("Invalid VECTOR_TABLE_NAME")
        return name

    @staticmethod
    def _embedding_literal(values: list[float]) -> str:
        return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"

    @staticmethod
    def _file_key(source: str) -> str:
        return source.split("#", maxsplit=1)[0]

    @staticmethod
    def _profile_patterns(profile: str | None) -> list[str]:
        if not profile:
            return []
        if profile == "petex":
            return [
                "tutorials/petex/%",
                "tutorials/gap/%",
                "tutorials/workflows/notebook_syntax_rules.txt%",
                "tutorials/workflows/notebook_global_rules.txt%",
                "%petex%",
                "%gap%",
                "%openserver%",
                "%inputs%",
                "%outputs%",
                "%notebook%",
            ]
        if profile == "tnav":
            return [
                "tutorials/tnav/%",
                "tutorials/tnavigator/%",
                "%tnav%",
                "%tnavigator%",
                "%t_navigator%",
                "%t-nav%",
            ]
        if profile == "pi":
            return [
                "tutorials/pi/%",
                "tutorials/pi_client/%",
                "%pi_client%",
                "%piwebapi%",
                "%osisoft%",
                "%\\pi\\%",
            ]
        if profile == "workflows":
            return [
                "tutorials/workflows/%",
                "%workflow%",
                "%notebook%",
                "%inputs%",
                "%outputs%",
            ]
        return []

    @staticmethod
    def _base_patterns() -> list[str]:
        # Always include only global, profile-agnostic notebook rules.
        return [
            "tutorials/workflows/notebook_syntax_rules.txt%",
            "tutorials/workflows/notebook_global_rules.txt%",
            "%notebook_syntax_rules%",
            "%notebook_global_rules%",
        ]

    @staticmethod
    def _dedupe_hits(hits: list[tuple[ChunkRecord, float]]) -> list[tuple[ChunkRecord, float]]:
        seen: set[tuple[str, str]] = set()
        unique: list[tuple[ChunkRecord, float]] = []
        for rec, score in hits:
            key = (rec.source, rec.text)
            if key in seen:
                continue
            seen.add(key)
            unique.append((rec, score))
        return unique

    def _connect(self):
        return psycopg.connect(self.database_url, autocommit=True)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {} (
                            id BIGSERIAL PRIMARY KEY,
                            source TEXT NOT NULL,
                            file TEXT NOT NULL,
                            file_hash TEXT NOT NULL,
                            content TEXT NOT NULL,
                            embedding VECTOR NOT NULL
                        )
                        """
                    ).format(sql.Identifier(self.table_name))
                )
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (file)").format(
                        sql.Identifier(f"idx_{self.table_name}_file"),
                        sql.Identifier(self.table_name),
                    )
                )

    def _load_existing_file_hashes(self) -> dict[str, str]:
        stmt = sql.SQL(
            "SELECT file, MAX(file_hash) AS file_hash FROM {} GROUP BY file"
        ).format(sql.Identifier(self.table_name))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(stmt)
                return {row[0]: row[1] for row in cur.fetchall()}

    def _delete_files(self, files: set[str]) -> None:
        if not files:
            return
        stmt = sql.SQL("DELETE FROM {} WHERE file = ANY(%s)").format(sql.Identifier(self.table_name))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(stmt, (list(files),))

    def build(self, pdf_dir: Path, tutorials_dir: Path) -> tuple[int, int, list[str]]:
        pages, skipped_files = load_documents(pdf_dir, tutorials_dir)

        docs_by_file: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for source, text in pages:
            docs_by_file[self._file_key(source)].append((source, text))

        current_hashes: dict[str, str] = {}
        for file_name, docs in docs_by_file.items():
            digest = sha256()
            for source, text in docs:
                digest.update(source.encode("utf-8", errors="ignore"))
                digest.update(b"\n")
                digest.update(text.encode("utf-8", errors="ignore"))
                digest.update(b"\n")
            current_hashes[file_name] = digest.hexdigest()

        existing_hashes = self._load_existing_file_hashes()
        current_files = set(current_hashes.keys())
        existing_files = set(existing_hashes.keys())

        removed_files = existing_files - current_files
        changed_files = {
            file_name
            for file_name, file_hash in current_hashes.items()
            if existing_hashes.get(file_name) != file_hash
        }

        self._delete_files(removed_files | changed_files)

        inserted_chunks = 0
        if not changed_files:
            return len(current_files), inserted_chunks, skipped_files

        insert_stmt = sql.SQL(
            "INSERT INTO {} (source, file, file_hash, content, embedding) VALUES (%s, %s, %s, %s, %s::vector)"
        ).format(sql.Identifier(self.table_name))

        with self._connect() as conn:
            with conn.cursor() as cur:
                with ThreadPoolExecutor(max_workers=self.embed_workers) as pool:
                    batch_rows: list[tuple[str, str, str, str]] = []
                    for file_name in sorted(changed_files):
                        file_hash = current_hashes[file_name]
                        for source, text in docs_by_file[file_name]:
                            for chunk in chunk_text(text, self.chunk_size, self.chunk_overlap):
                                batch_rows.append((source, file_name, file_hash, chunk))
                                if len(batch_rows) >= self.add_batch_size:
                                    inserted_chunks += self._flush_batch(cur, insert_stmt, batch_rows, pool)
                                    batch_rows = []

                    if batch_rows:
                        inserted_chunks += self._flush_batch(cur, insert_stmt, batch_rows, pool)

        return len(current_files), inserted_chunks, skipped_files

    def _flush_batch(self, cur, insert_stmt, batch_rows, pool) -> int:
        embedding_inputs = [row[3] for row in batch_rows]
        if self.embed_workers == 1:
            vectors = [self.ollama.embed(text) for text in embedding_inputs]
        else:
            vectors = list(pool.map(self.ollama.embed, embedding_inputs))

        rows = [
            (source, file_name, file_hash, content, self._embedding_literal(embed))
            for (source, file_name, file_hash, content), embed in zip(batch_rows, vectors)
        ]
        cur.executemany(insert_stmt, rows)
        return len(rows)

    def _search_by_patterns(
        self,
        query_vector: str,
        top_k: int,
        patterns: list[str] | None = None,
    ) -> list[tuple[ChunkRecord, float]]:
        if top_k <= 0:
            return []

        where_clause = sql.SQL("")
        params: list[object] = [query_vector]

        if patterns:
            like_clauses = [sql.SQL("source ILIKE %s") for _ in patterns]
            where_clause = sql.SQL(" WHERE ") + sql.SQL(" OR ").join(like_clauses)
            params.extend(patterns)

        stmt = sql.SQL(
            """
            SELECT source, content, GREATEST(0.0, 1.0 - (embedding <=> %s::vector)) AS score
            FROM {}
            {}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
        ).format(sql.Identifier(self.table_name), where_clause)

        params.extend([query_vector, top_k])

        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(stmt, params)
                    rows = cur.fetchall()
        except psycopg.errors.UndefinedTable:
            return []

        hits: list[tuple[ChunkRecord, float]] = []
        for source, content, score in rows:
            hits.append((ChunkRecord(source=source, text=content), float(score)))
        return hits

    def search(self, query: str, top_k: int, profile: str | None = None) -> list[tuple[ChunkRecord, float]]:
        if top_k <= 0:
            return []

        query_embedding = self.ollama.embed(query)
        query_vector = self._embedding_literal(query_embedding)

        profile_patterns = self._profile_patterns(profile) or None
        base_patterns = self._base_patterns()

        primary_hits = self._search_by_patterns(query_vector, top_k, patterns=profile_patterns)
        base_limit = max(1, min(2, top_k))
        base_hits = self._search_by_patterns(query_vector, base_limit, patterns=base_patterns)

        merged_hits = self._dedupe_hits(primary_hits + base_hits)
        return merged_hits[: top_k + base_limit]



