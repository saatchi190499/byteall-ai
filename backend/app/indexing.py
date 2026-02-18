from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import time
from typing import Callable
from uuid import uuid4


class IndexingManager:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="indexer")
        self._lock = Lock()
        self._jobs: dict[str, dict] = {}

    def start(self, run_fn: Callable[[], dict]) -> str:
        job_id = uuid4().hex
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "error": None,
                "result": None,
                "started_at": None,
                "finished_at": None,
            }
        self._executor.submit(self._run, job_id, run_fn)
        return job_id

    def _run(self, job_id: str, run_fn: Callable[[], dict]) -> None:
        with self._lock:
            self._jobs[job_id]["status"] = "running"
            self._jobs[job_id]["started_at"] = time()

        try:
            result = run_fn()
            with self._lock:
                self._jobs[job_id]["status"] = "completed"
                self._jobs[job_id]["result"] = result
                self._jobs[job_id]["finished_at"] = time()
        except Exception as exc:
            with self._lock:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["error"] = str(exc)
                self._jobs[job_id]["finished_at"] = time()

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None


indexing_manager = IndexingManager()
