import re

import httpx


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        chat_model: str,
        embedding_model: str,
        timeout_seconds: float = 120.0,
        chat_options: dict | None = None,
        keep_alive: str | None = None,
        fallback_chat_models: list[str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.timeout = max(float(timeout_seconds), 1.0)
        self.chat_options = dict(chat_options or {})
        self.keep_alive = str(keep_alive or "").strip() or None

        seen: set[str] = set()
        self.fallback_chat_models: list[str] = []
        for raw in fallback_chat_models or []:
            model = str(raw or "").strip()
            if not model or model == self.chat_model or model in seen:
                continue
            seen.add(model)
            self.fallback_chat_models.append(model)

    def embed(self, text: str) -> list[float]:
        payload = {"model": self.embedding_model, "prompt": text}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/api/embeddings", json=payload)
            self._raise_with_ollama_error(response)
            return response.json()["embedding"]

    def chat(self, prompt: str) -> str:
        base_payload = {
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a senior coding assistant. Use the provided context when relevant. "
                        "Return concise explanations and include a code block when code is requested."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        if self.chat_options:
            base_payload["options"] = self.chat_options
        if self.keep_alive:
            base_payload["keep_alive"] = self.keep_alive

        candidates = [self.chat_model, *self.fallback_chat_models]

        with httpx.Client(timeout=self.timeout) as client:
            for model in candidates:
                payload = dict(base_payload)
                payload["model"] = model
                response = client.post(f"{self.base_url}/api/chat", json=payload)

                if response.is_success:
                    return response.json()["message"]["content"]

                if self._is_missing_model_response(response) or self._is_retryable_model_response(response):
                    continue

                self._raise_with_ollama_error(response)

        tried = ", ".join(candidates)
        raise RuntimeError(f"Ollama chat model not available. Tried: {tried}")

    @staticmethod
    def _is_missing_model_response(response: httpx.Response) -> bool:
        if response.status_code not in {400, 404}:
            return False

        detail = (response.text or "").lower()
        return (
            ("model" in detail and "not found" in detail)
            or "pull a model" in detail
            or ("manifest" in detail and "not found" in detail)
        )

    @staticmethod
    def _is_retryable_model_response(response: httpx.Response) -> bool:
        if response.status_code < 500:
            return False

        detail = (response.text or "").lower()
        markers = [
            "runner process has terminated",
            "out of memory",
            "insufficient memory",
            "model requires more system memory",
            "cuda",
            "metal",
            "resource temporarily unavailable",
        ]
        return any(marker in detail for marker in markers)

    @staticmethod
    def _raise_with_ollama_error(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text.strip()
            raise RuntimeError(f"Ollama error: {detail}") from exc


def extract_code(markdown: str) -> str:
    match = re.search(r"```(?:\w+)?\n(.*?)```", markdown, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def strip_code_blocks(markdown: str) -> str:
    cleaned = re.sub(r"```(?:\w+)?\n.*?```", "", markdown, flags=re.DOTALL)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
