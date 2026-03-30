import re

import httpx


class OllamaClient:
    def __init__(self, base_url: str, chat_model: str, embedding_model: str, timeout_seconds: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.timeout = max(float(timeout_seconds), 1.0)

    def embed(self, text: str) -> list[float]:
        payload = {"model": self.embedding_model, "prompt": text}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/api/embeddings", json=payload)
            self._raise_with_ollama_error(response)
            return response.json()["embedding"]

    def chat(self, prompt: str) -> str:
        payload = {
            "model": self.chat_model,
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
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/api/chat", json=payload)
            self._raise_with_ollama_error(response)
            return response.json()["message"]["content"]

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
