#!/usr/bin/env sh
set -eu

python - <<'PY'
import json
import os
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

base = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b")
embed_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
fallback_raw = os.getenv("OLLAMA_CHAT_FALLBACK_MODELS", "llama3.1:latest,qwen2.5-coder:3b")
fallback_models = [m.strip() for m in fallback_raw.split(",") if m.strip() and m.strip() != chat_model]
chat_candidates = [chat_model, *fallback_models]
url = f"{base}/api/tags"


def has_model(name: str, names: set[str], normalized: set[str]) -> bool:
    if not name:
        return False
    if name in names or name in normalized:
        return True
    if not name.endswith(":latest") and f"{name}:latest" in names:
        return True
    return False


print(f"[startup] waiting for embedding model: {embed_model}")
print(f"[startup] waiting for chat models (first available): {', '.join(chat_candidates)}")

for attempt in range(1, 181):
    try:
        with urlopen(url, timeout=5) as response:
            payload = json.load(response)
        names = {item.get("name", "") for item in payload.get("models", [])}
        normalized = set(names)
        normalized.update({n.removesuffix(":latest") for n in names if n.endswith(":latest")})

        embed_ready = has_model(embed_model, names, normalized)
        chat_ready = any(has_model(model, names, normalized) for model in chat_candidates)

        if embed_ready and chat_ready:
            print("[startup] models ready")
            sys.exit(0)
    except URLError:
        pass
    except Exception as exc:
        print(f"[startup] probe error: {exc}")

    if attempt % 12 == 0:
        print(f"[startup] still waiting ({attempt * 5}s elapsed)")
    time.sleep(5)

print(f"[startup] timeout waiting for embedding model '{embed_model}' and chat models {chat_candidates} at {url}")
sys.exit(1)
PY

UVICORN_WORKERS="${UVICORN_WORKERS:-2}"
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers "$UVICORN_WORKERS"
