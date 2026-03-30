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
chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5-coder:3b")
embed_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
url = f"{base}/api/tags"
required = {chat_model, embed_model}

print(f"[startup] waiting for Ollama models: {', '.join(sorted(required))}")
for attempt in range(1, 181):
    try:
        with urlopen(url, timeout=5) as response:
            payload = json.load(response)
        names = {item.get("name", "") for item in payload.get("models", [])}
        normalized = set(names)
        normalized.update({n.removesuffix(":latest") for n in names if n.endswith(":latest")})
        if all(m in normalized or f"{m}:latest" in names for m in required):
            print(f"[startup] models ready: {', '.join(sorted(required))}")
            sys.exit(0)
    except URLError:
        pass
    except Exception as exc:
        print(f"[startup] probe error: {exc}")

    if attempt % 12 == 0:
        print(f"[startup] still waiting ({attempt * 5}s elapsed)")
    time.sleep(5)

print(f"[startup] timeout waiting for models {sorted(required)} at {url}")
sys.exit(1)
PY

UVICORN_WORKERS="${UVICORN_WORKERS:-2}"
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers "$UVICORN_WORKERS"
