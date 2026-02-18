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
model = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b")
url = f"{base}/api/tags"

print(f"[startup] waiting for Ollama model: {model}")
for attempt in range(1, 181):
    try:
        with urlopen(url, timeout=5) as response:
            payload = json.load(response)
        names = {item.get("name", "") for item in payload.get("models", [])}
        if model in names or f"{model}:latest" in names:
            print(f"[startup] model ready: {model}")
            sys.exit(0)
    except URLError:
        pass
    except Exception as exc:
        print(f"[startup] probe error: {exc}")

    if attempt % 12 == 0:
        print(f"[startup] still waiting ({attempt * 5}s elapsed)")
    time.sleep(5)

print(f"[startup] timeout waiting for model '{model}' at {url}")
sys.exit(1)
PY

exec uvicorn backend.main:app --host 0.0.0.0 --port 8000
