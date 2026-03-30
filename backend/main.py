from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

if __package__:
    from .app.api import router as agent_router
    from .app.config import settings
else:
    from app.api import router as agent_router
    from app.config import settings


app = FastAPI(title="CodeGen AI Agent", version="1.0.0")


def _parse_origins(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_origins(settings.cors_allow_origins),
    allow_origin_regex=(settings.cors_allow_origin_regex or None),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent_router)
