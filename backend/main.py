from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

if __package__:
    from .app.api import router as agent_router
else:
    from app.api import router as agent_router

app = FastAPI(title="CodeGen AI Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent_router)
