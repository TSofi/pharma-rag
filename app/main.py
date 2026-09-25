"""FastAPI app. Endpoints:
  GET  /api/health   -> is the service up, which LLM/vector DB are used
  GET  /api/drugs    -> drugs in the knowledge base
  GET  /api/fact     -> short "did you know?" fact for the loading screen
  POST /api/ask      -> {question} -> answer + cited sources
  GET  /docs         -> automatic Swagger UI
  GET  /             -> the frontend (served from ./frontend when running locally or on HF)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal

from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, rag

app = FastAPI(title="PharmaRAG", description="Drug-label Q&A with traceable sources", version="1.0")

# CORS lets the browser call this API from another domain (the Vercel frontend).
app.add_middleware(
    CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"]
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, examples=["What is metformin used for?"])
    top_k: int | None = Field(None, ge=1, le=15)
    answer_language: Literal["auto", "en", "uk", "pl"] = "auto"
    market: Literal["us"] = "us"


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "llm": config.LLM_PROVIDER,
        "vector_db": "qdrant-cloud" if config.QDRANT_URL else "qdrant-local",
        "drugs": len(rag.library()),
    }


@app.get("/api/drugs")
def drugs():
    return rag.library()


@app.get("/api/fact")
def fact(seen: str = "", lang: Literal["en", "uk", "pl"] = "en"):
    """A short curated fact for the loading screen. `seen` = drugs already shown (comma-separated)."""
    return rag.random_fact({d for d in seen.split(",") if d}, lang)


@app.post("/api/ask")
def ask(req: AskRequest):
    try:
        return rag.ask(req.question, req.top_k, req.answer_language, req.market)
    except Exception as e:  # surface a readable error to the UI instead of a bare 500
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e


# Serve the static frontend at "/" (must be mounted last so it doesn't shadow /api routes).
_frontend = config.ROOT / "frontend"
if _frontend.exists():
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")
