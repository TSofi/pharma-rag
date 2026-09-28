"""FastAPI app. Endpoints:
  GET  /api/health   -> is the service up, which LLM/vector DB are used
  GET  /api/drugs    -> drugs in the knowledge base
  GET  /api/fact     -> short "did you know?" fact for the loading screen
  POST /api/ask      -> {question} -> answer + cited sources
  GET  /docs         -> automatic Swagger UI
  GET  /             -> the frontend (served from ./frontend when running locally or on HF)
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal

from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, feedback, rag

from contextlib import asynccontextmanager

from . import store


@asynccontextmanager
async def lifespan(_app):
    # Warm-up: load the embedding model and open the Qdrant connection at startup,
    # so the FIRST user question doesn't pay for it (a few seconds on a small CPU).
    try:
        store.get_client()
        # Warm up the multilingual model (used by 5 of the 6 markets); the English one loads on first US question.
        store.embed_query("warm up", config.MULTI_EMBED_MODEL)
        for m in config.MARKETS:
            rag.markets.index(m)  # load the product indexes (~1 s) before the first question
    except Exception as e:  # noqa: BLE001 -- never block startup on warm-up
        print(f"warm-up skipped: {e}", flush=True)
    yield


app = FastAPI(title="PharmaRAG", description="Drug-label Q&A with traceable sources", version="1.0",
              lifespan=lifespan)

# CORS lets the browser call this API from another domain (the Vercel frontend).
app.add_middleware(
    CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"]
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, examples=["What is metformin used for?"])
    top_k: int | None = Field(None, ge=1, le=15)
    answer_language: Literal["auto", "en", "uk", "pl", "cs", "es", "fr"] = "auto"
    market: Literal["us", "pl", "ua", "cz", "es", "fr"] = "us"


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "llm": config.LLM_PROVIDER,
        "vector_db": "qdrant-cloud" if config.QDRANT_URL else "qdrant-local",
        "drugs": len(rag.library()),
        "markets": {m: len(rag.library(m)) for m in config.MARKETS},
    }


@app.get("/api/drugs")
def drugs(market: Literal["us", "pl", "ua", "cz", "es", "fr"] = "us"):
    return rag.library(market)


@app.get("/api/fact")
def fact(seen: str = "", lang: Literal["en", "uk", "pl", "cs", "es", "fr"] = "en"):
    """A short curated fact for the loading screen. `seen` = drugs already shown (comma-separated)."""
    return rag.random_fact({d for d in seen.split(",") if d}, lang)


@app.post("/api/ask")
def ask(req: AskRequest):
    try:
        return rag.ask(req.question, req.top_k, req.answer_language, req.market)
    except Exception as e:  # surface a readable error to the UI instead of a bare 500
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e


class FeedbackRequest(BaseModel):
    message: str = Field(..., min_length=5, max_length=3000)
    contact: str = Field("", max_length=200)
    context: str = Field("", max_length=6000)       # last question/answer, market, browser (if the user agrees)
    image: str | None = Field(None, max_length=6_000_000)  # screenshot as a data: URL
    website: str = ""                              # honeypot: real people never fill this hidden field


@app.post("/api/feedback")
def send_feedback(req: FeedbackRequest, request: Request):
    if req.website:  # a bot filled the hidden field: pretend it worked
        return {"ok": True}
    if not feedback.allowed(request.client.host if request.client else "?"):
        raise HTTPException(status_code=429, detail="Too many reports, please try again later.")
    try:
        feedback.send(req.message, req.contact, req.context, req.image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Could not send: {e}") from e
    return {"ok": True}


# Serve the static frontend at "/" (must be mounted last so it doesn't shadow /api routes).
_frontend = config.ROOT / "frontend"
if _frontend.exists():
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")
