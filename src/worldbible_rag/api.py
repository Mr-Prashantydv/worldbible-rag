"""HTTP API.

GET  /health          index status
GET  /search?q=&k=&mode=   ranked passages, no LLM call
POST /ask             grounded answer with checked citations
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .answer import Answerer, AnswerUnavailable
from .config import Settings
from .embed import FastEmbedder
from .index import Index
from .retrieve import Mode, Retriever


class Passage(BaseModel):
    id: str
    title: str
    section: str
    text: str
    score: float
    rank: int


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=20)
    mode: Mode = "hybrid"


class AskResponse(BaseModel):
    answer: str
    found: bool
    citations: list[Passage]
    rejected_citations: list[str]
    model: str


def default_retriever() -> Retriever:
    settings = Settings.from_env()
    try:
        index = Index.load(settings.index_dir)
    except FileNotFoundError as exc:
        raise RuntimeError(f"No index at {settings.index_dir}. Run `worldbible ingest` first.") from exc
    return Retriever(index, FastEmbedder(index.model))


def create_app(
    make_retriever: Callable[[], Retriever] = default_retriever,
    make_answerer: Callable[[], Answerer] = Answerer,
) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["retriever"] = make_retriever()
        yield

    app = FastAPI(title="World Bible RAG", lifespan=lifespan)

    def passages(hits) -> list[Passage]:
        return [
            Passage(id=h.chunk.id, title=h.chunk.title, section=h.chunk.section, text=h.chunk.text, score=h.score, rank=h.rank)
            for h in hits
        ]

    @app.get("/health")
    def health() -> dict:
        index = state["retriever"].index
        return {"status": "ok", "chunks": len(index.chunks), "embed_model": index.model, "fingerprint": index.fingerprint}

    @app.get("/search", response_model=list[Passage])
    def search(q: str = Query(min_length=1, max_length=500), k: int = Query(5, ge=1, le=20), mode: Mode = "hybrid"):
        return passages(state["retriever"].search(q, k=k, mode=mode))

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        hits = state["retriever"].search(req.question, k=req.k, mode=req.mode)
        try:
            if "answerer" not in state:
                state["answerer"] = make_answerer()
            result = state["answerer"].answer(req.question, hits)
        except AnswerUnavailable as exc:
            raise HTTPException(status_code=503, detail=f"Answering is unavailable: {exc}") from exc
        by_id = {p.id: p for p in passages(hits)}
        return AskResponse(
            answer=result.answer,
            found=result.found,
            citations=[by_id[c] for c in result.citations],
            rejected_citations=result.rejected_citations,
            model=result.model,
        )

    return app
