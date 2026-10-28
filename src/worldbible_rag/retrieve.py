"""Three ways to find the chunks that answer a question.

- vector:  cosine similarity between query and chunk embeddings (meaning)
- keyword: BM25 over word tokens (exact names, colours, numbers)
- hybrid:  reciprocal rank fusion of the two lists

Which one is used by default was decided by the retrieval eval (see evals/).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from .corpus import Chunk
from .embed import Embedder
from .index import Index

Mode = Literal["vector", "keyword", "hybrid"]
MODES: tuple[Mode, ...] = ("vector", "keyword", "hybrid")

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by does did do for from has have he her his how in is it its of on or "
    "she that the their them there they this to was what when where which who whose why will with".split()
)
# Standard RRF constant; larger values flatten the advantage of the very top ranks.
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int  # 1-based


class Retriever:
    def __init__(self, index: Index, embedder: Embedder):
        if embedder.name != index.model:
            raise ValueError(f"index was built with {index.model!r} but the embedder is {embedder.name!r}")
        self.index = index
        self.embedder = embedder
        self._bm25 = BM25Okapi([tokenize(c.text) for c in index.chunks])

    def _vector_ranking(self, query: str) -> list[tuple[int, float]]:
        scores = self.index.vectors @ self.embedder.embed_query(query)
        order = np.argsort(-scores)
        return [(int(i), float(scores[i])) for i in order]

    def _keyword_ranking(self, query: str) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(-scores, kind="stable")
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def _hybrid_ranking(self, query: str) -> list[tuple[int, float]]:
        fused: dict[int, float] = {}
        for ranking in (self._vector_ranking(query), self._keyword_ranking(query)):
            for rank, (i, _) in enumerate(ranking, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(fused.items(), key=lambda item: -item[1])

    def search(self, query: str, k: int = 5, mode: Mode = "hybrid") -> list[Hit]:
        if not query.strip():
            return []
        ranking = {
            "vector": self._vector_ranking,
            "keyword": self._keyword_ranking,
            "hybrid": self._hybrid_ranking,
        }[mode](query)
        return [
            Hit(chunk=self.index.chunks[i], score=score, rank=rank)
            for rank, (i, score) in enumerate(ranking[:k], start=1)
        ]
