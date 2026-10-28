"""Build, save and load the search index: chunks plus their embedding vectors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .corpus import Chunk
from .embed import Embedder


@dataclass
class Index:
    chunks: list[Chunk]
    vectors: np.ndarray  # shape (len(chunks), dim), unit-length rows
    model: str
    fingerprint: str  # hash of the chunk texts, to spot a stale index

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        (folder / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (folder / "meta.json").write_text(
            json.dumps({"model": self.model, "fingerprint": self.fingerprint, "chunks": len(self.chunks)}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, folder: Path) -> "Index":
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in json.loads((folder / "chunks.json").read_text(encoding="utf-8"))]
        vectors = np.load(folder / "vectors.npy")
        if len(chunks) != vectors.shape[0]:
            raise ValueError(f"index at {folder} is corrupt: {len(chunks)} chunks, {vectors.shape[0]} vectors")
        return cls(chunks=chunks, vectors=vectors, model=meta["model"], fingerprint=meta["fingerprint"])


def fingerprint(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.id.encode())
        digest.update(chunk.text.encode())
    return digest.hexdigest()[:16]


def build_index(chunks: list[Chunk], embedder: Embedder) -> Index:
    vectors = embedder.embed_documents([c.text for c in chunks])
    return Index(chunks=chunks, vectors=vectors, model=embedder.name, fingerprint=fingerprint(chunks))
