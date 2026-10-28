"""Text embeddings.

The default model runs locally through fastembed (ONNX, CPU, no API key), so
building the index costs nothing and works offline after the first download.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Unit-length rows, so a dot product is cosine similarity."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class FastEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        from fastembed import TextEmbedding  # imported here: loading the model is slow

        self.name = model_name
        self._model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return normalize(np.array(list(self._model.passage_embed(texts))))

    def embed_query(self, text: str) -> np.ndarray:
        # bge models expect a query instruction prefix; query_embed adds it.
        return normalize(np.array(next(iter(self._model.query_embed(text)))))
