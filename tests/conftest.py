import zlib
from pathlib import Path

import numpy as np
import pytest

from worldbible_rag.corpus import load_corpus
from worldbible_rag.embed import normalize
from worldbible_rag.index import build_index
from worldbible_rag.retrieve import Retriever, tokenize

SAMPLE = Path(__file__).resolve().parents[1] / "sample_corpus"


class HashEmbedder:
    """Bag-of-words hashed into a fixed vector: deterministic and offline.

    It is not a good embedding model. It lets the tests exercise the vector
    path without downloading one.
    """

    name = "test-hash-256"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(256, dtype=np.float32)
        for token in tokenize(text):
            v[zlib.crc32(token.encode()) % 256] += 1.0  # not hash(): that changes per process
        return v

    def embed_documents(self, texts):
        return normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text):
        return normalize(self._vec(text))


@pytest.fixture(scope="session")
def chunks():
    return load_corpus(SAMPLE)


@pytest.fixture(scope="session")
def index(chunks):
    return build_index(chunks, HashEmbedder())


@pytest.fixture(scope="session")
def retriever(index):
    return Retriever(index, HashEmbedder())
