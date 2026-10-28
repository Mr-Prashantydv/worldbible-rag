"""Settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .embed import DEFAULT_MODEL as DEFAULT_EMBED_MODEL

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path
    index_dir: Path
    embed_model: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            corpus_dir=Path(os.environ.get("WORLDBIBLE_CORPUS", REPO_ROOT / "sample_corpus")),
            index_dir=Path(os.environ.get("WORLDBIBLE_INDEX", REPO_ROOT / ".index")),
            embed_model=os.environ.get("WORLDBIBLE_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        )
