"""Retrieval eval: does the right passage come back for each question?

A case lists one or more expected targets. A target "slug#section" matches
that section (and its split pieces, e.g. "slug#section-2"); a bare "slug"
matches any section of that record. Metrics per search mode:

- hit@k: share of questions with an expected passage in the top k
- MRR:   mean of 1/rank of the first expected passage (0 if not in the top 10)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .retrieve import MODES, Mode, Retriever

KS = (1, 3, 5)
MRR_DEPTH = 10


@dataclass
class Case:
    id: str
    question: str
    expect: list[str]
    tag: str = ""


def load_cases(path: Path) -> list[Case]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cases = [Case(**item) for item in raw]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate case ids in {path}")
    return cases


def matches(chunk_id: str, target: str) -> bool:
    if "#" not in target:
        return chunk_id.split("#", 1)[0] == target
    # Split pieces of one section are numbered: "slug#notes-1", "slug#notes-2".
    return chunk_id == target or re.fullmatch(re.escape(target) + r"-\d+", chunk_id) is not None


def first_hit_rank(ranked_ids: list[str], expect: list[str]) -> int | None:
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if any(matches(chunk_id, t) for t in expect):
            return rank
    return None


def check_targets_exist(cases: list[Case], chunk_ids: list[str]) -> list[str]:
    """Targets that match no chunk at all: a typo in the eval file, not a retrieval miss."""
    return [
        f"{case.id}: {target}"
        for case in cases
        for target in case.expect
        if not any(matches(cid, target) for cid in chunk_ids)
    ]


def run(retriever: Retriever, cases: list[Case], modes: tuple[Mode, ...] = MODES) -> dict:
    missing = check_targets_exist(cases, [c.id for c in retriever.index.chunks])
    if missing:
        raise ValueError("eval targets not found in the corpus:\n  " + "\n  ".join(missing))

    report: dict = {"cases": len(cases), "modes": {}}
    for mode in modes:
        ranks: dict[str, int | None] = {}
        for case in cases:
            hits = retriever.search(case.question, k=MRR_DEPTH, mode=mode)
            ranks[case.id] = first_hit_rank([h.chunk.id for h in hits], case.expect)
        found = [r for r in ranks.values() if r is not None]
        report["modes"][mode] = {
            **{f"hit@{k}": sum(r <= k for r in found) / len(cases) for k in KS},
            "mrr": sum(1 / r for r in found) / len(cases),
            "misses@5": sorted(cid for cid, r in ranks.items() if r is None or r > 5),
            "ranks": ranks,
        }
    return report


def format_table(report: dict) -> str:
    header = f"{'mode':<8} " + " ".join(f"{f'hit@{k}':>6}" for k in KS) + f" {'MRR':>6}"
    lines = [f"{report['cases']} questions", header, "-" * len(header)]
    for mode, m in report["modes"].items():
        lines.append(f"{mode:<8} " + " ".join(f"{m[f'hit@{k}']:>6.2f}" for k in KS) + f" {m['mrr']:>6.3f}")
    for mode, m in report["modes"].items():
        if m["misses@5"]:
            lines.append(f"{mode} misses (not in top 5): {', '.join(m['misses@5'])}")
    return "\n".join(lines)
