"""Command line: worldbible ingest | search | ask | eval | serve"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .corpus import load_corpus
from .embed import FastEmbedder
from .index import Index, build_index, fingerprint
from .retrieve import MODES, Retriever


def _retriever(settings: Settings) -> Retriever:
    index = Index.load(settings.index_dir)
    current = fingerprint(load_corpus(settings.corpus_dir))
    if current != index.fingerprint:
        print("warning: the corpus changed since the index was built; run `worldbible ingest`", file=sys.stderr)
    return Retriever(index, FastEmbedder(index.model))


def cmd_ingest(settings: Settings, args) -> None:
    chunks = load_corpus(settings.corpus_dir)
    index = build_index(chunks, FastEmbedder(settings.embed_model))
    index.save(settings.index_dir)
    records = len({c.slug for c in chunks})
    print(f"indexed {len(chunks)} chunks from {records} records -> {settings.index_dir} ({index.model})")


def cmd_search(settings: Settings, args) -> None:
    for hit in _retriever(settings).search(args.query, k=args.k, mode=args.mode):
        first_line = hit.chunk.text.splitlines()[1] if "\n" in hit.chunk.text else ""
        print(f"{hit.rank}. {hit.chunk.id}  ({hit.score:.3f})\n   {first_line[:110]}")


def cmd_ask(settings: Settings, args) -> None:
    from .answer import Answerer

    retriever = _retriever(settings)
    hits = retriever.search(args.question, k=args.k, mode=args.mode)
    result = Answerer(effort=args.effort).answer(args.question, hits)
    print(result.answer)
    print(f"\nfound: {result.found}  model: {result.model}")
    for cid in result.citations:
        print(f"  cited: {cid}")
    for cid in result.rejected_citations:
        print(f"  rejected (not retrieved): {cid}")


def cmd_eval(settings: Settings, args) -> None:
    from . import evaluate

    report = evaluate.run(_retriever(settings), evaluate.load_cases(args.cases))
    print(evaluate.format_table(report))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nfull results -> {args.out}")


def cmd_serve(settings: Settings, args) -> None:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="worldbible", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="chunk and embed the corpus, save the index")

    p = sub.add_parser("search", help="show the top passages for a query")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--mode", choices=MODES, default="hybrid")

    p = sub.add_parser("ask", help="answer a question with Claude (needs ANTHROPIC_API_KEY)")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--mode", choices=MODES, default="hybrid")
    p.add_argument("--effort", choices=["low", "medium", "high"], default="medium")

    p = sub.add_parser("eval", help="score retrieval for every search mode")
    p.add_argument("cases", help="YAML file of eval questions")
    p.add_argument("--out", help="write the full JSON report here")

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    handlers = {"ingest": cmd_ingest, "search": cmd_search, "ask": cmd_ask, "eval": cmd_eval, "serve": cmd_serve}
    handlers[args.command](Settings.from_env(), args)


if __name__ == "__main__":
    main()
