"""Load a folder of markdown records and split them into retrievable chunks.

Each record is one thing in the world (a character, a place, a prop) with YAML
front matter and `## ` sections. A chunk is one section, prefixed with the
record's name and the section heading, so a chunk still makes sense when it is
read on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
WIKILINK = re.compile(r"\[\[([a-z0-9-]+)\]\]")
ONE_LINE = re.compile(r"\*\*One line:\*\*\s*(.+)")
# Sections longer than this are split again on paragraph breaks, so one long
# section does not crowd every other result out of the context window.
MAX_CHUNK_WORDS = 220


@dataclass
class Chunk:
    id: str  # "<slug>#<section-slug>[-<n>]", stable across rebuilds
    slug: str
    title: str
    kind: str
    section: str
    text: str
    links: list[str] = field(default_factory=list)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def split_long(body: str, limit: int = MAX_CHUNK_WORDS) -> list[str]:
    """Split on blank lines, packing paragraphs into pieces of at most `limit` words."""
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for para in re.split(r"\n\s*\n", body.strip()):
        words = len(para.split())
        if current and count + words > limit:
            pieces.append("\n\n".join(current))
            current, count = [], 0
        current.append(para)
        count += words
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def parse_record(path: Path, root: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    meta: dict = {}
    match = FRONT_MATTER.match(raw)
    if match:
        meta = yaml.safe_load(match.group(1)) or {}
        raw = raw[match.end():]

    slug = path.stem
    title = str(meta.get("name") or slug.replace("-", " ").title())
    kind = str(meta.get("kind") or path.parent.relative_to(root).as_posix() or "note")
    aliases = meta.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [a.strip() for a in aliases.split(",") if a.strip()]

    # Everything before the first "## " heading is the record's intro.
    sections: list[tuple[str, str]] = []
    heading = "Overview"
    lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith("## "):
            sections.append((heading, "\n".join(lines)))
            heading, lines = line[3:].strip(), []
        elif line.startswith("# "):
            continue  # the record title; already in `title`
        else:
            lines.append(line)
    sections.append((heading, "\n".join(lines)))

    # A section like "Director rules" rarely names whose rules they are. Repeating
    # the record's one-line summary in every section gives each chunk that context
    # (measured in evals/: it lifted MRR for every search mode).
    one_line = ONE_LINE.search(sections[0][1])
    context = f"About {title}: {WIKILINK.sub(lambda m: m.group(1).replace('-', ' '), one_line.group(1).strip())}" if one_line else ""

    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    for heading, body in sections:
        if not body.strip():
            continue
        header = f"{title} ({kind}) > {heading}"
        if heading == "Overview" and aliases:
            header += f"\nAlso called: {', '.join(map(str, aliases))}"
        elif heading != "Overview" and context:
            header += f"\n{context}"
        pieces = split_long(body)
        for n, piece in enumerate(pieces):
            base = slugify(heading)
            seen[base] = seen.get(base, 0) + 1
            suffix = "" if (len(pieces) == 1 and seen[base] == 1) else f"-{seen[base]}"
            chunks.append(
                Chunk(
                    id=f"{slug}#{base}{suffix}",
                    slug=slug,
                    title=title,
                    kind=kind,
                    section=heading,
                    text=f"{header}\n{piece.strip()}",
                    links=sorted(set(WIKILINK.findall(piece))),
                )
            )
    return chunks


def load_corpus(root: Path) -> list[Chunk]:
    """Every chunk from every markdown record under `root`, skipping INDEX/README files."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"corpus folder not found: {root}")
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root)
        if path.name.upper() in {"INDEX.MD", "README.MD"} or any(p.startswith(".") for p in rel.parts):
            continue
        chunks.extend(parse_record(path, root))
    if not chunks:
        raise ValueError(f"no markdown records found under {root}")
    return chunks
