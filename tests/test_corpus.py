from pathlib import Path

from worldbible_rag.corpus import load_corpus, parse_record, split_long

SAMPLE = Path(__file__).resolve().parents[1] / "sample_corpus"


def test_every_record_becomes_section_chunks():
    chunks = load_corpus(SAMPLE)
    slugs = {c.slug for c in chunks}
    assert slugs == {"asha", "bolt", "grandpa-ravi", "windy-hill", "ashas-workshop", "red-kite"}
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"


def test_chunk_carries_title_kind_and_section_header():
    chunks = {c.id: c for c in load_corpus(SAMPLE)}
    chunk = chunks["bolt#identity-locks-never-change"]
    assert chunk.title == "Bolt"
    assert chunk.kind == "character"
    assert chunk.text.startswith("Bolt (character) > Identity locks (never change)")
    assert "brass tag" in chunk.text


def test_aliases_are_searchable_in_the_overview():
    chunks = {c.id: c for c in load_corpus(SAMPLE)}
    assert "Also called: Nanu, Ravi" in chunks["grandpa-ravi#overview"].text
    # aliases given as a comma string instead of a list
    assert "Also called: the shed" in chunks["ashas-workshop#overview"].text


def test_wikilinks_are_collected():
    chunks = {c.id: c for c in load_corpus(SAMPLE)}
    assert chunks["asha#places-and-props"].links == ["ashas-workshop", "red-kite", "windy-hill"]


def test_long_sections_split_on_paragraphs():
    body = "\n\n".join(" ".join(["word"] * 100) for _ in range(5))
    pieces = split_long(body, limit=220)
    assert [len(p.split()) for p in pieces] == [200, 200, 100]


def test_repeated_long_section_gets_numbered_ids(tmp_path):
    long_para = " ".join(["word"] * 150)
    (tmp_path / "x.md").write_text(f"# X\n\n## Notes\n{long_para}\n\n{long_para}\n", encoding="utf-8")
    ids = [c.id for c in parse_record(tmp_path / "x.md", tmp_path)]
    assert ids == ["x#notes-1", "x#notes-2"]
