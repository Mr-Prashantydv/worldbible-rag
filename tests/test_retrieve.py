import pytest

from worldbible_rag.index import Index
from worldbible_rag.retrieve import MODES, Retriever, tokenize


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What colour is Bolt's collar?") == ["colour", "bolt", "s", "collar"]


@pytest.mark.parametrize("mode", MODES)
def test_each_mode_finds_the_exact_fact(retriever, mode):
    hits = retriever.search("brass tag shaped like a bun collar", k=3, mode=mode)
    assert hits[0].chunk.id == "bolt#identity-locks-never-change"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_keyword_mode_returns_nothing_for_unknown_words(retriever):
    assert retriever.search("xylophone zeppelin", mode="keyword") == []


def test_hybrid_fuses_both_rankings(retriever):
    vector_ids = [h.chunk.id for h in retriever.search("six blue ribbons", k=50, mode="vector")]
    hybrid = retriever.search("six blue ribbons", k=5, mode="hybrid")
    assert hybrid[0].chunk.id in vector_ids[:5]
    assert all(h.score > 0 for h in hybrid)


def test_blank_query_returns_nothing(retriever):
    assert retriever.search("   ") == []


def test_index_round_trips_through_disk(index, tmp_path):
    index.save(tmp_path)
    loaded = Index.load(tmp_path)
    assert [c.id for c in loaded.chunks] == [c.id for c in index.chunks]
    assert loaded.vectors.shape == index.vectors.shape
    assert loaded.fingerprint == index.fingerprint


def test_retriever_refuses_a_mismatched_embedder(index):
    class Other:
        name = "some-other-model"

    with pytest.raises(ValueError, match="built with"):
        Retriever(index, Other())
