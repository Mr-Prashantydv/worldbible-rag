from types import SimpleNamespace

import anthropic
import httpx2 as httpx  # the anthropic SDK is built on httpx2, not httpx
import pytest
from fastapi.testclient import TestClient

from worldbible_rag.answer import Answerer, AnswerUnavailable, GroundedAnswer, build_prompt
from worldbible_rag.api import create_app


class FakeMessages:
    """Stands in for client.beta.messages and records the request it was sent."""

    def __init__(self, parsed: GroundedAnswer | None, stop_reason: str = "end_turn"):
        self.parsed = parsed
        self.stop_reason = stop_reason
        self.request: dict | None = None

    def parse(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(parsed_output=self.parsed, stop_reason=self.stop_reason, model=kwargs["model"])


def fake_answerer(parsed, stop_reason="end_turn"):
    messages = FakeMessages(parsed, stop_reason)
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return Answerer(client=client), messages


def test_prompt_wraps_each_passage_with_its_id(retriever):
    hits = retriever.search("Bolt collar", k=2)
    prompt = build_prompt("What collar does Bolt wear?", hits)
    assert f'<passage id="{hits[0].chunk.id}">' in prompt
    assert prompt.endswith("Question: What collar does Bolt wear?")


def test_request_uses_structured_output_and_fallbacks(retriever):
    answerer, messages = fake_answerer(GroundedAnswer(answer="x", citations=[], found=False))
    answerer.answer("q", retriever.search("Bolt", k=1))
    req = messages.request
    assert req["model"] == "claude-opus-5-5"
    assert req["output_format"] is GroundedAnswer
    assert req["fallbacks"] == "default"
    assert req["betas"] == ["server-side-fallback-2026-07-01"]
    assert req["output_config"] == {"effort": "medium"}


def test_citations_outside_the_retrieved_passages_are_rejected(retriever):
    hits = retriever.search("brass tag bun collar", k=3)
    real = hits[0].chunk.id
    answerer, _ = fake_answerer(GroundedAnswer(answer="A blue collar.", citations=[real, "made-up#id", real], found=True))
    result = answerer.answer("What collar does Bolt wear?", hits)
    assert result.found is True
    assert result.citations == [real]
    assert result.rejected_citations == ["made-up#id"]


def test_answer_with_only_invented_citations_counts_as_not_found(retriever):
    hits = retriever.search("Bolt", k=2)
    answerer, _ = fake_answerer(GroundedAnswer(answer="Bolt is a wolf.", citations=["made-up#id"], found=True))
    result = answerer.answer("What is Bolt?", hits)
    assert result.found is False
    assert result.citations == []


def test_refusal_is_reported_not_read(retriever):
    answerer, _ = fake_answerer(None, stop_reason="refusal")
    result = answerer.answer("q", retriever.search("Bolt", k=1))
    assert result.refused is True and result.found is False


def test_no_passages_skips_the_model_call():
    answerer, messages = fake_answerer(GroundedAnswer(answer="x", citations=[], found=True))
    result = answerer.answer("anything", [])
    assert result.found is False
    assert messages.request is None


@pytest.fixture
def client(retriever):
    def make_answerer():
        hit_id = retriever.search("six blue ribbons", k=1)[0].chunk.id
        answerer, _ = fake_answerer(GroundedAnswer(answer="Exactly six.", citations=[hit_id], found=True))
        return answerer

    with TestClient(create_app(lambda: retriever, make_answerer)) as c:
        yield c


def test_health_reports_the_index(client, index):
    body = client.get("/health").json()
    assert body["chunks"] == len(index.chunks)
    assert body["fingerprint"] == index.fingerprint


def test_search_endpoint(client):
    rows = client.get("/search", params={"q": "brass tag bun collar", "k": 2, "mode": "hybrid"}).json()
    assert rows[0]["id"] == "bolt#identity-locks-never-change"
    assert [r["rank"] for r in rows] == [1, 2]


def test_search_rejects_an_unknown_mode(client):
    assert client.get("/search", params={"q": "x", "mode": "magic"}).status_code == 422


def test_ask_endpoint_returns_cited_passages(client):
    body = client.post("/ask", json={"question": "How many ribbons does the kite tail have?"}).json()
    assert body["found"] is True
    assert body["answer"] == "Exactly six."
    assert body["citations"][0]["id"].startswith("red-kite#")


def test_ask_without_credentials_returns_503(retriever, monkeypatch, tmp_path):
    # A real SDK client with no credentials (and an empty config dir, so a profile
    # on the developer's machine is never used): it fails when the request is sent.
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))
    with TestClient(create_app(lambda: retriever, Answerer)) as c:
        response = c.post("/ask", json={"question": "Who is Bolt?"})
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_api_errors_become_answer_unavailable(retriever):
    class Failing:
        def parse(self, **kwargs):
            raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))

    answerer = Answerer(client=SimpleNamespace(beta=SimpleNamespace(messages=Failing())))
    with pytest.raises(AnswerUnavailable, match="Could not reach"):
        answerer.answer("q", retriever.search("Bolt", k=1))
