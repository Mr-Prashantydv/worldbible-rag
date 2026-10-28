"""Answer a question from retrieved passages with Claude, and check its citations.

The model returns a typed object (answer, cited passage ids, found). Code then
throws out any citation that was not one of the passages it was given, and
treats an answer with no valid citation as "not found". The model never gets
the last word on whether an answer is grounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anthropic
from pydantic import BaseModel, Field

from .retrieve import Hit

DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM = """You answer questions about a story world, using only the passages from its world bible that the user provides.

- Answer only from the passages. Never add details from general knowledge, and never guess.
- Put the id of every passage you used in `citations`.
- If the passages do not contain the answer, set `found` to false, say in one sentence what the bible does not cover, and leave `citations` empty.
- Director rules ("never", "do not", "always") are facts too. Use them when they answer the question.
- Keep the answer to one to three sentences."""


class AnswerUnavailable(RuntimeError):
    """Claude could not be reached: no credentials, network failure or an API error."""


class GroundedAnswer(BaseModel):
    answer: str = Field(description="The answer, in one to three sentences.")
    citations: list[str] = Field(description="Ids of the passages the answer is based on.")
    found: bool = Field(description="False when the passages do not contain the answer.")


@dataclass
class AnswerResult:
    answer: str
    citations: list[str]
    found: bool
    model: str
    rejected_citations: list[str] = field(default_factory=list)
    refused: bool = False


def build_prompt(question: str, hits: list[Hit]) -> str:
    passages = "\n\n".join(f'<passage id="{h.chunk.id}">\n{h.chunk.text}\n</passage>' for h in hits)
    return f"<passages>\n{passages}\n</passages>\n\nQuestion: {question}"


def check_citations(parsed: GroundedAnswer, hits: list[Hit]) -> tuple[list[str], list[str]]:
    """Split the model's citations into ones it was given and ones it was not."""
    allowed = {h.chunk.id for h in hits}
    unique = list(dict.fromkeys(parsed.citations))
    return [c for c in unique if c in allowed], [c for c in unique if c not in allowed]


class Answerer:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str = DEFAULT_MODEL, effort: str = "medium"):
        if client is None:
            # Anthropic() reads ANTHROPIC_API_KEY (or an `ant auth login` profile).
            try:
                client = anthropic.Anthropic()
            except anthropic.CredentialsError as exc:
                raise AnswerUnavailable(f"Anthropic credentials could not be loaded ({exc}); set ANTHROPIC_API_KEY.") from exc
        self.client = client
        self.model = model
        self.effort = effort

    def answer(self, question: str, hits: list[Hit]) -> AnswerResult:
        if not hits:
            return AnswerResult("The world bible has nothing that matches this question.", [], False, self.model)

        try:
            response = self.client.beta.messages.parse(
                model=self.model,
                max_tokens=16000,
                # If a safety classifier declines, the API re-runs the request on a fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": self.effort},
                system=SYSTEM,
                messages=[{"role": "user", "content": build_prompt(question, hits)}],
                output_format=GroundedAnswer,
            )
        except anthropic.APIStatusError as exc:
            raise AnswerUnavailable(f"Claude API returned {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise AnswerUnavailable("Could not reach the Claude API.") from exc
        except anthropic.CredentialsError as exc:
            raise AnswerUnavailable(f"Anthropic credentials could not be loaded ({exc}); set ANTHROPIC_API_KEY.") from exc
        except TypeError as exc:
            # The SDK raises TypeError at request time when no credentials are configured.
            if "authentication" not in str(exc):
                raise
            raise AnswerUnavailable("No Anthropic credentials: set ANTHROPIC_API_KEY.") from exc

        if response.stop_reason == "refusal" or response.parsed_output is None:
            return AnswerResult("The model declined to answer this question.", [], False, response.model, refused=True)

        parsed = response.parsed_output
        valid, rejected = check_citations(parsed, hits)
        found = parsed.found and bool(valid)
        return AnswerResult(
            answer=parsed.answer,
            citations=valid if found else [],
            found=found,
            model=response.model,
            rejected_citations=rejected,
        )
