# worldbible-rag

Ask questions of a story world's "bible": the markdown records that fix how every character, place and prop looks and behaves. The service finds the right sections, asks Claude to answer from those sections only, and checks the citations in code before returning them.

I built it for the world bible of *Jevika & Joy's World*, a children's animated series at JJW by Botfit (32 records: characters, locations, props, style rules). That bible is company material, so it isn't in this repo. A small original sample world ships instead, and everything runs against it out of the box.

## What it does

```
question ──▶ retrieve top 5 sections ──▶ Claude answers with cited section ids ──▶ code checks every citation
             (vector + BM25, fused)       (structured output: answer, citations, found)   (drops any id it was not given)
```

- **Retrieval**: three modes. `vector` uses local embeddings (`BAAI/bge-small-en-v1.5` through fastembed: CPU, no API key). `keyword` uses BM25. `hybrid` fuses both rankings with reciprocal rank fusion.
- **Grounded answers**: Claude (`claude-opus-5-5`) returns a typed `GroundedAnswer` through structured outputs. Any citation that wasn't one of the retrieved passages is rejected, and an answer left with no valid citation is reported as *not found*. The model never decides on its own whether an answer is grounded.
- **Retrieval eval**: `worldbible eval` scores every mode with hit@1/3/5 and MRR on a question file, and lists the misses.
- **API**: FastAPI with `GET /health`, `GET /search`, `POST /ask`.

## Run it

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"      # macOS/Linux: .venv/bin/pip
.venv/Scripts/worldbible ingest            # chunk + embed sample_corpus/ into .index/
.venv/Scripts/worldbible search "Can the red kite be destroyed in an episode?"
.venv/Scripts/worldbible eval evals/sample.yaml
.venv/Scripts/python -m pytest
```

Answering needs an Anthropic API key:

```bash
export ANTHROPIC_API_KEY=...
.venv/Scripts/worldbible ask "Does Asha have a phone?"
.venv/Scripts/worldbible serve             # http://127.0.0.1:8000/docs
```

Without a key, `/ask` returns `503` with the reason. `/search` still works.

Use your own corpus with `WORLDBIBLE_CORPUS=/path/to/records` (and `WORLDBIBLE_INDEX` for the index folder). Records are markdown files with optional YAML front matter (`name`, `kind`, `aliases`) and `## ` sections.

## Decisions, and the measurements behind them

**Chunk by section, and give every chunk its record's context.** Each `## ` section becomes one chunk, headed with the record name, kind and section title. The first version stopped there. The eval showed that short sections such as "Director rules" or "Family and relationships" rarely say *whose* rules they are, so a question about "the grandfather" or "the robot" couldn't reach them. Repeating the record's one-line summary in every chunk raised MRR on the 26-question world-bible eval for every mode:

| mode | MRR before | MRR after |
|---|---|---|
| vector | 0.809 | **0.849** |
| keyword | 0.636 | 0.693 |
| hybrid | 0.690 | 0.799 |

**Hybrid is the default, though vector ranks the first result better.** Final scores on the world-bible eval:

| mode | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| vector | 0.77 | 0.92 | 0.96 | 0.849 |
| keyword | 0.62 | 0.69 | 0.85 | 0.693 |
| hybrid | 0.69 | 0.88 | 0.96 | 0.799 |

And on the public sample eval (15 questions, `evals/results/sample.json`):

| mode | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| vector | 0.67 | 0.87 | 0.93 | 0.783 |
| keyword | 0.60 | 0.80 | 0.93 | 0.727 |
| hybrid | 0.60 | 1.00 | 1.00 | 0.789 |

`/ask` gives Claude the top 5 passages, so hit@5 decides whether the answer *can* be right. Hybrid matches vector on hit@5 on the bible and beats it on the sample, where exact details ("brass tag", "six ribbons") matter. Vector puts the right section first more often, so it's the better choice if you only show one result. With 26 and 15 questions these are small samples: a one-question difference moves hit@5 by about 4 to 7 points.

**Local embeddings, not an embeddings API.** Building the index costs nothing, needs no key and runs offline, and a corpus this size (191 chunks) embeds in seconds on a CPU. The index records which model built it and refuses to load with a different one.

**Citations are checked in code, not trusted.** Structured outputs guarantee the *shape* of the answer, not that the citations are real. `check_citations` keeps only ids that were actually retrieved. The tests cover an answer citing an invented id, and one whose only citation is invented, which counts as not found.

**No prompt caching.** The system prompt is far below the minimum cacheable length, and the passages change with every question, so there's no stable prefix worth caching.

**Refusals are handled before the output is read.** The request opts into server-side fallbacks (`fallbacks: "default"`), and `stop_reason == "refusal"` is checked before `parsed_output`.

**An eval file can't silently lie.** `worldbible eval` refuses to run if an expected section id doesn't exist in the corpus, so a renamed heading shows up as an error, not as a "miss".

## What doesn't work yet

- The answer step hasn't been run against the live API in this repo's history: the development machine had no API key. It's tested with a fake client, which checks the request shape, citation filtering and refusal handling. **There's no answer-quality eval yet** (faithfulness, correct "not found" on unanswerable questions). That's the next thing to build.
- Retrieval doesn't follow `[[links]]` between records. A question that needs two records ("what does Joy's robot's tool look like") depends on both being retrieved separately.
- The eval sets are small and were written for this project. A question set from the people who actually use the bible would be a better test.
- One index at a time, held in memory: fine for hundreds of chunks, not for a large corpus.

## Layout

```
src/worldbible_rag/
  corpus.py     markdown records -> section chunks
  embed.py      local embedding model
  index.py      build / save / load the index
  retrieve.py   vector, BM25 and hybrid (RRF) search
  answer.py     Claude call, structured output, citation check
  evaluate.py   hit@k / MRR retrieval eval
  api.py        FastAPI app
  cli.py        worldbible ingest | search | ask | eval | serve
sample_corpus/  six records of an original sample world
evals/          question sets and results
tests/          31 tests (pytest)
```

Built by directing Claude Code (Anthropic's coding agent). The commit history shows each step, including the eval result that changed the chunking.
