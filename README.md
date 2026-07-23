# DSPy ReAct evaluation + optimization (HoVer)

Runnable port of the official DSPy **multi-hop ReAct agents** tutorial:

https://dspy.ai/tutorials/agents/

## What it does

1. Loads **HoVer** 3-hop claims (`vincentkoc/hover-parquet`).
2. Builds a **`dspy.ReAct`** agent with two tools: `search_wikipedia` and `lookup_wikipedia`.
3. Scores with the tutorial metric **`top5_recall`** via **`dspy.Evaluate`**.
4. Optionally optimizes prompts with **`dspy.MIPROv2`** (teacher LM) and re-evaluates.

| Piece | Role |
|--------|------|
| `dspy.ReAct` | Agent loop (thought → tool → observation → finish → extract titles) |
| `top5_recall` | Fraction of gold pages in `pred.titles[:5]`; perfect-only when `trace` is set (optimization) |
| `dspy.Evaluate` | Parallel baseline / post-opt scoring |
| `dspy.MIPROv2` | Jointly optimizes the two prompts inside ReAct |

## Setup

```bash
cd /home/mokt/dev/eval_research
uv sync

# OpenAI credentials (required for openai/* models)
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...
#
# Or export in the shell that starts the API:
#   export OPENAI_API_KEY=sk-...
```

The API loads `.env` from the repo root on startup. If the key is missing, `POST /api/evals` returns 400 instead of running a silent failed job.

## Search backend

The tutorial uses ColBERTv2 over **Wikipedia 2017 abstracts**:

`http://20.102.90.50:2017/wiki17_abstracts`

That host is often unreachable. This project:

- probes ColBERT first (`--backend auto`),
- falls back to the public **Wikipedia API** if ColBERT is down.

Fallback changes the retrieval corpus (not 2017 abstracts), so absolute scores may differ from the blog numbers (~8% → ~42%). The **eval + optimize loop** is the same.

Force a backend:

```bash
--backend colbert    # fail if ColBERT is down
--backend wikipedia  # always use Wikipedia API
```

## Commands

### Demo (one claim)

```bash
uv run python -m react_hover.run demo \
  --student-lm openai/gpt-5.4-mini \
  --claim "David Gregory was born in 1625."
```

### Baseline evaluation

Official tutorial uses train=100, dev=100, 16 threads. Start smaller to save cost:

```bash
uv run python -m react_hover.run evaluate \
  --student-lm openai/gpt-5.4-mini \
  --dev-size 5 \
  --train-size 5 \
  --num-threads 2 \
  --safe
# → appends artifacts/evals/<timestamp>_baseline_<model>.json
```

### Optimize (MIPROv2) + re-evaluate

Tutorial uses `auto="medium"` with GPT-4o as teacher (~30 min / a few USD at full size). Use `light` and small sets first:

```bash
uv run python -m react_hover.run optimize \
  --student-lm openai/gpt-5.4-mini \
  --teacher-lm openai/gpt-5.4 \
  --train-size 20 \
  --dev-size 10 \
  --auto light \
  --num-threads 4 \
  --safe \
  --save artifacts/optimized_react.json
```

Re-evaluate a saved program:

```bash
uv run python -m react_hover.run evaluate \
  --load artifacts/optimized_react.json \
  --dev-size 10 \
  --safe
```

## Eval history + React reviewer

Every `evaluate` / `optimize` run is **appended** to disk (never overwrites prior runs):

```text
artifacts/evals/
  <timestamp>_<kind>_<model>.json
```

Each file stores score, config (LM, backend, sizes, …), and per-example:

- claims / gold titles / pred titles / scores
- ReAct trajectory + extract reasoning (when present)
- **`llm_calls`**: full prompt messages and assistant outputs for every LM call on that example (for manual investigation)

Evals run **sequentially** so each example’s LM history can be attributed correctly.

### Review UI (React + FastAPI)

Terminal 1 — API (reads `artifacts/evals/`):

```bash
# Optional: LOG_LEVEL=DEBUG for request-level noise
uv run uvicorn react_hover.api:app --reload --port 8000
```

Eval progress is logged with **loguru** on stderr (`job.queued`, `eval.scoring_start`, `job.succeeded`, …).

Terminal 2 — React app (Vite proxies `/api` → `:8000`):

```bash
cd frontend && npm install && npm run dev
```

Open http://localhost:5173

- **Evaluate / Optimize** tabs: baseline eval or MIPROv2 (student + teacher, `auto` budget)
- Sidebar history of all persisted runs (auto-refreshes when a UI job finishes)
- Optimize jobs write linked `optimize_baseline` + `optimize_after` runs (with Δ and program path)
- Aggregate metrics + config
- Filter examples (all / perfect / partial / failed)
- Drill into gold vs predicted titles, reasoning, trajectory

API endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health + evals dir |
| GET | `/api/models` | Model dropdown options |
| GET | `/api/runs` | List run summaries (newest first) |
| GET | `/api/runs/{id}` | Full run with per-example results |
| POST | `/api/evals` | Start background eval (`student_lm`, sizes, …) |
| POST | `/api/optimize` | Start background MIPROv2 (student + teacher, `auto`, …) |
| GET | `/api/jobs/{id}` | Poll job status |

Legacy `artifacts/eval_results.json` is imported into history on API startup if present.

## Project layout

```
src/react_hover/
  data.py         # HoVer load + split
  tools.py        # ColBERT / Wikipedia tools
  metric.py       # top5_recall
  agent.py        # dspy.ReAct builder
  eval_runner.py  # shared baseline eval
  opt_runner.py   # shared MIPROv2 optimize
  history.py      # disk-backed eval run store
  jobs.py         # background eval/optimize jobs
  run.py          # CLI: demo | evaluate | optimize
  api.py          # FastAPI for eval history + jobs
frontend/         # React (Vite + TypeScript) review UI
artifacts/evals/  # persisted eval history
```

## Notes

- Metric returns continuous recall at eval time; returns `bool` (perfect only) when optimizers pass `trace` — same as the tutorial.
- `--safe` wraps the agent so a single bad trajectory does not abort the whole eval.
- Full official sizes: `--train-size 100 --dev-size 100 --auto medium --num-threads 16`.
