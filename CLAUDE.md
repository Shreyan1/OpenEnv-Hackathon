# CLAUDE.md

## Project Overview

**Memory Management Agent with RL + OpenEnv** — a reinforcement learning system that trains an LLM-based policy to manage memory optimally during multi-turn conversations.

The core question: *Given a stream of user interactions, how should an agent allocate limited memory resources to maximize future usefulness?*

## Repository Structure

```
src/memory_management_agent/    # Main package
├── schemas.py          # All dataclasses and enums (MemoryItem, Action, Observation, etc.)
├── environment.py      # MemoryManagementEnv (OpenEnv-compatible)
├── episode.py          # SyntheticEpisodeGenerator
├── tasks.py            # 3 TaskDefinitions (easy/medium/hard) + TaskEpisodeGenerator
├── memory_store.py     # MemoryStore (budget-constrained, LRU eviction)
├── agents.py           # 6 baseline agents
├── grader.py           # Grader + RewardComposer (output clamped to [0.0, 1.0])
├── evaluation.py       # run_episode(), evaluate_split(), BenchmarkReport
├── training.py         # Prompt building, rollout collection, TRL scaffold
├── analysis.py         # Failure analysis, memory evolution tracking
├── review.py           # Report rendering
└── utils.py            # Shared utilities

tests/
└── test_core.py        # Full test suite (unittest)

server.py                                 # FastAPI server (OpenEnv HTTP API)
run_baseline.py                           # Headless baseline inference script
openenv.yaml                              # OpenEnv spec manifest
Dockerfile                                # Container build (port 7860)
requirements.txt                          # fastapi, uvicorn, pydantic

memory-management-agent.md               # Design document
memory-management-agent-execution-plan.md  # Phase-by-phase implementation plan
```

## Key Architecture Concepts

### Data Flow

```
SyntheticEpisodeGenerator
    → MemoryManagementEnv (reset/step loop)
        → Agent (act) → Action
        → MemoryStore (add/query/update/delete)
        → Grader (score_episode) → GraderMetrics
        → RewardComposer (compose) → scalar reward
    → TRL Trainer (GRPO/PPO)
```

### Action Space (7 types)

| Action | Description |
|--------|-------------|
| `STORE(text)` | Save raw memory item |
| `STORE_SUMMARY(text)` | Save compressed version |
| `IGNORE` | Skip this turn |
| `RETRIEVE(ids)` | Fetch specific memories |
| `UPDATE(id, text)` | Modify existing memory |
| `DELETE(id)` | Remove memory |
| `ANSWER(text)` | Final response (terminal action) |

### Memory Types

- `PREFERENCE` — user preferences (e.g., "prefers ClickHouse")
- `CONSTRAINT` — hard constraints (e.g., "keep answers under 5 lines")
- `PROJECT_INFO` — facts about current project context

### Reward Formula

```
R = 0.45 * success
  + 0.20 * precision
  + 0.15 * recall
  + 0.10 * compactness
  + 0.10 * freshness
  - penalties
```

Dense (step-level) rewards fire immediately; delayed rewards fire at episode end.

## Development Setup

```bash
uv venv .venv
uv pip install -r requirements.txt --python .venv
```

## Development Commands

### Run Tests

```bash
.venv/bin/python -m unittest tests/test_core.py -v
```

### Start the HTTP Server (OpenEnv API)

```bash
.venv/bin/uvicorn server:app --host 0.0.0.0 --port 7860
```

Endpoints: `GET /health`, `GET /tasks`, `POST /reset`, `POST /step`, `POST /grader`, `GET /baseline`

### Run Baseline Inference Script

```bash
.venv/bin/python run_baseline.py
.venv/bin/python run_baseline.py --task easy_preference_recall
.venv/bin/python run_baseline.py --json    # machine-readable output
```

### Quick Smoke Test

```python
from src.memory_management_agent import (
    MemoryManagementEnv, RuleBasedMemoryAgent, run_episode
)
env = MemoryManagementEnv(memory_budget=200)
agent = RuleBasedMemoryAgent()
result = run_episode(agent, env, seed=42)
print(result.reward)
```

### Collect and Analyze Rollouts

```python
from src.memory_management_agent import collect_rollouts, analyze_rollouts, render_full_review
rollouts = collect_rollouts(agent, env, seeds=tuple(range(5)))
analysis = analyze_rollouts(rollouts)
render_full_review(analysis)
```

## Important Schemas

All schemas are in `schemas.py`. Key types:

- **`Observation`** — what the agent sees each step: `current_message`, `conversation_window`, `memory_bank`, `budget_remaining`, `step_number`
- **`Action`** — agent output: `type: ActionType`, optional `text`, optional `ids`
- **`MemoryItem`** — stored memory: `id`, `text`, `type`, `created_at`, `last_used`, `token_length`, `utility_score`
- **`GraderMetrics`** — 14 scoring dimensions including `success`, `precision`, `recall`, `compactness`, `freshness`, `non_interference`
- **`EpisodeResult`** — final outcome: `episode`, `answer`, `metrics`, `reward`, `trace`

## Baseline Agents (for comparison)

| Agent | Strategy |
|-------|----------|
| `NoMemoryAgent` | Always IGNORE |
| `StoreEverythingAgent` | Always STORE |
| `PreferenceOnlyAgent` | Store only PREFERENCE/CONSTRAINT/PROJECT_INFO |
| `KeywordRetrievalAgent` | Keyword-based retrieval heuristics |
| `EmbeddingRetrievalAgent` | Semantic embedding retrieval |
| `RuleBasedMemoryAgent` | Most sophisticated rule-based baseline |

The RL-trained policy must outperform all baselines on hidden eval seeds.

## Evaluation Design

- **Visible seeds** (1–4999): training + validation
- **Hidden seeds** (5000+): holdout, used only for final evaluation
- `generalization_gap = visible_reward - hidden_reward` — detects overfitting/reward hacking
- `evaluate_split()` returns a `BenchmarkReport` with both splits

## Environment Configuration

```python
MemoryManagementEnv(
    memory_budget=200,   # Token budget for memory store
    max_turns=8          # Max turns per episode
)
```

## Known Failure Modes

| Mode | Description | Penalty |
|------|-------------|---------|
| Memory hoarding | Stores everything, wastes budget | `memory_bloat_penalty` |
| Over-retrieval | Retrieves irrelevant memories | `-0.12` per irrelevant retrieval |
| Under-retrieval | Fails to retrieve key info at answer time | `-0.5` delayed |
| Stale memory | Stores correction but keeps old fact | `contradiction_penalty` |
| Reward hacking | Learns shortcuts that don't generalize | Caught by hidden eval |

## Three Tasks (OpenEnv)

| Task ID | Difficulty | Budget | Max Turns | Structure |
|---------|-----------|--------|-----------|-----------|
| `easy_preference_recall` | Easy | 300 tok | 5 | 1 preference, 1 distractor, final_query |
| `medium_preference_constraint_correction` | Medium | 200 tok | 7 | preference + constraint + correction + recall_check + final_query |
| `hard_full_memory_management` | Hard | 120 tok | 8 | preference + constraint + **2 corrections** + project_info + recall_check + final_query |

**Baseline score spread** (10 seeds 42–51):

| Agent | Easy | Medium | Hard |
|-------|------|--------|------|
| `no_memory` | 0.100 | 0.000 | 0.000 |
| `store_everything` | 0.541 | 0.356 | 0.286 |
| `preference_only` | 0.594 | 0.401 | 0.334 |
| `rule_based` | 0.594 | 0.676 | 0.609 |

Difficulty gradient for weak agents (store_everything): easy > medium > hard ✓.
Hard task is hardest structurally: 2 corrections require UPDATE/DELETE + tight budget penalises hoarding.

## Submission Checklist

- ✅ `openenv.yaml` — valid spec manifest
- ✅ `Dockerfile` — python:3.13-slim, port 7860, HEALTHCHECK
- ✅ `server.py` — FastAPI with `/health`, `/tasks`, `/reset`, `/step`, `/grader`, `/baseline`
- ✅ `run_baseline.py` — headless, all 3 tasks, exits 0
- ✅ 3 tasks with graded difficulty (easy/medium/hard)
- ✅ Grader returns [0.0, 1.0] (clamped in `RewardComposer.compose()`)
- ✅ Score variance confirmed (no_memory 0.0–0.1 vs rule_based 0.594–0.676)
- ✅ Cheat answers penalised (bare keyword now scores ~0.17 not 1.0)
- ✅ Answer keys hidden from agent observations (`episode_metadata` exposes only `turn_count`)
- ✅ Episode diversity: 100/100 unique episodes in seeds 1–100
- ✅ Precision metric is live (recall_check turns trigger RETRIEVE in rule_based)
- ✅ Hard task always has 2 corrections + project_info (every seed)
- ✅ `.gitignore` excludes `.venv/`, `__pycache__/`, artifacts
- 🔄 HF Space deployment (pending — target April 5)

## Phase Status

- ✅ Phase 0–5: Schemas, environment, memory store, grading, baselines, evaluation, training scaffold, analysis
- ✅ Phase 6a: OpenEnv HTTP server, 3 tasks, Dockerfile, baseline script, openenv.yaml
- 🔄 Phase 6b: TRL training loop connection (scaffold exists in `training.py`, needs model integration)
- Planned: HF Space deployment, Vector DB backend, memory decay, LLM judge grading

## Issue Tracking

This project uses **beads (bd)** for issue tracking. See `AGENTS.md` for workflow.
