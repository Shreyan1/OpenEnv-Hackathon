# CLAUDE.md

## Project Overview

This repository implements a reinforcement learning environment for memory management in long-running LLM conversations.

The core question is:

> Given a stream of realistic user messages and a fixed memory budget, what should an agent store, retrieve, update, refresh, or ignore so that its final answer is accurate, current, compact, and format-compliant?

## Repository Structure

```text
src/memory_management_agent/
├── schemas.py        # dataclasses and enums
├── environment.py    # MemoryManagementEnv reset/step loop
├── episode.py        # realistic conversation template pools + default generator
├── tasks.py          # easy / medium / hard task generators
├── memory_store.py   # budgeted memory with decay support
├── agents.py         # baseline heuristic agents
├── grader.py         # deterministic grader + reward composer
├── evaluation.py     # run_episode(), evaluate_split()
├── training.py       # prompt building and rollout collection
├── analysis.py       # rollout analysis helpers
├── review.py         # text report rendering
└── utils.py          # tokenization and similarity helpers

tests/test_core.py    # unittest suite
server.py             # FastAPI OpenEnv server
run_baseline.py       # baseline evaluation script
openenv.yaml          # OpenEnv manifest
README.md             # HF Space / project pitch
```

## Key Design Decisions

### Hidden turn kinds

The easy task exposes `current_turn_kind` as scaffolding.

The medium and hard tasks do not. They return `"unknown"` for the current turn, and recent conversation turns are also masked. This prevents schema-cheating and forces the agent to infer intent from natural language.

### Realistic templates

Episode templates are phrased like Slack or engineering-thread messages rather than synthetic declarations. Each preference, constraint, and project fact has multiple phrasings to improve diversity without changing the underlying task structure.

### Confabulation turns

The hard task includes misleading turns that mention plausible technologies in hypotheticals or third-party attributions. Storing them is incorrect. This tests attribution reasoning rather than naive keyword matching.

### Constraint adherence

The grader deterministically checks whether the final answer actually follows measurable formatting constraints such as bullet points, numbered lists, valid JSON, concise output, code examples, type annotations, and snake_case naming.

### Memory decay

Memories lose utility if they are not refreshed for several turns. This makes retrieval strategically useful beyond the final answer and pressures agents to maintain fresh memory instead of storing once and forgetting.

## Action Space

| Action | Meaning |
| --- | --- |
| `STORE(text)` | Save a raw memory item |
| `STORE_SUMMARY(text)` | Save a compressed memory |
| `IGNORE` | Skip the turn |
| `RETRIEVE(ids)` | Fetch specific memories |
| `UPDATE(id, text)` | Modify an existing memory |
| `DELETE(id)` | Remove a memory |
| `ANSWER(text)` | Finish the episode with a final answer |

## Reward Formula

```text
R = 0.40 * success
  + 0.18 * precision
  + 0.12 * recall
  + 0.10 * constraint_adherence
  + 0.05 * compactness
  + 0.05 * freshness
  + 0.10 * non_interference
  - penalties
```

Penalties include contradiction handling failures and memory bloat.

## Tasks

| Task | Difficulty | Budget | Max turns | Notes |
| --- | --- | --- | --- | --- |
| `easy_preference_recall` | Easy | 300 | 5 | turn kind exposed |
| `medium_preference_constraint_correction` | Medium | 200 | 7 | hidden turn kind, one correction |
| `hard_full_memory_management` | Hard | 120 | 8 | hidden turn kind, confabulation, project info, two corrections, decay |

## Important Schemas

- `Observation`: current user message, current turn kind, masked recent conversation, memory bank, remaining budget, step number
- `Action`: structured action chosen by the agent
- `MemoryItem`: stored memory with utility score and timestamps
- `GraderMetrics`: success, precision, recall, constraint adherence, compactness, freshness, non-interference, penalties, and memory stats
- `EpisodeResult`: final answer, metrics, reward, and action trace

## Development Commands

```bash
.venv/bin/python -m unittest tests/test_core.py -v
.venv/bin/python run_baseline.py
.venv/bin/uvicorn server:app --host 0.0.0.0 --port 7860
```

## Example Usage

```python
from src.memory_management_agent import MemoryManagementEnv, RuleBasedMemoryAgent, run_episode

env = MemoryManagementEnv(memory_budget=200)
result = run_episode(RuleBasedMemoryAgent(), env, seed=42)
print(result.reward)
print(result.metrics.constraint_adherence)
```

Task-aware environment:

```python
from src.memory_management_agent import TASK_HARD, generator_for_task
from src.memory_management_agent.environment import MemoryManagementEnv

env = MemoryManagementEnv(
    generator=generator_for_task(TASK_HARD),
    memory_budget=TASK_HARD.memory_budget,
    max_turns=TASK_HARD.max_turns,
    expose_turn_kind=TASK_HARD.expose_turn_kind,
    decay_rate=TASK_HARD.decay_rate,
)
```
