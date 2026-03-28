---
title: Memory Management RL Environment
emoji: "🧠"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# Memory Management RL Environment

An OpenEnv benchmark for training agents to selectively remember, update, retrieve, and forget under a hard memory budget.

## Why This Matters

Every long-running assistant has the same failure mode: context is expensive, user state changes over time, and naive memory policies either hoard everything or forget the only detail that mattered.

This environment tests the production-relevant version of that problem:

- preferences that should be remembered
- corrections that should supersede stale memory
- misleading confabulations that should be ignored
- formatting constraints that should actually be followed
- tight budgets that force selective storage instead of dumping the whole thread

## What Gets Tested

| Capability | How it is tested |
| --- | --- |
| Preference recall | The user states a stack or tool preference that must appear in the final answer |
| Correction handling | Later turns replace an earlier preference, so stale memory must be updated or ignored |
| Noise filtering | Distractors and confabulations mention plausible technologies that are not true preferences |
| Format compliance | The final answer is graded for bullet points, numbered lists, JSON, concise output, and similar constraints |
| Budget management | Memory is budgeted and decays when it is not refreshed |

## Task Design

| Task | Difficulty | Twist |
| --- | --- | --- |
| `easy_preference_recall` | Easy | Turn kind is exposed |
| `medium_preference_constraint_correction` | Medium | Turn kind is hidden, one correction, format matters |
| `hard_full_memory_management` | Hard | Hidden turn kind, confabulation, project context, two corrections, decay |

Medium and hard return `"unknown"` for `current_turn_kind`, so agents must infer intent from the message text rather than cheating off the schema.

## Reward Signal

Terminal reward is composed from deterministic metrics:

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

Dense step rewards are also emitted during the episode for storing, retrieving, ignoring, updating, deleting, and answering.

## Quick Start

```bash
uv venv .venv
uv pip install -r requirements.txt --python .venv
.venv/bin/python -m unittest tests/test_core.py -v
```

Run the OpenEnv server:

```bash
.venv/bin/uvicorn server:app --host 0.0.0.0 --port 7860
```

Run baseline agents:

```bash
.venv/bin/python run_baseline.py
.venv/bin/python run_baseline.py --task hard_full_memory_management
.venv/bin/python run_baseline.py --json
```

## Python Usage

```python
from src.memory_management_agent import MemoryManagementEnv, RuleBasedMemoryAgent, run_episode

env = MemoryManagementEnv(memory_budget=200)
agent = RuleBasedMemoryAgent()
result = run_episode(agent, env, seed=42)
print(result.reward)
print(result.metrics.constraint_adherence)
```

Task-aware environments:

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

## HTTP API

- `GET /health`
- `GET /tasks`
- `POST /reset`
- `POST /step`
- `POST /grader`
- `GET /baseline`

Example:

```bash
curl -s http://localhost:7860/tasks
```

```bash
curl -s -X POST http://localhost:7860/reset \
  -H 'Content-Type: application/json' \
  -d '{"task_id":"medium_preference_constraint_correction","seed":42}'
```

## Score Interpretation

| Score | Interpretation |
| --- | --- |
| `0.0 - 0.2` | Agent ignores most user state or answers with stale memory |
| `0.2 - 0.5` | Partial recall, weak correction handling, poor formatting compliance |
| `0.5 - 0.8` | Strong recall and update behavior, but still brittle on noise or hard constraints |
| `0.8 - 1.0` | Selective storage, fresh memory, correct updates, and compliant final answers |

## Repository Layout

```text
src/memory_management_agent/
  agents.py
  environment.py
  episode.py
  grader.py
  memory_store.py
  tasks.py
  training.py
tests/test_core.py
server.py
openenv.yaml
```
