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

An OpenEnv benchmark for training agents to selectively remember, update, retrieve, and forget under a fixed memory budget.

## Why This Matters

Long-running assistants constantly hit the same wall: context is expensive, user state changes, and simple memory policies either hoard everything or drop the one thing that actually mattered. This environment puts that problem front and center with realistic multi-turn conversations, tight budgets, and adversarial noise.

Specifically, agents are tested on:

- preferences that need to be stored and recalled later
- corrections that should replace stale memory, not stack on top of it
- confabulations and distractors that look relevant but should be ignored
- formatting constraints that the final answer must actually follow
- memory budgets that force prioritization rather than brute-force storage

## What Gets Tested

| Capability | How |
| --- | --- |
| Preference recall | User states a stack or tool preference; it must show up in the final answer |
| Correction handling | A later turn replaces an earlier preference; stale memory must be updated |
| Noise filtering | Distractors and confabulations mention plausible techs that are not real preferences |
| Format compliance | Final answer graded against bullet points, numbered lists, JSON, concise output, etc. |
| Budget management | Memory is token-budgeted and decays if not refreshed |

## Tasks

| Task | Difficulty | Notes |
| --- | --- | --- |
| `easy_preference_recall` | Easy | Turn kind is exposed |
| `medium_preference_constraint_correction` | Medium | Turn kind hidden, one correction, format matters |
| `hard_full_memory_management` | Hard | Turn kind hidden, confabulation, project context, two corrections, decay |

On medium and hard, `current_turn_kind` returns `"unknown"`. Agents have to read the message and figure out intent themselves rather than relying on the label.

## Reward

Terminal reward comes from deterministic grader metrics:

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

Dense step rewards fire on each store, retrieve, ignore, update, delete, and answer action during the episode.
Final task scores are clamped to the strict open interval `(0, 1)`, so externally reported scores are never exactly `0.0` or `1.0`.

## Quick Start

```bash
uv venv .venv
uv pip install -r requirements.txt --python .venv
.venv/bin/python -m unittest tests/test_core.py -v
```

Start the server:

```bash
.venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Run baseline agents:

```bash
.venv/bin/python run_baseline.py
.venv/bin/python run_baseline.py --task hard_full_memory_management
.venv/bin/python run_baseline.py --json
```

## LLM Agent Evaluation

```bash
# Anthropic
ANTHROPIC_API_KEY=sk-ant-... .venv/bin/python run_llm_agent.py

# OpenRouter
OPENROUTER_API_KEY=sk-or-... .venv/bin/python run_llm_agent.py \
  --provider openrouter --model anthropic/claude-haiku-4-5

# Single task, specific seeds
ANTHROPIC_API_KEY=... .venv/bin/python run_llm_agent.py \
  --task easy_preference_recall --seeds 42 43 44

# JSON output
ANTHROPIC_API_KEY=... .venv/bin/python run_llm_agent.py --json
```

### Benchmark Results (claude-haiku-4-5, 5 seeds)

| Task | LLM avg | rule_based avg | Delta |
| --- | --- | --- | --- |
| `easy_preference_recall` | 0.649 | 0.648 | +0.001 |
| `medium_preference_constraint_correction` | 0.868 | 0.934 | -0.065 |
| `hard_full_memory_management` | **0.908** | 0.663 | **+0.245** |

The LLM agent beats the rule-based baseline by a wide margin on the hard task.

## Python Usage

```python
from src.memory_management_agent import MemoryManagementEnv, RuleBasedMemoryAgent, run_episode

env = MemoryManagementEnv(memory_budget=200)
agent = RuleBasedMemoryAgent()
result = run_episode(agent, env, seed=42)
print(result.reward)
print(result.metrics.constraint_adherence)
```

Task-aware setup:

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
- `WS /ws` (OpenEnv WebSocket)

```bash
curl -s http://localhost:7860/tasks

curl -s -X POST http://localhost:7860/reset \
  -H 'Content-Type: application/json' \
  -d '{"task_id":"medium_preference_constraint_correction","seed":42}'
```

## Score Interpretation

| Score | What it means |
| --- | --- |
| 0.0 - 0.2 | Agent mostly ignores user state or answers from stale memory |
| 0.2 - 0.5 | Partial recall, weak correction handling, poor format compliance |
| 0.5 - 0.8 | Solid recall and updates, but brittle on noise or harder constraints |
| 0.8 - 1.0 | Selective storage, fresh memory, correct updates, format-compliant answers |

## Repository Layout

```text
src/memory_management_agent/
  agents.py         # baseline heuristic agents
  environment.py    # reset/step loop
  episode.py        # conversation template pools
  grader.py         # deterministic reward composer
  memory_store.py   # budgeted memory with decay
  tasks.py          # easy / medium / hard task generators
  training.py       # prompt building and rollout collection
server/
  app.py            # FastAPI server (HTTP + WebSocket)
tests/test_core.py
run_baseline.py     # rule-based baseline evaluation
run_llm_agent.py    # LLM agent evaluation (Anthropic / OpenRouter)
inference.py        # submission inference script
openenv.yaml        # OpenEnv manifest
```
