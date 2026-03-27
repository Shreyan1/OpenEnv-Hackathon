# Memory Management Agent (RL + OpenEnv)

A reinforcement learning system that trains an LLM-based policy to manage memory optimally during multi-turn conversations.

**Core question:** Given a stream of user interactions, how should an agent allocate limited memory resources to maximize future usefulness?

Instead of optimizing response quality directly, the system optimizes:

> Memory policy → Better future responses

---

## Overview

Most LLM systems either store everything (wasteful, noisy) or nothing (forgetful). This project frames memory management as a long-horizon decision problem and applies RL to learn an optimal strategy under a fixed token budget.

The agent observes each conversation turn and decides what to **store**, **ignore**, **retrieve**, **update**, **delete**, or **answer** — getting reward signals based on how well those decisions serve downstream task success.

---

## Requirements

- Python 3.9+
- pip

### Core dependencies (for environment + baselines)

No external dependencies are required to run the environment, baselines, and tests. The package uses only the Python standard library.

### Optional dependencies (for RL training)

To connect a language model and run actual RL training via TRL:

```bash
pip install trl transformers accelerate torch
```

For semantic embedding-based retrieval (`EmbeddingRetrievalAgent`):

```bash
pip install sentence-transformers
```

For experiment tracking:

```bash
pip install wandb          # Weights & Biases (optional)
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd memory-mgmt-with-rl-openenv
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\activate     # Windows
```

### 3. Install the package

```bash
pip install -e .
```

> If there is no `setup.py` yet, you can run directly from the repo root by setting `PYTHONPATH`:
> ```bash
> export PYTHONPATH=src   # macOS / Linux
> set PYTHONPATH=src      # Windows
> ```

### 4. (Optional) Install RL training dependencies

```bash
pip install trl transformers accelerate torch
```

---

## Project Structure

```
memory-mgmt-with-rl-openenv/
├── src/
│   └── memory_management_agent/
│       ├── schemas.py        # Data models: MemoryItem, Action, Observation, etc.
│       ├── environment.py    # MemoryManagementEnv (OpenEnv-compatible)
│       ├── episode.py        # Synthetic episode generator
│       ├── memory_store.py   # Budget-constrained memory store
│       ├── agents.py         # 6 baseline agents
│       ├── grader.py         # Grader + RewardComposer
│       ├── evaluation.py     # run_episode(), evaluate_split(), BenchmarkReport
│       ├── training.py       # Prompt building, rollout collection, TRL scaffold
│       ├── analysis.py       # Failure analysis, memory evolution tracking
│       ├── review.py         # Report rendering
│       └── utils.py          # Shared utilities
├── tests/
│   └── test_core.py          # Full test suite
├── memory-management-agent.md               # Design document
├── memory-management-agent-execution-plan.md # Implementation plan
├── CLAUDE.md                 # Developer reference
└── AGENTS.md                 # Agent workflow instructions
```

---

## Quick Start

### Run the test suite

```bash
python -m unittest tests/test_core.py -v
```

### Run a single episode with a baseline agent

```python
from src.memory_management_agent import (
    MemoryManagementEnv,
    RuleBasedMemoryAgent,
    run_episode,
)

env = MemoryManagementEnv(memory_budget=200, max_turns=8)
agent = RuleBasedMemoryAgent()
result = run_episode(agent, env, seed=42)

print(f"Reward:  {result.reward:.3f}")
print(f"Success: {result.metrics.success:.3f}")
print(f"Answer:  {result.answer}")
```

### Compare all baselines

```python
from src.memory_management_agent import (
    MemoryManagementEnv,
    NoMemoryAgent, StoreEverythingAgent, PreferenceOnlyAgent,
    KeywordRetrievalAgent, RuleBasedMemoryAgent,
    evaluate_split,
)

env = MemoryManagementEnv(memory_budget=200)
visible = tuple(range(1, 11))
hidden  = tuple(range(5000, 5010))

for AgentClass in [NoMemoryAgent, StoreEverythingAgent, PreferenceOnlyAgent,
                   KeywordRetrievalAgent, RuleBasedMemoryAgent]:
    report = evaluate_split(AgentClass(), env, visible, hidden)
    print(f"{AgentClass.__name__:30s}  visible={report.visible.avg_reward:.3f}  hidden={report.hidden.avg_reward:.3f}")
```

### Collect rollouts for training

```python
from src.memory_management_agent import (
    MemoryManagementEnv, RuleBasedMemoryAgent,
    collect_rollouts, export_rollouts_jsonl,
)

env   = MemoryManagementEnv(memory_budget=200)
agent = RuleBasedMemoryAgent()

rollouts = collect_rollouts(agent, env, seeds=tuple(range(50)))
export_rollouts_jsonl(rollouts, "rollouts.jsonl")
print(f"Exported {len(rollouts)} rollout episodes")
```

### Run a training experiment (with TRL)

```python
from src.memory_management_agent import (
    MemoryManagementEnv, RuleBasedMemoryAgent,
    run_training_experiment, TrainingConfig,
)

env    = MemoryManagementEnv(memory_budget=200)
agent  = RuleBasedMemoryAgent()
config = TrainingConfig(algorithm="grpo", run_name="first-run")

report = run_training_experiment(
    agent, env,
    train_seeds=tuple(range(50)),
    visible_eval_seeds=tuple(range(50, 60)),
    hidden_eval_seeds=tuple(range(5000, 5010)),
    output_dir="./checkpoints",
    config=config,
)
print(report)
```

### Analyze failures

```python
from src.memory_management_agent import (
    MemoryManagementEnv, RuleBasedMemoryAgent,
    collect_rollouts, analyze_rollouts, render_full_review,
)

env      = MemoryManagementEnv()
rollouts = collect_rollouts(RuleBasedMemoryAgent(), env, seeds=tuple(range(20)))
analysis = analyze_rollouts(rollouts)
render_full_review(analysis)
```

---

## How It Works

### Environment

`MemoryManagementEnv` presents a sequence of conversation turns to the agent. Each turn, the agent observes:

| Field | Description |
|-------|-------------|
| `current_message` | The current user turn text |
| `conversation_window` | Recent conversation history |
| `memory_bank` | Current contents of the memory store |
| `budget_remaining` | Remaining token budget |
| `step_number` | Current turn index |

### Action Space

| Action | Description |
|--------|-------------|
| `STORE(text)` | Save raw text to memory |
| `STORE_SUMMARY(text)` | Save a compressed version |
| `IGNORE` | Take no action this turn |
| `RETRIEVE(ids)` | Fetch specific memory items |
| `UPDATE(id, text)` | Replace an existing memory |
| `DELETE(id)` | Remove a memory item |
| `ANSWER(text)` | Produce the final response (ends episode) |

### Memory Store

The memory store enforces a fixed **token budget** (default: 200 tokens). When the budget is exceeded, the least-recently-used items are evicted. Duplicate detection uses normalized text comparison (Jaccard similarity).

### Grading & Reward

Each episode is scored on six metrics:

| Metric | Weight | What it measures |
|--------|-------:|-----------------|
| Success | 45% | Did stored memory enable the correct final answer? |
| Precision | 20% | Was retrieved memory relevant? |
| Recall | 15% | Were required memories retrieved? |
| Compactness | 10% | Was memory used efficiently? |
| Freshness | 10% | Were stale memories updated correctly? |
| Non-interference | — | Penalty for irrelevant memory pollution |

**Reward formula:**

```
R = 0.45 × success
  + 0.20 × precision
  + 0.15 × recall
  + 0.10 × compactness
  + 0.10 × freshness
  - penalties
```

Dense (step-level) rewards fire every turn; a delayed reward fires at episode end based on the final answer quality.

### Baseline Agents

| Agent | Strategy |
|-------|----------|
| `NoMemoryAgent` | Always IGNORE — never stores anything |
| `StoreEverythingAgent` | Always STORE — stores every turn |
| `PreferenceOnlyAgent` | Stores turns tagged as preference/constraint/project info |
| `KeywordRetrievalAgent` | Keyword-based heuristic retrieval |
| `EmbeddingRetrievalAgent` | Semantic embedding retrieval |
| `RuleBasedMemoryAgent` | Most sophisticated rule-based baseline |

The RL-trained policy is expected to outperform all of these on hidden evaluation seeds.

### Evaluation Design

To detect reward hacking and overfitting, evaluation uses two disjoint seed ranges:

- **Visible seeds** (1–4999): used for training and validation
- **Hidden seeds** (5000+): held out for final evaluation only

`generalization_gap = visible_reward − hidden_reward` — a large gap indicates the policy has overfit.

---

## Configuration

```python
MemoryManagementEnv(
    memory_budget=200,  # Token budget for the memory store (default: 200)
    max_turns=8,        # Maximum turns per episode (default: 8)
)

TrainingConfig(
    algorithm="grpo",          # "grpo" or "ppo"
    prompt_style="structured", # Prompt format
    max_prompt_tokens=4096,    # Max tokens in policy prompt
    checkpoint_dir="checkpoints",
    artifact_dir="artifacts",
    run_name="my-experiment",
)
```

---

## Development Status

| Phase | Status | Description |
|-------|--------|-------------|
| 0 – Schemas | ✅ Done | Data model frozen |
| 1 – Environment | ✅ Done | reset/step loop, episode generator |
| 2 – Memory store + baselines | ✅ Done | All 6 baseline agents |
| 3 – Grading + reward | ✅ Done | Deterministic grader, reward composer |
| 4 – RL training scaffold | ✅ Done | Prompt format, rollout collection, TRL scaffold |
| 5 – Analysis | ✅ Done | Failure detection, memory evolution, reporting |
| 6 – TRL model integration | 🔄 In progress | Wire real LLM into training loop |
| 7 – Advanced features | Planned | Vector DB, memory decay, LLM judge |

---

## Issue Tracking

This project uses **beads (`bd`)** for issue tracking.

```bash
bd ready                    # Find available work
bd show <id>                # View issue details
bd update <id> --claim      # Claim an issue
bd close <id>               # Mark complete
bd dolt push                # Sync issue data to remote
```

Run `bd prime` for the full workflow guide.

---

## Key Design Decisions

- **Deterministic seeding** — all episodes are reproducible via a seed parameter
- **Hard token budget** — strict memory cap with LRU eviction, not soft limits
- **Dense + delayed rewards** — step-level feedback prevents sparse-reward training problems
- **Typed schemas** — all data flows through immutable dataclasses
- **Hidden eval** — built-in train/visible/hidden split to catch reward hacking early
- **Synthetic episodes** — fully controlled data generation before moving to real conversations
