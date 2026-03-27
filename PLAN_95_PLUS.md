# Plan: Achieve 95+ Score

**Current estimated score:** ~80/100
**Target:** 95+/100
**Deadline:** April 8 — code freeze April 5 (HF Space must be live)

---

## Current Score Breakdown & Gap Analysis

| Criterion | Now | Target | Gap | Root Cause |
|---|---|---|---|---|
| Real-world utility | 22/30 | 28/30 | +6 | Templates are obviously synthetic; narrow domain; no articulated community value |
| Task & grader quality | 20/25 | 24/25 | +4 | `current_turn_kind` lets Nemotron trivially ace hard task; no constraint-adherence scoring; precision metric is weak |
| Environment design | 17/20 | 19/20 | +2 | `current_turn_kind` still exposed; no confabulation challenge; session memory leak |
| Code quality | 14/15 | 15/15 | +1 | No `pyproject.toml`; session TTL missing; HF Space README absent |
| Creativity | 7/10 | 9/10 | +2 | Memory decay mechanic missing; confabulation turns would be genuinely novel |
| **Total** | **80/100** | **95/100** | **+15** | |

---

## Phase A — Hide `current_turn_kind` for Medium & Hard (biggest single lever)

**Impact: +4–5 pts** (Task & grader quality +3, Environment design +1)
**Files:** `environment.py`, `schemas.py`, `tasks.py`, `agents.py`, `openenv.yaml`

### The problem
The observation currently passes `current_turn_kind` verbatim. Any LLM — including Nemotron — reads this and trivially acts:
- `preference/constraint/correction` → STORE
- `distractor` → IGNORE
- `recall_check` → RETRIEVE
- `final_query` → ANSWER

The hard task is **not actually hard** for a frontier model. Judges who inspect the observation schema will immediately spot this.

### The fix
Make `current_turn_kind` visibility a **per-task config option** on `TaskDefinition`:

```python
@dataclass(frozen=True)
class TaskDefinition:
    ...
    expose_turn_kind: bool = True   # NEW
```

In `MemoryManagementEnv._make_observation()`:
```python
turn_kind = self.current_turn.kind if task_exposes_turn_kind else "unknown"
```

| Task | `expose_turn_kind` | What agent sees |
|---|---|---|
| Easy | `True` | Exact kind — scaffolded for beginners |
| Medium | `False` | `"unknown"` — agent must infer from text |
| Hard | `False` | `"unknown"` + confabulation turns (Phase C) |

### Expected effect on scores
| Agent | Hard (before) | Hard (after) |
|---|---|---|
| `no_memory` | 0.000 | 0.000 |
| `rule_based` | 0.609 | ~0.250 (can't distinguish kinds) |
| Good LLM (Nemotron) | ~0.75 | ~0.50 (must reason from text) |

This is exactly the spread the judges want: rule-based fails, smart LLM does better.

### Implementation notes
- `RuleBasedMemoryAgent` falls back to text-based heuristics when `current_turn_kind == "unknown"` (already partially does this via `_looks_like_query`)
- Update `openenv.yaml` observation schema to document `current_turn_kind` as `"unknown" | actual_kind`
- Update `training.py` prompt template to not assume turn kind is always known

---

## Phase B — Realistic Conversation Templates

**Impact: +4–5 pts** (Real-world utility +4)
**Files:** `episode.py` only

### The problem
Current templates read like a requirements document, not a real conversation:
- `"My preference is ClickHouse."` — nobody talks like this
- `"Keep answers concise."` — robotic
- `"By the way, I also like coffee."` — comically obviously a distractor

A Meta engineer reviewing this will think "toy problem."

### The fix
Rewrite all templates to sound like actual Slack/code-review conversations:

**Preferences** (natural phrasings, not declarations):
```python
("ClickHouse", "clickhouse", [
    "We're running ClickHouse on the analytics side — that's the stack to target.",
    "I'd go with ClickHouse here, it handles our query volume better.",
    "The data team standardised on ClickHouse last quarter.",
    "ClickHouse is what I know best, let's stick with that.",
])
```
Each preference gets a **pool of 4 phrasings** — the generator picks one. This multiplies episode diversity by 4× with no structural changes.

**Constraints** (contextual, not commands):
```python
("bullet points", [
    "Can you keep it in bullet points? Easier to scan.",
    "Bullet-pointed list please — going into a doc.",
    "Format it as bullets, I'll paste this into Notion.",
])
```

**Distractors** (realistic noise, not absurdist):
```python
[
    "Sorry for the slow reply, been in back-to-back meetings.",
    "Can you also remind me — how do I set up a venv? (unrelated to the main task)",
    "Just checking — is this thread the right place to ask, or should I open a ticket?",
    "The deploy just went out, all green.",
    "Quick heads-up: I'll be OOO Thursday.",
]
```

**Corrections** (natural):
```python
"Actually, scratch the ClickHouse mention — the infra team said we're on PostgreSQL now.",
"Update: we're switching to {tech2}. Long story.",
"Forget what I said about {tech1} — {tech2} is what we need.",
```

### Schema change needed
`ConversationTurn` gets a `phrasing_index` metadata field so grader can verify the right content was stored.

---

## Phase C — Confabulation Turns (Novel + Hard Task Differentiator)

**Impact: +2–3 pts** (Task & grader quality +1, Creativity +2)
**Files:** `episode.py`, `environment.py`, `grader.py`, `schemas.py`

### What it is
**Confabulation turns** are conversation turns that mention a technology, preference, or constraint — but attributed to a third party, in a hypothetical, or explicitly as a non-preference. An agent that stores them is wrong.

Examples:
```
"My colleague keeps recommending Redis but honestly I don't care what he uses."
"Hypothetically, if we were using MongoDB, how would you approach this?"
"The old team used SQLite, but we moved away from it."
"I've heard GraphQL is popular but it's not relevant here."
```

These require **reading comprehension** to distinguish from genuine user preferences.

### Implementation
- New `ConversationTurn.kind = "confabulation"` (or keep as "distractor" but add `metadata={"type": "confabulation"}`)
- In grader: storing a confabulation turn is counted as `useless_store` (same as distractor)
- In `_utility_for_turn`: confabulation returns 0.0 utility
- Add 8–10 confabulation templates to `episode.py`
- Hard task: insert 1–2 confabulation turns (in addition to regular distractors)
- Only present when `expose_turn_kind = False` (otherwise trivially detectable)

### Why this is novel
No existing OpenEnv environment tests **attribution reasoning** — whether an agent can distinguish first-person user statements from third-party opinions or hypotheticals. This is a real problem in production memory agents.

---

## Phase D — Constraint-Adherence Scoring

**Impact: +3–4 pts** (Task & grader quality +3, Real-world utility +1)
**Files:** `grader.py`, `schemas.py`

### The problem
Currently, `success` only checks if the right **keywords** are in the answer. An agent that outputs:
```
"clickhouse bullet points"
```
scores nearly the same as one that outputs:
```
"• Use ClickHouse for the analytics layer
• Keep the query under 100ms
• No external dependencies"
```
even though the second response is actually following the constraint ("bullet points").

### The fix
Add `constraint_adherence: float` to `GraderMetrics`, computed deterministically:

```python
CONSTRAINT_FORMAT_CHECKERS = {
    "bullet points": lambda ans: sum(1 for line in ans.splitlines() if line.strip().startswith(("-", "•", "*"))) >= 2,
    "numbered list": lambda ans: sum(1 for line in ans.splitlines() if re.match(r"^\d+[\.\)]", line.strip())) >= 2,
    "five sentences": lambda ans: len(re.split(r"[.!?]+", ans.strip())) in range(4, 8),
    "concise": lambda ans: len(ans.split()) <= 50,
    "valid json": lambda ans: _is_valid_json(ans),
    "code examples": lambda ans: "```" in ans or "    " in ans,
    "type annotations": lambda ans: "->" in ans or ": " in ans,
    "snake_case": lambda ans: bool(re.search(r"\b[a-z]+_[a-z]+\b", ans)),
}
```

For each episode, extract the constraint keyword and run the matching checker against the final answer:
- `constraint_adherence = 1.0` if format matches
- `constraint_adherence = 0.0` if format doesn't match
- `constraint_adherence = 0.5` if constraint has no format checker (unmeasurable constraints)

### Reward formula update
Current weights sum to 1.0 (before penalties). Add `adherence` at 0.10, reduce `compactness` from 0.10 to 0.05 and `freshness` from 0.10 to 0.05:

```
R = 0.40 * success
  + 0.18 * precision
  + 0.12 * recall
  + 0.10 * adherence      # NEW — format compliance
  + 0.05 * compactness
  + 0.05 * freshness
  + 0.10 * non_interference_bonus
  - penalties
```

### Why this is important for utility score
A judge will ask: "Would we use this to evaluate a model that should follow formatting instructions?" Right now the answer is no — the grader doesn't check format. With adherence scoring, yes.

---

## Phase E — Memory Decay Mechanic

**Impact: +1–2 pts** (Creativity +2)
**Files:** `memory_store.py`, `environment.py`

### What it is
Memory items decay in relevance over time. Each step, `utility_score` of every item that wasn't accessed in the last 2 turns is reduced by a small `decay_rate`. Items at utility 0 get evicted by the budget enforcement on next add.

```python
@dataclass
class MemoryStore:
    decay_rate: float = 0.05   # per-turn decay for unaccessed items
    decay_window: int = 2      # turns before decay starts

    def apply_decay(self, current_turn: int) -> None:
        for item_id, item in list(self._items.items()):
            if current_turn - item.last_used > self.decay_window:
                new_score = max(0.0, item.utility_score - self.decay_rate)
                self._items[item_id] = replace(item, utility_score=new_score)
```

`MemoryManagementEnv.step()` calls `self.memory_store.apply_decay(self._step_index)` at the start of each step.

### Why this matters
- Forces the agent to **re-retrieve and re-store** important memories to keep them fresh
- Creates a genuinely different learning challenge from any existing env: agents that STORE once and never touch memories again will score worse than agents that actively maintain them
- Adds a fourth action (`RETRIEVE` mid-episode to refresh decay) that has real strategic value
- Makes the environment's reward dynamics genuinely novel and interesting

### Configuration
```python
MemoryManagementEnv(
    memory_budget=200,
    max_turns=8,
    decay_rate=0.05,    # tunable per task
    decay_window=2,
)
```

Hard task gets `decay_rate=0.08` (faster decay → tighter memory pressure).

---

## Phase F — HF Space README + Compelling Pitch

**Impact: +2 pts** (Real-world utility +2)
**Files:** `README.md` (new), `openenv.yaml` (description update)

### The problem
The judges score utility partly on whether they can immediately understand why this matters. The current README (if it exists) explains implementation, not value.

### The fix: write a README that answers the judge's mental checklist

```markdown
# Memory Management RL Environment

> An OpenEnv benchmark for training agents to selectively remember,
> update, and forget under a token budget — a capability gap in
> every production LLM assistant today.

## Why This Matters

Every long-running AI assistant faces the same problem: conversation
context is expensive. Current LLMs either:
- Store everything (wastes context, dilutes signal)
- Store nothing (forgets critical preferences)
- Use static rules (can't handle corrections or evolving user needs)

This environment trains RL policies that learn **when** to remember,
**when** to update, and **when** to discard — under a hard token budget.

## What Gets Tested

| Capability | How Tested |
|---|---|
| Preference recall | Store preference, recall at end |
| Correction handling | UPDATE stale memory when user changes their mind |
| Noise filtering | Distinguish user preferences from third-party opinions |
| Format compliance | Answer must match the user's stated formatting constraint |
| Budget management | Evict low-value memories under token pressure |

## Quick Start

[endpoint examples, curl commands]

## Score Interpretation

| Score | What It Means |
|---|---|
| 0.0–0.2 | Agent ignores all user state |
| 0.2–0.5 | Partial recall, poor update handling |
| 0.5–0.8 | Good recall, handles corrections |
| 0.8–1.0 | Near-perfect: selective storage + format compliance + freshness |
```

Also add the HF Spaces frontmatter to README:
```yaml
---
title: Memory Management RL Environment
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
license: mit
---
```

---

## Phase G — Code Quality Polish

**Impact: +1 pt** (Code quality 14→15)
**Files:** `pyproject.toml` (new), `server.py` (session cleanup), `openenv.yaml` (lint)

### Changes

1. **`pyproject.toml`** — makes the package installable, shows professionalism:
   ```toml
   [project]
   name = "memory-management-rl-openenv"
   version = "1.0.0"
   requires-python = ">=3.11"
   dependencies = ["fastapi>=0.110", "uvicorn[standard]>=0.29", "pydantic>=2.0"]
   ```

2. **Session TTL in `server.py`** — clean up sessions older than 30 minutes:
   ```python
   import time
   _sessions[session_id] = {"env": env, "task_id": task_id, "done": False, "created_at": time.time()}
   # On each /reset, evict sessions older than 1800s
   ```

3. **`openenv validate` compliance** — run the validator and fix any schema errors before submission

---

## Implementation Order & Timeline

Work backwards from April 5 (HF Space deployment day).

| Day | Work | Phase |
|---|---|---|
| March 28–29 | Phase B: realistic templates | Easy, no risk |
| March 29–30 | Phase A: hide turn_kind for medium/hard | High impact |
| March 30–31 | Phase C: confabulation turns | Novel mechanic |
| April 1 | Phase D: constraint-adherence scoring | Grader depth |
| April 1–2 | Phase E: memory decay | Creativity boost |
| April 2–3 | Phase F: README + HF Space README | Pitch matters |
| April 3–4 | Phase G: pyproject.toml, session TTL, validate | Polish |
| **April 4** | **End-to-end test + openenv validate** | Gate check |
| **April 5** | **HF Space deploy + smoke test** | Deadline |
| April 5–8 | Buffer for bugs | |

---

## File Impact Matrix

| File | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|
| `episode.py` | | ✓ | ✓ | | | | |
| `schemas.py` | ✓ | | ✓ | ✓ | | | |
| `environment.py` | ✓ | | ✓ | | ✓ | | |
| `grader.py` | | | ✓ | ✓ | | | |
| `memory_store.py` | | | | | ✓ | | |
| `agents.py` | ✓ | | | | | | |
| `tasks.py` | ✓ | | ✓ | | ✓ | | |
| `server.py` | | | | | | | ✓ |
| `openenv.yaml` | ✓ | | | ✓ | ✓ | ✓ | ✓ |
| `README.md` | | | | | | ✓ | |
| `pyproject.toml` | | | | | | | ✓ |
| `CLAUDE.md` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## Projected Score After Each Phase

| After Phase | Utility | Task Quality | Env Design | Code | Creativity | Total |
|---|---|---|---|---|---|---|
| **Current** | 22 | 20 | 17 | 14 | 7 | **80** |
| + A (hide turn_kind) | 22 | 23 | 19 | 14 | 7 | **85** |
| + B (realistic templates) | 26 | 23 | 19 | 14 | 7 | **89** |
| + C (confabulations) | 26 | 24 | 19 | 14 | 9 | **92** |
| + D (adherence scoring) | 27 | 25 | 19 | 14 | 9 | **94** |
| + E (memory decay) | 27 | 25 | 20 | 14 | 9 | **95** |
| + F (README/pitch) | 28 | 25 | 20 | 14 | 9 | **96** |
| + G (code polish) | 28 | 25 | 20 | 15 | 9 | **97** |

---

## Critical Design Decisions

### On hiding `current_turn_kind`
The risk: `RuleBasedMemoryAgent` and other baselines score much lower on medium/hard when the kind is hidden. That's **intentional** — it widens the gap between dumb baselines and smart LLMs. The judges will see this as evidence of a well-calibrated benchmark.

Do NOT hide it on easy — easy serves as the pedagogical entry point.

### On confabulation turns vs regular distractors
Confabulation turns are harder than distractors because they contain real technology keywords. An agent doing naïve keyword matching (store anything that mentions "PostgreSQL") will store them. Only an agent that understands attribution ("my colleague thinks") will correctly ignore them. This is a genuine reasoning challenge.

### On constraint-adherence
Some constraints don't have measurable format (e.g., "explain the tradeoffs"). For these, `constraint_adherence = 0.5` (neutral). The grader is still deterministic — same answer always gives same score. Only checkers that are 100% rule-based are used; no LLM judge dependency.

### On memory decay rate
`decay_rate=0.05` per turn means an item stored at turn 0 and never accessed again has `utility_score = 0.0` by turn 20. With max_turns=8, that means items stored at turn 0 decay to `0.6` by turn 8 — still above 0, so not evicted, but low enough that the budget enforcer prefers fresh items. Set `decay_rate=0.12` for hard task to create real pressure.

---

## What We Are NOT Doing

- **LLM-in-the-loop grading**: adds API dependency, non-deterministic, not reproducible
- **Vector DB backend**: adds heavy deps (faiss, sentence-transformers), not needed for 95+
- **Multi-episode memory persistence**: out of scope for OpenEnv single-episode model
- **Removing `current_turn_kind` from easy task**: hurts accessibility and the pedagogical value of the easy task

---

## Risks

| Risk | Mitigation |
|---|---|
| Hiding turn_kind breaks baseline agents too badly (score near 0) | Keep easy task with exposed turn_kind; baseline script reports all 5 agents |
| Constraint-adherence has edge cases (e.g., "valid json" when answer has json + extra text) | Use liberal checkers; partial match = 0.5 |
| Memory decay makes episodes too hard across the board | Tune decay_rate per task; default decay_rate=0 for easy |
| HF Space Docker build fails | Test locally with `docker build . && docker run -p 7860:7860` before April 5 |
| `openenv validate` schema errors | Run validator after every openenv.yaml change |
