# Memory Management Agent - Execution Plan

## 1. Purpose

Build an RL-driven memory management agent for multi-turn LLM interactions. The agent should learn when to:

- Store useful information
- Ignore irrelevant information
- Retrieve memory at the right time
- Update stale memory
- Compress memory to stay within budget

The primary objective is not better chat quality in the abstract. The objective is better future decisions under memory constraints.

## 2. MVP Definition

The MVP is complete when we have:

- A synthetic conversation environment with typed memory events
- A deterministic memory store with a fixed budget
- A rule-based baseline agent
- A grader that produces stable metrics
- A reward composer that maps metrics to scalar reward
- Baseline comparisons that show the environment is measurable
- A first TRL training loop connected to the environment

## 3. Non-Goals For v1

Do not build these in the first pass:

- Production-grade vector database integration
- Multi-agent memory sharing
- Human feedback loops
- Full UI or product surface
- Complex memory hierarchies
- External tool use beyond what is needed for evaluation

## 4. System Spec

### 4.1 Core Objects

#### Memory Item

Each memory entry should contain:

- `id`
- `text`
- `type` such as `preference`, `constraint`, or `project_info`
- `created_at`
- `last_used`
- `token_length`
- `utility_score`
- `source_turn`

#### Episode State

The environment state should include:

- Current user message
- Recent conversation window
- Current memory bank
- Remaining memory budget
- Step number
- Episode metadata

#### Actions

The action set for v1 should be small and explicit:

- `STORE(text)`
- `STORE_SUMMARY(text)`
- `IGNORE`
- `RETRIEVE(ids)`
- `UPDATE(id, text)`
- `DELETE(id)`
- `ANSWER(text)`

### 4.2 Environment Behavior

The environment should:

- Present a sequence of turns
- Expose memory state to the policy
- Accept memory management actions
- Track budget usage
- Record whether retrieved memories were useful
- End episodes when the final answer is produced or when the turn limit is reached

### 4.3 Grading Behavior

The grader should score:

- Future usefulness
- Retrieval precision
- Retrieval recall
- Compactness
- Freshness and update handling
- Non-interference

The grader should be deterministic for the baseline phase.

## 5. Suggested Repo Layout

If the code does not exist yet, create a structure like this:

- `src/env/` - environment implementation
- `src/memory/` - memory store and ranking logic
- `src/episode/` - synthetic episode generator
- `src/grader/` - scoring and reward composition
- `src/agents/` - rule-based and RL policies
- `src/training/` - TRL integration
- `src/eval/` - evaluation harness and baseline comparisons
- `tests/` - unit and integration tests
- `docs/` - design notes and experiment logs

## 6. Implementation Plan

### Current Status

Implemented in code:

- Core schemas
- Synthetic episode generation
- Memory store
- Environment reset/step loop
- Deterministic grading
- Reward composition
- Rule-based and baseline agents
- Episode evaluation harness
- Hidden evaluation splits
- Training prompt format and rollout collection scaffold
- Training run manifest and checkpoint export
- Analysis summaries and memory evolution charts
- Failure review text helpers

### Phase 0 - Scope and Contracts

**Goal:** Freeze the data model before coding the trainer.

Tasks:

- [x] Define the memory item schema
- [x] Define the environment observation schema
- [x] Define the action schema
- [x] Define grader output metrics
- [x] Define the reward composition formula
- [x] Define episode structure and turn limits

Acceptance criteria:

- Schemas are written down in markdown
- No ambiguous field names remain
- A baseline agent can be specified from the schema alone

### Phase 1 - Environment and Episode Generator

**Goal:** Produce a runnable environment with synthetic conversations.

Tasks:

- [x] Implement episode generation for preferences, constraints, distractors, corrections, and dependency tasks
- [x] Implement `reset()`
- [x] Implement `step()`
- [x] Implement turn progression and episode termination
- [x] Track memory budget consumption
- [x] Attach memory events to turns

Acceptance criteria:

- Environment resets deterministically with a seed
- Generated episodes are reproducible
- The environment supports all v1 actions
- Budget overrun behavior is defined and tested

### Phase 2 - Memory Store and Baseline Policies

**Goal:** Establish simple, measurable baselines before RL.

Tasks:

- [x] Implement the memory store class
- [x] Implement duplicate detection
- [x] Implement update and delete paths
- [x] Implement a store-everything baseline
- [x] Implement a no-memory baseline
- [x] Implement a preference-only baseline
- [x] Implement keyword retrieval baseline
- [x] Implement embedding retrieval baseline

Acceptance criteria:

- All baselines run on the same episodes
- Memory store state is inspectable after each turn
- Baseline outputs are logged in a consistent format

### Phase 3 - Grading and Reward

**Goal:** Make performance measurable and suitable for RL training.

Tasks:

- [x] Implement metric computation for precision, recall, compactness, and freshness
- [x] Implement rule-based useful-memory scoring
- [x] Implement reward shaping for store, retrieve, update, and answer actions
- [x] Implement final episode reward aggregation
- [x] Add penalty handling for contradictions and memory bloat
- [x] Add hidden evaluation episodes

Acceptance criteria:

- Scores are stable across repeated runs with the same seed
- Reward values are explainable at the turn level
- The grader can distinguish good and bad baseline behavior

### Phase 4 - RL Training Integration

**Goal:** Connect the environment to TRL and run the first training loop.

Tasks:

- [x] Pick the initial TRL algorithm
- [x] Define the policy prompt and output format
- [x] Connect environment rollouts to training data
- [x] Wire grader output into reward computation
- [x] Add logging for reward, memory usage, and success rate
- [x] Save checkpoints and experiment metadata

Acceptance criteria:

- A training run completes end-to-end
- Metrics are logged per rollout or per episode
- The policy can be compared against baselines

### Phase 5 - Analysis and Debugging

**Goal:** Understand what the agent learned and where it fails.

Tasks:

- [x] Track memory usage by type and turn
- [x] Analyze over-retrieval and under-retrieval
- [ ] Review failure episodes manually
- [x] Compare reward against downstream success
- [x] Check for reward hacking patterns
- [x] Visualize memory evolution over an episode

Acceptance criteria:

- Failure modes are documented with examples
- At least one clear improvement loop is identified
- The agent behavior is interpretable enough to debug

### Phase 6 - Advanced Features

**Goal:** Add features only after the MVP is working.

Possible additions:

- [ ] Memory decay
- [ ] Memory ranking
- [ ] Context-aware retrieval
- [ ] Vector database backend
- [ ] Multi-agent memory sharing
- [ ] LLM judge for semantic scoring

Acceptance criteria:

- Each feature has a clear reason to exist
- Each feature has a measurable effect on score or behavior

## 7. Working TODO Tracker

Use this as the live tracker during implementation.

| Status | Task | Priority | Depends On | Acceptance |
| --- | --- | ---: | --- | --- |
| [x] | Freeze schemas for state, action, memory item, and grader output | High | None | All schemas documented |
| [x] | Build synthetic episode generator | High | Schemas | Reproducible episodes with seed |
| [x] | Implement environment `reset()` and `step()` | High | Episode generator | Env runs through complete episode |
| [x] | Build memory store with add, update, delete, and query | High | Schemas | State persists correctly across turns |
| [x] | Implement deterministic grader | High | Env + memory store | Scores are stable and explainable |
| [x] | Implement reward composer | High | Grader | Reward maps cleanly from metrics |
| [x] | Add no-memory baseline | Medium | Env | Baseline runs end-to-end |
| [x] | Add store-everything baseline | Medium | Env + memory store | Baseline produces memory bloat |
| [x] | Add preference-only baseline | Medium | Memory store | Baseline is measurable on preference tasks |
| [x] | Add keyword retrieval baseline | Medium | Memory store | Retrieval works on lexical matches |
| [x] | Add embedding retrieval baseline | Medium | Memory store | Retrieval works on semantic matches |
| [x] | Add evaluation harness | High | Baselines + grader | Runs all baselines on same episodes |
| [ ] | Connect TRL training loop | High | Env + reward | First training run completes |
| [ ] | Add experiment logging | Medium | Training loop | Rewards and metrics are visible |
| [ ] | Review failure cases | Medium | Evaluation harness | At least 5 failures are categorized |
| [ ] | Decide whether to add vector DB backend | Low | MVP results | Decision is based on evidence |

## 8. Milestones

### Milestone 1 - Spec Frozen

Output:

- Final schemas
- Final action set
- Final grading metrics
- Final reward formula

### Milestone 2 - Environment Working

Output:

- Synthetic episodes
- Working reset/step loop
- Memory budget enforcement

### Milestone 3 - Baselines Working

Output:

- No-memory baseline
- Store-all baseline
- Retrieval baselines
- Benchmark table

### Milestone 4 - RL Training Working

Output:

- TRL integration
- Logged training runs
- First learned policy

### Milestone 5 - Analysis Complete

Output:

- Failure mode report
- Improvement backlog
- Decision on advanced features

## 9. Risks and Mitigations

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Reward hacking | Policy learns shortcuts instead of memory skill | Add hidden eval episodes and adversarial cases |
| Sparse feedback | Training signal may be too weak | Keep dense step-level rewards |
| Overfitting to synthetic episodes | Policy may not generalize | Vary episode templates and seed hidden evals |
| Memory bloat | Agent stores everything | Enforce strict budget penalties |
| Noisy grader | RL signal becomes unstable | Start with deterministic grading |
| Over-engineering | Scope grows before MVP lands | Gate advanced features behind milestone review |

## 10. Open Questions

Resolve these before expanding scope:

- What is the exact initial action format for the policy?
- Do we want text-only retrieval in v1, or text plus embeddings?
- What is the turn limit for the first environment?
- What does "success" mean for the first task family?
- Do we want the trainer to optimize per-turn reward or episode reward first?
- Which hidden eval cases best expose memory failure?

## 11. Definition of Done

The project is ready for the next iteration when:

- The environment is deterministic and documented
- The grader is stable
- Baselines are implemented and compared
- The RL loop runs end-to-end
- The failure modes are understood
- The next set of improvements is driven by evidence
