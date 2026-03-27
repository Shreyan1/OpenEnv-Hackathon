# Memory Management Agent (RL + OpenEnv)

## Project Design Document

## 1. Project Overview

### Objective
Build a reinforcement-learning-based memory management agent for LLM systems that learns:

- What information to store
- What to ignore
- When to retrieve memory
- How to update or compress memory

The goal is to maximize downstream task success while minimizing memory cost and noise.

### Core Problem Statement

> Given a stream of user interactions, how should an agent allocate limited memory resources to maximize future usefulness?

This is a long-horizon decision problem, which makes it a good fit for RL.

### Key Idea

Instead of optimizing for response quality directly, we optimize:

> Memory policy -> Better future responses

## 2. System Architecture

### High-Level Flow

```text
User Episodes -> OpenEnv Environment -> RL Policy (LLM) -> Memory Store
                                          |
                                          v
                                       Grader
                                          |
                                          v
                                     Reward Signal
                                          |
                                          v
                                      TRL Trainer
```

### Components Breakdown

1. OpenEnv environment
   - Simulates multi-turn conversations
   - Maintains memory state
   - Handles actions such as store, retrieve, and ignore

2. Memory store v1

   ```json
   {
     "id": "mem_001",
     "text": "User prefers ClickHouse",
     "type": "preference",
     "created_at": 1,
     "last_used": 5,
     "token_length": 6,
     "utility_score": 0.8
   }
   ```

3. RL policy (LLM)
   - Chooses actions based on state
   - Trained via TRL, with GRPO or PPO

4. Grader
   - Evaluates performance
   - Produces structured metrics

5. Reward composer
   - Converts metrics into a scalar reward

## 3. Environment Design (OpenEnv)

### Observation Space

```json
{
  "current_user_message": "...",
  "recent_conversation": [...],
  "memory_bank": [...],
  "memory_budget_remaining": 200,
  "step_number": 4
}
```

### Action Space

| Action | Description |
| --- | --- |
| `STORE(text)` | Save raw memory |
| `STORE_SUMMARY(text)` | Save compressed version |
| `IGNORE` | Do nothing |
| `RETRIEVE(ids)` | Fetch memory |
| `UPDATE(id, text)` | Modify memory |
| `DELETE(id)` | Remove memory |
| `ANSWER(text)` | Final response |

### Episode Structure

Typical episodes are 6 to 10 turns and include:

- Preferences
- Constraints
- Distractors
- Corrections
- Dependency tasks

### Example Episode

1. Turn 1: "I use ClickHouse"
2. Turn 2: unrelated
3. Turn 3: "Keep answers concise"
4. Turn 4: unrelated
5. Turn 5: "Write a SQL query for my dashboard"

The agent must:

- Store useful facts
- Retrieve memory at the correct time
- Ignore irrelevant information

## 4. Grading System

Total score: 100 points.

| Metric | Points | What it measures |
| --- | ---: | --- |
| Future usefulness | 35 | Did stored memory help later? |
| Retrieval precision | 20 | Was retrieved memory relevant? |
| Retrieval recall | 15 | Did the agent retrieve required memory? |
| Compactness | 15 | Was memory used efficiently? |
| Freshness / updates | 10 | Were corrections handled correctly? |
| Non-interference | 5 | Did the agent avoid irrelevant memory pollution? |

## 5. Reward System

### Dense Rewards (Step-Level)

| Event | Reward |
| --- | ---: |
| Store useful info | `+0.15` |
| Store useless info | `-0.05` |
| Good summary | `+0.10` |
| Duplicate memory | `-0.08` |
| Relevant retrieval | `+0.12` |
| Irrelevant retrieval | `-0.12` |
| Update stale memory | `+0.08` |
| Keep stale memory | `-0.10` |

### Delayed Rewards (Critical)

| Event | Reward |
| --- | ---: |
| Correct answer using memory | `+1.0` |
| Partial success | `+0.5` |
| Memory irrelevant | `0.0` |
| Missing memory -> failure | `-0.5` |
| Wrong memory -> wrong answer | `-0.4` |

### Final Episode Reward

- Compact memory: `+0.4`
- Memory bloat: `-0.3`
- Contradictions: `-0.3`

### Reward Formula

```text
R = 0.45 * success
  + 0.20 * precision
  + 0.15 * recall
  + 0.10 * compactness
  + 0.10 * freshness
  - penalties
```

## 6. Baselines

Before RL, implement:

- No memory
- Store everything
- Store only preferences
- Retrieve top-k by keyword
- Retrieve top-k by embeddings

The RL model must beat these baselines.

## 7. Implementation Phases

### Phase 1: Environment + Rule-Based System

**Goals**

- Build the OpenEnv environment
- Create a synthetic episode generator
- Implement a rule-based memory policy

**Tasks**

- Define the state and action schema
- Implement `reset()` and `step()`
- Build the memory store class
- Create the episode generator
- Add a basic rule-based grader
- Implement a simple baseline agent

**Output**

- Working environment
- Deterministic grading
- Debuggable system

### Phase 2: Evaluation + Baselines

**Goals**

- Validate environment quality
- Ensure the reward behaves sensibly

**Tasks**

- Run all baseline strategies
- Compare scores across episodes
- Check reward stability
- Add hidden evaluation episodes
- Detect reward hacking

**Output**

- Trusted grading system
- Benchmark metrics

### Phase 3: RL Integration (Hugging Face)

**Goals**

- Train the policy using TRL

**Stack**

- TRL, with GRPO recommended
- Accelerate
- Hugging Face Hub

**Tasks**

- Connect OpenEnv to TRL
- Define the policy model
- Configure the reward function
- Run the training loop
- Log metrics

**Output**

- First trained memory agent

### Phase 4: Analysis + Debugging

**Goals**

- Understand agent behavior

**Tasks**

- Analyze failure cases
- Track memory usage patterns
- Detect overfitting
- Visualize memory evolution

**Output**

- Insights into policy learning

### Phase 5: Advanced Features

Add:

- Vector DB memory, such as Chroma or Pinecone
- Memory ranking system
- Memory decay over time
- Context-aware retrieval
- Multi-agent memory sharing

### Phase 6: LLM Judge Integration (Optional)

**Goals**

- Improve grading quality

**Tasks**

- Add an LLM-based evaluator
- Score:
  - Summary quality
  - Contradiction detection
  - Semantic usefulness

## 8. Hugging Face Deployment

### Architecture

- OpenEnv -> Hugging Face Space
- TRL training -> local or Hugging Face
- Model -> pushed to the Hub

### Setup

- Environment server with FastAPI + OpenEnv
- TRL trainer script
- Accelerate config
- Logging, with Weights & Biases optional

## 9. Known Failure Modes

| Issue | Description |
| --- | --- |
| Memory hoarding | Stores everything |
| Over-retrieval | Uses too many memories |
| Under-retrieval | Misses key information |
| Stale memory | Uses outdated info |
| Reward hacking | Learns shortcuts |

## 10. MVP Scope

Start with:

- Memory types:
  - Preference
  - Constraint
  - Project info
- Actions:
  - Store
  - Ignore
  - Retrieve
  - Answer
- Episode length: 6 to 8 turns
- Fixed memory budget

## 11. Key Insight

This project is not about:

> Improving LLM responses

It is about:

> Learning optimal memory allocation under constraints to improve future decisions
