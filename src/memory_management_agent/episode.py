from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .schemas import ConversationTurn, Episode, MemoryType


_PREFERENCES: List[Tuple[str, str]] = [
    ("ClickHouse", "clickhouse"),
    ("PostgreSQL", "postgresql"),
    ("SQLite", "sqlite"),
    ("Python", "python"),
    ("FastAPI", "fastapi"),
    ("Redis", "redis"),
    ("MongoDB", "mongodb"),
    ("Rust", "rust"),
    ("Go", "golang"),
    ("TypeScript", "typescript"),
    ("DuckDB", "duckdb"),
    ("Kafka", "kafka"),
    ("Elasticsearch", "elasticsearch"),
    ("Celery", "celery"),
    ("GraphQL", "graphql"),
    ("gRPC", "grpc"),
    ("Parquet", "parquet"),
    ("Arrow", "arrow"),
    ("Pydantic", "pydantic"),
    ("SQLAlchemy", "sqlalchemy"),
]

_CONSTRAINTS: List[Tuple[str, str]] = [
    ("Keep answers concise.", "concise"),
    ("Use UTC timestamps.", "utc"),
    ("Avoid external dependencies.", "external dependencies"),
    ("Return bullet points.", "bullet points"),
    ("Explain the tradeoffs.", "tradeoffs"),
    ("Always include code examples.", "code examples"),
    ("Respond in under five sentences.", "five sentences"),
    ("Use snake_case for all identifiers.", "snake_case"),
    ("Never use synchronous blocking calls.", "blocking"),
    ("Include error handling in every snippet.", "error handling"),
    ("Output must be valid JSON.", "valid json"),
    ("Add type annotations to all functions.", "type annotations"),
    ("Use async/await patterns only.", "async"),
    ("Cite sources for any factual claim.", "cite sources"),
    ("Format output as a numbered list.", "numbered list"),
    ("Avoid global state at all costs.", "global state"),
    ("Prefer immutable data structures.", "immutable"),
    ("Always paginate large result sets.", "paginate"),
    ("Log every database call.", "log"),
    ("Default to read replicas for queries.", "read replicas"),
]

_PROJECT_FACTS: List[Tuple[str, str]] = [
    ("The project goal is a memory agent.", "memory agent"),
    ("The evaluation should be deterministic.", "deterministic"),
    ("We need a fixed memory budget.", "memory budget"),
    ("The baseline should be rule based.", "rule based"),
    ("We are targeting a production deployment.", "production"),
    ("The system must support multi-tenant workloads.", "multi-tenant"),
    ("Latency SLA is under 100 milliseconds.", "100 milliseconds"),
    ("The data pipeline runs on hourly cron jobs.", "cron"),
    ("All secrets are stored in Vault.", "vault"),
    ("The test suite uses pytest fixtures.", "pytest"),
    ("The CI pipeline runs on GitHub Actions.", "github actions"),
    ("We are migrating from monolith to microservices.", "microservices"),
]

_DISTRACTORS = [
    "By the way, I also like coffee.",
    "Let's talk about something unrelated.",
    "I walked my dog this morning.",
    "Can you ignore this random detail?",
    "The weather was nice yesterday.",
    "I recently finished a good book.",
    "My favourite colour is blue.",
    "I had a meeting that ran too long.",
    "Traffic was terrible on the way in.",
    "I need to schedule a dentist appointment.",
    "My laptop fan is louder than usual.",
    "We ordered pizza for the team last Friday.",
]

_RECALL_CHECK_TEMPLATES = [
    "Actually, remind me — what was the preference I mentioned earlier?",
    "Based on what I said before, what database or tool should we use?",
    "Going back to what I told you earlier — can you confirm my preference?",
    "What constraint did I give you at the start of our conversation?",
    "Just to double-check: what was the formatting rule I specified earlier?",
    "I forget — what did I say my technology preference was?",
    "Can you recall the constraint I mentioned? I want to make sure you have it.",
    "Before we continue, remind me what I told you about my preferences.",
]

_FINAL_QUERY_TEMPLATES = [
    "Given my previous preferences and constraints, answer the task in a way that matches them.",
    "Use what I told you earlier and respond to the request with the right format.",
    "Based on our earlier discussion, provide the final answer that respects my preferences.",
    "Now give me the final answer, keeping in mind everything I specified.",
    "Taking into account all my requirements from this conversation, provide your response.",
    "Synthesise everything I told you and give me the appropriate final answer.",
]


@dataclass
class SyntheticEpisodeGenerator:
    memory_budget: int = 200
    min_turns: int = 6
    max_turns: int = 8

    def generate(self, seed: int | None = None) -> Episode:
        rng = random.Random(seed if seed is not None else 0)
        episode_seed = seed if seed is not None else 0
        episode_id = f"episode_{episode_seed}_{rng.randint(1000, 9999)}"

        preference_text, preference_keyword = rng.choice(_PREFERENCES)
        constraint_text, constraint_keyword = rng.choice(_CONSTRAINTS)
        project_text, project_keyword = rng.choice(_PROJECT_FACTS)
        final_query = rng.choice(_FINAL_QUERY_TEMPLATES)
        recall_check_text = rng.choice(_RECALL_CHECK_TEMPLATES)

        turns: List[ConversationTurn] = []

        # Turn 0: preference
        turns.append(
            ConversationTurn(
                turn_id=0,
                text=f"My preference is {preference_text}.",
                kind="preference",
                memory_type=MemoryType.PREFERENCE,
                metadata={"keyword": preference_keyword},
            )
        )
        # Turn 1: distractor
        turns.append(
            ConversationTurn(
                turn_id=1,
                text=rng.choice(_DISTRACTORS),
                kind="distractor",
            )
        )
        # Turn 2: constraint
        turns.append(
            ConversationTurn(
                turn_id=2,
                text=constraint_text,
                kind="constraint",
                memory_type=MemoryType.CONSTRAINT,
                metadata={"keyword": constraint_keyword},
            )
        )
        # Turn 3: correction OR project_info (50/50)
        if rng.random() > 0.5:
            corrected_preference_text, corrected_preference_keyword = rng.choice(_PREFERENCES)
            turns.append(
                ConversationTurn(
                    turn_id=3,
                    text=f"Correction: actually use {corrected_preference_text} instead.",
                    kind="correction",
                    memory_type=MemoryType.PREFERENCE,
                    metadata={
                        "keyword": corrected_preference_keyword,
                        "correction_of": "preference",
                    },
                )
            )
            latest_preference_keyword = corrected_preference_keyword
            latest_preference_text = corrected_preference_text
        else:
            turns.append(
                ConversationTurn(
                    turn_id=3,
                    text=project_text,
                    kind="project_info",
                    memory_type=MemoryType.PROJECT_INFO,
                    metadata={"keyword": project_keyword},
                )
            )
            latest_preference_keyword = preference_keyword
            latest_preference_text = preference_text

        # Turn 4: recall_check — mid-episode question that requires RETRIEVE to score well.
        # The agent should look up a stored preference or constraint to answer.
        turns.append(
            ConversationTurn(
                turn_id=4,
                text=recall_check_text,
                kind="recall_check",
                memory_type=MemoryType.PREFERENCE,
                metadata={"keyword": latest_preference_keyword},
            )
        )

        # Turn 5: distractor (always present to pad before final_query)
        turns.append(
            ConversationTurn(
                turn_id=5,
                text=rng.choice(_DISTRACTORS),
                kind="distractor",
            )
        )

        # Final query: always last within max_turns
        turns.append(
            ConversationTurn(
                turn_id=6,
                text=(
                    f"{final_query} "
                    f"The answer should reflect {latest_preference_text} "
                    f"and {constraint_text.lower()}"
                ),
                kind="final_query",
            )
        )

        # Optionally insert one more distractor before final_query if budget allows
        if rng.random() > 0.6 and len(turns) < self.max_turns:
            turns.insert(
                -1,
                ConversationTurn(
                    turn_id=0,  # will be renumbered below
                    text=rng.choice(_DISTRACTORS),
                    kind="distractor",
                ),
            )

        turns = turns[: self.max_turns]
        # Re-number turn_ids sequentially
        for index, turn in enumerate(turns):
            turns[index] = ConversationTurn(
                turn_id=index,
                text=turn.text,
                kind=turn.kind,
                memory_type=turn.memory_type,
                tags=turn.tags,
                metadata=turn.metadata,
            )

        required_memory_types = [MemoryType.PREFERENCE.value, MemoryType.CONSTRAINT.value]
        required_keywords = [latest_preference_keyword, constraint_keyword]
        episode_metadata: Dict[str, object] = {
            # Grading ground-truth — hidden from agent observations (filtered in env._make_observation)
            "required_memory_types": required_memory_types,
            "required_keywords": required_keywords,
            "final_query": turns[-1].text,
            "latest_preference_keyword": latest_preference_keyword,
            "latest_constraint_keyword": constraint_keyword,
            "latest_project_keyword": project_keyword if any(t.kind == "project_info" for t in turns) else "",
            # Visible to agent
            "turn_count": len(turns),
        }

        return Episode(
            episode_id=episode_id,
            seed=episode_seed,
            turns=tuple(turns),
            memory_budget=self.memory_budget,
            metadata=episode_metadata,
        )
