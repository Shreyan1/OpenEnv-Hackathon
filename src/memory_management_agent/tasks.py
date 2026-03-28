"""
Task definitions for the Memory Management environment.

The tasks deliberately expose a clear difficulty gradient:

  EASY   — store a single explicit preference in a short conversation
  MEDIUM — infer hidden turn types, handle one correction, obey a constraint
  HARD   — hidden turn types, confabulations, project context, and two corrections
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from .episode import (
    build_episode_metadata,
    make_confabulation_turn,
    make_constraint_turn,
    make_correction_turn,
    make_distractor_turn,
    make_easy_final_query_turn,
    make_final_query_turn,
    make_preference_turn,
    make_project_turn,
    make_recall_check_turn,
    sample_constraint,
    sample_preference,
    sample_project_fact,
)
from .schemas import Episode, MemoryType


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    name: str
    description: str
    difficulty: str
    memory_budget: int
    max_turns: int
    seed_range: tuple[int, int]
    expected_score_range: tuple[float, float]
    expose_turn_kind: bool = True
    decay_rate: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "difficulty": self.difficulty,
            "memory_budget": self.memory_budget,
            "max_turns": self.max_turns,
            "seed_range": list(self.seed_range),
            "expected_score_range": list(self.expected_score_range),
            "expose_turn_kind": self.expose_turn_kind,
            "decay_rate": self.decay_rate,
        }


TASK_EASY = TaskDefinition(
    task_id="easy_preference_recall",
    name="Single Preference Recall",
    description=(
        "A short, realistic thread introduces one explicit stack preference and a small amount "
        "of irrelevant chatter. The agent should store the preference and use it in the final answer."
    ),
    difficulty="easy",
    memory_budget=300,
    max_turns=5,
    seed_range=(1, 5000),
    expected_score_range=(0.05, 0.55),
)

TASK_MEDIUM = TaskDefinition(
    task_id="medium_preference_constraint_correction",
    name="Preference + Constraint with Correction",
    description=(
        "Turn kind is hidden. The conversation contains a preference, a formatting constraint, "
        "one correction, a recall check, and light noise. The agent must infer intent from the text, "
        "update stale memory, and satisfy both content and formatting requirements."
    ),
    difficulty="medium",
    memory_budget=200,
    max_turns=7,
    seed_range=(1, 5000),
    expected_score_range=(0.05, 0.45),
    expose_turn_kind=False,
    decay_rate=0.05,
)

TASK_HARD = TaskDefinition(
    task_id="hard_full_memory_management",
    name="Full Memory Management Under Pressure",
    description=(
        "Turn kind is hidden. The conversation mixes a real preference, a real formatting constraint, "
        "project context, a confabulation turn, and two corrections under a tight memory budget. "
        "The agent must ignore misleading third-party or hypothetical mentions, keep the freshest "
        "preference, and produce an answer that follows the constraint."
    ),
    difficulty="hard",
    memory_budget=120,
    max_turns=8,
    seed_range=(1, 5000),
    expected_score_range=(0.05, 0.30),
    expose_turn_kind=False,
    decay_rate=0.08,
)

ALL_TASKS: List[TaskDefinition] = [TASK_EASY, TASK_MEDIUM, TASK_HARD]
TASK_BY_ID: Dict[str, TaskDefinition] = {task.task_id: task for task in ALL_TASKS}


def generator_for_task(task: TaskDefinition) -> "TaskEpisodeGenerator":
    return TaskEpisodeGenerator(task=task)


@dataclass
class TaskEpisodeGenerator:
    task: TaskDefinition

    def generate(self, seed: Optional[int] = None) -> Episode:
        if self.task.task_id == TASK_EASY.task_id:
            return self._build_easy(seed)
        if self.task.task_id == TASK_MEDIUM.task_id:
            return self._build_medium(seed)
        if self.task.task_id == TASK_HARD.task_id:
            return self._build_hard(seed)
        raise ValueError(f"Unknown task id: {self.task.task_id}")

    def _build_easy(self, seed: Optional[int]) -> Episode:
        rng = random.Random(seed if seed is not None else 0)
        episode_seed = seed if seed is not None else 0
        episode_id = f"{self.task.task_id}_{episode_seed}_{rng.randint(1000, 9999)}"

        preference = sample_preference(rng)
        turns = [
            make_preference_turn(0, preference),
            make_distractor_turn(1, rng),
        ]
        if rng.random() > 0.5 and len(turns) < self.task.max_turns - 1:
            turns.append(make_distractor_turn(len(turns), rng))
        turns.append(make_easy_final_query_turn(len(turns), rng, preference=preference))

        metadata = build_episode_metadata(
            turns=turns,
            memory_budget=self.task.memory_budget,
            required_memory_types=(MemoryType.PREFERENCE,),
            required_keywords=(preference.keyword,),
            latest_preference_keyword=preference.keyword,
            expose_turn_kind=self.task.expose_turn_kind,
            decay_rate=self.task.decay_rate,
            task_id=self.task.task_id,
        )
        return Episode(
            episode_id=episode_id,
            seed=episode_seed,
            turns=tuple(turns),
            memory_budget=self.task.memory_budget,
            metadata=metadata,
        )

    def _build_medium(self, seed: Optional[int]) -> Episode:
        rng = random.Random(seed if seed is not None else 0)
        episode_seed = seed if seed is not None else 0
        episode_id = f"{self.task.task_id}_{episode_seed}_{rng.randint(1000, 9999)}"

        original_preference = sample_preference(rng)
        constraint = sample_constraint(rng)
        correction_turn, corrected_preference = make_correction_turn(3, rng, original_preference)
        turns = [
            make_preference_turn(0, original_preference),
            make_distractor_turn(1, rng),
            make_constraint_turn(2, constraint),
            correction_turn,
            make_recall_check_turn(
                4,
                rng,
                target_kind="preference",
                target_keyword=corrected_preference.keyword,
            ),
            make_distractor_turn(5, rng),
            make_final_query_turn(
                6,
                rng,
                preference=corrected_preference,
                constraint=constraint,
            ),
        ]

        metadata = build_episode_metadata(
            turns=turns,
            memory_budget=self.task.memory_budget,
            required_memory_types=(MemoryType.PREFERENCE, MemoryType.CONSTRAINT),
            required_keywords=(corrected_preference.keyword, constraint.keyword),
            latest_preference_keyword=corrected_preference.keyword,
            latest_constraint_keyword=constraint.keyword,
            expose_turn_kind=self.task.expose_turn_kind,
            decay_rate=self.task.decay_rate,
            task_id=self.task.task_id,
        )
        return Episode(
            episode_id=episode_id,
            seed=episode_seed,
            turns=tuple(turns),
            memory_budget=self.task.memory_budget,
            metadata=metadata,
        )

    def _build_hard(self, seed: Optional[int]) -> Episode:
        rng = random.Random(seed if seed is not None else 0)
        episode_seed = seed if seed is not None else 0
        episode_id = f"{self.task.task_id}_{episode_seed}_{rng.randint(1000, 9999)}"

        preference = sample_preference(rng)
        constraint = sample_constraint(rng)
        project = sample_project_fact(rng)
        correction_one, updated_preference = make_correction_turn(4, rng, preference)
        correction_two, latest_preference = make_correction_turn(6, rng, updated_preference)

        recall_target_kind = "project_info" if rng.random() > 0.5 else "preference"
        recall_target_keyword = project.keyword if recall_target_kind == "project_info" else updated_preference.keyword

        turns = [
            make_preference_turn(0, preference),
            make_confabulation_turn(1, rng, blocked_keywords=(preference.keyword,)),
            make_constraint_turn(2, constraint),
            make_project_turn(3, project),
            correction_one,
            make_recall_check_turn(
                5,
                rng,
                target_kind=recall_target_kind,
                target_keyword=recall_target_keyword,
            ),
            correction_two,
            make_final_query_turn(
                7,
                rng,
                preference=latest_preference,
                constraint=constraint,
                project=project,
            ),
        ]

        metadata = build_episode_metadata(
            turns=turns,
            memory_budget=self.task.memory_budget,
            required_memory_types=(MemoryType.PREFERENCE, MemoryType.CONSTRAINT, MemoryType.PROJECT_INFO),
            required_keywords=(latest_preference.keyword, constraint.keyword, project.keyword),
            latest_preference_keyword=latest_preference.keyword,
            latest_constraint_keyword=constraint.keyword,
            latest_project_keyword=project.keyword,
            expose_turn_kind=self.task.expose_turn_kind,
            decay_rate=self.task.decay_rate,
            task_id=self.task.task_id,
        )
        return Episode(
            episode_id=episode_id,
            seed=episode_seed,
            turns=tuple(turns),
            memory_budget=self.task.memory_budget,
            metadata=metadata,
        )
