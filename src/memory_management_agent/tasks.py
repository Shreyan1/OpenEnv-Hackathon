"""
Task definitions for the Memory Management environment.

Three tasks with a genuine difficulty gradient:

  EASY   — single preference, no corrections, straightforward recall
  MEDIUM — preference + constraint, one correction (requires UPDATE), recall both
  HARD   — preference + constraint + project_info, multiple corrections,
            interleaved distractors, must reconcile all three types at answer time
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .episode import SyntheticEpisodeGenerator
from .schemas import ConversationTurn, Episode, MemoryType


# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    name: str
    description: str
    difficulty: str          # "easy" | "medium" | "hard"
    memory_budget: int
    max_turns: int
    seed_range: tuple[int, int]   # (start_inclusive, end_exclusive)
    expected_score_range: tuple[float, float]  # (random_agent_max, good_agent_min)
    expose_turn_kind: bool = True   # if False, agent sees "unknown" instead of the real kind
    decay_rate: float = 0.0         # per-turn memory utility decay for unaccessed items

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
        "A short conversation introduces one user preference and one distractor. "
        "The agent must store the preference and recall it verbatim in the final answer. "
        "No corrections occur. Budget is generous relative to content."
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
        "The conversation introduces a preference and a constraint, then issues one correction "
        "that supersedes the original preference. The agent must UPDATE the stored preference "
        "(or DELETE the stale one and STORE the new one) and incorporate both the corrected "
        "preference and the constraint in the final answer. "
        "Turn kind is hidden — the agent must infer from message text alone."
    ),
    difficulty="medium",
    memory_budget=200,
    max_turns=7,
    seed_range=(1, 5000),
    expected_score_range=(0.05, 0.45),
    expose_turn_kind=False,
    decay_rate=0.04,
)

TASK_HARD = TaskDefinition(
    task_id="hard_full_memory_management",
    name="Full Memory Management Under Pressure",
    description=(
        "A long conversation introduces a preference, a constraint, and project info — "
        "interspersed with multiple distractors, confabulation turns, and two corrections. "
        "The memory budget is tight. The agent must selectively store high-utility items, "
        "evict or delete stale information when corrected, distinguish first-person preferences "
        "from third-party opinions, and produce a final answer that references the latest "
        "preference, the constraint, and the project context. "
        "Turn kind is hidden; memory decays if not refreshed."
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
TASK_BY_ID: Dict[str, TaskDefinition] = {t.task_id: t for t in ALL_TASKS}


# ---------------------------------------------------------------------------
# Per-task episode generators
# ---------------------------------------------------------------------------

def generator_for_task(task: TaskDefinition) -> "TaskEpisodeGenerator":
    return TaskEpisodeGenerator(task=task)


@dataclass
class TaskEpisodeGenerator:
    """Wraps SyntheticEpisodeGenerator and applies task-specific constraints."""
    task: TaskDefinition

    def generate(self, seed: Optional[int] = None) -> Episode:
        base_gen = SyntheticEpisodeGenerator(
            memory_budget=self.task.memory_budget,
            max_turns=self.task.max_turns,
        )
        episode = base_gen.generate(seed=seed)

        if self.task.task_id == TASK_EASY.task_id:
            episode = self._make_easy(episode, seed)
        elif self.task.task_id == TASK_MEDIUM.task_id:
            episode = self._make_medium(episode, seed)
        elif self.task.task_id == TASK_HARD.task_id:
            episode = self._make_hard(episode, seed)

        return episode

    # ------------------------------------------------------------------
    # Easy: strip out correction turns, keep only preference + distractors
    # ------------------------------------------------------------------
    def _make_easy(self, episode: Episode, seed: Optional[int]) -> Episode:
        # Always generate a full episode (max_turns=8) so final_query isn't truncated,
        # then filter it down to an easy scenario.
        from .episode import SyntheticEpisodeGenerator as SEG
        full_gen = SEG(memory_budget=self.task.memory_budget, max_turns=8)
        full_episode = full_gen.generate(seed=seed)

        preference_turns = [t for t in full_episode.turns if t.kind == "preference"]
        distractor_turns = [t for t in full_episode.turns if t.kind == "distractor"]
        final_turns = [t for t in full_episode.turns if t.kind == "final_query"]

        if not preference_turns or not final_turns:
            return full_episode  # fallback

        selected_distractors = distractor_turns[:1]
        raw = preference_turns[:1] + selected_distractors + final_turns[:1]
        turns = tuple(
            ConversationTurn(
                turn_id=i,
                text=t.text,
                kind=t.kind,
                memory_type=t.memory_type,
                tags=t.tags,
                metadata=t.metadata,
            )
            for i, t in enumerate(raw)
        )

        # Rebuild metadata: only preference is required
        pref_keyword = preference_turns[0].metadata.get("keyword", "")
        metadata = dict(full_episode.metadata)
        metadata["required_memory_types"] = [MemoryType.PREFERENCE.value]
        metadata["required_keywords"] = [pref_keyword]
        metadata["final_query"] = final_turns[0].text
        metadata["turn_count"] = len(turns)
        metadata["latest_preference_keyword"] = pref_keyword
        metadata["latest_constraint_keyword"] = ""
        metadata["latest_project_keyword"] = ""

        from .schemas import Episode as Ep
        return Ep(
            episode_id=full_episode.episode_id + "_easy",
            seed=full_episode.seed,
            turns=turns,
            memory_budget=full_episode.memory_budget,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Medium: ensure exactly one correction occurs
    # ------------------------------------------------------------------
    def _make_medium(self, episode: Episode, seed: Optional[int]) -> Episode:
        has_correction = any(t.kind == "correction" for t in episode.turns)
        if has_correction:
            return episode

        # Regenerate until we get exactly one correction (base generator: correction prob ~0.5)
        import random
        base_gen = SyntheticEpisodeGenerator(
            memory_budget=self.task.memory_budget,
            max_turns=self.task.max_turns,
        )
        rng = random.Random(seed if seed is not None else 0)
        for _ in range(30):
            candidate_seed = rng.randint(0, 999_999)
            candidate = base_gen.generate(seed=candidate_seed)
            if any(t.kind == "correction" for t in candidate.turns):
                return candidate

        return episode  # fallback

    # ------------------------------------------------------------------
    # Hard: guarantee exactly 2 corrections + require all 3 memory types
    # ------------------------------------------------------------------
    def _make_hard(self, episode: Episode, seed: Optional[int]) -> Episode:
        import random
        from .episode import SyntheticEpisodeGenerator as SEG
        from .schemas import ConversationTurn as CT, MemoryType as MT, Episode as Ep
        from .episode import _PREFERENCES, _PROJECT_FACTS

        base_gen = SEG(memory_budget=self.task.memory_budget, max_turns=self.task.max_turns)
        rng = random.Random(seed if seed is not None else 0)

        # Step 1: find a base episode that has at least one correction.
        candidate = episode
        if not any(t.kind == "correction" for t in candidate.turns):
            for _ in range(30):
                c = base_gen.generate(seed=rng.randint(0, 999_999))
                if any(t.kind == "correction" for t in c.turns):
                    candidate = c
                    break

        turns = list(candidate.turns)

        # Step 2: ensure project_info is present; if not, replace the first distractor with one.
        if not any(t.kind == "project_info" for t in turns):
            rng2 = random.Random((seed or 0) + 7)
            from .episode import _PROJECT_FACTS as PF
            proj_text, proj_kw = rng2.choice(PF)
            for i, t in enumerate(turns):
                if t.kind == "distractor":
                    turns[i] = CT(
                        turn_id=i, text=proj_text, kind="project_info",
                        memory_type=MT.PROJECT_INFO,
                        tags=t.tags, metadata={"keyword": proj_kw},
                    )
                    break

        # Step 3: inject a SECOND correction just before the final_query.
        final_idx = next((i for i, t in enumerate(turns) if t.kind == "final_query"), len(turns) - 1)
        latest_pref_kw = str(candidate.metadata.get("latest_preference_keyword", ""))
        pool = [(pt, pk) for pt, pk in _PREFERENCES if pk != latest_pref_kw]
        if not pool:
            pool = list(_PREFERENCES)
        rng3 = random.Random((seed or 0) + 13)
        second_correction_text, second_correction_kw = rng3.choice(pool)

        second_correction = CT(
            turn_id=0,
            text=f"Wait, I changed my mind — switch to {second_correction_text} instead.",
            kind="correction",
            memory_type=MT.PREFERENCE,
            metadata={"keyword": second_correction_kw, "correction_of": "preference", "correction_index": 2},
        )
        insert_pos = max(0, final_idx - 1)
        turns.insert(insert_pos, second_correction)
        turns = turns[: self.task.max_turns]

        # Step 4: update final_query text to reference the SECOND correction's preference.
        constraint_text = next((t.text.lower() for t in turns if t.kind == "constraint"), "")
        turns_out: list[CT] = []
        for i, t in enumerate(turns):
            if t.kind == "final_query":
                new_text = (
                    f"Taking into account all my requirements, provide your response. "
                    f"Remember to use {second_correction_text} and {constraint_text}"
                )
                turns_out.append(CT(
                    turn_id=i, text=new_text, kind=t.kind,
                    memory_type=t.memory_type, tags=t.tags, metadata=t.metadata,
                ))
            else:
                turns_out.append(CT(
                    turn_id=i, text=t.text, kind=t.kind,
                    memory_type=t.memory_type, tags=t.tags, metadata=t.metadata,
                ))

        # Step 5: rebuild metadata — ground truth now points at the second correction.
        constraint_kw = str(candidate.metadata.get("latest_constraint_keyword", ""))
        new_meta = dict(candidate.metadata)
        new_meta["required_keywords"] = (
            [second_correction_kw, constraint_kw] if constraint_kw else [second_correction_kw]
        )
        new_meta["latest_preference_keyword"] = second_correction_kw
        new_meta["turn_count"] = len(turns_out)
        new_meta["final_query"] = turns_out[-1].text
        # Preserve project keyword if a project_info turn exists
        proj_kws = [t.metadata.get("keyword", "") for t in turns_out if t.kind == "project_info"]
        new_meta["latest_project_keyword"] = proj_kws[0] if proj_kws else ""

        return Ep(
            episode_id=candidate.episode_id + "_hard",
            seed=candidate.seed,
            turns=tuple(turns_out),
            memory_budget=candidate.memory_budget,
            metadata=new_meta,
        )
