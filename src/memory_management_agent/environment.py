from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

from .episode import SyntheticEpisodeGenerator
from .grader import Grader, RewardComposer
from .memory_store import MemoryStore
from .schemas import (
    Action,
    ActionRecord,
    ActionType,
    ConversationTurn,
    Episode,
    EpisodeResult,
    MemoryItem,
    MemoryType,
    Observation,
    StepResult,
)
from .utils import contains_any

# Keys that contain grading ground-truth — must never be visible to the agent.
_HIDDEN_METADATA_KEYS: frozenset[str] = frozenset({
    "required_memory_types",
    "required_keywords",
    "final_query",
    "latest_preference_keyword",
    "latest_constraint_keyword",
    "latest_project_keyword",
})


class MemoryManagementEnv:
    def __init__(
        self,
        *,
        generator: Optional[SyntheticEpisodeGenerator] = None,
        grader: Optional[Grader] = None,
        reward_composer: Optional[RewardComposer] = None,
        memory_budget: int = 200,
        max_turns: int = 8,
        expose_turn_kind: bool = True,
        decay_rate: float = 0.0,
    ):
        self.generator = generator or SyntheticEpisodeGenerator(memory_budget=memory_budget, max_turns=max_turns)
        self.grader = grader or Grader()
        self.reward_composer = reward_composer or RewardComposer()
        self.memory_budget = memory_budget
        self.max_turns = max_turns
        self.expose_turn_kind = expose_turn_kind
        self.decay_rate = decay_rate
        self.episode: Optional[Episode] = None
        self.memory_store: Optional[MemoryStore] = None
        self._step_index = 0
        self._done = False
        self._final_answer = ""
        self._trace: list[ActionRecord] = []

    @property
    def trace(self) -> tuple[ActionRecord, ...]:
        return tuple(self._trace)

    @property
    def done(self) -> bool:
        return self._done

    def reset(self, seed: Optional[int] = None) -> Observation:
        self.episode = self.generator.generate(seed=seed)
        self.memory_store = MemoryStore(budget_tokens=self.episode.memory_budget)
        self._step_index = 0
        self._done = False
        self._final_answer = ""
        self._trace = []
        return self._make_observation()

    def step(self, action: Action | Dict[str, Any]) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is already complete. Call reset() to start a new episode.")
        if self.episode is None or self.memory_store is None:
            raise RuntimeError("Environment has not been reset.")

        current_turn = self.current_turn
        normalized_action = self._normalize_action(action)
        reward = 0.0
        retrieved_items: tuple[MemoryItem, ...] = ()
        stored_item: Optional[MemoryItem] = None
        note = ""

        if normalized_action.type in {ActionType.STORE, ActionType.STORE_SUMMARY}:
            if normalized_action.text is None:
                normalized_action = replace(normalized_action, text=current_turn.text)
            memory_type = current_turn.memory_type or MemoryType.PROJECT_INFO
            utility_score = self._utility_for_turn(current_turn)
            stored_item, inserted, evicted = self.memory_store.add(
                normalized_action.text,
                memory_type,
                turn_index=self._step_index,
                utility_score=utility_score,
                source_turn=current_turn.turn_id,
                metadata={"turn_kind": current_turn.kind},
                is_summary=normalized_action.type == ActionType.STORE_SUMMARY,
            )
            if inserted:
                reward += 0.10 if normalized_action.type == ActionType.STORE_SUMMARY else 0.15
                reward += 0.03 if evicted else 0.0
            else:
                reward -= 0.08
                note = "duplicate_memory"
        elif normalized_action.type == ActionType.IGNORE:
            if current_turn.kind in {"preference", "constraint", "correction", "project_info"}:
                reward -= 0.10
                note = "ignored_useful_turn"
            elif current_turn.kind == "recall_check":
                # Ignoring a recall check is a mild penalty — agent should retrieve here
                reward -= 0.05
                note = "ignored_recall_check"
            else:
                reward += 0.02
        elif normalized_action.type == ActionType.RETRIEVE:
            if normalized_action.ids:
                retrieved = self.memory_store.retrieve(normalized_action.ids, turn_index=self._step_index)
            else:
                query_text = normalized_action.text or current_turn.text
                retrieved = self.memory_store.query(query_text, k=3)
            retrieved_items = tuple(retrieved)
            if self._retrieval_is_relevant(current_turn, retrieved_items):
                reward += 0.12
            else:
                reward -= 0.12
                note = "irrelevant_retrieval"
        elif normalized_action.type == ActionType.UPDATE:
            target_id = normalized_action.ids[0] if normalized_action.ids else ""
            updated = self.memory_store.update(
                target_id,
                normalized_action.text or current_turn.text,
                turn_index=self._step_index,
                metadata={"updated_from_turn_kind": current_turn.kind},
            )
            if updated is not None and current_turn.kind == "correction":
                reward += 0.08
            elif updated is not None:
                reward += 0.03
            else:
                reward -= 0.05
        elif normalized_action.type == ActionType.DELETE:
            target_id = normalized_action.ids[0] if normalized_action.ids else ""
            deleted = self.memory_store.delete(target_id)
            reward += 0.05 if deleted else -0.03
        elif normalized_action.type == ActionType.ANSWER:
            self._final_answer = normalized_action.text or ""
            if current_turn.kind != "final_query":
                reward -= 0.20
                self._done = True
            else:
                self._done = True
        else:
            raise ValueError(f"Unsupported action type: {normalized_action.type}")

        record = ActionRecord(
            turn_index=self._step_index,
            turn_kind=current_turn.kind,
            user_message=current_turn.text,
            action=normalized_action,
            retrieved_items=retrieved_items,
            stored_item=stored_item,
            note=note,
        )
        self._trace.append(record)

        if not self._done:
            self._step_index += 1
            if self._step_index >= len(self.episode.turns):
                self._done = True

        if self._done and self.current_turn.kind == "final_query" and normalized_action.type == ActionType.ANSWER:
            metrics = self.grader.score_episode(
                self.episode,
                self._trace,
                self._final_answer,
                self.memory_store.snapshot(),
            )
            final_reward = self.reward_composer.compose(metrics)
            reward += final_reward
            info = {
                "metrics": metrics.to_dict(),
                "final_reward": final_reward,
                "final_answer": self._final_answer,
                "memory_items": [item.to_dict() for item in self.memory_store.snapshot()],
            }
        elif self._done:
            metrics = self.grader.score_episode(
                self.episode,
                self._trace,
                self._final_answer,
                self.memory_store.snapshot(),
            )
            final_reward = self.reward_composer.compose(metrics)
            reward += final_reward
            info = {
                "metrics": metrics.to_dict(),
                "final_reward": final_reward,
                "final_answer": self._final_answer,
                "memory_items": [item.to_dict() for item in self.memory_store.snapshot()],
            }
        else:
            info = {
                "metrics": None,
                "final_reward": 0.0,
                "final_answer": self._final_answer,
                "memory_items": [item.to_dict() for item in self.memory_store.snapshot()],
            }

        observation = None if self._done else self._make_observation()
        return StepResult(observation=observation, reward=reward, done=self._done, info=info)

    @property
    def current_turn(self) -> ConversationTurn:
        if self.episode is None:
            raise RuntimeError("Environment has not been reset.")
        index = min(self._step_index, len(self.episode.turns) - 1)
        return self.episode.turns[index]

    def build_episode_result(self) -> EpisodeResult:
        if self.episode is None or self.memory_store is None:
            raise RuntimeError("Environment has not been reset.")
        metrics = self.grader.score_episode(
            self.episode,
            self._trace,
            self._final_answer,
            self.memory_store.snapshot(),
        )
        reward = self.reward_composer.compose(metrics)
        return EpisodeResult(
            episode=self.episode,
            final_answer=self._final_answer,
            metrics=metrics,
            reward=reward,
            trace=self.trace,
        )

    def _normalize_action(self, action: Action | Dict[str, Any]) -> Action:
        if isinstance(action, Action):
            return action
        action_type = ActionType(action["type"])
        text = action.get("text")
        ids = tuple(action.get("ids", ()))
        metadata = dict(action.get("metadata", {}))
        return Action(type=action_type, text=text, ids=ids, metadata=metadata)

    def _make_observation(self) -> Observation:
        if self.episode is None or self.memory_store is None:
            raise RuntimeError("Environment has not been reset.")
        recent_turns = self.episode.turns[max(0, self._step_index - 3) : self._step_index]
        memory_budget_remaining = max(0, self.episode.memory_budget - self.memory_store.total_tokens)
        # Strip answer-key fields so the agent cannot read the grading ground-truth.
        visible_metadata = {
            k: v for k, v in self.episode.metadata.items()
            if k not in _HIDDEN_METADATA_KEYS
        }
        return Observation(
            current_user_message=self.current_turn.text,
            current_turn_kind=self.current_turn.kind,
            recent_conversation=tuple(recent_turns),
            memory_bank=self.memory_store.snapshot(),
            memory_budget_remaining=memory_budget_remaining,
            step_number=self._step_index,
            episode_metadata=visible_metadata,
        )

    def _utility_for_turn(self, turn: ConversationTurn) -> float:
        if turn.kind in {"preference", "constraint", "correction", "project_info", "final_query"}:
            return 0.8
        if turn.kind == "recall_check":
            return 0.5
        return 0.0

    def _retrieval_is_relevant(self, turn: ConversationTurn, retrieved_items: tuple[MemoryItem, ...]) -> bool:
        if not retrieved_items:
            return False
        required_types = {
            MemoryType(value)
            for value in self.episode.metadata.get("required_memory_types", [])
            if value in MemoryType._value2member_map_
        }
        required_keywords = [str(value) for value in self.episode.metadata.get("required_keywords", [])]
        if turn.kind == "final_query":
            if any(item.type in required_types for item in retrieved_items):
                return True
            if required_keywords and any(contains_any(item.text, required_keywords) for item in retrieved_items):
                return True
            return False
        if turn.kind == "recall_check":
            # Relevant if retrieved item matches the turn's memory_type or contains the turn keyword
            turn_keyword = str(turn.metadata.get("keyword", ""))
            if any(item.type == turn.memory_type for item in retrieved_items):
                return True
            if turn_keyword and any(contains_any(item.text, [turn_keyword]) for item in retrieved_items):
                return True
            return False
        if turn.kind in {"preference", "constraint", "correction", "project_info"}:
            return any(item.type == turn.memory_type for item in retrieved_items)
        return False
