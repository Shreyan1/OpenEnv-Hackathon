from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .schemas import Action, ActionType, MemoryType, Observation
from .utils import contains_any, jaccard_similarity, tokenize


class BaseAgent(Protocol):
    def act(self, observation: Observation) -> Action:
        raise NotImplementedError


@dataclass
class NoMemoryAgent:
    def act(self, observation: Observation) -> Action:
        if observation.current_turn_kind == "final_query":
            return Action.answer("I do not have enough memory to answer precisely.")
        return Action.ignore()


@dataclass
class StoreEverythingAgent:
    def act(self, observation: Observation) -> Action:
        if observation.current_turn_kind == "final_query":
            return Action.answer(self._answer(observation))
        return Action.store(observation.current_user_message)

    def _answer(self, observation: Observation) -> str:
        preferred = self._first_memory_text(observation, MemoryType.PREFERENCE)
        constraint = self._first_memory_text(observation, MemoryType.CONSTRAINT)
        parts = ["Here is the answer"]
        if preferred:
            parts.append(f"using {preferred}")
        if constraint:
            parts.append(f"and keeping it {constraint}")
        return " ".join(parts)

    def _first_memory_text(self, observation: Observation, memory_type: MemoryType) -> str:
        for item in observation.memory_bank:
            if item.type == memory_type:
                return item.text
        return ""


@dataclass
class PreferenceOnlyAgent:
    def act(self, observation: Observation) -> Action:
        if observation.current_turn_kind == "final_query":
            return Action.answer(self._answer(observation))
        if observation.current_turn_kind in {"preference", "constraint", "correction", "project_info"}:
            return Action.store(observation.current_user_message)
        return Action.ignore()

    def _answer(self, observation: Observation) -> str:
        text = self._combine_memory(observation)
        return text if text else "I do not have enough memory to answer."

    def _combine_memory(self, observation: Observation) -> str:
        preference = self._first_memory_text(observation, MemoryType.PREFERENCE)
        constraint = self._first_memory_text(observation, MemoryType.CONSTRAINT)
        if preference and constraint:
            return f"Answering concisely with {preference} and {constraint}"
        if preference:
            return f"Answering with {preference}"
        if constraint:
            return f"Answering with {constraint}"
        return ""

    def _first_memory_text(self, observation: Observation, memory_type: MemoryType) -> str:
        for item in observation.memory_bank:
            if item.type == memory_type:
                return item.text
        return ""


@dataclass
class KeywordRetrievalAgent:
    def act(self, observation: Observation) -> Action:
        if observation.current_turn_kind == "final_query":
            return Action.answer(self._answer(observation))
        if observation.current_turn_kind in {"preference", "constraint", "correction", "project_info"}:
            return Action.store(observation.current_user_message)
        return Action.ignore()

    def _answer(self, observation: Observation) -> str:
        relevant = self._select_relevant_memory(observation)
        if relevant is None:
            return "I could not find the relevant memory."
        return f"Using {relevant.text}"

    def _select_relevant_memory(self, observation: Observation):
        query_tokens = set(tokenize(observation.current_user_message))
        scored = []
        for item in observation.memory_bank:
            item_tokens = set(tokenize(item.text))
            keyword_overlap = len(query_tokens & item_tokens)
            if keyword_overlap:
                scored.append((keyword_overlap, item))
        scored.sort(key=lambda pair: (-pair[0], pair[1].created_at, pair[1].id))
        return scored[0][1] if scored else None


@dataclass
class EmbeddingRetrievalAgent:
    def act(self, observation: Observation) -> Action:
        if observation.current_turn_kind == "final_query":
            return Action.answer(self._answer(observation))
        if observation.current_turn_kind in {"preference", "constraint", "correction", "project_info"}:
            return Action.store(observation.current_user_message)
        return Action.ignore()

    def _answer(self, observation: Observation) -> str:
        relevant = self._select_relevant_memory(observation)
        if relevant is None:
            return "I could not find the relevant memory."
        return f"Based on memory: {relevant.text}"

    def _select_relevant_memory(self, observation: Observation):
        scored = []
        for item in observation.memory_bank:
            similarity = jaccard_similarity(observation.current_user_message, item.text)
            if similarity > 0.0:
                scored.append((similarity, item))
        scored.sort(key=lambda pair: (-pair[0], pair[1].created_at, pair[1].id))
        return scored[0][1] if scored else None


@dataclass
class RuleBasedMemoryAgent:
    def act(self, observation: Observation) -> Action:
        if observation.current_turn_kind == "final_query":
            return Action.answer(self._compose_answer(observation))

        if observation.current_turn_kind == "recall_check":
            # Always retrieve when asked to recall something
            return Action.retrieve(text=observation.current_user_message)

        if observation.current_turn_kind in {"preference", "constraint", "correction", "project_info"}:
            return Action.store(observation.current_user_message)

        if self._looks_like_query(observation.current_user_message):
            return Action.retrieve(text=observation.current_user_message)

        return Action.ignore()

    def _looks_like_query(self, text: str) -> bool:
        return contains_any(text, ["remember", "earlier", "based on", "use what i told you", "respond", "answer"])

    def _compose_answer(self, observation: Observation) -> str:
        preference = self._first_memory_text(observation, MemoryType.PREFERENCE)
        constraint = self._first_memory_text(observation, MemoryType.CONSTRAINT)
        project = self._first_memory_text(observation, MemoryType.PROJECT_INFO)
        parts = ["Here is the answer"]
        if preference:
            parts.append(f"with {preference}")
        if constraint:
            parts.append(f"and {constraint}")
        if project:
            parts.append(f"for {project}")
        return " ".join(parts)

    def _first_memory_text(self, observation: Observation, memory_type: MemoryType) -> str:
        for item in observation.memory_bank:
            if item.type == memory_type:
                return item.text
        return ""
