from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .schemas import Action, MemoryType, Observation
from .utils import contains_any, jaccard_similarity, tokenize


_FINAL_QUERY_HINTS = (
    "final answer",
    "final response",
    "write the final",
    "draft the final",
    "wrap this up",
)
_RECALL_HINTS = ("remind me", "what did i say", "what stack", "what format", "what project context")
_CONSTRAINT_HINTS = (
    "bullet points",
    "numbered list",
    "five sentences",
    "concise",
    "valid json",
    "code example",
    "type annotations",
    "snake_case",
)
_CONFABULATION_HINTS = (
    "my colleague",
    "hypothetically",
    "the old team",
    "someone on another team",
    "not actually",
    "not relevant",
    "not asking you to use",
    "didn't adopt",
)
_CORRECTION_HINTS = ("actually", "correction", "update from the team", "change of plan", "scratch the", "swap out")
_PROJECT_HINTS = ("project", "evaluation", "memory budget", "latency", "production", "multi-tenant", "ci")


def _looks_like_final_query(observation: Observation) -> bool:
    return observation.current_turn_kind == "final_query" or contains_any(observation.current_user_message, _FINAL_QUERY_HINTS)


def _looks_like_recall_check(observation: Observation) -> bool:
    return observation.current_turn_kind == "recall_check" or contains_any(observation.current_user_message, _RECALL_HINTS)


def _looks_like_confabulation(text: str) -> bool:
    return contains_any(text, _CONFABULATION_HINTS)


def _looks_like_constraint(text: str) -> bool:
    return contains_any(text, _CONSTRAINT_HINTS)


def _looks_like_correction(text: str) -> bool:
    return contains_any(text, _CORRECTION_HINTS)


def _looks_like_project_info(text: str) -> bool:
    return contains_any(text, _PROJECT_HINTS)


def _looks_like_store_candidate(text: str) -> bool:
    if _looks_like_confabulation(text):
        return False
    return _looks_like_constraint(text) or _looks_like_correction(text) or _looks_like_project_info(text) or contains_any(
        text,
        ("let's", "we're", "we are", "prefer", "target", "use ", "stack", "shop"),
    )


class BaseAgent(Protocol):
    def act(self, observation: Observation) -> Action:
        raise NotImplementedError


@dataclass
class NoMemoryAgent:
    def act(self, observation: Observation) -> Action:
        if _looks_like_final_query(observation):
            return Action.answer("I do not have enough memory to answer precisely.")
        return Action.ignore()


@dataclass
class StoreEverythingAgent:
    def act(self, observation: Observation) -> Action:
        if _looks_like_final_query(observation):
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
        if _looks_like_final_query(observation):
            return Action.answer(self._answer(observation))
        if observation.current_turn_kind in {"preference", "constraint", "correction", "project_info"}:
            return Action.store(observation.current_user_message)
        if observation.current_turn_kind == "unknown" and _looks_like_store_candidate(observation.current_user_message):
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
        if _looks_like_final_query(observation):
            return Action.answer(self._answer(observation))
        if observation.current_turn_kind in {"preference", "constraint", "correction", "project_info"}:
            return Action.store(observation.current_user_message)
        if observation.current_turn_kind == "unknown" and _looks_like_store_candidate(observation.current_user_message):
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
        if _looks_like_final_query(observation):
            return Action.answer(self._answer(observation))
        if observation.current_turn_kind in {"preference", "constraint", "correction", "project_info"}:
            return Action.store(observation.current_user_message)
        if observation.current_turn_kind == "unknown" and _looks_like_store_candidate(observation.current_user_message):
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
        if _looks_like_final_query(observation):
            return Action.answer(self._compose_answer(observation))

        if _looks_like_recall_check(observation):
            # Always retrieve when asked to recall something
            return Action.retrieve(text=observation.current_user_message)

        if observation.current_turn_kind in {"preference", "constraint", "correction", "project_info"}:
            return Action.store(observation.current_user_message)

        if observation.current_turn_kind == "unknown":
            if _looks_like_store_candidate(observation.current_user_message):
                return Action.store(observation.current_user_message)
            if self._looks_like_query(observation.current_user_message):
                return Action.retrieve(text=observation.current_user_message)
            return Action.ignore()

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
