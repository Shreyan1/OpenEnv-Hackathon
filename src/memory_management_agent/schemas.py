from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class MemoryType(str, Enum):
    PREFERENCE = "preference"
    CONSTRAINT = "constraint"
    PROJECT_INFO = "project_info"


class ActionType(str, Enum):
    STORE = "store"
    STORE_SUMMARY = "store_summary"
    IGNORE = "ignore"
    RETRIEVE = "retrieve"
    UPDATE = "update"
    DELETE = "delete"
    ANSWER = "answer"


@dataclass(frozen=True)
class MemoryItem:
    id: str
    text: str
    type: MemoryType
    created_at: int
    last_used: int
    token_length: int
    utility_score: float = 0.0
    source_turn: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_updates(self, **changes: Any) -> "MemoryItem":
        return replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type.value
        return data


@dataclass(frozen=True)
class ConversationTurn:
    turn_id: int
    text: str
    kind: str
    memory_type: Optional[MemoryType] = None
    tags: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.memory_type is not None:
            data["memory_type"] = self.memory_type.value
        return data


@dataclass(frozen=True)
class Episode:
    episode_id: str
    seed: int
    turns: Tuple[ConversationTurn, ...]
    memory_budget: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "turns": [turn.to_dict() for turn in self.turns],
            "memory_budget": self.memory_budget,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Observation:
    current_user_message: str
    current_turn_kind: str
    recent_conversation: Tuple[ConversationTurn, ...]
    memory_bank: Tuple[MemoryItem, ...]
    memory_budget_remaining: int
    step_number: int
    episode_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_user_message": self.current_user_message,
            "current_turn_kind": self.current_turn_kind,
            "recent_conversation": [turn.to_dict() for turn in self.recent_conversation],
            "memory_bank": [item.to_dict() for item in self.memory_bank],
            "memory_budget_remaining": self.memory_budget_remaining,
            "step_number": self.step_number,
            "episode_metadata": self.episode_metadata,
        }


@dataclass(frozen=True)
class Action:
    type: ActionType
    text: Optional[str] = None
    ids: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def store(cls, text: str, *, summary: bool = False, metadata: Optional[Dict[str, Any]] = None) -> "Action":
        return cls(
            type=ActionType.STORE_SUMMARY if summary else ActionType.STORE,
            text=text,
            metadata=metadata or {},
        )

    @classmethod
    def retrieve(cls, ids: Optional[Tuple[str, ...]] = None, text: Optional[str] = None) -> "Action":
        return cls(type=ActionType.RETRIEVE, text=text, ids=ids or ())

    @classmethod
    def update(cls, memory_id: str, text: str) -> "Action":
        return cls(type=ActionType.UPDATE, text=text, ids=(memory_id,))

    @classmethod
    def delete(cls, memory_id: str) -> "Action":
        return cls(type=ActionType.DELETE, ids=(memory_id,))

    @classmethod
    def answer(cls, text: str) -> "Action":
        return cls(type=ActionType.ANSWER, text=text)

    @classmethod
    def ignore(cls) -> "Action":
        return cls(type=ActionType.IGNORE)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "text": self.text,
            "ids": list(self.ids),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class StepResult:
    observation: Optional[Observation]
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraderMetrics:
    success: float
    precision: float
    recall: float
    compactness: float
    freshness: float
    non_interference: float
    contradiction_penalty: float
    memory_bloat_penalty: float
    useful_store_ratio: float
    useless_store_ratio: float
    retrieval_count: int
    relevant_retrieval_count: int
    total_memory_items: int
    total_memory_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionRecord:
    turn_index: int
    turn_kind: str
    user_message: str
    action: Action
    retrieved_items: Tuple[MemoryItem, ...] = ()
    stored_item: Optional[MemoryItem] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "turn_kind": self.turn_kind,
            "user_message": self.user_message,
            "action": self.action.to_dict(),
            "retrieved_items": [item.to_dict() for item in self.retrieved_items],
            "stored_item": None if self.stored_item is None else self.stored_item.to_dict(),
            "note": self.note,
        }


@dataclass(frozen=True)
class EpisodeResult:
    episode: Episode
    final_answer: str
    metrics: GraderMetrics
    reward: float
    trace: Tuple[ActionRecord, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode": self.episode.to_dict(),
            "final_answer": self.final_answer,
            "metrics": self.metrics.to_dict(),
            "reward": self.reward,
            "trace": [record.to_dict() for record in self.trace],
        }
