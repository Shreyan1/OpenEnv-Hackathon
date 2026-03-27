from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List, Optional, Sequence, Tuple

from .schemas import MemoryItem, MemoryType
from .utils import jaccard_similarity, normalize_text, token_count


class MemoryStore:
    def __init__(self, budget_tokens: int = 200):
        self.budget_tokens = budget_tokens
        self._items: dict[str, MemoryItem] = {}
        self._counter = 0

    @property
    def items(self) -> Tuple[MemoryItem, ...]:
        return tuple(sorted(self._items.values(), key=lambda item: (item.created_at, item.id)))

    @property
    def total_tokens(self) -> int:
        return sum(item.token_length for item in self._items.values())

    def snapshot(self) -> Tuple[MemoryItem, ...]:
        return self.items

    def _next_id(self) -> str:
        self._counter += 1
        return f"mem_{self._counter:04d}"

    def _find_duplicate_id(self, text: str) -> Optional[str]:
        normalized = normalize_text(text)
        for item in self._items.values():
            if normalize_text(item.text) == normalized:
                return item.id
        return None

    def _coerce_memory_type(self, memory_type: MemoryType | str) -> MemoryType:
        if isinstance(memory_type, MemoryType):
            return memory_type
        return MemoryType(memory_type)

    def add(
        self,
        text: str,
        memory_type: MemoryType | str,
        *,
        turn_index: int,
        utility_score: float = 0.0,
        source_turn: int = 0,
        metadata: Optional[dict] = None,
        is_summary: bool = False,
    ) -> tuple[MemoryItem, bool, bool]:
        memory_type = self._coerce_memory_type(memory_type)
        duplicate_id = self._find_duplicate_id(text)
        if duplicate_id is not None:
            item = self._items[duplicate_id]
            updated = replace(
                item,
                last_used=turn_index,
                utility_score=max(item.utility_score, utility_score),
            )
            self._items[duplicate_id] = updated
            return updated, False, True

        item = MemoryItem(
            id=self._next_id(),
            text=text,
            type=memory_type,
            created_at=turn_index,
            last_used=turn_index,
            token_length=max(1, token_count(text) // (2 if is_summary else 1)),
            utility_score=utility_score,
            source_turn=source_turn,
            metadata=metadata or {},
        )
        self._items[item.id] = item
        evicted = self._enforce_budget()
        return self._items[item.id], True, evicted

    def update(
        self,
        memory_id: str,
        text: str,
        *,
        turn_index: int,
        utility_score: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[MemoryItem]:
        item = self._items.get(memory_id)
        if item is None:
            return None
        updated = replace(
            item,
            text=text,
            last_used=turn_index,
            token_length=token_count(text),
            utility_score=item.utility_score if utility_score is None else utility_score,
            metadata={**item.metadata, **(metadata or {})},
        )
        self._items[memory_id] = updated
        self._enforce_budget()
        return updated

    def delete(self, memory_id: str) -> bool:
        return self._items.pop(memory_id, None) is not None

    def retrieve(self, ids: Sequence[str], *, turn_index: int) -> list[MemoryItem]:
        retrieved: list[MemoryItem] = []
        for memory_id in ids:
            item = self._items.get(memory_id)
            if item is None:
                continue
            updated = replace(item, last_used=turn_index)
            self._items[memory_id] = updated
            retrieved.append(updated)
        return retrieved

    def query(
        self,
        query_text: str,
        *,
        k: int = 3,
        allowed_types: Optional[Iterable[MemoryType]] = None,
    ) -> list[MemoryItem]:
        allowed = set(allowed_types) if allowed_types is not None else None

        scored: list[tuple[float, MemoryItem]] = []
        for item in self._items.values():
            if allowed is not None and item.type not in allowed:
                continue
            similarity = jaccard_similarity(query_text, item.text)
            recency_bonus = 0.01 * item.last_used
            utility_bonus = 0.05 * item.utility_score
            score = similarity + recency_bonus + utility_bonus
            scored.append((score, item))

        scored.sort(key=lambda pair: (-pair[0], pair[1].created_at, pair[1].id))
        return [item for _, item in scored[:k]]

    def has_text(self, text: str) -> bool:
        return self._find_duplicate_id(text) is not None

    def _enforce_budget(self) -> list[str]:
        evicted: list[str] = []
        while self.total_tokens > self.budget_tokens and self._items:
            victim = min(
                self._items.values(),
                key=lambda item: (item.utility_score, item.last_used, item.created_at, item.id),
            )
            evicted.append(victim.id)
            del self._items[victim.id]
        return evicted
