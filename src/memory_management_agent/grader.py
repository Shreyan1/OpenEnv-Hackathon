from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from .schemas import ActionRecord, Episode, GraderMetrics, MemoryItem, MemoryType
from .utils import token_count, token_set

STRICT_SCORE_MIN = 0.0001
STRICT_SCORE_MAX = 0.9999


def normalize_task_score(score: float) -> float:
    """Clamp externally reported task scores into the strict open interval (0, 1)."""
    return min(STRICT_SCORE_MAX, max(STRICT_SCORE_MIN, score))


def _is_valid_json(answer: str) -> bool:
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, (dict, list))


def _sentence_count(answer: str) -> int:
    sentences = [part.strip() for part in re.split(r"[.!?]+", answer.strip()) if part.strip()]
    return len(sentences)


CONSTRAINT_FORMAT_CHECKERS: dict[str, Callable[[str], bool]] = {
    "bullet points": lambda answer: sum(
        1 for line in answer.splitlines() if line.strip().startswith(("-", "*", "•"))
    ) >= 2,
    "numbered list": lambda answer: sum(
        1 for line in answer.splitlines() if re.match(r"^\d+[\.\)]", line.strip())
    ) >= 2,
    "five sentences": lambda answer: 1 <= _sentence_count(answer) <= 5,
    "concise": lambda answer: token_count(answer) <= 50,
    "valid json": _is_valid_json,
    "code examples": lambda answer: "```" in answer or bool(re.search(r"^\s{4,}\S", answer, re.MULTILINE)),
    "type annotations": lambda answer: "->" in answer or bool(re.search(r"\b\w+\s*:\s*\w+", answer)),
    "snake_case": lambda answer: bool(re.search(r"\b[a-z]+(?:_[a-z0-9]+)+\b", answer)),
}


@dataclass
class RewardComposer:
    success_weight: float = 0.40
    precision_weight: float = 0.18
    recall_weight: float = 0.12
    adherence_weight: float = 0.10
    compactness_weight: float = 0.05
    freshness_weight: float = 0.05
    non_interference_weight: float = 0.10
    contradiction_penalty_weight: float = 0.20
    memory_bloat_penalty_weight: float = 0.10

    def compose(self, metrics: GraderMetrics) -> float:
        base = (
            self.success_weight * metrics.success
            + self.precision_weight * metrics.precision
            + self.recall_weight * metrics.recall
            + self.adherence_weight * metrics.constraint_adherence
            + self.compactness_weight * metrics.compactness
            + self.freshness_weight * metrics.freshness
            + self.non_interference_weight * metrics.non_interference
        )
        penalties = (
            self.contradiction_penalty_weight * metrics.contradiction_penalty
            + self.memory_bloat_penalty_weight * metrics.memory_bloat_penalty
        )
        return normalize_task_score(base - penalties)


class Grader:
    def score_episode(
        self,
        episode: Episode,
        trace: Sequence[ActionRecord],
        final_answer: str,
        memory_items: Sequence[MemoryItem],
    ) -> GraderMetrics:
        required_types = {
            MemoryType(value)
            for value in episode.metadata.get("required_memory_types", [])
            if value in MemoryType._value2member_map_
        }
        required_keywords = [str(value) for value in episode.metadata.get("required_keywords", []) if str(value)]
        final_tokens = token_set(final_answer)

        matched_keywords = sum(
            1
            for keyword in required_keywords
            if keyword.lower() in final_answer.lower() or keyword.lower() in final_tokens
        )
        if not required_keywords:
            raw_success = 0.0
        elif matched_keywords == len(required_keywords):
            raw_success = 1.0
        elif matched_keywords > 0:
            raw_success = matched_keywords / len(required_keywords)
        else:
            raw_success = 0.0

        answer_length_factor = min(1.0, token_count(final_answer) / 8.0) if final_answer.strip() else 0.0
        success = raw_success * answer_length_factor

        retrieval_records = [record for record in trace if record.action.type.value == "retrieve"]
        relevant_retrieval_count = 0
        retrieved_relevance_keys: set[str] = set()
        for record in retrieval_records:
            matched = False
            for item in record.retrieved_items:
                if item.type in required_types:
                    retrieved_relevance_keys.add(item.type.value)
                    matched = True
                for keyword in required_keywords:
                    if keyword and keyword.lower() in item.text.lower():
                        retrieved_relevance_keys.add(keyword.lower())
                        matched = True
            if matched:
                relevant_retrieval_count += 1

        retrieval_count = len(retrieval_records)
        precision = relevant_retrieval_count / retrieval_count if retrieval_count else 0.0
        recall_goal_keys = {memory_type.value for memory_type in required_types}
        recall_goal_keys.update(keyword.lower() for keyword in required_keywords)
        recall = len(retrieved_relevance_keys & recall_goal_keys) / max(1, len(recall_goal_keys))

        useful_store_count = 0
        useless_store_count = 0
        correction_turns = 0
        for record in trace:
            if record.action.type.value in {"store", "store_summary"}:
                if record.turn_kind in {"preference", "constraint", "correction", "project_info"}:
                    useful_store_count += 1
                else:
                    useless_store_count += 1
            if record.turn_kind == "correction":
                correction_turns += 1

        useful_store_total = useful_store_count + useless_store_count
        useful_store_ratio = useful_store_count / useful_store_total if useful_store_total else 0.0
        useless_store_ratio = useless_store_count / useful_store_total if useful_store_total else 0.0

        memory_usage_ratio = sum(item.token_length for item in memory_items) / max(1, episode.memory_budget)
        compactness = max(0.0, min(1.0, 1.0 - memory_usage_ratio))
        memory_bloat_penalty = max(0.0, memory_usage_ratio - 0.8)

        latest_keywords = {
            str(episode.metadata.get("latest_preference_keyword", "")).lower(),
            str(episode.metadata.get("latest_constraint_keyword", "")).lower(),
            str(episode.metadata.get("latest_project_keyword", "")).lower(),
        }
        latest_keywords.discard("")
        corrected_matches = 0
        for keyword in latest_keywords:
            if any(keyword in item.text.lower() for item in memory_items):
                corrected_matches += 1
        freshness = corrected_matches / max(1, len(latest_keywords)) if latest_keywords else 1.0
        contradiction_penalty = max(0.0, 1.0 - freshness) if correction_turns else 0.0

        constraint_keyword = str(episode.metadata.get("latest_constraint_keyword", "")).lower()
        checker = CONSTRAINT_FORMAT_CHECKERS.get(constraint_keyword)
        if not constraint_keyword:
            constraint_adherence = 0.5
        elif checker is None:
            constraint_adherence = 0.5
        else:
            constraint_adherence = 1.0 if checker(final_answer) else 0.0

        non_interference = max(0.0, min(1.0, 1.0 - useless_store_ratio))

        return GraderMetrics(
            success=max(0.0, min(1.0, success)),
            precision=max(0.0, min(1.0, precision)),
            recall=max(0.0, min(1.0, recall)),
            constraint_adherence=max(0.0, min(1.0, constraint_adherence)),
            compactness=max(0.0, min(1.0, compactness)),
            freshness=max(0.0, min(1.0, freshness)),
            non_interference=non_interference,
            contradiction_penalty=max(0.0, min(1.0, contradiction_penalty)),
            memory_bloat_penalty=max(0.0, min(1.0, memory_bloat_penalty)),
            useful_store_ratio=max(0.0, min(1.0, useful_store_ratio)),
            useless_store_ratio=max(0.0, min(1.0, useless_store_ratio)),
            retrieval_count=retrieval_count,
            relevant_retrieval_count=relevant_retrieval_count,
            total_memory_items=len(memory_items),
            total_memory_tokens=sum(item.token_length for item in memory_items),
        )
