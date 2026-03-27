from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .schemas import ActionRecord, Episode, GraderMetrics, MemoryItem, MemoryType
from .utils import contains_any, token_set


@dataclass
class RewardComposer:
    success_weight: float = 0.45
    precision_weight: float = 0.20
    recall_weight: float = 0.15
    compactness_weight: float = 0.10
    freshness_weight: float = 0.10
    contradiction_penalty_weight: float = 0.25
    memory_bloat_penalty_weight: float = 0.15
    non_interference_penalty_weight: float = 0.10

    def compose(self, metrics: GraderMetrics) -> float:
        base = (
            self.success_weight * metrics.success
            + self.precision_weight * metrics.precision
            + self.recall_weight * metrics.recall
            + self.compactness_weight * metrics.compactness
            + self.freshness_weight * metrics.freshness
        )
        penalties = (
            self.contradiction_penalty_weight * metrics.contradiction_penalty
            + self.memory_bloat_penalty_weight * metrics.memory_bloat_penalty
            + self.non_interference_penalty_weight * (1.0 - metrics.non_interference)
        )
        return max(0.0, min(1.0, base - penalties))


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
        required_keywords = [str(value) for value in episode.metadata.get("required_keywords", [])]
        final_tokens = token_set(final_answer)

        matched_keywords = sum(
            1 for keyword in required_keywords if keyword.lower() in final_answer.lower() or keyword.lower() in final_tokens
        )
        if not required_keywords:
            raw_success = 0.0
        elif matched_keywords == len(required_keywords):
            raw_success = 1.0
        elif matched_keywords > 0:
            raw_success = 0.5
        else:
            raw_success = 0.0

        # Penalise trivially short answers (bare keyword dumps).
        # A meaningful answer needs at least 6 tokens; below that, success is scaled down.
        _MIN_ANSWER_TOKENS = 6
        answer_token_count = len(final_tokens)
        length_factor = min(1.0, answer_token_count / _MIN_ANSWER_TOKENS)
        success = raw_success * length_factor

        retrieval_records = [record for record in trace if record.action.type.value == "retrieve"]
        relevant_retrieval_count = 0
        for record in retrieval_records:
            if any(item.type in required_types for item in record.retrieved_items):
                relevant_retrieval_count += 1
            elif required_keywords and any(
                contains_any(item.text, required_keywords) for item in record.retrieved_items
            ):
                relevant_retrieval_count += 1

        retrieval_count = len(retrieval_records)
        precision = relevant_retrieval_count / retrieval_count if retrieval_count else 0.0
        # Denominator = number of distinct required memory categories (types + any extra keyword slots)
        recall_denominator = max(len(required_types), len(required_keywords), 1)
        recall = relevant_retrieval_count / recall_denominator

        useful_store_count = 0
        useless_store_count = 0
        correction_turns = 0
        corrected_matches = 0
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
        for item in memory_items:
            if any(keyword in item.text.lower() for keyword in latest_keywords):
                corrected_matches += 1
        freshness = corrected_matches / max(1, len(latest_keywords))
        contradiction_penalty = max(0.0, 1.0 - freshness) if correction_turns else 0.0
        non_interference = 1.0 - useless_store_ratio

        return GraderMetrics(
            success=success,
            precision=max(0.0, min(1.0, precision)),
            recall=max(0.0, min(1.0, recall)),
            compactness=compactness,
            freshness=max(0.0, min(1.0, freshness)),
            non_interference=max(0.0, min(1.0, non_interference)),
            contradiction_penalty=max(0.0, min(1.0, contradiction_penalty)),
            memory_bloat_penalty=max(0.0, min(1.0, memory_bloat_penalty)),
            useful_store_ratio=max(0.0, min(1.0, useful_store_ratio)),
            useless_store_ratio=max(0.0, min(1.0, useless_store_ratio)),
            retrieval_count=retrieval_count,
            relevant_retrieval_count=relevant_retrieval_count,
            total_memory_items=len(memory_items),
            total_memory_tokens=sum(item.token_length for item in memory_items),
        )
