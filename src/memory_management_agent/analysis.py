from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from typing import Iterable, Sequence

from .evaluation import EvaluationSummary
from .training import RolloutEpisode


@dataclass(frozen=True)
class FailureCase:
    seed: int
    reward: float
    success: float
    categories: tuple[str, ...]
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "reward": self.reward,
            "success": self.success,
            "categories": list(self.categories),
            "note": self.note,
        }


@dataclass(frozen=True)
class MemoryTurnSnapshot:
    step_index: int
    memory_items: int
    memory_tokens: int
    useful_store_ratio: float
    useless_store_ratio: float

    def to_dict(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "memory_items": self.memory_items,
            "memory_tokens": self.memory_tokens,
            "useful_store_ratio": self.useful_store_ratio,
            "useless_store_ratio": self.useless_store_ratio,
        }


@dataclass(frozen=True)
class AnalysisReport:
    total_episodes: int
    average_reward: float
    average_success: float
    average_precision: float
    average_recall: float
    average_memory_items: float
    average_memory_tokens: float
    action_counts: dict[str, int]
    failure_cases: tuple[FailureCase, ...]
    memory_evolution: tuple[tuple[MemoryTurnSnapshot, ...], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_episodes": self.total_episodes,
            "average_reward": self.average_reward,
            "average_success": self.average_success,
            "average_precision": self.average_precision,
            "average_recall": self.average_recall,
            "average_memory_items": self.average_memory_items,
            "average_memory_tokens": self.average_memory_tokens,
            "action_counts": self.action_counts,
            "failure_cases": [case.to_dict() for case in self.failure_cases],
            "memory_evolution": [
                [snapshot.to_dict() for snapshot in rollout]
                for rollout in self.memory_evolution
            ],
        }


def analyze_rollouts(rollouts: Sequence[RolloutEpisode]) -> AnalysisReport:
    if not rollouts:
        return AnalysisReport(
            total_episodes=0,
            average_reward=0.0,
            average_success=0.0,
            average_precision=0.0,
            average_recall=0.0,
            average_memory_items=0.0,
            average_memory_tokens=0.0,
            action_counts={},
            failure_cases=(),
            memory_evolution=(),
        )

    rewards = [rollout.episode_result.reward for rollout in rollouts]
    success_values = [rollout.episode_result.metrics.success for rollout in rollouts]
    precision_values = [rollout.episode_result.metrics.precision for rollout in rollouts]
    recall_values = [rollout.episode_result.metrics.recall for rollout in rollouts]
    memory_items_values = [rollout.episode_result.metrics.total_memory_items for rollout in rollouts]
    memory_tokens_values = [rollout.episode_result.metrics.total_memory_tokens for rollout in rollouts]

    action_counter: Counter[str] = Counter()
    failure_cases: list[FailureCase] = []
    memory_evolution: list[tuple[MemoryTurnSnapshot, ...]] = []

    for rollout in rollouts:
        snapshots: list[MemoryTurnSnapshot] = []
        for step in rollout.steps:
            memory_items = step.info.get("memory_items", [])
            useful_ratio = 0.0
            useless_ratio = 0.0
            metrics = step.info.get("metrics")
            if isinstance(metrics, dict):
                useful_ratio = float(metrics.get("useful_store_ratio", 0.0))
                useless_ratio = float(metrics.get("useless_store_ratio", 0.0))
            snapshots.append(
                MemoryTurnSnapshot(
                    step_index=step.step_index,
                    memory_items=len(memory_items),
                    memory_tokens=sum(int(item.get("token_length", 0)) for item in memory_items),
                    useful_store_ratio=useful_ratio,
                    useless_store_ratio=useless_ratio,
                )
            )
            action_counter[str(step.action.get("type", "unknown"))] += 1
        memory_evolution.append(tuple(snapshots))

        categories = _categorize_failure(rollout)
        if categories:
            failure_cases.append(
                FailureCase(
                    seed=rollout.seed,
                    reward=rollout.episode_result.reward,
                    success=rollout.episode_result.metrics.success,
                    categories=tuple(categories),
                )
            )

    total = len(rollouts)
    return AnalysisReport(
        total_episodes=total,
        average_reward=sum(rewards) / total,
        average_success=sum(success_values) / total,
        average_precision=sum(precision_values) / total,
        average_recall=sum(recall_values) / total,
        average_memory_items=sum(memory_items_values) / total,
        average_memory_tokens=sum(memory_tokens_values) / total,
        action_counts=dict(sorted(action_counter.items())),
        failure_cases=tuple(failure_cases),
        memory_evolution=tuple(memory_evolution),
    )


def _categorize_failure(rollout: RolloutEpisode) -> list[str]:
    metrics = rollout.episode_result.metrics
    categories: list[str] = []
    if metrics.success <= 0.0:
        categories.append("answer_mismatch")
    if metrics.recall < 0.5:
        categories.append("retrieval_gap")
    if metrics.precision < 0.5 and metrics.retrieval_count > 0:
        categories.append("low_retrieval_precision")
    if metrics.useful_store_ratio < 0.5:
        categories.append("memory_noise")
    if metrics.memory_bloat_penalty > 0.0:
        categories.append("memory_bloat")
    if metrics.contradiction_penalty > 0.0:
        categories.append("stale_memory")
    return categories


def memory_evolution_text(rollout: RolloutEpisode) -> str:
    lines = [f"Seed {rollout.seed} memory evolution:"]
    for step in rollout.steps:
        memory_items = step.info.get("memory_items", [])
        memory_tokens = sum(int(item.get("token_length", 0)) for item in memory_items)
        bar = "#" * min(40, len(memory_items))
        lines.append(
            f"step {step.step_index}: items={len(memory_items):2d} tokens={memory_tokens:3d} {bar}"
        )
    return "\n".join(lines)


def summarize_memory_evolution(rollouts: Sequence[RolloutEpisode]) -> list[str]:
    return [memory_evolution_text(rollout) for rollout in rollouts]
