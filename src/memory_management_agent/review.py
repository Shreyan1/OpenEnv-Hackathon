from __future__ import annotations

from typing import Sequence

from .analysis import AnalysisReport, FailureCase, memory_evolution_text
from .training import TrainingRunReport


def render_training_run_summary(report: TrainingRunReport) -> str:
    lines = [
        f"Run: {report.config.run_name}",
        f"Algorithm: {report.config.algorithm}",
        f"Train reward: {report.train_summary.get('average_reward', 0.0):.3f}",
        f"Train success: {report.train_summary.get('average_success', 0.0):.3f}",
        f"Visible reward: {report.benchmark.visible.average_reward:.3f}",
        f"Visible success: {report.benchmark.visible.average_success:.3f}",
        f"Hidden reward: {report.benchmark.hidden.average_reward:.3f}",
        f"Hidden success: {report.benchmark.hidden.average_success:.3f}",
        f"Generalization gap: {report.benchmark.generalization_gap:.3f}",
        f"Manifest: {report.manifest_path}",
    ]
    return "\n".join(lines)


def render_failure_cases(report: AnalysisReport, *, limit: int = 10) -> str:
    lines = [f"Failure cases ({min(limit, len(report.failure_cases))} shown):"]
    for case in report.failure_cases[:limit]:
        lines.append(
            f"- seed={case.seed} reward={case.reward:.3f} success={case.success:.3f} "
            f"categories={', '.join(case.categories) if case.categories else 'none'}"
        )
    if not report.failure_cases:
        lines.append("- none")
    return "\n".join(lines)


def render_memory_evolution(rollouts: Sequence) -> str:
    return "\n\n".join(memory_evolution_text(rollout) for rollout in rollouts)


def render_full_review(report: TrainingRunReport, analysis: AnalysisReport | None = None) -> str:
    sections = [render_training_run_summary(report)]
    if analysis is not None:
        sections.append(
            "\n".join(
                [
                    f"Analysis episodes: {analysis.total_episodes}",
                    f"Average reward: {analysis.average_reward:.3f}",
                    f"Average success: {analysis.average_success:.3f}",
                    f"Average precision: {analysis.average_precision:.3f}",
                    f"Average recall: {analysis.average_recall:.3f}",
                    f"Average memory items: {analysis.average_memory_items:.3f}",
                    f"Average memory tokens: {analysis.average_memory_tokens:.3f}",
                    render_failure_cases(analysis),
                ]
            )
        )
    return "\n\n".join(sections)
