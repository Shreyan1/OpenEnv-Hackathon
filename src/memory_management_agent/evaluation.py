from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .agents import BaseAgent
from .environment import MemoryManagementEnv
from .schemas import EpisodeResult


@dataclass
class EvaluationSummary:
    results: tuple[EpisodeResult, ...]

    @property
    def average_reward(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.reward for result in self.results) / len(self.results)

    @property
    def average_success(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.metrics.success for result in self.results) / len(self.results)

    @property
    def average_precision(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.metrics.precision for result in self.results) / len(self.results)

    @property
    def average_recall(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.metrics.recall for result in self.results) / len(self.results)

    def to_dict(self) -> dict[str, object]:
        return {
            "average_reward": self.average_reward,
            "average_success": self.average_success,
            "average_precision": self.average_precision,
            "average_recall": self.average_recall,
            "results": [result.to_dict() for result in self.results],
        }


@dataclass
class BenchmarkReport:
    agent_name: str
    visible: EvaluationSummary
    hidden: EvaluationSummary

    @property
    def generalization_gap(self) -> float:
        return self.visible.average_reward - self.hidden.average_reward

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_name": self.agent_name,
            "visible": self.visible.to_dict(),
            "hidden": self.hidden.to_dict(),
            "generalization_gap": self.generalization_gap,
        }


def run_episode(agent: BaseAgent, env: MemoryManagementEnv, seed: int) -> EpisodeResult:
    observation = env.reset(seed=seed)
    done = False
    while not done:
        action = agent.act(observation)
        result = env.step(action)
        observation = result.observation
        done = result.done
    return env.build_episode_result()


def evaluate_agent(agent: BaseAgent, env: MemoryManagementEnv, seeds: Sequence[int]) -> EvaluationSummary:
    results = tuple(run_episode(agent, env, seed) for seed in seeds)
    return EvaluationSummary(results=results)


def evaluate_split(
    agent: BaseAgent,
    env: MemoryManagementEnv,
    visible_seeds: Sequence[int],
    hidden_seeds: Sequence[int],
) -> BenchmarkReport:
    visible = evaluate_agent(agent, env, visible_seeds)
    hidden = evaluate_agent(agent, env, hidden_seeds)
    agent_name = type(agent).__name__
    return BenchmarkReport(agent_name=agent_name, visible=visible, hidden=hidden)


def hidden_eval_seeds(count: int = 5, *, start: int = 10_000) -> tuple[int, ...]:
    return tuple(start + index for index in range(count))
