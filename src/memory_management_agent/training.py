from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence
import json

from .agents import BaseAgent
from .environment import MemoryManagementEnv
from .evaluation import BenchmarkReport, evaluate_split, hidden_eval_seeds as default_hidden_eval_seeds
from .schemas import Action, EpisodeResult, Observation


class PromptPolicy(Protocol):
    def act(self, observation: Observation) -> Action:
        raise NotImplementedError


@dataclass(frozen=True)
class TrainingConfig:
    algorithm: str = "grpo"
    prompt_style: str = "structured"
    max_prompt_tokens: int = 4096
    checkpoint_dir: str = "checkpoints"
    artifact_dir: str = "artifacts"
    run_name: str = "run"


@dataclass(frozen=True)
class PromptBundle:
    observation_prompt: str
    action_format: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutStep:
    step_index: int
    prompt: str
    action: dict[str, object]
    reward: float
    done: bool
    info: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "prompt": self.prompt,
            "action": self.action,
            "reward": self.reward,
            "done": self.done,
            "info": self.info,
        }


@dataclass(frozen=True)
class RolloutEpisode:
    seed: int
    episode_result: EpisodeResult
    steps: tuple[RolloutStep, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "episode_result": self.episode_result.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class TrainingRunReport:
    config: TrainingConfig
    train_summary: dict[str, float]
    benchmark: BenchmarkReport
    train_rollout_path: str
    benchmark_path: str
    checkpoint_path: str
    manifest_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.__dict__,
            "train_summary": self.train_summary,
            "benchmark": self.benchmark.to_dict(),
            "train_rollout_path": self.train_rollout_path,
            "benchmark_path": self.benchmark_path,
            "checkpoint_path": self.checkpoint_path,
            "manifest_path": self.manifest_path,
        }


def build_policy_prompt(observation: Observation) -> PromptBundle:
    memory_lines = [
        f"- {item.id} | {item.type.value} | {item.text}"
        for item in observation.memory_bank
    ]
    conversation_lines = [
        f"- turn {turn.turn_id} [{turn.kind}]: {turn.text}"
        for turn in observation.recent_conversation
    ]
    recent_conversation_section = ["Recent conversation:"]
    recent_conversation_section.extend(conversation_lines if conversation_lines else ["- none"])
    memory_bank_section = ["Memory bank:"]
    memory_bank_section.extend(memory_lines if memory_lines else ["- empty"])
    observation_prompt = "\n".join(
        [
            "You are a memory management policy.",
            "Choose exactly one action from: STORE, STORE_SUMMARY, IGNORE, RETRIEVE, UPDATE, DELETE, ANSWER.",
            "",
            f"Current turn kind: {observation.current_turn_kind}",
            f"Current user message: {observation.current_user_message}",
            f"Memory budget remaining: {observation.memory_budget_remaining}",
            f"Step number: {observation.step_number}",
            "",
            *recent_conversation_section,
            "",
            *memory_bank_section,
            "",
            "Respond in the following format:",
            "ACTION: <STORE|STORE_SUMMARY|IGNORE|RETRIEVE|UPDATE|DELETE|ANSWER>",
            "TEXT: <optional>",
            "IDS: <comma-separated ids or empty>",
        ]
    )
    action_format = "\n".join(
        [
            "ACTION: STORE|STORE_SUMMARY|IGNORE|RETRIEVE|UPDATE|DELETE|ANSWER",
            "TEXT: optional free-form text",
            "IDS: optional memory ids",
        ]
    )
    return PromptBundle(observation_prompt=observation_prompt, action_format=action_format)


def parse_action_block(text: str) -> Action:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    values: dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip().lower()] = value.strip()

    action_type = values.get("action", "ignore").upper()
    action_text = values.get("text") or None
    ids_raw = values.get("ids", "")
    ids = tuple(part.strip() for part in ids_raw.split(",") if part.strip())

    if action_type == "STORE_SUMMARY":
        return Action.store(action_text or "", summary=True)
    if action_type == "STORE":
        return Action.store(action_text or "")
    if action_type == "IGNORE":
        return Action.ignore()
    if action_type == "RETRIEVE":
        return Action.retrieve(ids=ids, text=action_text)
    if action_type == "UPDATE":
        if not ids:
            return Action.ignore()
        return Action.update(ids[0], action_text or "")
    if action_type == "DELETE":
        if not ids:
            return Action.ignore()
        return Action.delete(ids[0])
    if action_type == "ANSWER":
        return Action.answer(action_text or "")
    return Action.ignore()


def collect_rollouts(agent: BaseAgent, env: MemoryManagementEnv, seeds: Sequence[int]) -> tuple[RolloutEpisode, ...]:
    episodes: list[RolloutEpisode] = []
    for seed in seeds:
        observation = env.reset(seed=seed)
        steps: list[RolloutStep] = []
        done = False
        while not done:
            prompt = build_policy_prompt(observation)
            action = agent.act(observation)
            result = env.step(action)
            steps.append(
                RolloutStep(
                    step_index=observation.step_number,
                    prompt=prompt.observation_prompt,
                    action=action.to_dict(),
                    reward=result.reward,
                    done=result.done,
                    info=dict(result.info),
                )
            )
            observation = result.observation if result.observation is not None else observation
            done = result.done
        episodes.append(
            RolloutEpisode(
                seed=seed,
                episode_result=env.build_episode_result(),
                steps=tuple(steps),
            )
        )
    return tuple(episodes)


def export_rollouts_jsonl(rollouts: Iterable[RolloutEpisode], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for rollout in rollouts:
            handle.write(json.dumps(rollout.to_dict(), ensure_ascii=True))
            handle.write("\n")
    return output_path


def run_training_data_collection(
    agent: BaseAgent,
    env: MemoryManagementEnv,
    seeds: Sequence[int],
    output_path: str | Path,
) -> Path:
    rollouts = collect_rollouts(agent, env, seeds)
    return export_rollouts_jsonl(rollouts, output_path)


def summarize_rollouts(rollouts: Sequence[RolloutEpisode]) -> dict[str, float]:
    if not rollouts:
        return {"average_reward": 0.0, "average_success": 0.0}
    average_reward = sum(rollout.episode_result.reward for rollout in rollouts) / len(rollouts)
    average_success = sum(rollout.episode_result.metrics.success for rollout in rollouts) / len(rollouts)
    return {
        "average_reward": average_reward,
        "average_success": average_success,
    }


def run_training_experiment(
    agent: BaseAgent,
    env: MemoryManagementEnv,
    *,
    train_seeds: Sequence[int],
    visible_eval_seeds: Sequence[int],
    hidden_eval_seeds: Sequence[int] | None = None,
    output_dir: str | Path = ".",
    config: TrainingConfig | None = None,
) -> TrainingRunReport:
    config = config or TrainingConfig()
    hidden_eval_seeds = tuple(hidden_eval_seeds or default_hidden_eval_seeds())

    root_dir = Path(output_dir) / config.artifact_dir / config.run_name
    checkpoint_dir = root_dir / config.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_rollouts = collect_rollouts(agent, env, train_seeds)
    train_summary = summarize_rollouts(train_rollouts)
    benchmark = evaluate_split(agent, env, visible_eval_seeds, hidden_eval_seeds)

    train_rollout_path = export_rollouts_jsonl(train_rollouts, root_dir / "train_rollouts.jsonl")
    benchmark_path = root_dir / "benchmark.json"
    checkpoint_path = checkpoint_dir / "checkpoint.json"
    manifest_path = root_dir / "manifest.json"

    benchmark_path.write_text(json.dumps(benchmark.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")
    checkpoint_payload = {
        "config": config.__dict__,
        "train_summary": train_summary,
        "benchmark": benchmark.to_dict(),
        "note": "checkpoint placeholder for future TRL model state",
    }
    checkpoint_path.write_text(json.dumps(checkpoint_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    manifest_payload = {
        "config": config.__dict__,
        "train_seeds": list(train_seeds),
        "visible_eval_seeds": list(visible_eval_seeds),
        "hidden_eval_seeds": list(hidden_eval_seeds),
        "train_summary": train_summary,
        "benchmark": benchmark.to_dict(),
        "artifacts": {
            "train_rollouts": str(train_rollout_path),
            "benchmark": str(benchmark_path),
            "checkpoint": str(checkpoint_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    return TrainingRunReport(
        config=config,
        train_summary=train_summary,
        benchmark=benchmark,
        train_rollout_path=str(train_rollout_path),
        benchmark_path=str(benchmark_path),
        checkpoint_path=str(checkpoint_path),
        manifest_path=str(manifest_path),
    )
