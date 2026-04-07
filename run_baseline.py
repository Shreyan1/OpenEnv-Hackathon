#!/usr/bin/env python3
"""
Baseline inference script for the Memory Management RL environment.

Runs all built-in baseline agents against all 3 tasks, prints per-task scores,
and exits 0 on success. Designed to run headlessly — no interactive input required.

Usage:
    python run_baseline.py
    python run_baseline.py --seeds 42 43 44 45 46
    python run_baseline.py --task easy_preference_recall
"""
from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.memory_management_agent import (
    ALL_TASKS,
    TASK_BY_ID,
    NoMemoryAgent,
    StoreEverythingAgent,
    PreferenceOnlyAgent,
    KeywordRetrievalAgent,
    RuleBasedMemoryAgent,
    normalize_task_score,
    run_episode,
)
from src.memory_management_agent.environment import MemoryManagementEnv
from src.memory_management_agent.tasks import generator_for_task


DEFAULT_SEEDS = list(range(42, 52))   # 10 seeds


def make_env(task_id: str) -> MemoryManagementEnv:
    task = TASK_BY_ID[task_id]
    gen = generator_for_task(task)
    return MemoryManagementEnv(
        generator=gen,
        memory_budget=task.memory_budget,
        max_turns=task.max_turns,
    )


def run_agent_on_task(agent, task_id: str, seeds: list[int]) -> dict:
    env = make_env(task_id)
    scores = []
    for seed in seeds:
        result = run_episode(agent, env, seed=seed)
        score = normalize_task_score(result.reward)
        scores.append(score)
    return {
        "scores": [round(s, 4) for s in scores],
        "average": round(sum(scores) / len(scores), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Memory Management baseline evaluation")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="RNG seeds to evaluate on (default: 42–51)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Evaluate only this task_id (default: all tasks)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON instead of human-readable text",
    )
    args = parser.parse_args()

    seeds = args.seeds
    tasks = [TASK_BY_ID[args.task]] if args.task else ALL_TASKS

    if args.task and args.task not in TASK_BY_ID:
        print(f"ERROR: unknown task_id {args.task!r}. Valid: {list(TASK_BY_ID)}", file=sys.stderr)
        return 1

    agents = {
        "no_memory":       NoMemoryAgent(),
        "store_everything": StoreEverythingAgent(),
        "preference_only": PreferenceOnlyAgent(),
        "keyword_retrieval": KeywordRetrievalAgent(),
        "rule_based":      RuleBasedMemoryAgent(),
    }

    all_results: dict = {}

    for task in tasks:
        task_results: dict = {}
        for agent_name, agent in agents.items():
            stats = run_agent_on_task(agent, task.task_id, seeds)
            task_results[agent_name] = stats

        all_results[task.task_id] = {
            "task_name": task.name,
            "difficulty": task.difficulty,
            "agents": task_results,
        }

    if args.json:
        print(json.dumps(all_results, indent=2))
        return 0

    # Human-readable output
    print("=" * 70)
    print("Memory Management RL Environment — Baseline Evaluation")
    print(f"Seeds: {seeds}")
    print("=" * 70)

    for task in tasks:
        tr = all_results[task.task_id]
        print(f"\nTask: {tr['task_name']}  [{tr['difficulty'].upper()}]")
        print(f"  task_id: {task.task_id}")
        print(f"  memory_budget: {task.memory_budget} tokens  |  max_turns: {task.max_turns}")
        print()
        print(f"  {'Agent':<22} {'Avg':>6}  {'Min':>6}  {'Max':>6}")
        print(f"  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*6}")
        for agent_name, stats in tr["agents"].items():
            print(
                f"  {agent_name:<22} {stats['average']:>6.3f}  "
                f"{stats['min']:>6.3f}  {stats['max']:>6.3f}"
            )

    print("\n" + "=" * 70)
    print("Baseline complete. Exit 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
