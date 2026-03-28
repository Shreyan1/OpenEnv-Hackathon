#!/usr/bin/env python3
"""
Run an LLM agent against the Memory Management RL Environment via OpenEnv WebSocket.

Providers:
    anthropic   — Anthropic API directly (requires: uv pip install anthropic)
    openrouter  — OpenRouter (OpenAI-compatible, 100+ models)

Usage:
    # Anthropic (default)
    ANTHROPIC_API_KEY=sk-ant-... python run_llm_agent.py

    # OpenRouter
    OPENROUTER_API_KEY=sk-or-... python run_llm_agent.py --provider openrouter --model anthropic/claude-haiku-4-5

    # Other options
    python run_llm_agent.py --server https://chiragsehra-memory-mgmt-rl.hf.space
    python run_llm_agent.py --server http://localhost:7860 --seeds 42 43 44
    python run_llm_agent.py --task easy_preference_recall --seeds 42
    python run_llm_agent.py --json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.request
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openenv.core.generic_client import GenericEnvClient
from openenv.core.client_types import StepResult
from openenv.core.llm_client import create_llm_client

from src.memory_management_agent.training import parse_action_block
from src.memory_management_agent.tasks import ALL_TASKS, TASK_BY_ID


DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_SERVER = "http://localhost:7860"


# ---------------------------------------------------------------------------
# OpenRouter client (OpenAI-compatible, but needs /api/v1 path)
# ---------------------------------------------------------------------------

class _OpenRouterClient:
    """Thin async wrapper around AsyncOpenAI pointed at OpenRouter."""

    def __init__(self, model: str, api_key: str) -> None:
        from openai import AsyncOpenAI
        self.model = model
        self._client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    async def complete(self, prompt: str, max_tokens: int = 256, **kwargs: Any) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=kwargs.get("temperature", 0.0),
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Extended client that captures score/metrics from the done=True payload
# ---------------------------------------------------------------------------

class _MemoryEnvClient(GenericEnvClient):
    """GenericEnvClient subclass that also captures score/metrics on episode end."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_score: float = 0.0
        self.last_metrics: Dict[str, Any] = {}
        self.last_final_answer: str = ""

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[Dict[str, Any]]:
        result = super()._parse_result(payload)
        if payload.get("done"):
            self.last_score = float(payload.get("score", 0.0))
            self.last_metrics = payload.get("metrics", {})
            self.last_final_answer = payload.get("final_answer", "")
        return result


# ---------------------------------------------------------------------------
# Prompt building (dict-based, mirrors build_policy_prompt from training.py)
# ---------------------------------------------------------------------------

def _build_prompt_from_dict(obs: Dict[str, Any]) -> str:
    memory_lines = [
        f"- {m['id']} | {m['type']} | {m['text']}"
        for m in obs.get("memory_bank", [])
    ]
    conv_lines = [
        f"- turn {t['turn_id']} [{t['kind']}]: {t['text']}"
        for t in obs.get("recent_conversation", [])
    ]
    return "\n".join([
        "You are a memory management policy for a multi-turn conversation assistant.",
        "Your goal: store the right information so the final answer is accurate, fresh, and format-compliant.",
        "",
        "## Action rules (follow strictly)",
        "",
        "Turn kind    → Action",
        "preference   → STORE the preference verbatim as TEXT.",
        "constraint   → STORE the formatting/output constraint verbatim as TEXT.",
        "project_info → STORE the project fact verbatim as TEXT.",
        "correction   → If a stored memory is now stale/wrong: UPDATE it (IDS=<old id>, TEXT=<new value>).",
        "               If no matching memory exists yet: STORE the corrected value.",
        "recall_check → RETRIEVE the relevant memory by ID so it stays fresh (IDS=<id>).",
        "               If no matching memory exists: STORE the answer from context.",
        "distractor   → IGNORE. Do not store speculative, third-party, or hypothetical mentions.",
        "confabulation→ IGNORE. Do not store anything the user attributes to others or poses as hypothetical.",
        "final_query  → ANSWER. Compose the final answer from your memory bank.",
        "               The answer MUST include every stored preference, constraint, and project fact.",
        "               The answer MUST obey any stored formatting constraints (bullets, JSON, etc.).",
        "unknown      → Infer intent from the message text and recent conversation context,",
        "               then apply the matching rule above.",
        "",
        "## Critical",
        "- NEVER send ANSWER unless the current turn kind is 'final_query' (or clearly a final question).",
        "- NEVER store distractors, confabulations, or speculative/third-party mentions.",
        "- When correcting: always UPDATE an existing memory rather than storing a duplicate.",
        "- Memory budget is limited — prefer STORE_SUMMARY for long texts.",
        "",
        f"Current turn kind: {obs['current_turn_kind']}",
        f"Current user message: {obs['current_user_message']}",
        f"Memory budget remaining: {obs['memory_budget_remaining']}",
        f"Step number: {obs['step_number']}",
        "",
        "Recent conversation:",
        *(conv_lines if conv_lines else ["- none"]),
        "",
        "Memory bank (id | type | text):",
        *(memory_lines if memory_lines else ["- empty"]),
        "",
        "Respond in the following format (no extra text):",
        "ACTION: <STORE|STORE_SUMMARY|IGNORE|RETRIEVE|UPDATE|DELETE|ANSWER>",
        "TEXT: <required for STORE/STORE_SUMMARY/UPDATE/ANSWER, else empty>",
        "IDS: <comma-separated memory ids for RETRIEVE/UPDATE/DELETE, else empty>",
    ])


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_episode(
    server_url: str,
    llm: Any,
    task_id: str,
    seed: int,
) -> Dict[str, Any]:
    """Run one episode via WebSocket, return score and metadata."""
    env = _MemoryEnvClient(base_url=server_url, message_timeout_s=120.0)
    async with env:
        result = await env.reset(task_id=task_id, seed=seed)
        obs = result.observation or {}
        steps = 0

        while not result.done:
            prompt = _build_prompt_from_dict(obs)
            try:
                llm_text = await llm.complete(prompt, max_tokens=256)
            except Exception as exc:
                llm_text = f"ACTION: ignore\nTEXT: (error: {exc})\nIDS:"

            action_dict = parse_action_block(llm_text).to_dict()
            result = await env.step(action_dict)
            obs = result.observation or obs
            steps += 1

        return {
            "task_id": task_id,
            "seed": seed,
            "score": env.last_score,
            "reward": result.reward,
            "steps": steps,
            "final_answer": env.last_final_answer,
            "metrics": env.last_metrics,
        }


# ---------------------------------------------------------------------------
# Baseline fetcher
# ---------------------------------------------------------------------------

def _fetch_baseline_scores(server_url: str) -> Dict[str, float]:
    """GET /baseline and return {task_id: rule_based_avg} mapping."""
    url = server_url.rstrip("/") + "/baseline"
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            data = json.loads(r.read())
        result = {}
        for task_id, agents in data.get("baseline_scores", {}).items():
            rb = agents.get("rule_based", {})
            result[task_id] = round(rb.get("average", 0.0), 4)
        return result
    except Exception as exc:
        print(f"  [warn] Could not fetch baseline scores: {exc}", file=sys.stderr)
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _run_all(args: argparse.Namespace) -> int:
    if args.provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY environment variable is not set.", file=sys.stderr)
            return 1
        llm = _OpenRouterClient(model=args.model, api_key=api_key)
    else:  # anthropic (default)
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
            return 1
        llm = create_llm_client("anthropic", model=args.model, api_key=api_key)

    tasks = [TASK_BY_ID[args.task]] if args.task else ALL_TASKS
    seeds: List[int] = args.seeds

    # Fetch rule_based baseline for comparison
    if not args.json:
        print(f"Fetching rule_based baseline from {args.server}...")
    baseline = _fetch_baseline_scores(args.server)

    all_results: Dict[str, Any] = {}

    for task in tasks:
        task_scores: List[float] = []
        seed_results: List[Dict[str, Any]] = []

        for seed in seeds:
            ep = await run_episode(args.server, llm, task.task_id, seed)
            task_scores.append(ep["score"])
            seed_results.append(ep)

        avg = round(sum(task_scores) / len(task_scores), 4)
        rb_avg = baseline.get(task.task_id)

        all_results[task.task_id] = {
            "task_name": task.name,
            "difficulty": task.difficulty,
            "model": args.model,
            "average": avg,
            "min": round(min(task_scores), 4),
            "max": round(max(task_scores), 4),
            "scores": [round(s, 4) for s in task_scores],
            "rule_based_avg": rb_avg,
            "delta": round(avg - rb_avg, 4) if rb_avg is not None else None,
            "episodes": seed_results,
        }

    if args.json:
        print(json.dumps(all_results, indent=2))
        return 0

    # Human-readable output
    print()
    print("=" * 70)
    print(f"Memory Management RL — LLM Agent Evaluation")
    print(f"Provider: {args.provider}  Model: {args.model}")
    print(f"Server: {args.server}")
    print(f"Seeds:  {seeds}")
    print("=" * 70)

    for task in tasks:
        tr = all_results[task.task_id]
        rb = tr["rule_based_avg"]
        delta = tr["delta"]
        delta_str = f"{delta:+.3f}" if delta is not None else "n/a"
        direction = ""
        if delta is not None:
            direction = " [ABOVE BASELINE]" if delta >= 0 else " [BELOW BASELINE]"

        print(f"\nTask: {tr['task_name']}  [{tr['difficulty'].upper()}]")
        print(f"  task_id: {task.task_id}")
        print()
        print(f"  {'Metric':<22} {'LLM':>8}  {'rule_based':>10}")
        print(f"  {'-'*22}  {'-'*8}  {'-'*10}")
        print(f"  {'Average':<22} {tr['average']:>8.3f}  {rb if rb is not None else 'n/a':>10}")
        print(f"  {'Min':<22} {tr['min']:>8.3f}")
        print(f"  {'Max':<22} {tr['max']:>8.3f}")
        print(f"  {'Delta vs rule_based':<22} {delta_str:>8}{direction}")
        print()
        print(f"  Per-seed scores:")
        for ep, score in zip(seed_results, tr["scores"]):
            print(f"    seed {ep['seed']}: {score:.3f}  ({ep['steps']} steps)")

    print("\n" + "=" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an LLM agent against the Memory Management RL Environment")
    parser.add_argument("--server", default=DEFAULT_SERVER, help=f"Server base URL (default: {DEFAULT_SERVER})")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="RNG seeds")
    parser.add_argument("--task", default=None, choices=list(TASK_BY_ID), help="Single task to run (default: all)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model ID (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openrouter"],
        help="LLM provider (default: anthropic)",
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    return asyncio.run(_run_all(args))


if __name__ == "__main__":
    sys.exit(main())
