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

from src.memory_management_agent.grader import normalize_task_score
from src.memory_management_agent.training import parse_action_block
from src.memory_management_agent.tasks import ALL_TASKS, TASK_BY_ID


DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_SERVER = "http://localhost:7860"


def _require_task_score(raw_score: Any) -> float:
    if raw_score is None:
        raise ValueError("Missing terminal task score in environment response.")
    score = normalize_task_score(float(raw_score))
    if not 0.0 < score < 1.0:
        raise ValueError(f"Invalid terminal task score: {raw_score!r}")
    return score


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
        self.last_score: Optional[float] = None
        self.last_metrics: Dict[str, Any] = {}
        self.last_final_answer: str = ""

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[Dict[str, Any]]:
        result = super()._parse_result(payload)
        if payload.get("done"):
            self.last_score = _require_task_score(payload.get("score"))
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
        "You are a memory management policy. Choose exactly one action per turn.",
        "",
        "## Action rules",
        "preference    → STORE the preference verbatim.",
        "constraint    → STORE the constraint keyword verbatim.",
        "project_info  → STORE the project fact verbatim.",
        "correction    → UPDATE the existing memory (IDS=<old id>, TEXT=<new value>).",
        "                If no existing memory exists yet: STORE instead.",
        "                NEVER leave IDS empty when updating — set it to the old memory's id.",
        "recall_check  → RETRIEVE the relevant memory ID (IDS=<id>). Do NOT answer yet.",
        "distractor    → IGNORE.",
        "confabulation → IGNORE. Never store hypothetical/third-party/colleague mentions.",
        "final_query   → ANSWER using stored memories (see format rules below).",
        "unknown       → classify the message using the signal phrases below, then apply the rule.",
        "",
        "## Classifying unknown turns — signal phrases",
        "preference:    'we standardized on X', 'I prefer X', 'let's use X', 'team is on X',",
        "               'if we want X, Y is the obvious choice', 'X is what the platform uses',",
        "               'let's keep this on X', 'targeting X'",
        "constraint:    'bullet points', 'numbered list', 'five sentences', 'concise',",
        "               'valid json', 'code example', 'type annotations', 'snake_case'",
        "correction:    'scratch the X', 'forget X use Y', 'Correction:', 'Change of plan:',",
        "               'Update from the team:', 'swap out X'",
        "recall_check:  'what stack did I say', 'what formatting rule', 'what context did I mention',",
        "               'before I send this around', 'do you remember what'",
        "distractor:    small talk, scheduling, deploy status, 'I'll be out', 'meetings'",
        "confabulation: 'my colleague', 'hypothetically', 'the old team used', 'I've heard people'",
        "final_query:   'write the final response', 'draft the final answer', 'please draft'",
        "",
        "## Format rules for ANSWER",
        "Check the memory bank for a stored constraint and follow it exactly:",
        "  'bullet points'    → 2+ lines each starting with '- ' or '* '",
        "  'numbered list'    → 2+ lines each starting with '1. ' '2. ' etc.",
        "  'five sentences'   → at most 5 sentences",
        "  'concise'          → 1-2 sentences, 10-25 words. MUST include the word 'concise' in the text",
        "                       (e.g. 'Keeping this concise: use Python as the standardized stack.').",
        "                       Include the preference keyword and any project keywords.",
        "  'valid json'       → output ONLY a raw JSON object — NO ``` fences, NO prose.",
        "                       MUST include preference keyword as a value AND a 'constraint' key set to 'valid json'.",
        "                       Example: {\"stack\": \"FastAPI\", \"constraint\": \"valid json\"}",
        "  'code examples'    → include a fenced ``` code block",
        "  'type annotations' → include '->' or 'param: type' in examples",
        "  'snake_case'       → use snake_case for all identifiers",
        "The answer MUST include every stored preference and project fact.",
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
        "Respond in this exact format (no extra text):",
        "ACTION: <STORE|STORE_SUMMARY|IGNORE|RETRIEVE|UPDATE|DELETE|ANSWER>",
        "TEXT: <value for STORE/STORE_SUMMARY/UPDATE/ANSWER, else empty>",
        "IDS: <id(s) for RETRIEVE/UPDATE/DELETE — REQUIRED for those actions, else empty>",
    ])


# ---------------------------------------------------------------------------
# Episode runner — WebSocket transport
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
                llm_text = await llm.complete(prompt, max_tokens=512)
            except Exception as exc:
                llm_text = f"ACTION: ignore\nTEXT: (error: {exc})\nIDS:"

            action_dict = parse_action_block(llm_text).to_dict()
            result = await env.step(action_dict)
            obs = result.observation or obs
            steps += 1

        return {
            "task_id": task_id,
            "seed": seed,
            "score": _require_task_score(env.last_score),
            "reward": result.reward,
            "steps": steps,
            "final_answer": env.last_final_answer,
            "metrics": env.last_metrics,
        }


# ---------------------------------------------------------------------------
# Episode runner — HTTP transport (fallback for proxies that block WebSocket)
# ---------------------------------------------------------------------------

def _http_post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous HTTP POST helper using stdlib only."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


async def run_episode_http(
    server_url: str,
    llm: Any,
    task_id: str,
    seed: int,
) -> Dict[str, Any]:
    """Run one episode via HTTP REST endpoints (/reset, /step, /grader)."""
    base = server_url.rstrip("/")

    reset_resp = _http_post(f"{base}/reset", {"task_id": task_id, "seed": seed})
    session_id = reset_resp["session_id"]
    obs = reset_resp["observation"]
    steps = 0
    done = False
    last_reward = 0.0

    while not done:
        prompt = _build_prompt_from_dict(obs)
        try:
            llm_text = await llm.complete(prompt, max_tokens=512)
        except Exception as exc:
            llm_text = f"ACTION: ignore\nTEXT: (error: {exc})\nIDS:"

        action_dict = parse_action_block(llm_text).to_dict()
        step_resp = _http_post(f"{base}/step", {"session_id": session_id, "action": action_dict})
        done = step_resp.get("done", False)
        last_reward = step_resp.get("reward", 0.0)
        if step_resp.get("observation"):
            obs = step_resp["observation"]
        steps += 1

    grader_resp = _http_post(f"{base}/grader", {"session_id": session_id})
    return {
        "task_id": task_id,
        "seed": seed,
        "score": _require_task_score(grader_resp.get("score")),
        "reward": last_reward,
        "steps": steps,
        "final_answer": grader_resp.get("final_answer", ""),
        "metrics": grader_resp.get("metrics", {}),
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
            average = rb.get("average")
            if average is not None:
                result[task_id] = round(_require_task_score(average), 4)
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

    # Choose transport: explicit flag, or auto-detect HF Space URLs
    use_http = args.transport == "http" or (
        args.transport == "auto" and ".hf.space" in args.server
    )
    _run_ep = run_episode_http if use_http else run_episode
    transport_label = "http" if use_http else "websocket"

    # Fetch rule_based baseline for comparison
    if not args.json:
        print(f"Fetching rule_based baseline from {args.server}... (transport: {transport_label})")
    baseline = _fetch_baseline_scores(args.server)

    all_results: Dict[str, Any] = {}

    for task in tasks:
        task_scores: List[float] = []
        seed_results: List[Dict[str, Any]] = []

        for seed in seeds:
            ep = await _run_ep(args.server, llm, task.task_id, seed)
            task_scores.append(_require_task_score(ep["score"]))
            seed_results.append(ep)

        avg = round(_require_task_score(sum(task_scores) / len(task_scores)), 4)
        rb_avg = baseline.get(task.task_id)

        all_results[task.task_id] = {
            "task_name": task.name,
            "difficulty": task.difficulty,
            "model": args.model,
            "average": avg,
            "min": round(_require_task_score(min(task_scores)), 4),
            "max": round(_require_task_score(max(task_scores)), 4),
            "scores": [round(_require_task_score(s), 4) for s in task_scores],
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
    print(f"Server: {args.server}  Transport: {transport_label}")
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
    parser.add_argument(
        "--transport",
        default="auto",
        choices=["auto", "websocket", "http"],
        help="Transport: auto (default) uses HTTP for *.hf.space, WebSocket otherwise",
    )
    args = parser.parse_args()

    return asyncio.run(_run_all(args))


if __name__ == "__main__":
    sys.exit(main())
