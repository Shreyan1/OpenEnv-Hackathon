"""
inference.py — Memory Management RL Environment inference script

Required environment variables:
    API_BASE_URL   The LLM API endpoint (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your Hugging Face / API key
Optional environment variables:
    LOCAL_IMAGE_NAME  Optional local Docker image name when using from_docker_image()
"""
from __future__ import annotations

import os
import re
import sys
import textwrap
import time
from typing import Any, Dict, List

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.memory_management_agent.environment import MemoryManagementEnv
from src.memory_management_agent.grader import normalize_task_score
from src.memory_management_agent.logging_utils import elapsed_ms, log_event, now_monotonic
from src.memory_management_agent.tasks import ALL_TASKS, generator_for_task

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME: str | None = os.getenv("LOCAL_IMAGE_NAME")

SEEDS: List[int] = [42, 43, 44]
MAX_TOKENS: int = 512
TEMPERATURE: float = 0.0

# ---------------------------------------------------------------------------
# System prompt (action rules + format rules for ANSWER)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are a memory management policy. Choose exactly one action per turn.

    ## Action rules
    preference    → STORE the preference verbatim.
    constraint    → STORE the constraint keyword verbatim.
    project_info  → STORE the project fact verbatim.
    correction    → UPDATE the existing memory (IDS=<old id>, TEXT=<new value>).
                    If no existing memory exists yet: STORE instead.
                    NEVER leave IDS empty when updating — set it to the old memory's id.
    recall_check  → RETRIEVE the relevant memory ID (IDS=<id>). Do NOT answer yet.
    distractor    → IGNORE.
    confabulation → IGNORE. Never store hypothetical/third-party/colleague mentions.
    final_query   → ANSWER using stored memories (see format rules below).
    unknown       → classify the message using the signal phrases below, then apply the rule.

    ## Classifying unknown turns — signal phrases
    preference:    'we standardized on X', 'I prefer X', 'let's use X', 'team is on X',
                   'if we want X, Y is the obvious choice', 'X is what the platform uses',
                   'let's keep this on X', 'targeting X'
    constraint:    'bullet points', 'numbered list', 'five sentences', 'concise',
                   'valid json', 'code example', 'type annotations', 'snake_case'
    correction:    'scratch the X', 'forget X use Y', 'Correction:', 'Change of plan:',
                   'Update from the team:', 'swap out X'
    recall_check:  'what stack did I say', 'what formatting rule', 'what context did I mention',
                   'before I send this around', 'do you remember what'
    distractor:    small talk, scheduling, deploy status, 'I'll be out', 'meetings'
    confabulation: 'my colleague', 'hypothetically', 'the old team used', 'I've heard people'
    final_query:   'write the final response', 'draft the final answer', 'please draft'

    ## Format rules for ANSWER
    Check the memory bank for a stored constraint and follow it exactly:
      'bullet points'    → 2+ lines each starting with '- ' or '* '
      'numbered list'    → 2+ lines each starting with '1. ' '2. ' etc.
      'five sentences'   → at most 5 sentences
      'concise'          → 1-2 sentences, 10-25 words. MUST include the word 'concise'.
      'valid json'       → output ONLY a raw JSON object — NO ``` fences, NO prose.
                           MUST include preference keyword as a value AND a 'constraint' key.
      'code examples'    → include a fenced ``` code block
      'type annotations' → include '->' or 'param: type' in examples
      'snake_case'       → use snake_case for all identifiers
    The answer MUST include every stored preference and project fact.

    Respond in this exact format (no extra text):
    ACTION: <STORE|STORE_SUMMARY|IGNORE|RETRIEVE|UPDATE|DELETE|ANSWER>
    TEXT: <value for STORE/STORE_SUMMARY/UPDATE/ANSWER, else empty>
    IDS: <id(s) for RETRIEVE/UPDATE/DELETE — REQUIRED for those actions, else empty>
""").strip()

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(obs: Dict[str, Any]) -> str:
    memory_lines = [
        f"- {m['id']} | {m['type']} | {m['text']}"
        for m in obs.get("memory_bank", [])
    ]
    conv_lines = [
        f"- turn {t['turn_id']} [{t['kind']}]: {t['text']}"
        for t in obs.get("recent_conversation", [])
    ]
    return "\n".join([
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
    ])

# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(r"ACTION:\s*(\w+)", re.IGNORECASE)
_TEXT_RE = re.compile(r"TEXT:\s*(.*?)(?=\nIDS:|\Z)", re.DOTALL | re.IGNORECASE)
_IDS_RE = re.compile(r"IDS:\s*(.*)", re.IGNORECASE)
_VALID_ACTIONS = {"store", "store_summary", "ignore", "retrieve", "update", "delete", "answer"}
_ACTION_ALIASES = {
    "recall_check": "retrieve",
    "project_info": "store",
    "preference": "store",
    "constraint": "store",
    "correction": "update",
    "final_query": "answer",
    "distractor": "ignore",
    "confabulation": "ignore",
    "unknown": "ignore",
}


def _parse_action(text: str) -> Dict[str, Any]:
    action_match = _ACTION_RE.search(text)
    text_match = _TEXT_RE.search(text)
    ids_match = _IDS_RE.search(text)

    raw_action_type = action_match.group(1).lower() if action_match else "ignore"
    action_type = _ACTION_ALIASES.get(raw_action_type, raw_action_type)
    if action_type not in _VALID_ACTIONS:
        log_event("STEP", "invalid_action_type", raw_action_type=raw_action_type, fallback_action="ignore")
        action_type = "ignore"
    payload_text = text_match.group(1).strip() if text_match else ""
    ids_raw = ids_match.group(1).strip() if ids_match else ""
    ids = [i.strip() for i in ids_raw.split(",") if i.strip()] if ids_raw else []

    return {"type": action_type, "text": payload_text, "ids": ids}

# ---------------------------------------------------------------------------
# Episode runner (direct env import — no server required)
# ---------------------------------------------------------------------------

def _run_episode(env: MemoryManagementEnv, client: OpenAI, seed: int) -> tuple[float, int]:
    obs = env.reset(seed=seed)
    obs_dict = obs.to_dict()
    done = False
    step_count = 0

    while not done:
        user_prompt = _build_prompt(obs_dict)
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            log_event("STEP", "llm_call_failed", seed=seed, error=str(exc))
            print(f"  [warn] LLM call failed: {exc}", file=sys.stderr)
            response_text = "ACTION: ignore\nTEXT:\nIDS:"

        action_dict = _parse_action(response_text)
        result = env.step(action_dict)
        done = result.done
        step_count += 1
        if result.observation is not None:
            obs_dict = result.observation.to_dict()

    ep_result = env.build_episode_result()
    final_score = normalize_task_score(ep_result.reward)
    log_event(
        "STEP",
        "seed_run",
        seed=seed,
        step_count=step_count,
        score=round(final_score, 4),
    )
    return final_score, step_count

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    run_started_at = now_monotonic()
    if not HF_TOKEN:
        log_event("END", "inference_run", status="error", error="HF_TOKEN environment variable is not set")
        print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    log_event(
        "START",
        "inference_run",
        model_name=MODEL_NAME,
        api_base_url=API_BASE_URL,
        seeds=SEEDS,
        task_count=len(ALL_TASKS),
    )

    all_scores: Dict[str, Any] = {}

    global_step = 0
    for task in ALL_TASKS:
        task_started_at = now_monotonic()
        log_event("START", "task_run", task_id=task.task_id, difficulty=task.difficulty)
        print(f"[START] task={task.task_id}", flush=True)
        env = MemoryManagementEnv(
            generator=generator_for_task(task),
            memory_budget=task.memory_budget,
            max_turns=task.max_turns,
            expose_turn_kind=task.expose_turn_kind,
            decay_rate=task.decay_rate,
        )

        scores: List[float] = []
        total_steps = 0
        for seed in SEEDS:
            seed_started_at = time.perf_counter()
            score, n_steps = _run_episode(env, client, seed)
            scores.append(score)
            total_steps += n_steps
            global_step += n_steps
            log_event(
                "STEP",
                "seed_result",
                task_id=task.task_id,
                seed=seed,
                score=round(score, 4),
                elapsed_ms=elapsed_ms(seed_started_at),
            )
            print(f"[STEP] step={global_step} reward={round(score, 4)}", flush=True)

        avg = normalize_task_score(sum(scores) / len(scores))
        all_scores[task.task_id] = {
            "average": round(avg, 4),
            "scores": [round(s, 4) for s in scores],
        }
        log_event(
            "END",
            "task_run",
            task_id=task.task_id,
            average=round(avg, 4),
            elapsed_ms=elapsed_ms(task_started_at),
            status="ok",
        )
        print(f"[END] task={task.task_id} score={round(avg, 4)} steps={total_steps}", flush=True)

    # Verify all scores are strictly within (0.0, 1.0)
    for task_id, result in all_scores.items():
        for s in result["scores"]:
            assert 0.0 < s < 1.0, f"Score out of range: {task_id} = {s}"
        assert 0.0 < result["average"] < 1.0, f"Average out of range: {task_id} = {result['average']}"
    log_event(
        "END",
        "inference_run",
        status="ok",
        task_ids=list(all_scores.keys()),
        elapsed_ms=elapsed_ms(run_started_at),
    )


if __name__ == "__main__":
    main()
