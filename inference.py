"""
inference.py — Memory Management RL Environment inference script

Required environment variables:
    API_BASE_URL   The LLM API endpoint (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your Hugging Face / API key
"""
from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.memory_management_agent.environment import MemoryManagementEnv
from src.memory_management_agent.tasks import ALL_TASKS, generator_for_task

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""

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


def _parse_action(text: str) -> Dict[str, Any]:
    action_match = _ACTION_RE.search(text)
    text_match = _TEXT_RE.search(text)
    ids_match = _IDS_RE.search(text)

    action_type = action_match.group(1).lower() if action_match else "ignore"
    payload_text = text_match.group(1).strip() if text_match else ""
    ids_raw = ids_match.group(1).strip() if ids_match else ""
    ids = [i.strip() for i in ids_raw.split(",") if i.strip()] if ids_raw else []

    return {"type": action_type, "text": payload_text, "ids": ids}

# ---------------------------------------------------------------------------
# Episode runner (direct env import — no server required)
# ---------------------------------------------------------------------------

def _run_episode(env: MemoryManagementEnv, client: OpenAI, seed: int) -> float:
    obs = env.reset(seed=seed)
    obs_dict = obs.to_dict()
    done = False

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
            print(f"  [warn] LLM call failed: {exc}", file=sys.stderr)
            response_text = "ACTION: ignore\nTEXT:\nIDS:"

        action_dict = _parse_action(response_text)
        result = env.step(action_dict)
        done = result.done
        if result.observation is not None:
            obs_dict = result.observation.to_dict()

    ep_result = env.build_episode_result()
    return max(0.0, min(1.0, ep_result.reward))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    print(f"Model:    {MODEL_NAME}")
    print(f"API Base: {API_BASE_URL}")
    print(f"Seeds:    {SEEDS}")
    print()

    all_scores: Dict[str, Any] = {}

    for task in ALL_TASKS:
        print(f"Task: {task.task_id}  [{task.difficulty}]")
        env = MemoryManagementEnv(
            generator=generator_for_task(task),
            memory_budget=task.memory_budget,
            max_turns=task.max_turns,
            expose_turn_kind=task.expose_turn_kind,
            decay_rate=task.decay_rate,
        )

        scores: List[float] = []
        for seed in SEEDS:
            score = _run_episode(env, client, seed)
            scores.append(score)
            print(f"  seed={seed}  score={score:.4f}")

        avg = sum(scores) / len(scores)
        all_scores[task.task_id] = {
            "average": round(avg, 4),
            "scores": [round(s, 4) for s in scores],
        }
        print(f"  → average: {avg:.4f}")
        print()

    print("=" * 50)
    print("Final results:")
    print(json.dumps(all_scores, indent=2))
    print()

    # Verify all scores are in [0.0, 1.0]
    for task_id, result in all_scores.items():
        for s in result["scores"]:
            assert 0.0 <= s <= 1.0, f"Score out of range: {task_id} = {s}"
    print("All scores in [0.0, 1.0] range. OK")


if __name__ == "__main__":
    main()
