"""
OpenEnv-compliant HTTP server for the Memory Management environment.

Endpoints:
  POST /reset           — start a new episode
  POST /step            — advance the episode with an action
  GET  /tasks           — list available tasks and action schema
  POST /grader          — score a completed episode
  GET  /baseline        — run baseline agents across all tasks and return scores
  GET  /health          — liveness check
"""
from __future__ import annotations

import sys
import os
import time

# Add project root to path so `src.memory_management_agent` is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.memory_management_agent import (
    ALL_TASKS,
    TASK_BY_ID,
    RuleBasedMemoryAgent,
    NoMemoryAgent,
    StoreEverythingAgent,
    run_episode,
)
from src.memory_management_agent.environment import MemoryManagementEnv
from src.memory_management_agent.schemas import Action, ActionType
from src.memory_management_agent.tasks import generator_for_task

app = FastAPI(
    title="Memory Management RL Environment",
    description=(
        "OpenEnv-compatible environment for training and evaluating LLM-based memory "
        "management policies. The agent must decide what to remember, update, and forget "
        "across a multi-turn conversation to maximise a future recall reward."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# In-memory session store  (keyed by session_id)
# ---------------------------------------------------------------------------

_sessions: Dict[str, Dict[str, Any]] = {}
_SESSION_TTL_SECONDS = 1800


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = None   # defaults to easy task
    seed: Optional[int] = None
    session_id: Optional[str] = None  # caller may supply; else auto-generated


class ResetResponse(BaseModel):
    session_id: str
    task_id: str
    observation: Dict[str, Any]


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


class StepResponse(BaseModel):
    session_id: str
    observation: Optional[Dict[str, Any]]
    reward: float
    done: bool
    info: Dict[str, Any]


class GraderRequest(BaseModel):
    session_id: str


class GraderResponse(BaseModel):
    session_id: str
    task_id: str
    score: float           # 0.0 – 1.0  (normalised reward)
    metrics: Dict[str, Any]
    final_answer: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(task_id: str) -> MemoryManagementEnv:
    task = TASK_BY_ID[task_id]
    gen = generator_for_task(task)
    return MemoryManagementEnv(
        generator=gen,
        memory_budget=task.memory_budget,
        max_turns=task.max_turns,
        expose_turn_kind=task.expose_turn_kind,
        decay_rate=task.decay_rate,
    )


def _cleanup_expired_sessions(now: Optional[float] = None) -> None:
    now = now if now is not None else time.time()
    expired = [
        session_id
        for session_id, session in _sessions.items()
        if now - float(session.get("last_accessed_at", session.get("created_at", now))) > _SESSION_TTL_SECONDS
    ]
    for session_id in expired:
        _sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    """Return available tasks and the action schema."""
    return {
        "tasks": [t.to_dict() for t in ALL_TASKS],
        "action_schema": {
            "type": "object",
            "required": ["type"],
            "properties": {
                "type": {
                    "type": "string",
                    "enum": [a.value for a in ActionType],
                    "description": "Action type",
                },
                "text": {
                    "type": "string",
                    "description": "Text payload (for store, store_summary, update, answer, retrieve-by-query)",
                },
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Memory IDs (for retrieve, update, delete)",
                },
            },
        },
        "observation_schema": {
            "type": "object",
            "properties": {
                "current_user_message": {"type": "string"},
                "current_turn_kind": {
                    "type": "string",
                    "enum": [
                        "preference",
                        "constraint",
                        "correction",
                        "project_info",
                        "distractor",
                        "confabulation",
                        "recall_check",
                        "final_query",
                        "unknown",
                    ],
                },
                "recent_conversation": {"type": "array"},
                "memory_bank": {"type": "array"},
                "memory_budget_remaining": {"type": "integer"},
                "step_number": {"type": "integer"},
                "episode_metadata": {"type": "object"},
            },
        },
    }


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest) -> ResetResponse:
    _cleanup_expired_sessions()
    task_id = request.task_id or ALL_TASKS[0].task_id
    if task_id not in TASK_BY_ID:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id!r}. Valid: {list(TASK_BY_ID)}")

    session_id = request.session_id or str(uuid.uuid4())
    env = _make_env(task_id)
    observation = env.reset(seed=request.seed)

    _sessions[session_id] = {
        "env": env,
        "task_id": task_id,
        "done": False,
        "created_at": time.time(),
        "last_accessed_at": time.time(),
    }

    return ResetResponse(
        session_id=session_id,
        task_id=task_id,
        observation=observation.to_dict(),
    )


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    _cleanup_expired_sessions()
    session = _sessions.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {request.session_id!r}")
    if session["done"]:
        raise HTTPException(status_code=400, detail="Episode is already done. Call /reset to start a new one.")

    env: MemoryManagementEnv = session["env"]
    action_dict = request.action

    # Validate action type
    try:
        ActionType(action_dict.get("type", ""))
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action type {action_dict.get('type')!r}. Valid: {[a.value for a in ActionType]}",
        )

    result = env.step(action_dict)
    session["done"] = result.done
    session["last_accessed_at"] = time.time()

    return StepResponse(
        session_id=request.session_id,
        observation=result.observation.to_dict() if result.observation else None,
        reward=result.reward,
        done=result.done,
        info=result.info,
    )


@app.post("/grader", response_model=GraderResponse)
def grader(request: GraderRequest) -> GraderResponse:
    _cleanup_expired_sessions()
    session = _sessions.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {request.session_id!r}")

    env: MemoryManagementEnv = session["env"]
    episode_result = env.build_episode_result()
    session["last_accessed_at"] = time.time()

    return GraderResponse(
        session_id=request.session_id,
        task_id=session["task_id"],
        score=max(0.0, min(1.0, episode_result.reward)),
        metrics=episode_result.metrics.to_dict(),
        final_answer=episode_result.final_answer,
    )


@app.get("/baseline")
def baseline() -> Dict[str, Any]:
    """
    Run the built-in baseline agents on all 3 tasks and return their scores.
    Uses 5 seeds per task to keep response time reasonable.
    """
    seeds = list(range(42, 47))
    baseline_agents = {
        "no_memory": NoMemoryAgent(),
        "store_everything": StoreEverythingAgent(),
        "rule_based": RuleBasedMemoryAgent(),
    }

    results: Dict[str, Any] = {}

    for task in ALL_TASKS:
        task_results: Dict[str, Any] = {}
        for agent_name, agent in baseline_agents.items():
            env = _make_env(task.task_id)
            scores = []
            for seed in seeds:
                ep_result = run_episode(agent, env, seed=seed)
                scores.append(max(0.0, min(1.0, ep_result.reward)))
            avg = sum(scores) / len(scores)
            task_results[agent_name] = {
                "scores": scores,
                "average": round(avg, 4),
            }
        results[task.task_id] = task_results

    return {
        "baseline_scores": results,
        "seeds_used": seeds,
        "note": (
            "rule_based is the strongest baseline. "
            "An RL-trained policy should exceed rule_based on hidden seeds."
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
