"""
OpenEnv-compliant HTTP server for the Memory Management environment.

Endpoints:
  POST /reset           — start a new episode
  POST /step            — advance the episode with an action
  GET  /tasks           — list available tasks and action schema
  POST /grader          — score a completed episode
  GET  /baseline        — run baseline agents across all tasks and return scores
  GET  /health          — liveness check
  WS   /ws              — OpenEnv-native WebSocket endpoint
"""
from __future__ import annotations

import json
import sys
import os
import time

# Add project root to path so `src.memory_management_agent` is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, List, Optional
import uuid

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.memory_management_agent import (
    ALL_TASKS,
    TASK_BY_ID,
    RuleBasedMemoryAgent,
    NoMemoryAgent,
    StoreEverythingAgent,
    normalize_task_score,
    run_episode,
)
from src.memory_management_agent.environment import MemoryManagementEnv
from src.memory_management_agent.logging_utils import elapsed_ms, log_event, now_monotonic
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
    score: float           # strictly between 0.0 and 1.0 (normalised reward)
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
def reset(request: ResetRequest | None = Body(default=None)) -> ResetResponse:
    started_at = now_monotonic()
    status = "ok"
    request = request or ResetRequest()
    task_id = request.task_id or ALL_TASKS[0].task_id
    session_id = request.session_id
    log_event("START", "http_reset", task_id=task_id, session_id=session_id)
    _cleanup_expired_sessions()
    try:
        if task_id not in TASK_BY_ID:
            status = "error"
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
        log_event("STEP", "http_reset_session_created", task_id=task_id, session_id=session_id, seed=request.seed)
        return ResetResponse(
            session_id=session_id,
            task_id=task_id,
            observation=observation.to_dict(),
        )
    except Exception:
        status = "error"
        raise
    finally:
        log_event(
            "END",
            "http_reset",
            task_id=task_id,
            session_id=session_id,
            status=status,
            elapsed_ms=elapsed_ms(started_at),
        )


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    started_at = now_monotonic()
    status = "ok"
    task_id: Optional[str] = None
    log_event("START", "http_step", session_id=request.session_id)
    _cleanup_expired_sessions()
    try:
        session = _sessions.get(request.session_id)
        if session is None:
            status = "error"
            raise HTTPException(status_code=404, detail=f"Session not found: {request.session_id!r}")
        if session["done"]:
            status = "error"
            raise HTTPException(status_code=400, detail="Episode is already done. Call /reset to start a new one.")

        task_id = session["task_id"]
        env: MemoryManagementEnv = session["env"]
        action_dict = request.action
        log_event(
            "STEP",
            "http_step_action_received",
            session_id=request.session_id,
            task_id=task_id,
            action_type=action_dict.get("type"),
        )

        # Validate action type
        try:
            ActionType(action_dict.get("type", ""))
        except ValueError:
            status = "error"
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action type {action_dict.get('type')!r}. Valid: {[a.value for a in ActionType]}",
            )

        result = env.step(action_dict)
        session["done"] = result.done
        session["last_accessed_at"] = time.time()
        response_reward = result.reward
        if result.done:
            response_reward = normalize_task_score(env.build_episode_result().reward)
        log_event(
            "STEP",
            "http_step_result",
            session_id=request.session_id,
            task_id=task_id,
            done=result.done,
            reward=round(response_reward, 4),
        )

        return StepResponse(
            session_id=request.session_id,
            observation=result.observation.to_dict() if result.observation else None,
            reward=response_reward,
            done=result.done,
            info=result.info,
        )
    except Exception:
        status = "error"
        raise
    finally:
        log_event(
            "END",
            "http_step",
            session_id=request.session_id,
            task_id=task_id,
            status=status,
            elapsed_ms=elapsed_ms(started_at),
        )


@app.post("/grader", response_model=GraderResponse)
def grader(request: GraderRequest) -> GraderResponse:
    started_at = now_monotonic()
    status = "ok"
    task_id: Optional[str] = None
    log_event("START", "http_grader", session_id=request.session_id)
    _cleanup_expired_sessions()
    try:
        session = _sessions.get(request.session_id)
        if session is None:
            status = "error"
            raise HTTPException(status_code=404, detail=f"Session not found: {request.session_id!r}")

        task_id = session["task_id"]
        env: MemoryManagementEnv = session["env"]
        episode_result = env.build_episode_result()
        session["last_accessed_at"] = time.time()
        score = normalize_task_score(episode_result.reward)
        log_event("STEP", "http_grader_result", session_id=request.session_id, task_id=task_id, score=round(score, 4))

        return GraderResponse(
            session_id=request.session_id,
            task_id=task_id,
            score=score,
            metrics=episode_result.metrics.to_dict(),
            final_answer=episode_result.final_answer,
        )
    except Exception:
        status = "error"
        raise
    finally:
        log_event(
            "END",
            "http_grader",
            session_id=request.session_id,
            task_id=task_id,
            status=status,
            elapsed_ms=elapsed_ms(started_at),
        )


@app.get("/baseline")
def baseline() -> Dict[str, Any]:
    """
    Run the built-in baseline agents on all 3 tasks and return their scores.
    Uses 5 seeds per task to keep response time reasonable.
    """
    started_at = now_monotonic()
    status = "ok"
    log_event("START", "http_baseline")
    try:
        seeds = list(range(42, 47))
        baseline_agents = {
            "no_memory": NoMemoryAgent(),
            "store_everything": StoreEverythingAgent(),
            "rule_based": RuleBasedMemoryAgent(),
        }

        results: Dict[str, Any] = {}

        for task in ALL_TASKS:
            task_results: Dict[str, Any] = {}
            log_event("STEP", "http_baseline_task_started", task_id=task.task_id)
            for agent_name, agent in baseline_agents.items():
                env = _make_env(task.task_id)
                scores = []
                for seed in seeds:
                    ep_result = run_episode(agent, env, seed=seed)
                    scores.append(normalize_task_score(ep_result.reward))
                avg = sum(scores) / len(scores)
                task_results[agent_name] = {
                    "scores": scores,
                    "average": round(avg, 4),
                }
                log_event(
                    "STEP",
                    "http_baseline_agent_result",
                    task_id=task.task_id,
                    agent_name=agent_name,
                    average=round(avg, 4),
                )
            results[task.task_id] = task_results

        return {
            "baseline_scores": results,
            "seeds_used": seeds,
            "note": (
                "rule_based is the strongest baseline. "
                "An RL-trained policy should exceed rule_based on hidden seeds."
            ),
        }
    except Exception:
        status = "error"
        raise
    finally:
        log_event("END", "http_baseline", status=status, elapsed_ms=elapsed_ms(started_at))


# ---------------------------------------------------------------------------
# WebSocket endpoint (OpenEnv native protocol)
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    OpenEnv-native WebSocket endpoint.

    Each connection is its own session — no session_id needed over the wire.

    Client → server message types:
      {"type": "reset", "data": {"task_id": "...", "seed": 42}}
      {"type": "step",  "data": {"type": "store", "text": "..."}}
      {"type": "state"}
      {"type": "close"}

    Server → client responses:
      {"type": "observation", "data": {"observation": {...}, "reward": float|null, "done": bool}}
      {"type": "state",       "data": {"episode_id": ..., "step_count": int, "task_id": ...}}
      {"type": "error",       "data": {"message": "...", "code": "..."}}
    On done=True the observation data also includes score, metrics, final_answer.
    """
    await websocket.accept()
    env: Optional[MemoryManagementEnv] = None
    task_id: Optional[str] = None
    episode_done: bool = False
    connection_id = str(uuid.uuid4())
    started_at = now_monotonic()
    final_status = "ok"
    log_event("START", "ws_session", connection_id=connection_id)

    async def _err(message: str, code: str) -> None:
        log_event("STEP", "ws_error", connection_id=connection_id, task_id=task_id, code=code)
        await websocket.send_json({"type": "error", "data": {"message": message, "code": code}})

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _err("Invalid JSON", "INVALID_JSON")
                continue

            msg_type = msg.get("type")
            data = msg.get("data") or {}
            log_event("STEP", "ws_message_received", connection_id=connection_id, task_id=task_id, message_type=msg_type)

            if msg_type == "reset":
                tid = data.get("task_id") or ALL_TASKS[0].task_id
                if tid not in TASK_BY_ID:
                    await _err(f"Unknown task_id: {tid!r}. Valid: {list(TASK_BY_ID)}", "VALIDATION_ERROR")
                    continue
                task_id = tid
                env = _make_env(tid)
                obs = env.reset(seed=data.get("seed"))
                episode_done = False
                log_event("STEP", "ws_reset_completed", connection_id=connection_id, task_id=task_id, seed=data.get("seed"))
                await websocket.send_json({
                    "type": "observation",
                    "data": {"observation": obs.to_dict(), "reward": None, "done": False},
                })

            elif msg_type == "step":
                if env is None:
                    await _err("No active episode. Send reset first.", "SESSION_ERROR")
                    continue
                if episode_done:
                    await _err("Episode is done. Send reset to start a new one.", "SESSION_ERROR")
                    continue
                try:
                    ActionType(data.get("type", ""))
                except ValueError:
                    await _err(
                        f"Invalid action type {data.get('type')!r}. Valid: {[a.value for a in ActionType]}",
                        "VALIDATION_ERROR",
                    )
                    continue
                result = env.step(data)
                episode_done = result.done
                response_reward = result.reward
                ep = None
                if result.done:
                    ep = env.build_episode_result()
                    response_reward = normalize_task_score(ep.reward)
                log_event(
                    "STEP",
                    "ws_step_result",
                    connection_id=connection_id,
                    task_id=task_id,
                    done=result.done,
                    reward=round(response_reward, 4),
                )
                resp: Dict[str, Any] = {
                    "observation": result.observation.to_dict() if result.observation else None,
                    "reward": response_reward,
                    "done": result.done,
                }
                if result.done:
                    assert ep is not None
                    resp["score"] = normalize_task_score(ep.reward)
                    resp["metrics"] = ep.metrics.to_dict()
                    resp["final_answer"] = ep.final_answer
                await websocket.send_json({"type": "observation", "data": resp})

            elif msg_type == "state":
                await websocket.send_json({
                    "type": "state",
                    "data": {
                        "episode_id": env.episode.episode_id if env and env.episode else None,
                        "step_count": env._step_index if env else 0,
                        "task_id": task_id,
                    },
                })

            elif msg_type == "close":
                final_status = "closed"
                break

            else:
                await _err(f"Unknown message type: {msg_type!r}", "UNKNOWN_TYPE")

    except WebSocketDisconnect:
        final_status = "disconnect"
    except Exception as exc:
        final_status = "error"
        try:
            await _err(str(exc), "EXECUTION_ERROR")
        except Exception:
            pass
    finally:
        log_event(
            "END",
            "ws_session",
            connection_id=connection_id,
            task_id=task_id,
            status=final_status,
            elapsed_ms=elapsed_ms(started_at),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
