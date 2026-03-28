"""
Shared data models for the Memory Management RL Environment.

These mirror the server request/response schemas and can be used by
clients, training scripts, and evaluation pipelines.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# API request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


class GraderRequest(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# API response models
# ---------------------------------------------------------------------------

class ResetResponse(BaseModel):
    session_id: str
    task_id: str
    observation: Dict[str, Any]


class StepResponse(BaseModel):
    session_id: str
    observation: Optional[Dict[str, Any]]
    reward: float
    done: bool
    info: Dict[str, Any]


class GraderResponse(BaseModel):
    session_id: str
    task_id: str
    score: float
    metrics: Dict[str, Any]
    final_answer: str


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def store(text: str) -> Dict[str, Any]:
    return {"type": "store", "text": text}

def store_summary(text: str) -> Dict[str, Any]:
    return {"type": "store_summary", "text": text}

def retrieve(ids: List[str]) -> Dict[str, Any]:
    return {"type": "retrieve", "ids": ids}

def update(id: str, text: str) -> Dict[str, Any]:
    return {"type": "update", "ids": [id], "text": text}

def delete(id: str) -> Dict[str, Any]:
    return {"type": "delete", "ids": [id]}

def ignore() -> Dict[str, Any]:
    return {"type": "ignore"}

def answer(text: str) -> Dict[str, Any]:
    return {"type": "answer", "text": text}
