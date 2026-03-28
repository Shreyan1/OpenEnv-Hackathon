"""
Python client for the Memory Management RL Environment HTTP API.

Usage:
    from client import MemoryManagementClient

    client = MemoryManagementClient("http://localhost:7860")
    session_id, obs = client.reset(task_id="easy_preference_recall", seed=42)

    done = False
    while not done:
        action = {"type": "store", "text": obs["current_user_message"]}
        obs, reward, done, info = client.step(session_id, action)

    result = client.grade(session_id)
    print(result["score"])
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

try:
    import httpx as _http_lib
    _USE_HTTPX = True
except ImportError:
    import urllib.request as _urllib
    import json as _json
    _USE_HTTPX = False


class MemoryManagementClient:
    """Thin HTTP client for the Memory Management RL Environment."""

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        if _USE_HTTPX:
            resp = _http_lib.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        else:
            data = _json.dumps(payload).encode()
            req = _urllib.Request(url, data=data, headers={"Content-Type": "application/json"})
            with _urllib.urlopen(req, timeout=self.timeout) as r:
                return _json.loads(r.read())

    def _get(self, path: str) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        if _USE_HTTPX:
            resp = _http_lib.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        else:
            with _urllib.urlopen(url, timeout=self.timeout) as r:
                return _json.loads(r.read())

    def health(self) -> Dict[str, str]:
        """Check server liveness."""
        return self._get("/health")

    def tasks(self) -> Dict[str, Any]:
        """List available tasks and the action/observation schema."""
        return self._get("/tasks")

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new episode.

        Returns:
            (session_id, observation)
        """
        payload: Dict[str, Any] = {}
        if task_id is not None:
            payload["task_id"] = task_id
        if seed is not None:
            payload["seed"] = seed
        if session_id is not None:
            payload["session_id"] = session_id

        resp = self._post("/reset", payload)
        return resp["session_id"], resp["observation"]

    def step(
        self,
        session_id: str,
        action: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], float, bool, Dict[str, Any]]:
        """
        Advance the episode with an action.

        Returns:
            (observation, reward, done, info)
        """
        resp = self._post("/step", {"session_id": session_id, "action": action})
        return resp["observation"], resp["reward"], resp["done"], resp["info"]

    def grade(self, session_id: str) -> Dict[str, Any]:
        """
        Finalize the episode and return graded metrics.

        Returns dict with keys: session_id, task_id, score, metrics, final_answer
        """
        return self._post("/grader", {"session_id": session_id})

    def baseline(self) -> Dict[str, Any]:
        """Run baseline agents across all tasks and return scores."""
        return self._get("/baseline")
