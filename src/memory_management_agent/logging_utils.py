from __future__ import annotations

import json
import sys
import time
from typing import Any


def now_monotonic() -> float:
    return time.perf_counter()


def elapsed_ms(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)


def log_event(stage: str, event: str, **fields: Any) -> None:
    normalized_stage = stage.upper()
    parts = [f"[{normalized_stage}]", f"event={json.dumps(event)}"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={json.dumps(value, sort_keys=True)}")
    # Keep operational logs off stdout so validator parsers only see intended score lines.
    print(" ".join(parts), flush=True, file=sys.stderr)
