"""Emit runs in the shared trace schema (../trace-schema.md, v0.1)."""

import json
import subprocess
from datetime import datetime
from pathlib import Path

from .engine import SimResult

SCHEMA_VERSION = "0.1"


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def write_traces(result: SimResult, out_dir: Path, run_id: str, strategy: str, config: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "traces.jsonl", "w") as f:
        for r in sorted(result.requests, key=lambda x: x.rid):
            f.write(json.dumps({
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "source": "sim",
                "request_id": r.rid,
                "ts_arrival_s": r.arrival,
                "strategy": strategy,
                "model": "simulated",
                "endpoint": "sim",
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "queue_wait_s": r.queue_wait,
                "ttft_s": r.ttft,
                "tpot_s": r.tpot,
                "latency_s": r.latency,
                "error": None,
                "cost_usd": None,
                "meta": r.meta,
            }) + "\n")

    (out_dir / "run.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "source": "sim",
        "created_at": datetime.now().astimezone().isoformat(),
        "repo": "llm-inference-queueing",
        "config": config,
        "environment": {"machine": "m3-pro", "notes": "discrete-event simulation"},
        "n_requests": len(result.requests),
        "git_commit": _git_commit(),
    }, indent=2))
    return out_dir
