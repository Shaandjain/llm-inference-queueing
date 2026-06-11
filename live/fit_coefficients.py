"""Fit CostModel coefficients from live traces (concurrency-1 sweep).

prefill: least-squares TTFT = base + per_token * input_tokens
decode:  decode_base = median TPOT (per-seq term needs concurrent runs; left 0)

Usage:
  uv run python live/fit_coefficients.py results/live-m3pro-qwen3b --label m3pro-ollama-qwen2.5-3b
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--label", required=True)
    ap.add_argument("--kv-capacity", type=int, default=32_768)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(Path(args.trace_dir) / "traces.jsonl")]
    ok = [r for r in rows if not r["error"] and r["input_tokens"] and r["ttft_s"]]
    if len(ok) < 10:
        raise SystemExit(f"only {len(ok)} usable traces, need more")

    x = np.array([r["input_tokens"] for r in ok], dtype=float)
    y = np.array([r["ttft_s"] for r in ok])
    per_token, base = np.polyfit(x, y, 1)
    pred = base + per_token * x
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

    tpots = np.array([r["tpot_s"] for r in ok if r["tpot_s"]])
    decode_base = float(np.median(tpots))

    profile = {
        "prefill_base_s": round(float(base), 6),
        "prefill_per_token_s": round(float(per_token), 8),
        "decode_base_s": round(decode_base, 6),
        "decode_per_seq_s": 0.0,  # needs a concurrency sweep to fit; week 2
        "decode_per_kv_token_s": 0.0,
        "kv_capacity_tokens": args.kv_capacity,
        "label": args.label,
    }
    out = Path(__file__).parent.parent / "profiles" / f"{args.label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(profile, indent=2))

    print(f"n={len(ok)} traces")
    print(f"prefill: ttft = {base * 1000:.1f}ms + {per_token * 1000:.4f}ms/token  (R²={r2:.3f})")
    print(f"  -> implied prefill throughput ~{1 / per_token:.0f} tok/s")
    print(f"decode: {decode_base * 1000:.1f}ms/token  (~{1 / decode_base:.0f} tok/s at batch 1)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
