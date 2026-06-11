"""Predicted vs observed: run the simulator on the exact workload that was
replayed against the real server, with fitted coefficients, and compare.

Usage:
  uv run python analysis/validate.py --profile-json profiles/modal-l4-vllm-qwen2.5-7b.json \
      --replay results/replay-low results/replay-high --max-batch 64
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulator.engine import simulate_continuous  # noqa: E402
from simulator.request import Request  # noqa: E402
from simulator.server import CostModel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_replay(trace_dir: Path):
    rows = [json.loads(l) for l in open(trace_dir / "traces.jsonl")]
    run = json.loads((trace_dir / "run.json").read_text())
    ok = [r for r in rows if not r["error"] and r["ttft_s"]]
    return ok, run


def simulate_replay(rows, cost: CostModel, max_batch: int, request_overhead_s: float = 0.0):
    """request_overhead_s reattributes that much of the fitted prefill
    intercept from per-iteration GPU-blocking time to per-request,
    non-blocking overhead (network RTT, API processing, tokenization).
    The sim charges prefill_base_s on EVERY prefill iteration while all
    decodes stall; if the intercept is mostly per-request overhead, that
    phantom serialized time inflates predicted e2e latency under load."""
    if request_overhead_s:
        cost.prefill_base_s = max(cost.prefill_base_s - request_overhead_s, 0.005)
    reqs = [
        Request(
            rid=r["request_id"],
            arrival=r["meta"]["intended_arrival_s"],
            input_tokens=r["input_tokens"],        # actual server-counted tokens
            output_tokens=r["output_tokens"],
        )
        for r in rows
    ]
    simulate_continuous(reqs, cost, max_batch=max_batch)
    if request_overhead_s:
        for r in reqs:
            r.first_token += request_overhead_s
            r.finish += request_overhead_s
    return {r.rid: r for r in reqs}


def pcts(vals):
    return {f"p{q}": float(np.percentile(vals, q)) for q in (50, 95)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-json", required=True)
    ap.add_argument("--replay", nargs="+", required=True)
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--request-overhead-s", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cost = CostModel.load(Path(args.profile_json))
    fig, axes = plt.subplots(2, len(args.replay), figsize=(5.5 * len(args.replay), 8), squeeze=False)
    report = {"profile": cost.label, "runs": []}

    for col, rdir in enumerate(args.replay):
        rdir = Path(rdir)
        rows, run = load_replay(rdir)
        sim = simulate_replay(rows, CostModel.load(Path(args.profile_json)), args.max_batch,
                              args.request_overhead_s)

        obs_ttft = [r["ttft_s"] for r in rows]
        obs_lat = [r["latency_s"] for r in rows]
        pred_ttft = [sim[r["request_id"]].ttft for r in rows]
        pred_lat = [sim[r["request_id"]].latency for r in rows]

        entry = {
            "run_id": run["run_id"],
            "rate_rps": run["config"]["rate"],
            "n": len(rows),
            "observed": {"ttft": pcts(obs_ttft), "latency": pcts(obs_lat)},
            "predicted": {"ttft": pcts(pred_ttft), "latency": pcts(pred_lat)},
        }
        for metric in ("ttft", "latency"):
            entry[f"{metric}_err_pct"] = {
                q: round(100 * (entry["predicted"][metric][q] - entry["observed"][metric][q])
                         / entry["observed"][metric][q], 1)
                for q in ("p50", "p95")
            }
        report["runs"].append(entry)

        rate = run["config"]["rate"]
        for row_i, (obs, pred, name) in enumerate(
            [(obs_ttft, pred_ttft, "TTFT"), (obs_lat, pred_lat, "end-to-end latency")]
        ):
            ax = axes[row_i][col]
            for vals, label, color in [(obs, "observed (vLLM)", "#1a7f64"),
                                       (pred, "predicted (simulator)", "#e07b39")]:
                xs = np.sort(vals)
                ax.plot(xs, np.linspace(0, 1, len(xs)), label=label, color=color, linewidth=1.8)
            ax.set_xlabel(f"{name} (s)")
            ax.set_ylabel("CDF")
            ax.set_title(f"{name} @ {rate} rps (n={len(rows)})", fontsize=10)
            ax.legend(frameon=False, fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.25, linewidth=0.5)

    fig.suptitle(f"Simulator prediction vs real vLLM — {cost.label}", fontsize=12)
    fig.tight_layout()
    suffix = f"_{args.tag}" if args.tag else ""
    out_png = ROOT / "analysis" / "plots" / f"05_validation{suffix}.png"
    fig.savefig(out_png, dpi=150)

    report["request_overhead_s"] = args.request_overhead_s
    out_json = ROOT / "results" / f"validation{suffix}.json"
    out_json.write_text(json.dumps(report, indent=2))

    for e in report["runs"]:
        print(f"\n{e['run_id']} ({e['rate_rps']} rps, n={e['n']}):")
        for m in ("ttft", "latency"):
            o, p, err = e["observed"][m], e["predicted"][m], e[f"{m}_err_pct"]
            print(f"  {m:8s} p50 obs {o['p50']:.2f}s pred {p['p50']:.2f}s ({err['p50']:+.1f}%)   "
                  f"p95 obs {o['p95']:.2f}s pred {p['p95']:.2f}s ({err['p95']:+.1f}%)")
    print(f"\nwrote {out_png}\nwrote {out_json}")


if __name__ == "__main__":
    main()
