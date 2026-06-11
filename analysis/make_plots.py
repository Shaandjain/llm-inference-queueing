"""Generate all figures from results/summary.csv. No hand-made charts.

Usage: uv run python analysis/make_plots.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "analysis" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

COLORS = {"continuous_64": "#1a7f64", "continuous_8": "#7fc7b4",
          "static_2": "#f4b942", "static_8": "#e07b39", "static_32": "#b33a3a"}
LABELS = {"continuous_64": "continuous (max batch 64)", "continuous_8": "continuous (max batch 8)",
          "static_2": "static (batch 2)", "static_8": "static (batch 8)", "static_32": "static (batch 32)"}


def load_rows():
    with open(ROOT / "results" / "summary.csv") as f:
        return [{k: (v if k in ("exp", "policy", "profile", "arrival") else float(v))
                 for k, v in row.items()} for row in csv.DictReader(f)]


def band(rows, xkey, ykey, groupkeys):
    """Group rows, return x, mean(y), min(y), max(y) across seeds."""
    g = defaultdict(list)
    for r in rows:
        g[tuple(round(r[k], 4) if isinstance(r[k], float) else r[k] for k in groupkeys) + (round(r[xkey], 4),)].append(r[ykey])
    pts = defaultdict(list)
    for key, ys in sorted(g.items()):
        pts[key[:-1]].append((key[-1], np.mean(ys), np.min(ys), np.max(ys)))
    return pts


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.5)


def plot_load_sweep(rows):
    rows = [r for r in rows if r["exp"] == "load_sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ykey, title in [(axes[0], "ttft_p95_s", "p95 time-to-first-token"),
                            (axes[1], "latency_p95_s", "p95 end-to-end latency")]:
        for (policy,), pts in band(rows, "rate_rps", ykey, ["policy"]).items():
            x, m, lo, hi = zip(*pts)
            ax.plot(x, m, "o-", color=COLORS[policy], label=LABELS[policy], markersize=4)
            ax.fill_between(x, lo, hi, color=COLORS[policy], alpha=0.15)
        ax.set_yscale("log")
        ax.set_xlabel("offered load (requests/s)")
        ax.set_ylabel("seconds (log)")
        ax.set_title(title, fontsize=11)
        style(ax)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Continuous vs static batching, chat workload (5 seeds, min–max band)", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOTS / "01_load_sweep.png", dpi=150)


def plot_hockey(rows):
    """Two findings in one figure: (a) the tail-latency hockey stick vs load,
    (b) busy-time 'utilization' is ~1.0 at EVERY load level — the server is
    never idle, it just runs underfilled batches. Busy % is a useless health
    signal for batched LLM serving; batch occupancy is the real one."""
    rows = [r for r in rows if r["exp"] == "hockey"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    g = defaultdict(list)
    for r in rows:
        g[round(r["offered_load"], 3)].append(r)
    loads = sorted(g)
    p95 = [np.mean([r["latency_p95_s"] for r in g[k]]) for k in loads]
    p95_lo = [np.min([r["latency_p95_s"] for r in g[k]]) for k in loads]
    p95_hi = [np.max([r["latency_p95_s"] for r in g[k]]) for k in loads]
    busy = [np.mean([r["utilization"] for r in g[k]]) for k in loads]
    batch = [np.mean([r["mean_decode_batch"] for r in g[k]]) for k in loads]

    ax = axes[0]
    ax.plot(loads, p95, "o-", color="#1a7f64", markersize=4)
    ax.fill_between(loads, p95_lo, p95_hi, color="#1a7f64", alpha=0.15)
    ax.set_xlabel("offered load (fraction of measured capacity)")
    ax.set_ylabel("p95 end-to-end latency (s)")
    ax.set_title("The hockey stick: tail latency vs load", fontsize=11)
    style(ax)

    ax = axes[1]
    ax.plot(loads, busy, "o-", color="#b33a3a", markersize=4, label="busy time / makespan")
    ax.plot(loads, np.array(batch) / 64, "o-", color="#1a7f64", markersize=4,
            label="mean decode batch / max batch (64)")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("offered load (fraction of measured capacity)")
    ax.set_ylabel("fraction")
    ax.set_title("“Busy” saturates immediately; occupancy is the real signal", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    style(ax)

    fig.suptitle("Continuous batching, chat workload (5 seeds)", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOTS / "02_p95_vs_utilization.png", dpi=150)


def plot_bursty(rows):
    rows = [r for r in rows if r["exp"] == "bursty"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = {"poisson": "#1a7f64", "bursty": "#b33a3a"}
    for (arrival,), pts in band(rows, "offered_load", "ttft_p95_s", ["arrival"]).items():
        x, m, lo, hi = zip(*pts)
        ax.plot(x, m, "o-", color=colors[arrival], label=f"{arrival} arrivals", markersize=4)
        ax.fill_between(x, lo, hi, color=colors[arrival], alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("mean offered load (fraction of capacity)")
    ax.set_ylabel("p95 TTFT, seconds (log)")
    ax.set_title("Same mean rate, different tails: burstiness and TTFT\n(agent workload, continuous batching, 5 seeds)", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(PLOTS / "03_burstiness.png", dpi=150)


def plot_frontier(rows):
    rows = [r for r in rows if r["exp"] == "frontier"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for (policy,), pts in band(rows, "offered_load", "throughput_rps", ["policy"]).items():
        # for the frontier we need (throughput, p95) pairs per load level
        sub = [r for r in rows if r["policy"] == policy]
        g = defaultdict(lambda: ([], []))
        for r in sub:
            g[round(r["offered_load"], 3)][0].append(r["throughput_rps"])
            g[round(r["offered_load"], 3)][1].append(r["latency_p95_s"])
        x = [np.mean(t) for t, _ in (g[k] for k in sorted(g))]
        y = [np.mean(l) for _, l in (g[k] for k in sorted(g))]
        ax.plot(x, y, "o-", color=COLORS[policy], label=LABELS[policy], markersize=4)
    ax.set_yscale("log")
    ax.set_xlabel("achieved throughput (requests/s)")
    ax.set_ylabel("p95 end-to-end latency, seconds (log)")
    ax.set_title("Throughput–latency frontier by batching config (chat workload)", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(PLOTS / "04_frontier.png", dpi=150)


if __name__ == "__main__":
    rows = load_rows()
    plot_load_sweep(rows)
    plot_hockey(rows)
    plot_bursty(rows)
    plot_frontier(rows)
    print(f"wrote 4 plots -> {PLOTS}")
