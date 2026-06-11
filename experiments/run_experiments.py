"""Run all week-1/2 experiments. Writes results/summary.csv (one row per sim)
and one representative trace dir in the shared schema.

Load levels are expressed as fractions of each policy's MEASURED saturation
throughput (probed by overloading it), so policies are compared fairly.

Usage: uv run python experiments/run_experiments.py
"""

import csv
import itertools
from pathlib import Path

import numpy as np

from simulator import CostModel, make_workload, simulate_continuous, simulate_static, summarize
from simulator.traces import write_traces

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
N = 1500
SEEDS = [0, 1, 2, 3, 4]

POLICIES = {
    "continuous_64": lambda reqs, cost: simulate_continuous(reqs, cost, max_batch=64),
    "continuous_8": lambda reqs, cost: simulate_continuous(reqs, cost, max_batch=8),
    "static_2": lambda reqs, cost: simulate_static(reqs, cost, batch_size=2),
    "static_8": lambda reqs, cost: simulate_static(reqs, cost, batch_size=8),
    "static_32": lambda reqs, cost: simulate_static(reqs, cost, batch_size=32),
}


def saturation_rps(policy: str, profile: str, cost: CostModel) -> float:
    reqs = make_workload(profile, rate_rps=80, n=800, seed=99)
    return summarize(POLICIES[policy](reqs, cost))["throughput_rps"]


def run_one(policy, profile, rate, seed, cost, arrival="poisson"):
    reqs = make_workload(profile, rate_rps=rate, n=N, seed=seed, arrival_process=arrival)
    res = POLICIES[policy](reqs, cost)
    return summarize(res), res


def main():
    cost = CostModel()
    RESULTS.mkdir(exist_ok=True)
    rows = []
    caps = {}

    def cap(policy, profile):
        if (policy, profile) not in caps:
            caps[(policy, profile)] = saturation_rps(policy, profile, cost)
            print(f"  capacity {policy}/{profile}: {caps[(policy, profile)]:.2f} rps")
        return caps[(policy, profile)]

    def record(exp, policy, profile, arrival, rate, seed, summary):
        rows.append({
            "exp": exp, "policy": policy, "profile": profile, "arrival": arrival,
            "rate_rps": round(rate, 4), "seed": seed,
            "capacity_rps": round(cap(policy, profile), 4),
            "offered_load": round(rate / cap(policy, profile), 4),
            **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in summary.items()},
        })

    # ---- exp 1: head-to-head load sweep, chat profile, absolute rates ----
    print("exp1: load_sweep")
    rates = np.linspace(0.25, 4.75, 10)
    for policy, rate, seed in itertools.product(["continuous_64", "static_8"], rates, SEEDS):
        s, _ = run_one(policy, "chat", rate, seed, cost)
        record("load_sweep", policy, "chat", "poisson", rate, seed, s)

    # ---- exp 2: p95 vs utilization hockey stick, continuous only ----
    print("exp2: hockey")
    c = cap("continuous_64", "chat")
    for frac, seed in itertools.product(np.linspace(0.30, 0.98, 12), SEEDS):
        s, _ = run_one("continuous_64", "chat", c * frac, seed, cost)
        record("hockey", "continuous_64", "chat", "poisson", c * frac, seed, s)

    # ---- exp 3: poisson vs bursty at the same mean rate, agent profile ----
    print("exp3: bursty")
    c = cap("continuous_64", "agent")
    for frac, arrival, seed in itertools.product(np.linspace(0.3, 0.9, 7), ["poisson", "bursty"], SEEDS):
        s, _ = run_one("continuous_64", "agent", c * frac, seed, cost, arrival=arrival)
        record("bursty", "continuous_64", "agent", arrival, c * frac, seed, s)

    # ---- exp 4: throughput vs p95 latency frontier across batch configs ----
    print("exp4: frontier")
    for policy in POLICIES:
        c = cap(policy, "chat")
        for frac, seed in itertools.product(np.linspace(0.40, 0.95, 8), SEEDS[:3]):
            s, _ = run_one(policy, "chat", c * frac, seed, cost)
            record("frontier", policy, "chat", "poisson", c * frac, seed, s)

    out = RESULTS / "summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")

    # representative trace in the shared schema (committed as an example)
    s, res = run_one("continuous_64", "chat", cap("continuous_64", "chat") * 0.7, 0, cost)
    write_traces(
        res, RESULTS / "example-run", run_id="qsim-chat-cont64-rho070-seed0",
        strategy="continuous_64",
        config={"profile": "chat", "offered_load": 0.7, "seed": 0, "n": N,
                "cost_model": cost.label, "engine": "simulate_continuous"},
    )
    print("wrote example trace -> results/example-run/")


if __name__ == "__main__":
    main()
