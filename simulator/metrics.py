import numpy as np

from .engine import SimResult


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def summarize(result: SimResult, warmup_frac: float = 0.1) -> dict:
    """Aggregate a sim run. Drops the first warmup_frac of requests (by
    arrival) so transient startup doesn't pollute steady-state numbers.
    Throughput is computed over the kept window only."""
    reqs = sorted(result.requests, key=lambda r: r.arrival)
    kept = reqs[int(len(reqs) * warmup_frac):]
    if not kept:
        raise ValueError("no requests left after warmup trim")

    window = max(r.finish for r in kept) - kept[0].arrival
    out_tokens = sum(r.output_tokens for r in kept)
    ttfts = [r.ttft for r in kept]
    lats = [r.latency for r in kept]
    waits = [r.queue_wait for r in kept]
    tpots = [r.tpot for r in kept if r.tpot is not None]

    return {
        "n": len(kept),
        "throughput_rps": len(kept) / window,
        "throughput_tok_s": out_tokens / window,
        "ttft_p50_s": _pct(ttfts, 50),
        "ttft_p95_s": _pct(ttfts, 95),
        "ttft_p99_s": _pct(ttfts, 99),
        "latency_p50_s": _pct(lats, 50),
        "latency_p95_s": _pct(lats, 95),
        "latency_p99_s": _pct(lats, 99),
        "queue_wait_p95_s": _pct(waits, 95),
        "tpot_mean_s": float(np.mean(tpots)) if tpots else float("nan"),
        "tpot_p95_s": _pct(tpots, 95),
        "utilization": result.utilization,
        "mean_decode_batch": float(np.mean(result.batch_size_samples)) if result.batch_size_samples else 0.0,
        "max_kv_reserved": result.max_kv_reserved,
    }
