"""Open-loop replay of a simulator workload against a real endpoint.

Generates a workload with simulator.workloads.make_workload, dispatches each
request at its scheduled arrival time (open loop: no waiting for earlier
responses), forces exact output lengths (max_tokens + ignore_eos), and writes
traces. The simulator must then PREDICT these traces from fitted coefficients
— that comparison is analysis/validate.py.

Usage:
  uv run python live/replay_workload.py --base-url <url> --model <model> \
      --api-key <key> --profile chat --rate 1.0 --n 250 --seed 42 --out results/replay-r10
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import httpx

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulator.workloads import make_workload  # noqa: E402
from capture_traces import build_prompt  # noqa: E402


async def fire(client, base_url, model, req, run_id, t_start):
    prompt = build_prompt(req.input_tokens, f"{run_id}-{req.rid}")
    t0 = time.perf_counter()
    ttft = None
    usage = None
    n_chunks = 0
    err = None
    try:
        async with client.stream(
            "POST", f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": req.output_tokens,
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
                "ignore_eos": True,
            },
            timeout=600,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if choices and (choices[0].get("delta") or {}).get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n_chunks += 1
    except Exception as e:
        err = str(e)
    total = time.perf_counter() - t0
    out_tok = usage["completion_tokens"] if usage else n_chunks
    in_tok = usage["prompt_tokens"] if usage else None
    tpot = (total - ttft) / (out_tok - 1) if not err and ttft and out_tok and out_tok > 1 else None
    return {
        "schema_version": "0.1", "run_id": run_id, "source": "live",
        "request_id": req.rid,
        "ts_arrival_s": t0 - t_start,  # actual dispatch time
        "strategy": "open_loop_replay", "model": model, "endpoint": base_url,
        "input_tokens": in_tok, "output_tokens": out_tok, "queue_wait_s": None,
        "ttft_s": ttft, "tpot_s": tpot, "latency_s": total,
        "error": err, "cost_usd": None,
        "meta": {"intended_arrival_s": req.arrival, "intended_input_tokens": req.input_tokens,
                 "intended_output_tokens": req.output_tokens},
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--profile", default="chat")
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arrival", default="poisson", choices=["poisson", "bursty"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--machine", default="modal-l4")
    args = ap.parse_args()

    reqs = make_workload(args.profile, rate_rps=args.rate, n=args.n,
                         seed=args.seed, arrival_process=args.arrival)
    run_id = f"replay-{args.profile}-r{args.rate}-{datetime.now():%Y%m%d-%H%M%S}"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    async with httpx.AsyncClient(headers=headers,
                                 limits=httpx.Limits(max_connections=128)) as client:
        # warmup outside the measured window
        await fire(client, args.base_url, args.model, reqs[0].__class__(
            rid=-1, arrival=0, input_tokens=64, output_tokens=8), run_id, time.perf_counter())

        t_start = time.perf_counter()

        async def schedule(req):
            delay = req.arrival - (time.perf_counter() - t_start)
            if delay > 0:
                await asyncio.sleep(delay)
            return await fire(client, args.base_url, args.model, req, run_id, t_start)

        records = await asyncio.gather(*[schedule(r) for r in reqs])

    with open(out_dir / "traces.jsonl", "w") as f:
        for r in sorted(records, key=lambda x: x["request_id"]):
            f.write(json.dumps(r) + "\n")
    (out_dir / "run.json").write_text(json.dumps({
        "schema_version": "0.1", "run_id": run_id, "source": "live",
        "created_at": datetime.now().astimezone().isoformat(),
        "repo": "llm-inference-queueing",
        "config": vars(args),
        "environment": {"machine": args.machine, "notes": "open-loop workload replay for sim validation"},
        "n_requests": len(records),
        "git_commit": None,
    }, indent=2))
    ok = [r for r in records if not r["error"]]
    lag = [r["ts_arrival_s"] - r["meta"]["intended_arrival_s"] for r in ok]
    print(f"{len(ok)}/{len(records)} ok; max dispatch lag {max(lag) * 1000:.0f}ms; "
          f"window {max(r['ts_arrival_s'] + r['latency_s'] for r in ok):.1f}s")
    print(f"wrote -> {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
