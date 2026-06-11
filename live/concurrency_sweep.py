"""Fit the decode batch-size term: fire c identical-shape requests
simultaneously and measure per-request TPOT as batch size varies.

All requests in a round share prompt length (~512 tok target) and a FORCED
output length (max_tokens + ignore_eos), so the decode batch stays ≈ c for
the whole round and TPOT cleanly reflects batch-size cost.

Usage:
  uv run python live/concurrency_sweep.py --base-url <url> --model <model> \
      --api-key <key> --out results/conc-sweep
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import httpx

from capture_traces import build_prompt  # same filler/nonce logic

LEVELS = [1, 2, 4, 8, 16, 32]
ROUNDS = 2
PROMPT_TOKENS = 512
OUTPUT_TOKENS = 128


async def one_request(client, base_url, model, prompt, max_tokens, t_start):
    t0 = time.perf_counter()
    ttft = None
    usage = None
    n_chunks = 0
    async with client.stream(
        "POST", f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": True,  # vLLM extension: force exact output length
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
    total = time.perf_counter() - t0
    out_tok = usage["completion_tokens"] if usage else n_chunks
    in_tok = usage["prompt_tokens"] if usage else None
    tpot = (total - ttft) / (out_tok - 1) if out_tok and out_tok > 1 else None
    return {
        "arrival": t0 - t_start, "ttft": ttft, "total": total,
        "in_tok": in_tok, "out_tok": out_tok, "tpot": tpot,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--out", default="results/conc-sweep")
    ap.add_argument("--machine", default="modal-l4")
    args = ap.parse_args()

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    run_id = f"conc-{datetime.now():%Y%m%d-%H%M%S}"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    async with httpx.AsyncClient(headers=headers, limits=httpx.Limits(max_connections=64)) as client:
        # warmup
        await one_request(client, args.base_url, args.model,
                          build_prompt(64, "warmup"), 16, time.perf_counter())
        t_start = time.perf_counter()
        for c in LEVELS:
            for rnd in range(ROUNDS):
                prompts = [build_prompt(PROMPT_TOKENS, f"{run_id}-{c}-{rnd}-{i}") for i in range(c)]
                results = await asyncio.gather(*[
                    one_request(client, args.base_url, args.model, p, OUTPUT_TOKENS, t_start)
                    for p in prompts
                ], return_exceptions=True)
                for i, r in enumerate(results):
                    if isinstance(r, Exception):
                        records.append({"error": str(r), "meta": {"concurrency": c, "round": rnd}})
                        continue
                    records.append({
                        "schema_version": "0.1", "run_id": run_id, "source": "live",
                        "request_id": len(records), "ts_arrival_s": r["arrival"],
                        "strategy": f"concurrent_{c}", "model": args.model,
                        "endpoint": args.base_url, "input_tokens": r["in_tok"],
                        "output_tokens": r["out_tok"], "queue_wait_s": None,
                        "ttft_s": r["ttft"], "tpot_s": r["tpot"], "latency_s": r["total"],
                        "error": None, "cost_usd": None,
                        "meta": {"concurrency": c, "round": rnd},
                    })
                ok = [r for r in results if not isinstance(r, Exception) and r["tpot"]]
                if ok:
                    med = sorted(r["tpot"] for r in ok)[len(ok) // 2]
                    print(f"  c={c:2d} round={rnd}: median tpot {med * 1000:.1f}ms over {len(ok)} reqs")

    with open(out_dir / "traces.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (out_dir / "run.json").write_text(json.dumps({
        "schema_version": "0.1", "run_id": run_id, "source": "live",
        "created_at": datetime.now().astimezone().isoformat(),
        "repo": "llm-inference-queueing",
        "config": {**vars(args), "levels": LEVELS, "rounds": ROUNDS,
                   "prompt_tokens": PROMPT_TOKENS, "output_tokens": OUTPUT_TOKENS},
        "environment": {"machine": args.machine, "notes": "concurrency sweep for decode_per_seq fit"},
        "n_requests": len(records),
        "git_commit": None,
    }, indent=2))
    print(f"wrote {len(records)} traces -> {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
