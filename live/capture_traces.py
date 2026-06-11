"""Capture live traces from any OpenAI-compatible endpoint into the shared
trace schema. Sequential requests (concurrency 1) with prompt lengths swept
log-scale — built for fitting prefill/decode coefficients, not load testing.

Usage:
  uv run python live/capture_traces.py --base-url http://localhost:11434/v1 \
      --model qwen2.5:3b --out results/live-m3pro-qwen3b
"""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import httpx

FILLER = (
    "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
    "Industrial systems exhibit queueing behavior whenever arrival variability "
    "meets finite service capacity, and language model inference is no exception. "
)


def build_prompt(target_tokens: int, nonce: str) -> str:
    # ~4 chars/token heuristic; the fit uses the server-reported prompt_tokens,
    # so this only needs to be roughly right.
    # The nonce goes FIRST: servers cache by longest matching prefix, and a
    # unique first token defeats the cache. With identical prompts, Ollama's
    # prompt cache skips prefill entirely and TTFT measures nothing (observed:
    # flat ~0.13s TTFT on repeats vs 1.6s cold at 2.8k tokens).
    body = (FILLER * (target_tokens * 4 // len(FILLER) + 1))[: target_tokens * 4]
    return f"[run {nonce}] " + body + "\n\nSummarize the above in one sentence."


def stream_request(client: httpx.Client, base_url: str, model: str, prompt: str, max_tokens: int):
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    usage = None
    with client.stream(
        "POST",
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        timeout=300,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
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
    in_tok = usage["prompt_tokens"] if usage else None
    out_tok = usage["completion_tokens"] if usage else n_chunks
    return ttft, total, in_tok, out_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--out", default="results/live-capture")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--machine", default="m3-pro")
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    lengths = [32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"live-{args.model.replace(':', '-').replace('/', '-')}-{datetime.now():%Y%m%d-%H%M%S}"

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    client = httpx.Client(headers=headers)
    print("warmup...")
    for _ in range(2):
        stream_request(client, args.base_url, args.model, build_prompt(64, "warmup"), 16)

    records = []
    t_start = time.perf_counter()
    for target in lengths:
        for rep in range(args.reps):
            prompt = build_prompt(target, f"{run_id}-{target}-{rep}")
            err = None
            try:
                ttft, total, in_tok, out_tok = stream_request(
                    client, args.base_url, args.model, prompt, args.max_tokens
                )
            except Exception as e:  # record failures, don't die
                ttft = total = in_tok = out_tok = None
                err = str(e)
            tpot = (total - ttft) / (out_tok - 1) if not err and out_tok and out_tok > 1 else None
            records.append({
                "schema_version": "0.1",
                "run_id": run_id,
                "source": "live",
                "request_id": len(records),
                "ts_arrival_s": time.perf_counter() - t_start,
                "strategy": "sequential_sweep",
                "model": args.model,
                "endpoint": args.base_url,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "queue_wait_s": None,
                "ttft_s": ttft,
                "tpot_s": tpot,
                "latency_s": total,
                "error": err,
                "cost_usd": None,
                "meta": {"target_tokens": target, "rep": rep, "concurrency": 1},
            })
            status = err or f"in={in_tok} ttft={ttft:.3f}s tpot={tpot * 1000:.1f}ms" if tpot else err or "short"
            print(f"  target={target:5d} rep={rep}: {status}")

    with open(out_dir / "traces.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        commit = None
    (out_dir / "run.json").write_text(json.dumps({
        "schema_version": "0.1",
        "run_id": run_id,
        "source": "live",
        "created_at": datetime.now().astimezone().isoformat(),
        "repo": "llm-inference-queueing",
        "config": vars(args),
        "environment": {"machine": args.machine, "notes": "sequential prompt-length sweep for coefficient fitting"},
        "n_requests": len(records),
        "git_commit": commit,
    }, indent=2))
    ok = sum(1 for r in records if not r["error"])
    print(f"wrote {ok}/{len(records)} successful traces -> {out_dir}")


if __name__ == "__main__":
    main()
