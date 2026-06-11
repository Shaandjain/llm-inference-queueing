# llm-inference-queueing

Discrete-event simulator for LLM inference serving: how batching policy, arrival process, context length, and prefill/decode scheduling shape tail latency and utilization.

**Research question:** When does batching improve throughput while hurting tail latency?

**Status:** in progress (week 1 of 2). Simulator, experiments, and live coefficient fitting working end-to-end.

## Quickstart

```bash
uv sync
uv run pytest                                  # 10 tests
uv run python experiments/run_experiments.py   # 350 sims -> results/summary.csv (~1 min)
uv run python analysis/make_plots.py           # -> analysis/plots/*.png
```

Live capture against any OpenAI-compatible endpoint (e.g. local Ollama):

```bash
uv run python live/capture_traces.py --base-url http://localhost:11434/v1 --model qwen2.5:3b --out results/live-m3pro-qwen3b
uv run python live/fit_coefficients.py results/live-m3pro-qwen3b --label m3pro-ollama-qwen2.5-3b
```

## Model

Each engine iteration is either a **prefill** (cost = base + per-token × prompt tokens) or a **decode step** (one token for every active sequence; cost = base + per-seq × batch). Two schedulers:

- **Continuous (iteration-level) batching** — Orca/vLLM-style: new requests join between decode steps, prefill prioritized.
- **Static (request-level) batching** — batch of N prefills together, decodes until the *longest* request finishes; finished sequences pad the batch.

Workload profiles (`chat`, `rag`, `agent`) are lognormal token-length distributions with Poisson or bursty (on/off) arrivals. All runs emit the [shared trace schema](../trace-schema.md).

## Findings so far (simulated, placeholder A10-ish cost coefficients)

1. **Continuous batching carries ~5× the load of static batch-8** on the chat workload (measured saturation: 5.10 vs 1.05 rps). Static's p95 TTFT passes 100s at loads continuous serves at sub-second TTFT. ([plot](analysis/plots/01_load_sweep.png))
2. **Bigger static batches don't fix it:** static batch-32 saturates at 1.70 rps — still below continuous batching capped at batch 8 (2.18 rps). Padding waste dominates. ([plot](analysis/plots/04_frontier.png))
3. **"Busy" is a useless health signal:** busy-time utilization is ≥0.97 at *every* load level from 30% to 98% of capacity — the server is never idle, it just runs underfilled batches. Batch occupancy is the metric that actually tracks load. ([plot](analysis/plots/02_p95_vs_utilization.png))
4. **Burstiness ~doubles p95 TTFT at the same mean rate** (agent workload, on/off arrivals at 4× burst intensity vs Poisson). Mean load is not enough to predict tails. ([plot](analysis/plots/03_burstiness.png))

## Live validation (M3 Pro, Ollama, qwen2.5:3b)

The linear prefill model fits real hardware well: TTFT = 10.2ms + 1.93ms/token, **R² = 0.999** over a 63–2,869-token sweep (40 requests). Implied prefill throughput ~519 tok/s; decode 24.1ms/token (~42 tok/s at batch 1). Fitted profile: `profiles/m3pro-ollama-qwen2.5-3b.json`.

**Gotcha worth keeping:** the first capture run produced garbage repetitions — Ollama's prompt cache skips prefill for identical prompts (flat 0.13s TTFT on repeats vs 5.6s cold at 2.8k tokens). Fixed by putting a unique nonce at the *start* of every prompt, which breaks longest-prefix matching. Cache-aware benchmarking is mandatory.

## Limitations (honest list)

- Simulation experiments use hand-set placeholder coefficients (`CostModel` defaults), not fitted GPU numbers. Magnitudes are illustrative; the *comparisons* (continuous vs static, burstiness) are the claims.
- KV footprint (input + output) is reserved at admission — the scheduler effectively knows output lengths in advance. No preemption/recompute is modeled.
- Decode batch-size coefficient (`decode_per_seq_s`) is unfitted (needs a concurrency sweep; Ollama serializes by default).
- Prefill-prioritized scheduling only; no chunked-prefill/decode interleaving cost model.
- Apple Silicon numbers validate the *model structure*, not NVIDIA magnitudes.

## Next (week 2)

- Concurrency sweep to fit `decode_per_seq_s`; validate simulator predictions against a real vLLM endpoint on Modal (predicted vs observed p50/p95).
- Additional schedulers: shortest-prefill-first, deadline-aware.
- Writeup: "LLM Inference as an Operations Research Problem."
