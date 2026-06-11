# llm-inference-queueing

Discrete-event simulator for LLM inference serving: how batching policy, arrival process, context length, and prefill/decode scheduling shape tail latency and utilization.

**Research question:** When does batching improve throughput while hurting tail latency?

**Status:** core complete. Simulator initially validated on one real vLLM/L4 deployment: e2e latency predicted within 13% after error-attribution ablation (see Validation).

## Quickstart

```bash
uv sync
uv run python -m pytest                       # 10 tests
uv run python experiments/run_experiments.py   # 350 sims -> results/summary.csv (~1 min)
uv run python analysis/make_plots.py           # -> analysis/plots/*.png
```

Live capture against any OpenAI-compatible endpoint (e.g. local Ollama):

```bash
uv run python live/capture_traces.py --base-url http://localhost:11434/v1 --model qwen2.5:3b --out results/live-m3pro-qwen3b
uv run python live/fit_coefficients.py results/live-m3pro-qwen3b --label m3pro-ollama-qwen2.5-3b
```

## Claims / non-claims

**Claims.** A compact queueing model reproduces the core batching and tail-latency behavior of LLM serving; frontend-only fitting can predict end-to-end latency surprisingly well after separating per-request overhead from GPU-blocking iteration time; GPU-busy time alone is a poor load/headroom signal for batched serving.

**Non-claims.** This is not a full vLLM simulator, does not yet model vLLM V1 chunked-prefill/decode-first scheduling, does not validate across GPUs/models/workloads, and does not independently identify network/API/tokenization/GPU overhead.

## Model

Each engine iteration is either a **prefill** (cost = base + per-token × prompt tokens) or a **decode step** (one token for every active sequence; cost = base + per-seq × batch). Two schedulers:

- **Continuous (iteration-level) batching**, simplified Orca/vLLM-style: new requests join between decode steps, prefill prioritized when admissible work is waiting. It does **not** model modern vLLM V1 chunked prefill or decode-first token-budget packing.
- **Static (request-level) batching**: batch of N prefills together, decodes until the *longest* request finishes; finished sequences pad the batch.

Workload profiles (`chat`, `rag`, `agent`) are lognormal token-length distributions with Poisson or bursty (on/off) arrivals. KV footprint is reserved as input + actual output length at admission, so output-length knowledge is an explicit limitation. All runs emit the [shared trace schema](../trace-schema.md).

## Findings so far (simulated with placeholder A10-ish cost coefficients)

These figures use hand-set illustrative coefficients, not the fitted L4 profile. Treat the policy comparisons as the claim, not the absolute rps magnitudes.

1. **Continuous batching carries ~5× the load of static batch-8** on the chat workload (measured saturation: 5.10 vs 1.05 rps). Static's p95 TTFT passes 100s at loads continuous serves at sub-second TTFT. ([plot](analysis/plots/01_load_sweep.png))
2. **Bigger static batches don't fix it:** static batch-32 saturates at 1.70 rps, still below continuous batching capped at batch 8 (2.18 rps). Padding waste dominates. ([plot](analysis/plots/04_frontier.png))
3. **Busy is not headroom:** busy-time utilization is ≥0.97 at *every* load level from 30% to 98% of capacity. The server is almost never idle; it just runs underfilled batches. Batch occupancy, queue depth, TTFT/TPOT, and token throughput are the metrics that actually track load. ([plot](analysis/plots/02_p95_vs_utilization.png))
4. **Burstiness ~doubles p95 TTFT at the same mean rate** (agent workload, on/off arrivals at 4× burst intensity vs Poisson). Mean load is not enough to predict tails. ([plot](analysis/plots/03_burstiness.png))

![Continuous vs static batching](analysis/plots/01_load_sweep.png)

![Busy saturates immediately; occupancy is the real signal](analysis/plots/02_p95_vs_utilization.png)

## Validation against one real vLLM deployment (Modal L4, Qwen2.5-7B-Instruct)

| Field | Value |
|---|---|
| Engine | vLLM 0.11.0 |
| Model | Qwen2.5-7B-Instruct |
| Precision | bf16 |
| GPU / platform | NVIDIA L4 on Modal |
| Client location | Toronto laptop over public internet |
| `max_num_seqs` | 64 |
| `max_model_len` | 8192 |
| `max_num_batched_tokens` | vLLM default (not explicitly set) |
| Chunked prefill | vLLM default for this engine version/config (not explicitly set) |
| Scheduling policy | vLLM default (not explicitly set) |
| Streaming | yes |
| Tokenization location | server-side OpenAI-compatible endpoint |
| Output control | forced lengths via `ignore_eos` + `max_tokens` |

Can the simulator *predict* a real serving system? Protocol: fit coefficients from a sequential prompt-length sweep (prefill: 271.6ms + 0.288ms/token, R²=0.887) and a concurrency sweep (decode: 56.9ms + 0.67ms/seq over c∈[1,32], R²=0.702), then replay two held-out Poisson workloads (250 requests each, exact output lengths forced via `ignore_eos`) open-loop against the server, and compare predicted vs observed distributions on identical request sets.

| | p50 TTFT err | p95 TTFT err | p50 e2e err | p95 e2e err |
|---|---|---|---|---|
| raw fitted model @ 0.73 rps | −12% | −7% | +37% | +41% |
| raw fitted model @ 1.17 rps | −10% | −3% | +60% | +51% |
| overhead-attributed @ 0.73 rps | −15% | −33% | **+6%** | **+9%** |
| overhead-attributed @ 1.17 rps | −15% | −33% | **+13%** | **+11%** |

**The error had a diagnosable cause.** The raw model overestimated e2e latency 37–60% because the fitted 272ms prefill intercept, which bundles per-request overhead (network RTT, API processing, tokenization) with GPU-blocking iteration time, gets charged by the simulator as GPU time on *every prefill iteration*, stalling all decodes. Real vLLM does not serialize frontend/request overhead this way, and modern vLLM can interleave chunked prefill with decode. Reattributing 240ms of the intercept to non-blocking per-request overhead (`--request-overhead-s` ablation) collapses e2e error to +6–13% at both load levels ([CDFs](analysis/plots/05_validation_overhead-ablation.png), `results/validation*.json`). This is error attribution, not an independently measured overhead split; the remaining TTFT p95 gap (−33%) is real because observed TTFT has scheduling variance a constant offset can't model.

![Predicted vs observed CDFs](analysis/plots/05_validation_overhead-ablation.png)

Run it: `serving/vllm_modal.py` (deploy/stop on Modal), then `capture_traces.py` → `concurrency_sweep.py` → `fit_coefficients.py` → `replay_workload.py` → `analysis/validate.py`. GPU cost for the full protocol: ~$1 on an L4.

## Coefficient fitting also works locally (M3 Pro, Ollama, qwen2.5:3b)

The linear prefill model fits real hardware well: TTFT = 10.2ms + 1.93ms/token, **R² = 0.999** over a 63–2,869-token sweep (40 requests). Implied prefill throughput ~519 tok/s; decode 24.1ms/token (~42 tok/s at batch 1). Fitted profile: `profiles/m3pro-ollama-qwen2.5-3b.json`.

**Gotcha worth keeping:** the first capture run produced garbage repetitions: Ollama's prompt cache skips prefill for identical prompts (flat 0.13s TTFT on repeats vs 5.6s cold at 2.8k tokens). Fixed by putting a unique nonce at the *start* of every prompt, which breaks longest-prefix matching. Cache-aware benchmarking is mandatory.

## Limitations (honest list)

- Simulation experiments use hand-set placeholder coefficients (`CostModel` defaults), not fitted GPU numbers. Magnitudes are illustrative; the *comparisons* (continuous vs static, burstiness) are the claims.
- KV footprint (input + output) is reserved at admission, so the scheduler effectively knows output lengths in advance. No preemption/recompute is modeled.
- Prefill-prioritized scheduling only; no chunked-prefill/decode interleaving cost model; this is the main source of e2e overprediction, partially corrected by the overhead ablation rather than properly modeled.
- The 240ms request-overhead split is an ablation estimate, not independently measured (would need server-side metrics or a localhost benchmark to separate network/API/tokenize from GPU time).
- TTFT prediction is a point estimate; observed TTFT variance (scheduling jitter) is unmodeled, so predicted tails are too tight.
- Validation covers one model/GPU/workload (Qwen2.5-7B on L4, chat profile, Poisson) at two load levels below saturation.

## Next

- Collect server-side vLLM timings/metrics to independently split network, API, tokenization, queueing, prefill, decode, and streaming time.
- Model chunked prefill (mixed prefill/decode iterations) instead of the overhead ablation.
- Run a near-saturation replay and multiple seeds per load if spend stays low.
- Additional schedulers: shortest-prefill-first, deadline-aware.
- Writeup: "LLM Inference as an Operations Research Problem."
