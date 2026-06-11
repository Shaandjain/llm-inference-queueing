<!-- Website draft for shaanjain.com. Images live in analysis/plots/ in the repo;
     copy 01, 02, 03, 05_validation_overhead-ablation into the site's assets. -->

# Queueing the Transformer: why AI agents feel slow before the GPU is full

I build AI systems for clients during the day. The complaint is never "the model is dumb." It's "the agent is slow." This summer I went one layer down to understand why, and the answer turned out to live in my industrial engineering coursework, not in the model weights: LLM serving is a queueing problem.

So I built a discrete-event simulator of batched LLM inference, the kind of thing you'd build to model a factory floor or a hospital ER, and then did the part that usually gets skipped: I tested whether it could predict a real serving system, not just draw plausible curves.

**TL;DR:** a two-coefficient-per-phase linear cost model, fitted from 40 cheap probe requests, predicted real vLLM end-to-end latency within 6 to 13 percent on held-out workloads. Getting there required diagnosing a 40 to 60 percent systematic error first, and that diagnosis is the most useful thing in this post. Total GPU bill for the validation: about $1.

## The model is almost embarrassingly simple

An LLM server runs iterations. A prefill iteration ingests prompts and costs `base + per_token × prompt_tokens`. A decode iteration emits one token for every active sequence and costs `base + per_seq × batch_size`. Requests arrive randomly, wait in a queue, get batched, and leave. That's the whole model. Four coefficients.

From just that structure, the simulator reproduces the canonical serving results. Continuous batching (vLLM-style, where requests join between decode steps) carries about 5x the load of classic static batching. Cranking static batch size to 32 doesn't fix it; padding waste means it still loses to continuous batching capped at 8.

![Continuous vs static batching](01_load_sweep.png)

## The finding I didn't expect: "busy" is a lie

I went to plot the classic queueing hockey stick, p95 latency against utilization, and got a degenerate chart. Busy-time utilization was at least 97 percent at every load level, from 30 percent of capacity all the way to 98 percent.

The server never idles. At low load it just runs tiny, underfilled batches, paying nearly the full iteration cost to produce a trickle of tokens. The metric that actually tracks load is decode batch occupancy.

![Busy saturates immediately](02_p95_vs_utilization.png)

If your GPU dashboard says "99 percent busy," you have learned approximately nothing about your headroom. Watch batch occupancy and queue depth instead.

Also from the simulator: at the same mean arrival rate, bursty traffic (think agents doing tool-call fan-outs and retries) roughly doubles p95 time-to-first-token compared to smooth Poisson arrivals. Capacity planning on mean load systematically under-provisions agentic workloads.

## Then I made it face reality

A simulator that only makes pretty charts is decoration. The test protocol:

1. Deploy vLLM with Qwen2.5-7B-Instruct on an NVIDIA L4 (Modal, scale-to-zero).
2. Fit the four coefficients from cheap probes: a 40-request prompt-length sweep for prefill, a concurrency sweep for decode.
3. Have the simulator predict the L4's saturation point (it said 1.46 requests/s).
4. Replay two held-out workloads the fitting never saw, open-loop, at 50 and 80 percent of that predicted capacity.
5. Compare predicted vs observed latency distributions on identical request sets.

The fits themselves were satisfying. Prefill: TTFT = 271.6ms + 0.288ms per token. Decode: 56.9ms + 0.67ms per sequence, which means batching is nearly free up to batch 16. That one fitted line is the entire economic argument for continuous batching, measured directly.

## The 60 percent error that taught me the most

First validation results: time-to-first-token predicted within 12 percent. End-to-end latency overpredicted by 37 to 60 percent. Ouch.

But the error had a shape, and the shape pointed at the cause. That 272ms prefill intercept I fitted? It's mostly not GPU time. It's network round trips from my laptop in Toronto, API processing, tokenization. Per-request overhead. My simulator was charging it as GPU-blocking time on every prefill iteration, during which every other request's decode stalls. Real vLLM overlaps prefill with decode, and a network round trip blocks nobody.

Reattribute 240ms of the intercept as non-blocking per-request overhead and the end-to-end error collapses to 6 to 13 percent at both load levels. The predicted and observed latency CDFs nearly coincide.

![Predicted vs observed](05_validation_overhead-ablation.png)

The transferable lesson: **a latency intercept fitted over a network conflates two kinds of overhead, and any model that serializes per-request overhead across concurrent work will overpredict latency under load.** If you're benchmarking inference from outside the datacenter, you will hit this.

Honorable mention for a second trap: my first fitting run repeated identical prompts, and the server's prefix cache silently skipped prefill on the repeats. Flat 0.13s TTFT at every prompt length, versus 5.6s cold. Three of my four "repetitions" were measuring the cache. Every benchmark prompt now starts with a unique nonce.

## What this is and isn't

None of the simulation findings are new science; Orca and the vLLM paper established the batching results, and serious serving simulators like Vidur exist. What I wanted, and got, was the full loop: measure a real system, model it, predict, be wrong, diagnose why, quantify the fix, and state the remaining error honestly (predicted TTFT tails are too tight by 33 percent, because a constant offset can't model scheduling jitter; it's in the limitations section).

Everything is open: simulator, experiments, raw traces, fitted profiles, the paper, and one-command reproduction. The repo is [github.com/Shaandjain/llm-inference-queueing](https://github.com/Shaandjain/llm-inference-queueing).

Next up in the series: deriving headroom rules from queueing theory ("given this burstiness, run at X percent utilization to hold your p95") and checking them against the traces I already have. After that, the same measurement discipline goes after a bigger question: when should an agent use RAG, long context, or memory? Context is an inference budget. More soon.
