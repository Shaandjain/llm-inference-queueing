<!-- Website draft for shaanjain.com. Images live in analysis/plots/ in the repo;
     copy 01, 02, 03, 05_validation_overhead-ablation into the site's assets. -->

# Queueing the Transformer: why AI agents feel slow before the GPU is full

There's a specific moment I've lived through more times than I'd like. I'm demoing an AI system to a client, someone asks it a question, and then we all sit there watching the cursor blink. Three seconds. Five. The answer that eventually streams out is good, but the silence already did its damage. Nobody in that room is thinking "the model is dumb." They're thinking "this is slow."

That's my day job. I'm a forward-deployed engineer at an AI studio, which means I build agents, retrieval systems, and MCP servers that real teams at real companies use, things like document Q&A over dense legal text and multi-step agent workflows. I live at the last mile between a model and a user. And at the last mile, latency isn't a metric. It's the product.

The question that started this project came from a client, paraphrased: "this works great in the demo, but what happens when our whole team is on it at once?" My honest answer at the time was a shrug with extra words. The serving layer underneath my apps was a black box I paid by the token. I decided to stop shrugging.

## The part where my degree finally pays off

Here's the thing: I'm an industrial engineering student. I've spent semesters on queueing theory and discrete-event simulation, modeling factory lines and service systems where things arrive randomly, wait in line, get processed in batches, and leave. Wait times, utilization, tail behavior under load.

At some point it clicked that a GPU serving an LLM is exactly that object. Requests arrive stochastically. They wait in a queue. They get batched. They occupy a server through two distinct phases: prefill (reading your prompt) and decode (writing the answer one token at a time). They contend for memory. Every reason my clients' agents feel slow lives in that pipeline, and it's the same math as the factory floor.

So I built the thing my coursework trained me to build: a discrete-event simulator of LLM serving. Then I did the part that usually gets skipped in side projects. I tested whether it could predict a real serving system, not just draw plausible curves.

**TL;DR:** a cost model with two linear coefficients per phase, fitted from 40 cheap probe requests, predicted real vLLM end-to-end latency within 6 to 13 percent on workloads it had never seen. Getting there required diagnosing a 40 to 60 percent systematic error first, and that diagnosis taught me more than the success did. Total GPU bill: about a dollar.

## What the simulator showed

The model is almost embarrassingly simple. A prefill step costs `base + per_token × prompt_tokens`. A decode step emits one token for every active request and costs `base + per_seq × batch_size`. Four coefficients. From just that structure, the canonical serving results fall out: continuous batching (vLLM-style, where requests join between decode steps) carries about 5x the load of old-school static batching, and cranking the static batch size up doesn't save it.

![Continuous vs static batching](01_load_sweep.png)

But the finding that changed how I'll think about production systems was an accident. I went to plot the classic queueing hockey stick, p95 latency against utilization (p95 meaning the experience of your unluckiest one-in-twenty users), and got a broken-looking chart. Busy-time utilization was pinned at 97+ percent at every load level, from 30 percent of capacity to 98.

The server never idles. At low load it just runs tiny, underfilled batches, paying nearly the full cost of each step to produce a trickle of tokens. The signal that actually tracks load is batch occupancy: how full the batches are, not whether the GPU looks busy.

![Busy saturates immediately](02_p95_vs_utilization.png)

This one matters for my day job directly. "GPU busy: 99%" on a dashboard tells you nothing about headroom. And the related result: bursty traffic, which is exactly what agents generate with their tool-call fan-outs and retries, roughly doubles tail latency compared to smooth traffic at the same average rate. When a client asks "can this handle the team?", average load was never the right thing to estimate. The bursts are.

## Making it face reality

A simulator that only makes charts is decoration. So: deploy vLLM with a 7B model on a rented NVIDIA L4, fit the four coefficients from cheap probes, have the simulator predict the GPU's saturation point (it said 1.46 requests/s), then replay two held-out workloads against the real server at 50 and 80 percent of that predicted capacity, open-loop, meaning requests fire on schedule whether or not earlier ones have finished, just like real users. Compare predicted versus observed latency distributions.

First result: time-to-first-token predicted within 12 percent. End-to-end latency overpredicted by 37 to 60 percent. Ouch.

But the error had a shape, and the shape pointed at the cause. The 272ms intercept I'd fitted onto prefill was mostly not GPU time at all. It was network round trips from my laptop in Toronto, API processing, tokenization. Per-request overhead. My simulator was charging it as GPU-blocking time on every prefill, during which every other request's decode stalls. The real server overlaps that work; a network round trip blocks nobody. Reattribute 240ms of the intercept as non-blocking overhead and rerun: end-to-end error collapses to 6 to 13 percent. The predicted and observed distributions nearly coincide.

![Predicted vs observed](05_validation_overhead-ablation.png)

The transferable lesson, and the one I'd put on a poster for anyone benchmarking inference from outside the datacenter: **a latency intercept fitted over a network mixes two kinds of overhead, and any model that serializes per-request overhead across concurrent work will overpredict latency under load.**

Runner-up lesson: my first fitting run repeated identical prompts, and the server's prefix cache silently skipped prefill on the repeats. Three of my four "measurements" at each prompt length were measuring the cache, not the model. Every benchmark prompt I send now starts with a unique nonce. If your eval or load test reuses prompts, check whether you're testing the model or its cache.

## Why an applications engineer should care about any of this

I didn't do this to become a kernel engineer. I did it because the inference layer decides what my products can feel like, and "the model is thinking" stopped being an acceptable thing for me to say to a client. After two weeks and a dollar of GPU time, I can answer capacity questions with a fitted model instead of a shrug, I know which dashboard metrics are lies, and I know two benchmarking traps that would have quietly corrupted any latency numbers I put in front of a stakeholder.

None of the simulation findings are new science; the batching results date to the Orca and vLLM papers, and serious serving simulators exist. What I wanted was the full loop: measure a real system, model it, predict, be wrong, diagnose why, quantify the fix, and state the remaining error honestly (predicted TTFT tails are too tight by 33 percent; it's in the limitations section of the writeup).

Everything is open: the simulator, raw traces, fitted profiles, a paper-style writeup, and one-command reproduction. [github.com/Shaandjain/llm-inference-queueing](https://github.com/Shaandjain/llm-inference-queueing).

Next in the series: turning the queueing frame into operating rules ("given this burstiness, run at X percent utilization to hold your p95"), and then pointing the same measurement discipline at the question my client work actually revolves around: when should an agent use RAG, long context, or memory? Context is an inference budget. More soon.
