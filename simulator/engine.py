from collections import deque
from dataclasses import dataclass, field

from .request import Request
from .server import CostModel


@dataclass
class SimResult:
    requests: list[Request]
    busy_s: float = 0.0
    makespan_s: float = 0.0
    max_kv_reserved: int = 0
    decode_iterations: int = 0
    prefill_iterations: int = 0
    batch_size_samples: list[int] = field(default_factory=list)

    @property
    def utilization(self) -> float:
        return self.busy_s / self.makespan_s if self.makespan_s > 0 else 0.0


def simulate_continuous(
    requests: list[Request],
    cost: CostModel,
    max_batch: int = 64,
    max_prefill_tokens: int = 8192,
) -> SimResult:
    """Simplified iteration-level (continuous) batching.

    Each engine iteration is either a prefill of newly admitted requests or
    one decode step for every active sequence. Prefill is prioritized when
    admissible work is waiting. This captures the queueing structure of
    Orca/vLLM-style continuous batching, but it is not a full model of modern
    vLLM V1 scheduling: chunked prefill and decode-first token-budget packing
    are intentionally left for the next fidelity step.

    Simplification: the full KV footprint (input + output tokens) is reserved
    at admission, i.e. the scheduler knows output lengths in advance. This
    avoids modeling preemption/recompute. It makes admission conservative;
    it does not change the queueing-level tradeoffs we study here.
    """
    pending = deque(sorted(requests, key=lambda r: r.arrival))
    waiting: deque[Request] = deque()
    active: list[Request] = []
    kv_reserved = 0
    clock = 0.0
    res = SimResult(requests=requests)

    def pull_arrivals() -> None:
        while pending and pending[0].arrival <= clock:
            waiting.append(pending.popleft())

    while pending or waiting or active:
        pull_arrivals()

        if not waiting and not active:
            clock = pending[0].arrival
            continue

        # admission: fill the prefill token budget from the FIFO queue
        admit: list[Request] = []
        prefill_tokens = 0
        # the head request is always admissible regardless of the prefill
        # token budget (real engines chunk oversized prefills; cost still
        # scales with its tokens, so timing stays honest)
        while (
            waiting
            and len(active) + len(admit) < max_batch
            and kv_reserved + waiting[0].kv_footprint <= cost.kv_capacity_tokens
            and (not admit or prefill_tokens + waiting[0].input_tokens <= max_prefill_tokens)
        ):
            r = waiting.popleft()
            kv_reserved += r.kv_footprint
            prefill_tokens += r.input_tokens
            admit.append(r)

        if admit:
            for r in admit:
                r.admitted = clock
            dt = cost.prefill_time(prefill_tokens)
            clock += dt
            res.busy_s += dt
            res.prefill_iterations += 1
            for r in admit:
                r.first_token = clock
                r.tokens_done = 1
                if r.output_tokens <= 1:
                    r.finish = clock
                    kv_reserved -= r.kv_footprint
                else:
                    active.append(r)
        elif active:
            resident = sum(r.input_tokens + r.tokens_done for r in active)
            dt = cost.decode_step_time(len(active), resident)
            clock += dt
            res.busy_s += dt
            res.decode_iterations += 1
            res.batch_size_samples.append(len(active))
            for r in active[:]:
                r.tokens_done += 1
                if r.tokens_done >= r.output_tokens:
                    r.finish = clock
                    kv_reserved -= r.kv_footprint
                    active.remove(r)
        else:
            # waiting requests exist but none admissible (kv/batch full with
            # nothing active can't happen since active empty frees kv; this
            # branch means a single request exceeds capacity)
            r = waiting[0]
            raise RuntimeError(
                f"request {r.rid} (kv footprint {r.kv_footprint}) exceeds kv capacity {cost.kv_capacity_tokens}"
            )

        res.max_kv_reserved = max(res.max_kv_reserved, kv_reserved)

    res.makespan_s = max(r.finish for r in requests)
    return res


def simulate_static(
    requests: list[Request],
    cost: CostModel,
    batch_size: int = 8,
) -> SimResult:
    """Classic static (request-level) batching.

    The server takes up to `batch_size` requests, prefills them together,
    then decodes until the LONGEST request in the batch finishes. Finished
    sequences pad the batch: every decode step pays for `batch_size` slots
    and the padded KV. No new request joins mid-batch.
    """
    pending = deque(sorted(requests, key=lambda r: r.arrival))
    queue: deque[Request] = deque()
    clock = 0.0
    res = SimResult(requests=requests)

    while pending or queue:
        while pending and pending[0].arrival <= clock:
            queue.append(pending.popleft())
        if not queue:
            clock = pending[0].arrival
            continue

        # build the batch, respecting kv capacity with padding to the longest output
        batch: list[Request] = []
        sum_inputs = 0
        max_out = 0
        while queue and len(batch) < batch_size:
            r = queue[0]
            new_max = max(max_out, r.output_tokens)
            padded_kv = sum_inputs + r.input_tokens + (len(batch) + 1) * new_max
            if padded_kv > cost.kv_capacity_tokens:
                break
            queue.popleft()
            batch.append(r)
            sum_inputs += r.input_tokens
            max_out = new_max
        if not batch:
            raise RuntimeError("single request exceeds kv capacity in static batching")

        b = len(batch)
        for r in batch:
            r.admitted = clock
        dt = cost.prefill_time(sum_inputs)
        clock += dt
        res.busy_s += dt
        res.prefill_iterations += 1
        for r in batch:
            r.first_token = clock
            r.tokens_done = 1
            if r.output_tokens <= 1:
                r.finish = clock

        for step in range(1, max_out):  # steps generate token (step+1)
            resident = sum_inputs + b * (step + 1)  # padded kv
            dt = cost.decode_step_time(b, resident)
            clock += dt
            res.busy_s += dt
            res.decode_iterations += 1
            res.batch_size_samples.append(b)
            for r in batch:
                if r.finish is None and r.output_tokens == step + 1:
                    r.finish = clock
                    r.tokens_done = r.output_tokens
        res.max_kv_reserved = max(res.max_kv_reserved, sum_inputs + b * max_out)

    res.makespan_s = max(r.finish for r in requests)
    return res
