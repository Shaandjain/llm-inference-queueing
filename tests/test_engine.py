import numpy as np
import pytest

from simulator import CostModel, make_workload, simulate_continuous, simulate_static, summarize
from simulator.request import Request


@pytest.fixture
def cost():
    return CostModel()


def make_reqs(specs):
    return [Request(rid=i, arrival=a, input_tokens=inp, output_tokens=out)
            for i, (a, inp, out) in enumerate(specs)]


@pytest.mark.parametrize("engine", [simulate_continuous, simulate_static])
def test_conservation_and_ordering(cost, engine):
    reqs = make_workload("chat", rate_rps=5, n=300, seed=1)
    res = engine(reqs, cost)
    for r in res.requests:
        assert r.finish is not None, "every request completes"
        assert r.tokens_done == r.output_tokens
        assert r.arrival <= r.admitted <= r.first_token <= r.finish
        assert r.ttft > 0 and r.latency >= r.ttft


@pytest.mark.parametrize("engine", [simulate_continuous, simulate_static])
def test_kv_capacity_respected(engine):
    cost = CostModel(kv_capacity_tokens=8_000)
    reqs = make_workload("chat", rate_rps=20, n=200, seed=2)
    res = engine(reqs, cost)
    assert res.max_kv_reserved <= cost.kv_capacity_tokens


def test_single_request_timing(cost):
    # one request, no queueing: ttft == prefill time, latency adds decode steps
    reqs = make_reqs([(0.0, 100, 5)])
    res = simulate_continuous(reqs, cost)
    r = res.requests[0]
    assert r.ttft == pytest.approx(cost.prefill_time(100))
    expected_decode = sum(
        cost.decode_step_time(1, 100 + k) for k in range(1, 5)
    )
    assert r.latency == pytest.approx(cost.prefill_time(100) + expected_decode)


def test_static_padding_penalty(cost):
    # a short request batched with a long one is held hostage in static batching
    reqs = make_reqs([(0.0, 100, 2), (0.0, 100, 200)])
    static = simulate_static(reqs, cost, batch_size=2)
    short_static = static.requests[0]
    reqs2 = make_reqs([(0.0, 100, 2), (0.0, 100, 200)])
    cont = simulate_continuous(reqs2, cost)
    short_cont = cont.requests[0]
    # the short request itself finishes at the same time in both (it exits the
    # batch logically), but the SERVER stays busy with padding in static:
    assert static.busy_s > cont.busy_s * 0.9
    # and a third arrival behind the static batch waits much longer
    reqs3 = make_reqs([(0.0, 100, 2), (0.0, 100, 200), (0.1, 100, 2)])
    static3 = simulate_static(reqs3, cost, batch_size=2)
    reqs4 = make_reqs([(0.0, 100, 2), (0.0, 100, 200), (0.1, 100, 2)])
    cont3 = simulate_continuous(reqs4, cost)
    assert static3.requests[2].ttft > cont3.requests[2].ttft


def test_deterministic_with_seed(cost):
    a = simulate_continuous(make_workload("agent", 4, 200, seed=7), cost)
    b = simulate_continuous(make_workload("agent", 4, 200, seed=7), cost)
    assert [r.finish for r in a.requests] == [r.finish for r in b.requests]


def test_workload_mean_rate():
    reqs = make_workload("chat", rate_rps=10, n=5000, seed=3)
    duration = reqs[-1].arrival
    assert 5000 / duration == pytest.approx(10, rel=0.1)
    bursty = make_workload("chat", rate_rps=10, n=5000, seed=3, arrival_process="bursty")
    assert 5000 / bursty[-1].arrival == pytest.approx(10, rel=0.15)


def test_summarize_sanity(cost):
    res = simulate_continuous(make_workload("chat", 5, 500, seed=4), cost)
    s = summarize(res)
    assert s["ttft_p50_s"] <= s["ttft_p95_s"] <= s["ttft_p99_s"]
    assert 0 < s["utilization"] <= 1.0
    assert s["throughput_rps"] > 0


def test_oversized_request_raises():
    cost = CostModel(kv_capacity_tokens=100)
    reqs = make_reqs([(0.0, 500, 10)])
    with pytest.raises(RuntimeError):
        simulate_continuous(reqs, cost)
