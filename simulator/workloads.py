import numpy as np

from .request import Request

# (input_tokens lognormal mean/sigma, output_tokens lognormal mean/sigma)
# Rough shapes: chat = short in / medium out, rag = long in / short out,
# agent = medium in / medium out with high variance (tool loops).
PROFILES = {
    "chat":  {"in": (5.5, 0.6), "out": (5.0, 0.7)},    # ~median 245 in, 148 out
    "rag":   {"in": (7.6, 0.5), "out": (4.6, 0.6)},    # ~median 2000 in, 100 out
    "agent": {"in": (6.6, 0.8), "out": (5.3, 0.9)},    # ~median 735 in, 200 out
}


def poisson_arrivals(rate_rps: float, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.cumsum(rng.exponential(1.0 / rate_rps, size=n))


def bursty_arrivals(
    rate_rps: float,
    n: int,
    rng: np.random.Generator,
    burst_factor: float = 4.0,
    on_fraction: float = 0.2,
) -> np.ndarray:
    """On/off arrival process with the same long-run mean rate as Poisson.

    During 'on' periods requests arrive at burst_factor * base; 'off' periods
    are quiet. on_fraction of time is 'on'. Calibrated so the overall mean
    rate equals rate_rps.
    """
    on_rate = rate_rps * burst_factor
    off_rate = rate_rps * (1 - on_fraction * burst_factor) / (1 - on_fraction)
    if off_rate <= 0:
        raise ValueError("burst_factor * on_fraction must be < 1")
    period = 10.0  # seconds per on+off cycle
    on_len = period * on_fraction
    times, t = [], 0.0
    while len(times) < n:
        phase = t % period
        rate = on_rate if phase < on_len else off_rate
        t += rng.exponential(1.0 / rate)
        times.append(t)
    return np.array(times[:n])


def sample_tokens(profile: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    p = PROFILES[profile]
    ins = np.clip(rng.lognormal(*p["in"], size=n), 8, 16384).astype(int)
    outs = np.clip(rng.lognormal(*p["out"], size=n), 1, 4096).astype(int)
    return ins, outs


def make_workload(
    profile: str,
    rate_rps: float,
    n: int,
    seed: int = 0,
    arrival_process: str = "poisson",
    **arrival_kwargs,
) -> list[Request]:
    rng = np.random.default_rng(seed)
    if arrival_process == "poisson":
        arrivals = poisson_arrivals(rate_rps, n, rng)
    elif arrival_process == "bursty":
        arrivals = bursty_arrivals(rate_rps, n, rng, **arrival_kwargs)
    else:
        raise ValueError(f"unknown arrival process: {arrival_process}")
    ins, outs = sample_tokens(profile, n, rng)
    return [
        Request(rid=i, arrival=float(arrivals[i]), input_tokens=int(ins[i]), output_tokens=int(outs[i]))
        for i in range(n)
    ]
