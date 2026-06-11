from dataclasses import dataclass, field


@dataclass
class Request:
    rid: int
    arrival: float          # seconds, relative to run start
    input_tokens: int
    output_tokens: int      # total tokens to generate, >= 1

    # filled in by the engine
    admitted: float | None = None      # entered the engine (left the queue)
    first_token: float | None = None   # end of prefill; prefill emits token 1
    finish: float | None = None
    tokens_done: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def ttft(self) -> float:
        return self.first_token - self.arrival

    @property
    def latency(self) -> float:
        return self.finish - self.arrival

    @property
    def queue_wait(self) -> float:
        return self.admitted - self.arrival

    @property
    def tpot(self) -> float | None:
        if self.output_tokens <= 1:
            return None
        return (self.finish - self.first_token) / (self.output_tokens - 1)

    @property
    def kv_footprint(self) -> int:
        return self.input_tokens + self.output_tokens
