import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class CostModel:
    """Linear cost model for one engine iteration.

    A prefill iteration processes a batch of prompts; cost scales with total
    prompt tokens. A decode iteration generates one token for every active
    sequence; cost scales with batch size and (optionally) resident KV tokens.
    Defaults are hand-set placeholders in the ballpark of a 7B model on an
    A10-class GPU — they are NOT fitted. Fit real coefficients with
    live/fit_coefficients.py and load a profile instead.
    """

    prefill_base_s: float = 0.006
    prefill_per_token_s: float = 0.0002      # ~5k prompt tok/s
    decode_base_s: float = 0.012
    decode_per_seq_s: float = 0.0004
    decode_per_kv_token_s: float = 0.0       # off by default; enable to model attention cost growth
    kv_capacity_tokens: int = 160_000        # max resident (prompt + generated) tokens
    label: str = "placeholder-7b-a10"

    def prefill_time(self, total_prompt_tokens: int) -> float:
        return self.prefill_base_s + self.prefill_per_token_s * total_prompt_tokens

    def decode_step_time(self, batch_size: int, kv_tokens: int) -> float:
        return (
            self.decode_base_s
            + self.decode_per_seq_s * batch_size
            + self.decode_per_kv_token_s * kv_tokens
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "CostModel":
        return cls(**json.loads(Path(path).read_text()))
