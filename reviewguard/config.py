from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SENTIMENT_REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"


@dataclass
class Config:
    """Design weights and operational limits; not empirical constants."""

    weights: dict = field(
        default_factory=lambda: dict(D=0.30, C=0.25, L=0.20, R=0.15, E=0.10)
    )
    duplicate_threshold: float = 0.85
    embedding_backend: str = "sentence-transformer"
    sentiment_backend: str = "transformer"
    max_batch_size: int = 5000
    seed: int = 42

    def __post_init__(self):
        if (
            set(self.weights) != set("DCLRE")
            or any(v < 0 for v in self.weights.values())
            or abs(sum(self.weights.values()) - 1) > 1e-6
        ):
            raise ValueError("Signal weights must be nonnegative and sum to 1.")
        if not 0 < self.duplicate_threshold <= 1:
            raise ValueError("Duplicate threshold must be in (0, 1].")
