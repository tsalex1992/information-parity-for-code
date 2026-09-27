"""Result dataclass for curated-corpus scoring."""

from dataclasses import dataclass


@dataclass
class CodeNLLProfile:
    """Scalar summary of a model forward pass on a code snippet."""

    total_bits: float
    num_tokens: int
    num_chars: int
    num_words: int
    mean_nll_nats: float
    bpt: float
    bpc: float
    mean_entropy: float
    total_entropy: float

    def to_dict(self) -> dict:
        return {
            "total_bits": self.total_bits,
            "num_tokens": self.num_tokens,
            "num_chars": self.num_chars,
            "num_words": self.num_words,
            "mean_nll_nats": self.mean_nll_nats,
            "bpt": self.bpt,
            "bpc": self.bpc,
            "mean_entropy": self.mean_entropy,
            "total_entropy": self.total_entropy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CodeNLLProfile":
        return cls(
            total_bits=d["total_bits"],
            num_tokens=d["num_tokens"],
            num_chars=d["num_chars"],
            num_words=d["num_words"],
            mean_nll_nats=d["mean_nll_nats"],
            bpt=d["bpt"],
            bpc=d["bpc"],
            mean_entropy=d["mean_entropy"],
            total_entropy=d["total_entropy"],
        )
