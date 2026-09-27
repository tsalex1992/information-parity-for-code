"""Score curated-corpus code snippets with a causal language model."""

import math
import re
import types
from functools import lru_cache

import torch

from ..schemas import CodeNLLProfile


class _ById:
    """Make an unhashable object usable as an lru_cache key by identity."""

    __slots__ = ("obj",)

    def __init__(self, obj):
        self.obj = obj

    def __hash__(self):
        return id(self.obj)

    def __eq__(self, other):
        return self.obj is other.obj


def entropy_from_logits(logits: torch.Tensor) -> tuple[float, float]:
    """Shannon entropy H = -sum p(x) log2 p(x) from raw logits."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    per_token = -(probs * log_probs).sum(dim=-1) / math.log(2)
    return float(per_token.mean().item()), float(per_token.sum().item())


def nll_bits(loss_nats: float, n_predicted: int) -> float:
    """Convert mean cross-entropy loss (nats) to total NLL in bits."""
    return loss_nats * n_predicted / math.log(2)


def count_words(code: str) -> int:
    """Count word-like spans for tokenizer fertility."""
    return len(re.findall(r"\w+", code, flags=re.UNICODE))


def score_code(model, tokenizer, code: str, max_len: int) -> CodeNLLProfile:
    """Run a forward pass on code and return its full NLL profile."""
    return _score_code_cached(_ById(model), _ById(tokenizer), code, max_len)


@lru_cache(maxsize=None)
def _score_code_cached(model_w, tokenizer_w, code: str, max_len: int) -> CodeNLLProfile:
    model, tokenizer = model_w.obj, tokenizer_w.obj
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = inputs["input_ids"].to(model.device)

    n_tokens = int(input_ids.shape[1])
    n_predicted = n_tokens - 1
    n_chars = len(code)
    n_words = count_words(code)

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)

    total = nll_bits(out.loss.item(), n_predicted)
    mean_entropy, total_entropy = entropy_from_logits(out.logits[0, :-1])

    return CodeNLLProfile(
        total_bits=total,
        num_tokens=n_tokens,
        num_chars=n_chars,
        num_words=n_words,
        mean_nll_nats=out.loss.item(),
        bpt=total / max(n_predicted, 1),
        bpc=total / max(n_chars, 1),
        mean_entropy=mean_entropy,
        total_entropy=total_entropy,
    )


def count_code_tokens(tokenizer, code: str, max_len: int) -> types.MappingProxyType:
    """Tokenizer-only statistics (no model forward pass)."""
    return _count_code_tokens_cached(_ById(tokenizer), code, max_len)


@lru_cache(maxsize=None)
def _count_code_tokens_cached(tokenizer_w, code: str, max_len: int) -> types.MappingProxyType:
    tokenizer = tokenizer_w.obj
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = inputs["input_ids"]
    return types.MappingProxyType(
        {
            "num_tokens": int(input_ids.shape[1]),
            "num_chars": len(code),
            "num_words": count_words(code),
        }
    )


def clear_scoring_caches():
    """Clear all lru_cache instances."""
    _score_code_cached.cache_clear()
    _count_code_tokens_cached.cache_clear()
