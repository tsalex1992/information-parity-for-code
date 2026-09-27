"""Baseline and surface-level metrics for the curated corpus."""

from .model_scoring import count_code_tokens, score_code


def token_parity(tokenizer, lang_code: str, ref_code: str, max_len: int) -> float:
    """Token count ratio: tokens(lang) / tokens(ref)."""
    lang_stats = count_code_tokens(tokenizer, lang_code, max_len)
    ref_stats = count_code_tokens(tokenizer, ref_code, max_len)
    return lang_stats["num_tokens"] / ref_stats["num_tokens"]


def char_parity(lang_code: str, ref_code: str) -> float:
    """Character count ratio: chars(lang) / chars(ref)."""
    return len(lang_code) / len(ref_code)


def loc_parity(lang_code: str, ref_code: str) -> float:
    """Line count ratio: lines(lang) / lines(ref)."""
    ref_lines = max(ref_code.count("\n") + 1, 1)
    lang_lines = lang_code.count("\n") + 1
    return lang_lines / ref_lines


def bpc(model, tokenizer, code: str, max_len: int) -> float:
    """Bits per character."""
    profile = score_code(model, tokenizer, code, max_len)
    return profile.total_bits / max(profile.num_chars, 1)


def bpt(model, tokenizer, code: str, max_len: int) -> float:
    """Bits per predicted token."""
    profile = score_code(model, tokenizer, code, max_len)
    return profile.total_bits / max(profile.num_tokens - 1, 1)


def tokenizer_fertility(tokenizer, code: str, max_len: int) -> float:
    """Average tokenizer pieces per word-like span."""
    stats = count_code_tokens(tokenizer, code, max_len)
    return stats["num_tokens"] / max(stats["num_words"], 1)
