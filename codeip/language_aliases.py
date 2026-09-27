"""Shared language-label alias handling for benchmark alignment."""

from __future__ import annotations


LANGUAGE_ALIASES = {
    "Shell": "Bash",
    "Bash": "Shell",
}


def resolve_metric_language(
    benchmark_language: str,
    available_languages: set[str] | list[str] | tuple[str, ...],
) -> str | None:
    """Map a benchmark language label onto locally available metric labels."""
    available = set(available_languages)
    if benchmark_language in available:
        return benchmark_language
    alias = LANGUAGE_ALIASES.get(benchmark_language)
    if alias in available:
        return alias
    return None
