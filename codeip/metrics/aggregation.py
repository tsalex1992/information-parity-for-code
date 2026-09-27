"""Cross-metric aggregation over cached NLL profiles.

Iterates over cached `CodeNLLProfile` dicts and computes the per-task metric
values used by the main paper, then averages them per language.
"""

import re
from collections import defaultdict

import numpy as np


def aggregate_per_language(
    all_bits: dict[str, dict[str, dict]],
    ref_lang: str,
    tasks: dict[str, dict[str, str]],
) -> dict[str, dict[str, float]]:
    """Apply Code IP + baseline metrics per task, average per language.

    Args:
        all_bits: {task_name: {lang: CodeNLLProfile-as-dict}}
        ref_lang: reference language key (e.g. "Python")
        tasks:    {task_name: {lang: source_code}}

    Returns:
        {lang: {metric_key: mean_value, metric_key_median: median_value,
                n_tasks: int}}
    """
    per_lang: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list),
    )

    for task_name, lang_data in all_bits.items():
        if ref_lang not in lang_data:
            continue
        ref = lang_data[ref_lang]
        ref_code = tasks.get(task_name, {}).get(ref_lang, "")

        for lang, data in lang_data.items():
            if lang == ref_lang:
                continue
            lang_code = tasks.get(task_name, {}).get(lang, "")
            lang_words = data.get(
                "num_words",
                len(re.findall(r"\w+", lang_code, flags=re.UNICODE)),
            )

            # Code IP: NLL(lang) / NLL(ref)
            if ref.get("total_bits", 0) > 0 and data.get("total_bits", 0) > 0:
                per_lang[lang]["code_ip"].append(
                    data["total_bits"] / ref["total_bits"]
                )

            # Token parity: tokens(lang) / tokens(ref)
            if ref["num_tokens"] > 0:
                per_lang[lang]["token_ratio"].append(
                    data["num_tokens"] / ref["num_tokens"]
                )
                per_lang[lang]["tokenizer_parity"].append(
                    data["num_tokens"] / ref["num_tokens"]
                )

            # Char parity: chars(lang) / chars(ref)
            if ref["num_chars"] > 0:
                per_lang[lang]["char_ratio"].append(
                    data["num_chars"] / ref["num_chars"]
                )

            # LOC parity: lines(lang) / lines(ref)
            ref_lines = max(ref_code.count("\n") + 1, 1)
            lang_lines = lang_code.count("\n") + 1
            per_lang[lang]["loc_ratio"].append(lang_lines / ref_lines)

            if data.get("total_bits", 0) > 0:
                # Bits per character
                per_lang[lang]["bpc"].append(
                    data["total_bits"] / max(data["num_chars"], 1)
                )
                # Bits per predicted token  (n_tokens - 1 predictions)
                per_lang[lang]["bpt"].append(
                    data["total_bits"] / max(data["num_tokens"] - 1, 1)
                )

            # Tokenizer fertility: tokens / word-like spans
            if lang_words > 0:
                per_lang[lang]["tokenizer_fertility"].append(
                    data["num_tokens"] / lang_words
                )

    result = {}
    for lang, metrics in per_lang.items():
        result[lang] = {k: float(np.mean(v)) for k, v in metrics.items()}
        for k, v in metrics.items():
            result[lang][k + "_median"] = float(np.median(v))
        result[lang]["n_tasks"] = max(
            (len(v) for v in metrics.values()), default=0,
        )
    return result
