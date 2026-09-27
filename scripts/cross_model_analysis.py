"""Main-paper-only aggregation for the submission code artifact.

Rebuilds the main paper tables from the shipped metric files, benchmark score
files, and the bundled Stack v2 language-volume proxy.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from codeip.config import discover_models  # noqa: E402
from codeip.language_aliases import resolve_metric_language  # noqa: E402
from codeip.loaders import load_benchmarks  # noqa: E402
from codeip.stats_utils import fisher_pooled_summary  # noqa: E402

RESULTS_DIR = ROOT / "results"
PROXY_PATH = ROOT / "proxy_data" / "stack_v2_swh_full_gb.json"
REFERENCE_LANG = "Python"
MIN_N = 4

MAIN_TABLE_SLICES = [
    "MultiPL-E (BigCode)",
    "MultiPL-E (Qwen paper)",
    "CRUXEval-X (output)",
    "IFEvalCode English Correctness",
    "IFEvalCode Chinese Correctness",
    "FullStack Bench - Advanced Programming",
    "FullStack Bench - Basic Programming",
    "CRUXEval-X (input)",
    "IFEvalCode English Instruction",
    "IFEvalCode Chinese Instruction",
    "Multi-LCB",
]

VOLUME_TABLE_SLICES = [
    "MultiPL-E (BigCode)",
    "MultiPL-E (Qwen paper)",
    "CRUXEval-X (output)",
    "Multi-LCB",
]

SLICE_LABELS = {
    "MultiPL-E (BigCode)": "MultiPL-E (BigCode)",
    "MultiPL-E (Qwen paper)": "MultiPL-E (Qwen pool)",
    "CRUXEval-X (output)": "CRUXEval-X output",
    "IFEvalCode English Correctness": "IFEvalCode correctness (En)",
    "IFEvalCode Chinese Correctness": "IFEvalCode correctness (Ch)",
    "FullStack Bench - Advanced Programming": "FullStack (Adv. Prog.)",
    "FullStack Bench - Basic Programming": "FullStack (Basic Prog.)",
    "CRUXEval-X (input)": "CRUXEval-X input",
    "IFEvalCode English Instruction": "IFEvalCode instruction (En)",
    "IFEvalCode Chinese Instruction": "IFEvalCode instruction (Ch)",
    "Multi-LCB": "Multi-LCB",
}

CONTROL_METRICS = [
    ("bpt", "B/T"),
    ("bpt_ratio", "B/T ratio"),
    ("tokenizer_parity", "TP"),
    ("tokenizer_fertility", "TF"),
]


def load_metric_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_curated_metric_values(result_file: str, metric: str) -> dict[str, float]:
    data = load_metric_file(RESULTS_DIR / result_file)
    curated = data.get("results", {}).get("curated", {})

    values = {}
    for lang, payload in curated.items():
        if metric == "bpt_ratio":
            code_ip = payload.get("code_ip")
            tokenizer_parity = payload.get("tokenizer_parity")
            if (
                isinstance(code_ip, (int, float))
                and isinstance(tokenizer_parity, (int, float))
                and math.isfinite(code_ip)
                and math.isfinite(tokenizer_parity)
                and tokenizer_parity > 0
            ):
                values[lang] = float(code_ip) / float(tokenizer_parity)
            continue

        value = payload.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values[lang] = float(value)
    return values


def align_metric_and_benchmark(
    metric_values: dict[str, float],
    benchmark_scores: dict[str, float],
    *,
    allowed_languages: set[str] | None = None,
) -> tuple[list[str], list[float], list[float]]:
    available = set(metric_values)
    languages: list[str] = []
    x_vals: list[float] = []
    y_vals: list[float] = []

    for bench_lang, raw_score in benchmark_scores.items():
        if bench_lang == REFERENCE_LANG:
            continue
        if allowed_languages is not None and bench_lang not in allowed_languages:
            continue
        metric_lang = resolve_metric_language(bench_lang, available)
        if metric_lang is None:
            continue
        try:
            x_val = float(metric_values[metric_lang])
            y_val = float(raw_score)
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(x_val) and math.isfinite(y_val)):
            continue
        languages.append(bench_lang)
        x_vals.append(x_val)
        y_vals.append(y_val)
    return languages, x_vals, y_vals


def spearman_only(x_vals: list[float], y_vals: list[float]) -> dict | None:
    if len(x_vals) < MIN_N:
        return None
    if np.allclose(x_vals, x_vals[0]) or np.allclose(y_vals, y_vals[0]):
        return None
    sp = stats.spearmanr(x_vals, y_vals)
    if not (math.isfinite(sp.statistic) and math.isfinite(sp.pvalue)):
        return None
    return {
        "spearman_r": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "n": len(x_vals),
    }


def fisher_pooled_with_ci(
    rs: list[float],
    ns: list[int],
    *,
    controls: int = 0,
) -> dict | None:
    min_effect_n = controls + 4
    valid = []
    for r_val, n_val in zip(rs, ns):
        try:
            rho = float(r_val)
            n = int(n_val)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(rho) or n < min_effect_n:
            continue
        weight = n - controls - 3
        if weight <= 0:
            continue
        valid.append((float(np.clip(rho, -0.999999, 0.999999)), n, weight))

    if len(valid) < 2:
        return None

    zs = [np.arctanh(rho) for rho, _, _ in valid]
    weights = [weight for _, _, weight in valid]
    total_weight = float(sum(weights))
    z_mean = float(np.average(zs, weights=weights))
    se = 1.0 / math.sqrt(total_weight)
    z_test = z_mean / se
    p_val = float(2 * (1 - stats.norm.cdf(abs(z_test))))
    ci_lo = float(np.tanh(z_mean - 1.96 * se))
    ci_hi = float(np.tanh(z_mean + 1.96 * se))

    return {
        "rho": float(np.tanh(z_mean)),
        "p": p_val,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "k_models": len(valid),
    }


def partial_spearman(
    metric_values: list[float],
    benchmark_values: list[float],
    proxy_values: list[float],
) -> dict | None:
    n = len(metric_values)
    if n < MIN_N:
        return None

    x = np.asarray(metric_values, dtype=float)
    y = np.asarray(benchmark_values, dtype=float)
    z = np.asarray(proxy_values, dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]) or np.allclose(z, z[0]):
        return None

    r_xy = float(stats.spearmanr(x, y).statistic)
    r_xz = float(stats.spearmanr(x, z).statistic)
    r_yz = float(stats.spearmanr(y, z).statistic)
    if not all(math.isfinite(v) for v in (r_xy, r_xz, r_yz)):
        return None

    denom_sq = (1.0 - r_xz * r_xz) * (1.0 - r_yz * r_yz)
    if denom_sq <= 1e-12:
        return None

    rho = (r_xy - r_xz * r_yz) / math.sqrt(denom_sq)
    rho = float(np.clip(rho, -0.999999, 0.999999))
    df = n - 3
    if df <= 0:
        return None
    t_stat = rho * math.sqrt(df / max(1e-12, 1.0 - rho * rho))
    p_val = float(2 * stats.t.sf(abs(t_stat), df))
    if not math.isfinite(p_val):
        return None
    return {"spearman_r": rho, "spearman_p": p_val, "n": n}


def load_stack_v2_proxy() -> dict[str, float]:
    with open(PROXY_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return {
        lang: math.log10(value)
        for lang, value in raw.items()
        if isinstance(value, (int, float)) and value > 0
    }


def summarize_table_1(discovered: dict[str, dict]) -> dict:
    rows = []
    for slice_name in MAIN_TABLE_SLICES:
        per_model = []
        rs: list[float] = []
        ns: list[int] = []
        language_counts: list[int] = []

        for model_name, model_info in discovered.items():
            metric_values = load_curated_metric_values(model_info["metrics_file"], "code_ip")
            if not metric_values:
                continue
            benchmarks = load_benchmarks(model_info["model_path"], exact_ifeval=True)
            benchmark_scores = benchmarks.get(slice_name)
            if not benchmark_scores:
                continue

            languages, x_vals, y_vals = align_metric_and_benchmark(metric_values, benchmark_scores)
            summary = spearman_only(x_vals, y_vals)
            if summary is None:
                continue

            summary["languages"] = sorted(languages)
            summary["model"] = model_name
            per_model.append(summary)
            rs.append(summary["spearman_r"])
            ns.append(summary["n"])
            language_counts.append(summary["n"])

        pooled = fisher_pooled_with_ci(rs, ns)
        if pooled is None:
            continue

        rows.append({
            "slice_key": slice_name,
            "slice_label": SLICE_LABELS[slice_name],
            "models": len(per_model),
            "language_range": [min(language_counts), max(language_counts)],
            "pooled": pooled,
            "per_model": per_model,
        })
    return {"rows": rows}


def summarize_table_2(discovered: dict[str, dict]) -> dict:
    rows = []
    for slice_name in MAIN_TABLE_SLICES:
        metric_entries = {}
        for metric_key, metric_label in CONTROL_METRICS:
            per_model = []
            rs: list[float] = []
            ns: list[int] = []

            for model_name, model_info in discovered.items():
                metric_values = load_curated_metric_values(model_info["metrics_file"], metric_key)
                if not metric_values:
                    continue
                benchmarks = load_benchmarks(model_info["model_path"], exact_ifeval=True)
                benchmark_scores = benchmarks.get(slice_name)
                if not benchmark_scores:
                    continue

                languages, x_vals, y_vals = align_metric_and_benchmark(metric_values, benchmark_scores)
                summary = spearman_only(x_vals, y_vals)
                if summary is None:
                    continue

                summary["languages"] = sorted(languages)
                summary["model"] = model_name
                per_model.append(summary)
                rs.append(summary["spearman_r"])
                ns.append(summary["n"])

            pooled = fisher_pooled_summary(rs, ns)
            metric_entries[metric_key] = {
                "label": metric_label,
                "pooled": None
                if pooled is None
                else {"rho": pooled[0], "p": pooled[1], "k_models": pooled[2]},
                "per_model": per_model,
            }

        rows.append({
            "slice_key": slice_name,
            "slice_label": SLICE_LABELS[slice_name],
            "metrics": metric_entries,
        })
    return {"rows": rows}


def summarize_table_3(discovered: dict[str, dict]) -> dict:
    proxy = load_stack_v2_proxy()
    allowed_languages = set(proxy)
    rows = []

    for slice_name in VOLUME_TABLE_SLICES:
        raw_rows = []
        proxy_rows = []
        partial_rows = []

        for model_name, model_info in discovered.items():
            metric_values = load_curated_metric_values(model_info["metrics_file"], "code_ip")
            if not metric_values:
                continue
            benchmarks = load_benchmarks(model_info["model_path"], exact_ifeval=True)
            benchmark_scores = benchmarks.get(slice_name)
            if not benchmark_scores:
                continue

            languages, code_ip_vals, benchmark_vals = align_metric_and_benchmark(
                metric_values,
                benchmark_scores,
                allowed_languages=allowed_languages,
            )
            if len(languages) < MIN_N:
                continue

            stack_vals = [proxy[lang] for lang in languages]

            raw = spearman_only(code_ip_vals, benchmark_vals)
            proxy_only = spearman_only(stack_vals, benchmark_vals)
            partial = partial_spearman(code_ip_vals, benchmark_vals, stack_vals)

            if raw is not None:
                raw["model"] = model_name
                raw["languages"] = sorted(languages)
                raw_rows.append(raw)
            if proxy_only is not None:
                proxy_only["model"] = model_name
                proxy_only["languages"] = sorted(languages)
                proxy_rows.append(proxy_only)
            if partial is not None:
                partial["model"] = model_name
                partial["languages"] = sorted(languages)
                partial_rows.append(partial)

        raw_pooled = fisher_pooled_summary(
            [row["spearman_r"] for row in raw_rows],
            [row["n"] for row in raw_rows],
        )
        proxy_pooled = fisher_pooled_summary(
            [row["spearman_r"] for row in proxy_rows],
            [row["n"] for row in proxy_rows],
        )
        partial_pooled = fisher_pooled_with_ci(
            [row["spearman_r"] for row in partial_rows],
            [row["n"] for row in partial_rows],
            controls=1,
        )

        rows.append({
            "slice_key": slice_name,
            "slice_label": SLICE_LABELS[slice_name],
            "code_ip": None
            if raw_pooled is None
            else {"rho": raw_pooled[0], "p": raw_pooled[1], "k_models": raw_pooled[2]},
            "stack_v2": None
            if proxy_pooled is None
            else {"rho": proxy_pooled[0], "p": proxy_pooled[1], "k_models": proxy_pooled[2]},
            "partial_code_ip": partial_pooled,
            "per_model": {
                "code_ip": raw_rows,
                "stack_v2": proxy_rows,
                "partial_code_ip": partial_rows,
            },
        })
    return {"rows": rows}


def sig_marker(p_val: float | None) -> str:
    if p_val is None:
        return ""
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    return ""


def render_markdown(summary: dict) -> str:
    lines = [
        "# Main-Paper Correlation Summary",
        "",
        "This report is generated from the submission code artifact only.",
        "",
        "## Table 1",
        "",
        "| Slice | Models | Languages | Code IP rho | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["table_1"]["rows"]:
        pooled = row["pooled"]
        lang_lo, lang_hi = row["language_range"]
        lang_text = str(lang_lo) if lang_lo == lang_hi else f"{lang_lo}--{lang_hi}"
        lines.append(
            f"| {row['slice_label']} | {row['models']} | {lang_text} | "
            f"{pooled['rho']:+.3f} | [{pooled['ci_lo']:+.3f}, {pooled['ci_hi']:+.3f}] |"
        )

    lines.extend([
        "",
        "## Table 2",
        "",
        "| Slice | B/T | B/T ratio | TP | TF |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in summary["table_2"]["rows"]:
        cells = []
        for metric_key, _ in CONTROL_METRICS:
            pooled = row["metrics"][metric_key]["pooled"]
            cells.append("n/a" if pooled is None else f"{pooled['rho']:+.3f}{sig_marker(pooled['p'])}")
        lines.append(f"| {row['slice_label']} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Table 3",
        "",
        "| Slice | Code IP | Stack v2 | Partial Code IP |",
        "|---|---:|---:|---:|",
    ])
    for row in summary["table_3"]["rows"]:
        raw = row["code_ip"]
        proxy = row["stack_v2"]
        partial = row["partial_code_ip"]
        raw_text = "n/a" if raw is None else f"{raw['rho']:+.3f}"
        proxy_text = "n/a" if proxy is None else f"{proxy['rho']:+.3f}"
        partial_text = "n/a" if partial is None else f"{partial['rho']:+.3f}"
        lines.append(
            f"| {row['slice_label']} | {raw_text} | {proxy_text} | {partial_text} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- IFEvalCode rows use the exact-overlap benchmark export bundled with the artifact.",
        "- FullStack rows use the processed per-category files bundled with the artifact.",
        "- `B/T ratio` is reconstructed as `Code IP / TP`, matching the exact paper decomposition.",
        "- Partial correlations control for `log10(Stack v2 SWH-full GB)` on the 16-language covered subset.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    discovered = discover_models(RESULTS_DIR)
    summary = {
        "table_1": summarize_table_1(discovered),
        "table_2": summarize_table_2(discovered),
        "table_3": summarize_table_3(discovered),
    }

    json_path = RESULTS_DIR / "cross_model_analysis.json"
    md_path = RESULTS_DIR / "correlation_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(summary))

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
