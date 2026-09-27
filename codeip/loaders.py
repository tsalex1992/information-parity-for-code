import json
import math
from collections import defaultdict
from pathlib import Path, PurePosixPath

from .config import (
    BENCH_DIR,
    CRUXEVAL_LANG_MAP,
    CRUXEVAL_MODEL_MAP,
    CRUXEVAL_WEB_MODEL_MAP,
    DATA_DIR,
    FULLSTACK_MODEL_MAP,
    MODEL_BENCHMARK_MAP,
    MULTI_COL_TO_LANG,
    MULTI_LCB_MODEL_MAP,
    QWEN_PAPER_EXCLUDE_LANGS,
    QWEN_PAPER_MODEL_MAP,
)


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize_model_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _model_path_basename(model_path: str) -> str:
    normalized = model_path.replace("\\", "/").rstrip("/")
    if not normalized:
        return ""
    return PurePosixPath(normalized).name


def _resolve_benchmark_model(model_path: str) -> str | None:
    direct = MODEL_BENCHMARK_MAP.get(model_path)
    if direct:
        return direct

    substring = next(
        (val for key, val in MODEL_BENCHMARK_MAP.items() if key in model_path),
        None,
    )
    if substring:
        return substring

    requested_basename = _model_path_basename(model_path)
    if not requested_basename:
        return None

    for key, val in MODEL_BENCHMARK_MAP.items():
        if requested_basename == _model_path_basename(key):
            return val
    return None


def load_curated_corpus(
    path: Path | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Load the main-paper curated corpus from JSON."""
    corpus_path = path or (DATA_DIR / "curated_corpus.json")
    with open(corpus_path, encoding="utf-8") as f:
        raw = json.load(f)

    tasks: dict[str, dict[str, str]] = {}
    descriptions: dict[str, str] = {}
    for algo, payload in raw.items():
        descriptions[algo] = payload.get("_description", "")
        tasks[algo] = {
            lang: code for lang, code in payload.items() if lang != "_description"
        }
    return tasks, descriptions


def summarize_language_coverage(tasks: dict[str, dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for langs in tasks.values():
        for lang in langs:
            counts[lang] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _extract_ifevalcode_scores(
    model_path: str,
    bench_model: str,
    *,
    exact_overlap: bool,
) -> dict[str, dict[str, float]]:
    if exact_overlap:
        path = BENCH_DIR / "ifevalcode" / "overlap_scores.json"
        if not path.exists():
            return {}
        data = _load_json(path)
        mapping = data.get("local_model_map", {}).get(model_path)
        if not isinstance(mapping, dict) or mapping.get("mapping_quality") != "exact":
            return {}
        matched_name = mapping.get("benchmark_model")
        if not isinstance(matched_name, str):
            return {}
        raw_scores = data.get("models", {}).get(matched_name, {}).get("scores", {})
    else:
        path = BENCH_DIR / "ifevalcode" / "scores.json"
        if not path.exists():
            return {}
        data = _load_json(path)
        target_norm = _normalize_model_name(bench_model)
        matched_name = next(
            (
                model_name
                for model_name in data.get("models", {})
                if _normalize_model_name(model_name) == target_norm
            ),
            None,
        )
        if matched_name is None:
            return {}
        raw_scores = data["models"][matched_name].get("scores", {})

    slice_map = {
        "english_correctness": "IFEvalCode English Correctness",
        "english_instruction": "IFEvalCode English Instruction",
        "chinese_correctness": "IFEvalCode Chinese Correctness",
        "chinese_instruction": "IFEvalCode Chinese Instruction",
    }
    out: dict[str, dict[str, float]] = {}
    for raw_key, slice_name in slice_map.items():
        slice_scores = raw_scores.get(raw_key, {})
        cleaned = {}
        for lang, val in slice_scores.items():
            try:
                numeric = float(val)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric) and numeric >= 0:
                cleaned[lang] = numeric
        if cleaned:
            out[slice_name] = cleaned
    return out


def _extract_fullstack_processed_scores(bench_model: str) -> dict[str, dict[str, float]]:
    processed_dir = DATA_DIR / "fullstack_bench" / "processed"
    if not processed_dir.exists():
        return {}

    target_norm = _normalize_model_name(bench_model)
    matched_path = None
    for path in sorted(processed_dir.glob("*.json")):
        payload = _load_json(path)
        canonical_model = payload.get("canonical_model")
        source_file = payload.get("source_file", "")
        path_stem = path.stem[:-4] if path.stem.endswith("_fsb") else path.stem
        source_stem = Path(source_file).stem
        source_stem = source_stem[:-4] if source_stem.endswith("_fsb") else source_stem

        candidate_norms = {
            _normalize_model_name(value)
            for value in (canonical_model, path_stem, source_stem)
            if value
        }
        if target_norm in candidate_norms:
            matched_path = path
            break

    if matched_path is None:
        return {}

    payload = _load_json(matched_path)
    canonical_categories = [
        "Advanced Programming",
        "Basic Programming",
        "Data Analysis",
        "Desktop and Web Development",
        "Operating System",
        "Others",
        "Software Engineering",
    ]

    out: dict[str, dict[str, float]] = {}
    by_category = payload.get("by_category", {})
    for category in canonical_categories:
        cat_scores = by_category.get(category, {})
        cleaned = {}
        for lang, stats in cat_scores.items():
            try:
                numeric = float(stats["pass@1"])
            except (TypeError, ValueError, KeyError):
                continue
            if math.isfinite(numeric) and numeric >= 0:
                cleaned[lang] = numeric
        if cleaned:
            out[f"FullStack Bench - {category}"] = cleaned
    return out


def load_benchmarks(
    model_path: str,
    *,
    exact_ifeval: bool = False,
) -> dict[str, dict[str, float]]:
    """Load the benchmark slices used in the main paper."""
    bench_model = _resolve_benchmark_model(model_path)
    if not bench_model:
        return {}

    benchmarks: dict[str, dict[str, float]] = {}

    bigcode_path = BENCH_DIR / "multiple_bigcode" / "scores.json"
    if bigcode_path.exists():
        bigcode_data = _load_json(bigcode_path)
        col_to_lang = bigcode_data.get("column_to_language", MULTI_COL_TO_LANG)
        normalized_bench_model = _normalize_model_name(bench_model)
        for model_key, model_scores in bigcode_data.get("models", {}).items():
            if normalized_bench_model != _normalize_model_name(model_key):
                continue
            scores = {}
            for col, val in model_scores.items():
                lang = col_to_lang.get(col, col)
                try:
                    numeric = float(val)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(numeric) and numeric >= 0:
                    scores[lang] = numeric
            if scores:
                benchmarks["MultiPL-E (BigCode)"] = scores
            break

    multi_lcb_path = BENCH_DIR / "multi_lcb" / "scores.json"
    multi_lcb_key = MULTI_LCB_MODEL_MAP.get(bench_model)
    if multi_lcb_key and multi_lcb_path.exists():
        multi_lcb_data = _load_json(multi_lcb_path)
        model_scores = multi_lcb_data.get("models", {}).get(multi_lcb_key, {})
        cleaned = {}
        for lang, val in model_scores.items():
            try:
                numeric = float(val)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric) and numeric >= 0:
                cleaned[lang] = numeric
        if cleaned:
            benchmarks["Multi-LCB"] = cleaned

    qwen_path = BENCH_DIR / "multiple_qwen" / "scores.json"
    qwen_key = QWEN_PAPER_MODEL_MAP.get(bench_model)
    if qwen_key and qwen_path.exists():
        qwen_data = _load_json(qwen_path)
        model_scores = qwen_data.get("models", {}).get(qwen_key, {})
        if model_scores:
            benchmarks["MultiPL-E (Qwen paper)"] = {
                lang: float(score)
                for lang, score in model_scores.items()
                if lang not in QWEN_PAPER_EXCLUDE_LANGS
            }

    crux_path = BENCH_DIR / "cruxeval_x" / "scores_paper.json"
    if crux_path.exists() and bench_model in CRUXEVAL_MODEL_MAP:
        crux_data = _load_json(crux_path)
        for crux_name, crux_size in CRUXEVAL_MODEL_MAP[bench_model]:
            for entry in crux_data.get("models", []):
                if entry.get("name") != crux_name or entry.get("size") != crux_size:
                    continue
                input_scores = {}
                output_scores = {}
                for code, lang in CRUXEVAL_LANG_MAP.items():
                    if code in entry.get("input_prediction", {}):
                        input_scores[lang] = entry["input_prediction"][code]
                    if code in entry.get("output_prediction", {}):
                        output_scores[lang] = entry["output_prediction"][code]
                if input_scores:
                    benchmarks["CRUXEval-X (input)"] = input_scores
                if output_scores:
                    benchmarks["CRUXEval-X (output)"] = output_scores
                break

    crux_web_path = BENCH_DIR / "cruxeval_x" / "scores_web.json"
    crux_web_key = CRUXEVAL_WEB_MODEL_MAP.get(bench_model)
    if crux_web_key and crux_web_path.exists():
        crux_web_data = _load_json(crux_web_path)
        entry = crux_web_data.get(crux_web_key, {})
        pass_at_1 = entry.get("pass@1", {})
        input_scores = {}
        output_scores = {}
        for key, val in pass_at_1.items():
            if "_" not in key:
                continue
            code, split = key.rsplit("_", 1)
            lang = CRUXEVAL_LANG_MAP.get(code)
            if lang is None:
                continue
            try:
                numeric = float(val)
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(numeric) and numeric >= 0):
                continue
            if split == "input":
                input_scores[lang] = numeric
            elif split == "output":
                output_scores[lang] = numeric
        if input_scores and "CRUXEval-X (input)" not in benchmarks:
            benchmarks["CRUXEval-X (input)"] = input_scores
        if output_scores and "CRUXEval-X (output)" not in benchmarks:
            benchmarks["CRUXEval-X (output)"] = output_scores

    benchmarks.update(
        _extract_ifevalcode_scores(model_path, bench_model, exact_overlap=exact_ifeval)
    )
    benchmarks.update(
        _extract_fullstack_processed_scores(FULLSTACK_MODEL_MAP.get(bench_model, bench_model))
    )

    return benchmarks
