"""Minimal CLI for the submission code artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath

from transformers import AutoModelForCausalLM, AutoTokenizer

from codeip.config import MAX_SEQ_LEN, RESULTS_DIR
from codeip.loaders import load_curated_corpus
from codeip.metrics.aggregation import aggregate_per_language
from codeip.metrics.model_scoring import score_code


def _model_slug(model_path: str) -> str:
    name = PurePosixPath(model_path.replace("\\", "/")).name
    return name.lower().replace("/", "_").replace(" ", "_")


def score_curated(args: argparse.Namespace) -> int:
    if not args.corpus.exists():
        print(f"ERROR: {args.corpus} not found")
        print("Download curated_corpus.json from the dataset artifact and pass --corpus.")
        return 1

    tasks, _ = load_curated_corpus(args.corpus)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    all_bits = {}
    for task_name, langs in tasks.items():
        all_bits[task_name] = {}
        for lang, code in langs.items():
            profile = score_code(model, tokenizer, code, args.max_seq_len)
            all_bits[task_name][lang] = profile.to_dict()

    metrics = aggregate_per_language(all_bits, args.reference_lang, tasks)
    output_path = args.output or (RESULTS_DIR / f"metrics_{_model_slug(args.model)}.json")
    cache_path = output_path.with_name(
        output_path.name.replace("metrics_", "total_bits_cache_curated_", 1)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(all_bits, f, indent=2)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "results": {"curated": metrics},
            },
            f,
            indent=2,
        )

    print(f"Wrote {cache_path}")
    print(f"Wrote {output_path}")
    return 0


def validate_curated(args: argparse.Namespace) -> int:
    from scripts.validate_corpus import main as validate_main

    argv = sys.argv[:1] + ["--corpus", str(args.corpus)]
    if args.summary:
        argv.append("--summary")
    sys.argv = argv
    return validate_main()


def run_analysis(_args: argparse.Namespace) -> int:
    from scripts.cross_model_analysis import main as analysis_main

    analysis_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submission code artifact CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score-curated", help="Score the curated corpus for one model")
    score.add_argument("--model", required=True, help="Hugging Face model path or local model path")
    score.add_argument(
        "--corpus",
        type=Path,
        default=Path("data") / "curated_corpus.json",
        help="Path to curated_corpus.json",
    )
    score.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination metrics JSON (defaults to results/metrics_<model>.json)",
    )
    score.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    score.add_argument("--reference-lang", default="Python")
    score.add_argument("--trust-remote-code", action="store_true")
    score.set_defaults(func=score_curated)

    validate = subparsers.add_parser("validate-curated", help="Validate the curated corpus")
    validate.add_argument(
        "--corpus",
        type=Path,
        default=Path("data") / "curated_corpus.json",
        help="Path to curated_corpus.json",
    )
    validate.add_argument("--summary", action="store_true")
    validate.set_defaults(func=validate_curated)

    analyze = subparsers.add_parser(
        "analyze-main-paper",
        help="Rebuild the main-paper tables from the shipped metrics and benchmark files",
    )
    analyze.set_defaults(func=run_analysis)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
