# Repository manifest

- `main.py`: minimal command-line interface for corpus validation and scoring.
- `codeip/`: loaders, language alignment, scoring, metrics, and aggregation.
- `scripts/cross_model_analysis.py`: rebuilds the main analysis tables from
  shipped values.
- `scripts/validate_corpus.py`: checks corpus structure and excluded
  boilerplate.
- `scripts/validate_corpus_execution.py`: runs shared execution-equivalence
  checks across supported language toolchains.
- `benchmark_evals/`: benchmark score summaries used by the main analysis.
- `data/curated_corpus_scored_v1.0.json`: exact corpus snapshot used for the
  reported NLL results.
- `data/curated_corpus_corrected_v1.1.json`: corrected variant recommended for
  future use.
- `data/corrections_v1.1.json`: correction provenance and checksums.
- `data/execution_validation_*.json`: scored and corrected execution reports.
- `data/renamed_corpus.json` and
  `data/portable_kernel_function_only_corpus.json`: robustness corpora.
- `data/fullstack_bench/processed/`: processed public FullStack score inputs.
- `proxy_data/`: the Stack v2 public language-volume proxy.
- `results/`: precomputed metric values and aggregated main-analysis outputs.
- `paper_figures/`: figure-building scripts.

The same first-party dataset files will be archived in Harvard Dataverse with
the dataset cards and Croissant metadata.
