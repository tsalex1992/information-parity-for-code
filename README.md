# Information Parity for Code

Code and compact reproducibility inputs for **Information Parity for Code: The
Scope of Transfer in Multilingual Code Models** by Alexander Tsvetkov and Alon
Kipnis.

This repository contains the minimal analysis path needed to inspect and
reproduce the paper's main-text tables and figures:

- Code Information Parity scoring and aggregation code;
- benchmark score summaries used in the reported analyses;
- precomputed per-model, per-language metric values;
- the Stack v2 public language-volume proxy;
- the paper's small first-party corpus files and validation records;
- corpus structure and execution validators; and
- scripts for rebuilding the main tables and figures.

The same dataset files are archived in
[Harvard Dataverse](https://doi.org/10.7910/DVN/KJ7HIH).
Raw third-party benchmark datasets and model weights are not redistributed
here.

## Dataset variants

- `data/curated_corpus_scored_v1.0.json` is the exact snapshot used for the
  paper's reported NLL results.
- `data/curated_corpus_corrected_v1.1.json` fixes two Bash discrepancies found
  during post-scoring execution validation and is recommended for future use.
- `data/corrections_v1.1.json` records the two changes and both file checksums.
- `data/renamed_corpus.json` and
  `data/portable_kernel_function_only_corpus.json` support the paper's
  robustness analyses.

The scored snapshot passes 438/440 implementation checks; the corrected
variant passes 440/440. The corresponding reports are included under `data/`.

## Setup

The locked environment uses Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Reproduce the main tables

```bash
uv run python scripts/cross_model_analysis.py
```

This rebuilds the aggregated values used in Tables 1--3 from the shipped
metric and benchmark-summary files.

## Rebuild the figures

```bash
uv run python paper_figures/build_regime_dependence_figure.py
uv run python paper_figures/build_figure_c_enhanced.py
```

## Validate or score the curated corpus

Pass either curated-corpus variant explicitly:

```bash
uv run python main.py validate-curated --corpus data/curated_corpus_scored_v1.0.json
uv run python scripts/validate_corpus_execution.py \
  --corpus data/curated_corpus_corrected_v1.1.json
```

Full execution validation requires the corresponding language toolchains on
`PATH`. A tool can instead be supplied with a `CODEIP_*` environment variable
(for example, `CODEIP_SWIFTC` or `CODEIP_RUSTC`).

To score one model on the corpus (GPU required):

```bash
uv run python main.py score-curated \
  --model Qwen/Qwen2.5-Coder-1.5B \
  --corpus data/curated_corpus_scored_v1.0.json
```

Full multi-model rescoring is not required to inspect the reported analyses;
the per-model metric files used by the aggregation scripts are included under
`results/`.
