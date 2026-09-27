# Dataset mirror

This directory mirrors the paper's small first-party datasets. Harvard
Dataverse will serve as the canonical archival release; the public DOI will be
added before this repository is made public.

## Curated corpus

- `curated_corpus_scored_v1.0.json`: exact snapshot used for all reported NLL
  results.
- `curated_corpus_corrected_v1.1.json`: recommended variant fixing the Bash
  `edit_distance` and `kmp_search` discrepancies found after scoring.
- `corrections_v1.1.json`: machine-readable correction record and checksums.
- `execution_validation_scored_v1.0.json`: full report for the scored snapshot
  (438/440 implementations pass).
- `execution_validation_corrected_v1.1.json`: full report for the corrected
  variant (440/440 implementations pass).

## Robustness corpora

- `renamed_corpus.json`: variable-renaming diagnostic corpus.
- `portable_kernel_function_only_corpus.json`: non-canonical portable-kernel
  corpus.

## Benchmark-side inputs

Files under `fullstack_bench/processed/` are processed public benchmark score
inputs used to reproduce the paper's FullStack analysis. They are not the
FullStack benchmark dataset itself.
