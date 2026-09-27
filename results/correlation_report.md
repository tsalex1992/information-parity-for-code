# Main-Paper Correlation Summary

This report is generated from the submission code artifact only.

## Table 1

| Slice | Models | Languages | Code IP rho | 95% CI |
|---|---:|---:|---:|---:|
| MultiPL-E (BigCode) | 27 | 11--15 | -0.813 | [-0.851, -0.767] |
| MultiPL-E (Qwen pool) | 28 | 5--6 | -0.675 | [-0.778, -0.538] |
| CRUXEval-X output | 18 | 18 | -0.429 | [-0.521, -0.326] |
| IFEvalCode correctness (En) | 21 | 7 | -0.545 | [-0.678, -0.377] |
| IFEvalCode correctness (Ch) | 21 | 7 | -0.607 | [-0.725, -0.454] |
| FullStack (Adv. Prog.) | 15 | 9 | -0.567 | [-0.691, -0.411] |
| FullStack (Basic Prog.) | 15 | 10 | -0.598 | [-0.707, -0.461] |
| CRUXEval-X input | 18 | 18 | -0.122 | [-0.237, -0.003] |
| IFEvalCode instruction (En) | 21 | 7 | -0.142 | [-0.343, +0.071] |
| IFEvalCode instruction (Ch) | 21 | 7 | -0.041 | [-0.250, +0.171] |
| Multi-LCB | 16 | 11 | +0.105 | [-0.068, +0.272] |

## Table 2

| Slice | B/T | B/T ratio | TP | TF |
|---|---:|---:|---:|---:|
| MultiPL-E (BigCode) | -0.777*** | -0.785*** | +0.062 | +0.140* |
| MultiPL-E (Qwen pool) | -0.637*** | -0.638*** | -0.654*** | -0.341** |
| CRUXEval-X output | -0.301*** | -0.307*** | -0.318*** | +0.041 |
| IFEvalCode correctness (En) | -0.584*** | -0.632*** | -0.207 | +0.341** |
| IFEvalCode correctness (Ch) | -0.582*** | -0.641*** | -0.272* | +0.249* |
| FullStack (Adv. Prog.) | -0.517*** | -0.510*** | +0.183 | +0.213* |
| FullStack (Basic Prog.) | -0.420*** | -0.455*** | -0.106 | +0.099 |
| CRUXEval-X input | -0.022 | -0.029 | -0.182** | +0.026 |
| IFEvalCode instruction (En) | -0.037 | -0.054 | -0.080 | +0.467*** |
| IFEvalCode instruction (Ch) | +0.067 | +0.059 | +0.011 | +0.566*** |
| Multi-LCB | -0.006 | -0.018 | +0.236** | -0.192* |

## Table 3

| Slice | Code IP | Stack v2 | Partial Code IP |
|---|---:|---:|---:|
| MultiPL-E (BigCode) | -0.708 | +0.610 | -0.550 |
| MultiPL-E (Qwen pool) | -0.675 | +0.525 | -0.550 |
| CRUXEval-X output | -0.330 | +0.175 | -0.262 |
| Multi-LCB | +0.117 | +0.574 | +0.214 |

## Notes

- IFEvalCode rows use the exact-overlap benchmark export bundled with the artifact.
- FullStack rows use the processed per-category files bundled with the artifact.
- `B/T ratio` is reconstructed as `Code IP / TP`, matching the exact paper decomposition.
- Partial correlations control for `log10(Stack v2 SWH-full GB)` on the 16-language covered subset.
