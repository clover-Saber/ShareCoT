# ShareCoT Experimental Results Verification Summary

**Code and configuration:** The results in Tables 4, 5, 11, and 12 were generated using the implementation in `code/` and its corresponding configurations provided in this package. See [README.md](README.md) for the mapping between the code and data directories.

**Scope:** Tables 4, 5, 11, and 12, plus the supplementary fixed 5:1 experiments, covering 47 prediction files.

**Data checks:** Predictions align with the same 1,000 test instances, with unique sample IDs and matching ground-truth labels.

## Three-run summaries

Values are mean ± sample standard deviation.

| Method | Accuracy | Macro-F1 |
| --- | --- | --- |
| ShareCoT (adaptive) | 0.9600 ± 0.0017 | 0.9413 ± 0.0024 |
| Direct-FT | 0.9520 ± 0.0020 | 0.9300 ± 0.0022 |
| Single-CoT | 0.9397 ± 0.0055 | 0.9083 ± 0.0093 |
| DHRD | 0.9357 ± 0.0051 | 0.9071 ± 0.0077 |
| ShareCoT (fixed 5:1) | 0.9593 ± 0.0032 | 0.9402 ± 0.0048 |

## Paired comparisons

Differences are ShareCoT minus comparator.

| Comparator | Macro-F1 difference | 95% CI | p-value |
| --- | --- | --- | --- |
| Direct-FT | 0.0088 | [0.0028, 0.0165] | 0.0312 |
| Shuffled | 0.0138 | [0.0016, 0.0268] | 0.0468 |

## Statistical settings

Records are paired by `sample_id`. Paired bootstrap and two-sided permutation tests each use 5,000 resamples. Permutation p-values use the plus-one correction; sample standard deviations use `ddof=1`.
