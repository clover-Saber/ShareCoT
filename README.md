# ShareCoT Code and Data

This directory contains the ShareCoT implementation, experimental data, rationale corpora, and per-instance predictions.

The suffixes `seed_1`, `seed_2`, and `seed_3` correspond to training random seeds 44, 42, and 46, respectively.

## Code: `code/`

| Folder | Contents |
| --- | --- |
| `models_and_baselines/` | Training and prediction code for ShareCoT, Direct-FT, Single-CoT, DHRD, and other baselines, including input serialization and loss weighting. |
| `data_and_rationales/` | Scripts for data preparation, rationale generation and filtering, and construction of ablation variants. |
| `evaluation_and_statistics/` | Classification metrics, repeated-training summaries, paired statistical tests, rationale evaluation, and efficiency analysis. |
| `runtime_tools/` | Experiment execution utilities and the dependency file `requirements.txt`. |

## Data: `data/`

| Folder | Contents and corresponding experiments |
| --- | --- |
| `source_data/` | Training, validation, and test splits for the main experiment; `rationales.jsonl` contains the main rationale corpus, and `DHRD_rationales.jsonl` contains the DHRD corpus. |
| `main_experiment/` | Per-instance predictions for the methods in Table 4. |
| `multi_seed/` | Per-instance predictions from three independent training runs per method, corresponding to Table 5. |
| `cot_construction_ablation/` | Predictions for No-SHAP, No-Judge, LogitBoost-CoT, and XGBoost-CoT, corresponding to Table 11. |
| `loss_weight_sensitivity/` | Predictions for adaptive and fixed loss weighting in Table 12, together with three training runs using fixed 5:1 weighting. |
| `data_scale_robustness/` | The 50K and 100K datasets and corresponding model predictions. |
| `chronological_split/` | Data and model predictions for the chronological-split experiment. |
| `regional_robustness/` | Data and model predictions for the urban-center and suburban experiments. |
| `rationale_quality/` | Student-generated explanations, ratings from two evaluators, and records of feature replacement and classification changes. |

## How the Code Relates to the Data

- **Data and rationale preparation:** Scripts in `data_and_rationales/` prepare training data, rationale corpora, and experimental variants.
- **Training and prediction:** Code in `models_and_baselines/` reads the data splits and rationale corpus for the corresponding experiment. Prediction files are organized in the `data/` subfolders listed above.
- **Metrics and statistical verification:** `evaluation_and_statistics/main_statistics.py` reads prediction CSV files to compute classification metrics, means and sample standard deviations across three training runs, and paired comparisons of ShareCoT against Direct-FT and the shuffled-rationale control.
- **Rationale evaluation:** The relevant scripts in `evaluation_and_statistics/` analyze the ratings and feature-replacement records in `rationale_quality/`.

Prediction files identify instances using `sample_id`; `y_true` is the ground-truth label, and `y_pred` is the predicted class. Paired comparisons align records by `sample_id`. In the source data, `shared_trip_authorized` is 0 for Reject and 1 for Accept.
