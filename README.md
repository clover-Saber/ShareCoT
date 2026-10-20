# ShareCoT

Code and available datasets for **ShareCoT: A Real-Time Expert Model for Ridesharing Willingness Prediction via Dual-Head LLM Distillation**.

ShareCoT predicts recorded ride-sharing authorization from nine trip features. A dual-head Qwen2.5-7B-Instruct student combines classification labels for all training records with auxiliary rationale supervision for an annotated subset. Classification uses the final prompt-token representation without autoregressive rationale generation.

This release includes the main-comparison implementations, repeated-run statistics, and shuffled-rationale control. The rationale corpus, historical predictions, run logs, and model weights are not included. The loss formula, serialization, and dependency configuration incorporate manuscript-alignment changes; historical results have not been revalidated with these changes.

## Repository structure

| Path | Contents |
| --- | --- |
| `code/protocol.py` | Nine-feature input protocol and prompt serialization |
| `code/train_llm.py`, `code/loss_weighting.py` | ShareCoT / Direct-FT training and adaptive or fixed loss weighting |
| `code/train_singlecot.py` | Single-head, instruction-split mixed supervision; rationale and label generation at inference |
| `code/dhrd_model.py`, `code/train_dhrd.py` | DHRD adaptation with a two-layer classification head and full-sequence LM supervision |
| `code/baselines_suite.py` | Logistic Regression, Random Forest, XGBoost, LogitBoost, and Gradient Boosting |
| `code/run_adaboost.py` | AdaBoost training and validation-based parameter selection |
| `code/baselines_text_suite.py` | Vanilla Transformer, TextCNN, and BERT text baselines |
| `code/api_suite.py` | Zero-shot and four-shot API evaluation, parsing, retries, and caching |
| `code/shuffle_rationales.py` | Seed-44 cyclic rationale reassignment for the shuffled control |
| `code/main_statistics.py` | Metrics, bootstrap intervals, repeated-run summaries, and paired tests |
| `configs/main_experiment.json` | Main-experiment settings and evaluation conventions |
| `SOURCE_MANIFEST.json` | Source provenance, code hashes, and documented packaging changes |
| `tests/` | Offline regression checks |

## Included data

| Directory | Contents |
| --- | --- |
| `data/01_existing_upstream/` | An existing merged 100,000-record upstream table, including trip metadata, weather, and POI fields |
| `data/02_feature_tables/` | Processed 10,000-, 50,000-, and 100,000-record feature tables; these retain historical fields such as `tip_paid` |
| `data/03_current_formal_main/` | Current main-experiment `train.csv`, `val.csv`, and `test.csv`, containing 8,000, 1,000, and 1,000 records |

Use the splits in `data/03_current_formal_main/` for the main experiment. They contain the registered nine inputs, sample IDs, and labels, with tipping-derived inputs excluded. These directories represent different processing stages and should not be concatenated into one dataset.

## Environment

Use a dedicated Python 3.10.0 environment and install the dependencies in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Target versions include PyTorch 2.5.1, Transformers 5.3.0, PEFT 0.12.0, and scikit-learn 1.7.2. The scikit-learn pin supports Python 3.10. Install a PyTorch build compatible with the target CUDA environment. The 7B training scripts require a CUDA GPU with BF16 support and locally available model weights.

## Prepare an experiment

Use a separate `ROOT` directory for each method and seed. Copy the three current main-experiment CSV files into this layout:

```text
ROOT/
  data/
    train.csv
    val.csv
    test.csv
    rationales.jsonl
```

Each CSV must contain exactly `sample_id`, `shared_trip_authorized`, and these nine input columns:

```text
fare
trip_miles
trip_seconds
additional_charges
pickup_to_center_distance_km
dropoff_to_center_distance_km
trip_start_timestamp
dropoff_poi_category
weather_dewpoint_2m
```

Labels are binary: `0` for Reject and `1` for Accept. Sample IDs must be unique within each split and disjoint across splits. The scripts preserve the supplied partitions.

Timestamps and POI categories in the current formal splits are readable text. Encoded category indices require their actual mappings in `ROOT/data/semantic_dictionary.json`, following `configs/semantic_dictionary.example.json`. Use the same vocabulary across all splits. Its file hash is recorded in the run configuration. Tabular baselines require parseable date-time values for the timestamp field.

Rationales must be supplied separately. Each JSONL record is aligned to a training `sample_id` and contains `reason` and `accepted`, with an optional matching `label`. The main configuration uses 946 accepted training rationales. ShareCoT and Single-CoT use the SHAP-guided corpus; DHRD requires an independent plain-teacher corpus without SHAP guidance. Direct-FT also requires the rationale file to bind input provenance, but optimizes classification loss only. Rationale construction and review scripts are outside this release's implementation scope.

## Train the LLM models

Run commands from the repository root. Replace `ROOT`, `QWEN_DIR`, and `BERT_DIR` with actual paths. Use a new experiment directory when changing the method, seed, data, or configuration.

```bash
python code/train_llm.py --root ROOT --method sharecot --model-dir QWEN_DIR --seed 44 --epochs 5 --weighting adaptive
python code/train_llm.py --root ROOT --method direct_ft --model-dir QWEN_DIR --seed 44 --epochs 5 --weighting fixed_3
python code/train_singlecot.py --root ROOT --model-dir QWEN_DIR --seed 44 --epochs 5
python code/train_dhrd.py --root ROOT --method dhrd_style --model-dir QWEN_DIR --seed 44 --epochs 5 --weighting fixed_1
```

The `fixed_3` argument for Direct-FT is required by its command-line interface; it does not activate rationale training. ShareCoT averages rationale loss over valid target tokens within each annotated record, then combines losses across all sampled records. Missing rationales mask both the rationale loss and its adaptive regularizer. The adaptive objective includes the one-half coefficients in manuscript Eq. (6), giving initial task coefficients of 0.5. Fixed reasoning-to-classification coefficients are 1:1, 3:1, and 5:1.

Checkpoints are selected by the highest validation Macro-F1. For repeated training, run seeds 43, 44, and 50 with the same data, backbone, and training settings. Seed 44 is the main-table reference run.

## Train the tabular and neural text baselines

```bash
python code/baselines_suite.py --root ROOT --method logistic_regression --seed 44 --device cpu
python code/baselines_suite.py --root ROOT --method random_forest --seed 44 --device cpu
python code/baselines_suite.py --root ROOT --method xgboost --seed 44 --device cpu
python code/baselines_suite.py --root ROOT --method logit_boost --seed 44 --device cpu
python code/baselines_suite.py --root ROOT --method gradient_boosting --seed 44 --device cpu
python code/run_adaboost.py --root ROOT --method adaboost --seed 44 --device cpu
python code/baselines_text_suite.py --root ROOT --method transformer --seed 44 --device cpu --bert-model BERT_DIR --max-length 256
python code/baselines_text_suite.py --root ROOT --method textcnn --seed 44 --device cpu --bert-model BERT_DIR --max-length 256
python code/baselines_text_suite.py --root ROOT --method bert --seed 44 --device cpu --bert-model BERT_DIR --max-length 256
```

The three neural text baselines use English serialization of the nine features and the same local BERT tokenizer. Qwen uses the Chinese chat-template prompts implemented in `protocol.py`. Both serialization paths apply category mappings when needed and include numerical units. The CPU commands retain the main-experiment training configuration.

## Shuffled-rationale control

Prepare an independent `SHUFFLED_ROOT` with the same train, validation, and test CSV files and any required semantic dictionary. Reassign accepted training rationales while preserving each recipient's classification label and rationale-coverage mask:

```bash
python code/shuffle_rationales.py --input MAIN_ROOT/data/rationales.jsonl --output SHUFFLED_ROOT/data/rationales.jsonl
python code/train_llm.py --root SHUFFLED_ROOT --method sharecot --model-dir QWEN_DIR --seed 44 --epochs 5 --weighting adaptive
```

## API baselines

Set the environment variables shown in `.env.example` in your shell. The template is not loaded automatically.

```bash
python code/api_suite.py --root API_ROOT --workers 6
python code/api_suite.py --root NEW_GEMINI_ROOT --workers 6 --models gemini-3.1-pro-preview --shots 0
```

Four-shot evaluation uses `train.csv` and accepted `rationales.jsonl` entries to select two training demonstrations per class. Zero-shot-only evaluation requires `test.csv` and a semantic dictionary only when category values are encoded. Sample IDs are read as strings, preserving leading zeros. The supplied `test.csv` defines the evaluated records.

API calls use temperature 0 and a 512-token output limit. Provider aliases and server-side versions can change. Cached responses are bound to the script, serialization protocol, semantic dictionary, and endpoint; changed conditions require a new experiment directory.

## Statistical evaluation

```bash
python code/main_statistics.py model --predictions PREDICTIONS.csv --resamples 1000
python code/main_statistics.py paired --reference SHARECOT.csv --comparator DIRECT_FT.csv --resamples 5000
python code/main_statistics.py paired --reference SHARECOT.csv --comparator SHUFFLED.csv --resamples 5000
python code/main_statistics.py multiseed --predictions SEED43.csv SEED44.csv SEED50.csv
```

Prediction files require `sample_id,y_true,y_pred`. Paired comparisons align records by ID and verify matching ground-truth labels. The difference is reference Macro-F1 minus comparator Macro-F1. Confidence intervals use paired percentile bootstrap resampling. Two-sided permutation tests independently swap the two predictions for each record with probability 0.5 and apply an add-one correction. Repeated-run summaries use the sample standard deviation.

## Baseline supervision protocols

- **Single-CoT:** annotated records request and supervise `reason,pred`; label-only records request and supervise `pred`. Validation and test prompts request `reason,pred`. The model uses a single language-model head.
- **DHRD:** a two-layer MLP classification head reads the prompt boundary. Full-sequence LM supervision is combined with classification at a fixed 1:1 ratio, using independent plain-teacher rationales.
- **Neural text baselines:** Vanilla Transformer, TextCNN, and BERT share the English feature serialization and BERT tokenizer implemented in `baselines_text_suite.py`.

## Offline regression checks

```bash
python -B -m unittest discover -s tests -v
```

The checks cover Eq. (6), missing-rationale masking, coefficient reporting, category dictionaries, units, feature-only input boundaries, string IDs, zero-shot independence, and API cache identity. Formula checks execute the actual loss methods through a scalar adapter; they do not replace PyTorch autograd or full Qwen/PEFT integration tests. API tests use an offline transport substitute and make no external calls.

Scripts write outputs to the specified experiment directories. Full training reproduction additionally requires the corresponding rationale corpora, local backbone weights, and suitable hardware.
