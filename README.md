# ShareCoT

ShareCoT predicts recorded ride-sharing authorization using SHAP-guided rationale supervision and a dual-head language model.

## Current manuscript implementation

The current code is in **[ShareCoT_main_code_20261020](ShareCoT_main_code_20261020/)**. Start with its **[README](ShareCoT_main_code_20261020/README.md)** for environment setup, input schemas, training, baselines, and statistical evaluation.

The package includes:

- ShareCoT, Direct-FT, Single-CoT, and DHRD training implementations.
- Tabular, neural text, and API baselines, plus the shuffled-rationale control.
- Main-experiment configuration, dependency requirements, source hashes, and offline regression tests.

The current package uses nine registered input features and targets Qwen2.5-7B-Instruct. It contains code and configuration only; formal data, accepted rationale corpora, base model weights, and historical run artifacts must be supplied separately. The package README records the changes requiring new runs and the scope of the included experiments.

Run the offline checks from the current package directory:

```bash
cd ShareCoT_main_code_20261020
python -B -m unittest discover -s tests -v
```

## Earlier implementation

The repository-root `train.py`, `evaluation.py`, and the `data/`, `cot/`, and `model/` directories belong to the earlier implementation. Use the current package and its documented input schema for the revised manuscript. Earlier files and checkpoints are not inputs or reproduced outputs of the current implementation.
