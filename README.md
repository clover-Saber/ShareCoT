# Ridesharing Dual-Head (Open Code)

This repository provides a dual-head framework for **ridesharing authorization prediction**:

- **Head A (LM Head)**: generates structured CoT/rationales in JSON format.
- **Head B (Classification Head)**: predicts binary labels directly (`0/1`).

The key idea is to reuse Qwen hidden states and jointly optimize generation and classification in one model.

---

## 1. Project Structure

```text
open_code/
├── train.py                  # Dual-head joint fine-tuning (LoRA)
├── evaluation.py             # Dual-head evaluation (cls head + optional gen head)
├── data/
│   ├── chicago_train.csv
│   ├── chicago_val.csv
│   └── chicago_test.csv
├── cot/
│   ├── train_with_cot.csv
│   └── val_with_cot.csv
└── model/
    ├── checkpoint-*/
    └── final/
```

---

## 2. Requirements

- Python 3.10+
- CUDA + PyTorch (compatible with your GPU driver)
- Main dependencies:
  - `torch`
  - `transformers`
  - `peft`
  - `accelerate`
  - `pandas`
  - `scikit-learn`
  - `tqdm`

A dedicated virtual environment is recommended.

---

## 3. Data Format

### 3.1 Base tables (`data/chicago_*.csv`)

Must include the target column:

- `shared_trip_authorized` (`0/1`)

Other columns are model features (currently Top-10 feature setup).

### 3.2 CoT tables (`cot/train_with_cot.csv`, `cot/val_with_cot.csv`)

At minimum:

- `row_index`: aligned with row index in base tables
- `reason`: JSON-like text (with `reason` / `pred`) or plain rationale text
- `user_choice`

If `USE_PASS_BOTH_ONLY=True`, these are also required:

- `judge1_pass`
- `judge2_pass`

---

## 4. Training (`train.py`)

Run:

```bash
python train.py
```

### 4.1 Key Configurations (script top)

- Paths:
  - `BASE_MODEL_DIR`
  - `TRAIN_BASE_CSV` / `VAL_BASE_CSV`
  - `TRAIN_COT_CSV` / `VAL_COT_CSV`
  - `OUT_DIR`
- Hyperparameters:
  - `MAX_LENGTH`, `LEARNING_RATE`, `NUM_EPOCHS`
  - `TRAIN_BATCH_SIZE`, `GRAD_ACC_STEPS`
- Joint loss weights:
  - `LOSS_WEIGHT_LM`
  - `LOSS_WEIGHT_CLS`
- LoRA:
  - `LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT`
- GPU:
  - `TRAIN_CUDA_VISIBLE_DEVICES`

### 4.2 Training Outputs

- LoRA checkpoints: `model/checkpoint-*` (or your `OUT_DIR`)
- Final adapter: `model/final`

---

## 5. Evaluation (`evaluation.py`)

Run:

```bash
python evaluation.py
```

### 5.1 Key Configurations

- Paths:
  - `BASE_MODEL_DIR`
  - `ADAPTER_ROOT_DIR`
  - `TEST_CSV`
  - `OUT_CSV`
  - `METRICS_JSON`
- Checkpoint selection:
  - `CHECKPOINT_STEP` (higher priority)
  - `MAX_TRAIN_EPOCH_FOR_EVAL`
- Inference strategy:
  - `TRY_GENERATION`: whether to enable generation head
  - `blend_cls_gen_pred(...)`: final label fusion rule
- GPU:
  - `EVAL_CUDA_VISIBLE_DEVICES`

### 5.2 Evaluation Outputs

- `predictions_*.csv`: per-sample predictions
- `metrics_*.json`: aggregate metrics (`accuracy`, `f1`, `auc`, reports)

---

## 6. Reproducibility Tips

1. Start from classification-only mode (`TRY_GENERATION=False`) to establish a robust baseline.
2. Enable generation head afterward and evaluate JSON parse rate and fusion gain.
3. For smaller models (e.g., 1.5B), tune first:
   - `LOSS_WEIGHT_LM` / `LOSS_WEIGHT_CLS`
   - `MAX_LENGTH`
   - `LORA_RANK`
4. Keep prompt logic consistent between training and evaluation to avoid distribution mismatch.

---

## 7. FAQ

- **Q: Generated outputs are noisy or JSON parsing fails.**  
  A: Start with classification-only mode, then revisit prompt design, `MAX_NEW_TOKENS`, and fusion rule.

- **Q: Out-of-memory during training.**  
  A: Reduce batch size, increase gradient accumulation, and keep gradient checkpointing enabled.

- **Q: Why is the 7B vs 1.5B gap large?**  
  A: Structured generation is highly scale-sensitive; tune each model size separately.

---

## 8. Open-Source Notes

For paper-level reproducibility, report:

- data version (Top-10 features vs full-feature variant),
- base model version (e.g., Qwen2.5-1.5B / 7B),
- LoRA settings and fusion rule.

Providing full training/evaluation configs is strongly recommended.
