"""
双头（LM + 分类）模型评估：分类头 + 可选贪心 JSON 生成，结果写入 CSV / JSON。

配置项见下方常量。按 epoch 选 checkpoint：MAX_TRAIN_EPOCH_FOR_EVAL；或 CHECKPOINT_STEP。
训练脚本优先加载同目录 `2.2-dual_head联合微调.py`。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# ===== GPU（必须在 import torch 之前设置 CUDA_VISIBLE_DEVICES）=====
# 单卡示例 "0"；多卡示例 "0,1"。若启动前已 export CUDA_VISIBLE_DEVICES，则以环境变量为准。
EVAL_CUDA_VISIBLE_DEVICES = "0"
if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
    os.environ["CUDA_VISIBLE_DEVICES"] = EVAL_CUDA_VISIBLE_DEVICES

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoTokenizer

TARGET_COL = "shared_trip_authorized"
PROJECT_DIR = Path(__file__).resolve().parent

# 含 QwenDualHeadForRidesharing 与 prompt 的训练脚本（二选一存在即可）
_TRAIN_SCRIPT_CANDIDATES = (
    PROJECT_DIR / "2.2-dual_head联合微调.py",
)

TEST_CSV = PROJECT_DIR / "chicago_test.csv"
OUT_CSV = PROJECT_DIR / "predictions_dual_head_1.5B_x.csv"
METRICS_JSON = PROJECT_DIR / "metrics_dual_head_1.5B_x.json"

BASE_MODEL_DIR = "/home/yuhao/project/models/Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_ROOT_DIR = PROJECT_DIR / "finetune/output_dual_head_x/qwen2.5-1.5b/lora/sft_dual_head"

CHECKPOINT_STEP: Optional[int] = None
MAX_TRAIN_EPOCH_FOR_EVAL: Optional[float] = 3.0

LIMIT = 1000
MAX_LENGTH = 512
# 以分类头输出为准：默认关闭生成与融合
TRY_GENERATION = False
MAX_NEW_TOKENS = 256
MAX_GENERATE_TOTAL_LEN = 1024

# cls_prob < 该阈值且 gen≠cls 时，y_pred_final 采用 gen
BLEND_LOW_CONF_THRESHOLD = 0.5


def _resolve_train_script() -> Path:
    for p in _TRAIN_SCRIPT_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "未找到双头训练脚本，请放置以下之一：\n"
        + "\n".join(f"  - {x}" for x in _TRAIN_SCRIPT_CANDIDATES)
    )


def _load_train_module():
    train_script = _resolve_train_script()
    spec = importlib.util.spec_from_file_location("ridesharing_dual_train_eval", train_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块：{train_script}")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ridesharing_dual_train_eval"] = m
    spec.loader.exec_module(m)
    return m


def _pick_latest_checkpoint(root: Path) -> Path:
    ckpts: list[tuple[int, Path]] = []
    for p in root.glob("checkpoint-*"):
        if p.is_dir():
            try:
                step = int(p.name.split("-", 1)[1])
            except Exception:
                step = -1
            ckpts.append((step, p))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint-* under {root}")
    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


def _read_trainer_state_epoch_step(ckpt_dir: Path) -> Optional[tuple[float, int]]:
    state_path = ckpt_dir / "trainer_state.json"
    if not state_path.is_file():
        return None
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        ep = float(data.get("epoch", 0.0))
        step = int(data.get("global_step", 0))
        return ep, step
    except Exception:
        return None


def _pick_checkpoint_at_most_epoch(root: Path, max_epoch: float) -> Optional[Path]:
    dirs: list[Path] = [p for p in root.glob("checkpoint-*") if p.is_dir()]
    final = root / "final"
    if final.is_dir() and any(final.iterdir()):
        dirs.append(final)

    candidates: list[tuple[float, int, Path]] = []
    for p in dirs:
        parsed = _read_trainer_state_epoch_step(p)
        if parsed is None:
            continue
        ep, step = parsed
        if ep <= max_epoch + 1e-5:
            candidates.append((ep, step, p))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


def _pick_checkpoint(root: Path) -> Path:
    if CHECKPOINT_STEP is not None:
        cand = root / f"checkpoint-{CHECKPOINT_STEP}"
        if cand.is_dir():
            return cand
        raise FileNotFoundError(f"Missing checkpoint: {cand}")

    if MAX_TRAIN_EPOCH_FOR_EVAL is not None:
        picked = _pick_checkpoint_at_most_epoch(root, MAX_TRAIN_EPOCH_FOR_EVAL)
        if picked is not None:
            return picked
        print(
            f"警告：未找到 epoch≤{MAX_TRAIN_EPOCH_FOR_EVAL} 且含 trainer_state.json 的 checkpoint，"
            "回退为 final / 最新 checkpoint-*。"
        )

    final = root / "final"
    if final.is_dir() and any(final.iterdir()):
        return final
    return _pick_latest_checkpoint(root)


def _extract_json(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.strip("`").replace("json", "", 1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        a, b = s.find("{"), s.rfind("}")
        if a != -1 and b != -1 and b > a:
            return json.loads(s[a : b + 1])
        raise


def blend_cls_gen_pred(cls_pred: int, cls_prob: float, gen_pred: Optional[int]) -> int:
    if gen_pred is None:
        return int(cls_pred)
    return int(gen_pred)


@torch.inference_mode()
def predict_class_head(
    model: torch.nn.Module,
    tokenizer: Any,
    features: dict[str, Any],
    m: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[int, float]:
    up = m._build_user_prompt(features)
    prompt = m._build_chat_prompt(up, m.INSTRUCTION_WITH_REASON)
    enc = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    attn = enc.get("attention_mask")
    if attn is not None:
        last_i = int(attn[0].sum().item()) - 1
    else:
        last_i = enc["input_ids"].shape[1] - 1
    last_i = max(0, last_i)
    enc["cls_token_index"] = torch.tensor([last_i], device=device, dtype=torch.long)
    _, cls_logits = model(**enc)
    logit = cls_logits.squeeze(-1).float()
    prob = torch.sigmoid(logit).item()
    pred = 1 if prob >= 0.5 else 0
    return pred, prob


def _forward_lm_logits(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    if isinstance(out, tuple) and len(out) >= 1:
        return out[0]
    if hasattr(out, "logits"):
        return out.logits
    raise TypeError(f"Unexpected model forward output: {type(out)}")


@torch.inference_mode()
def generate_cot_json_greedy(
    model: torch.nn.Module,
    tokenizer: Any,
    features: dict[str, Any],
    m: Any,
    device: torch.device,
) -> tuple[Optional[int], Optional[str], str]:
    up = m._build_user_prompt(features)
    prompt = m._build_chat_prompt(up, m.INSTRUCTION_WITH_REASON)
    enc = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    ids = enc["input_ids"].to(device)
    att = enc["attention_mask"].to(device) if enc.get("attention_mask") is not None else torch.ones_like(ids)
    prompt_len = int(ids.shape[1])
    eos_id = tokenizer.eos_token_id

    for _ in range(MAX_NEW_TOKENS):
        if ids.shape[1] >= MAX_GENERATE_TOTAL_LEN:
            break
        lm_logits = _forward_lm_logits(model, ids, att)
        next_id = int(lm_logits[:, -1, :].argmax(dim=-1).item())
        if eos_id is not None and next_id == eos_id:
            break
        nid = torch.tensor([[next_id]], device=device, dtype=torch.long)
        ids = torch.cat([ids, nid], dim=1)
        att = torch.cat([att, torch.ones_like(nid, device=device)], dim=1)

        gen_slice = ids[0, prompt_len:]
        text = tokenizer.decode(gen_slice, skip_special_tokens=True).strip()
        if "}" not in text or "pred" not in text:
            continue
        try:
            obj = _extract_json(text)
            pred = int(obj.get("pred"))
            if pred not in (0, 1):
                continue
            reason = obj.get("reason")
            rs = str(reason).strip() if reason is not None else None
            if not rs:
                continue
            return pred, rs, text
        except Exception:
            continue

    text = tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=True).strip()
    try:
        obj = _extract_json(text)
        pred = int(obj.get("pred"))
        if pred not in (0, 1):
            return None, None, text
        reason = obj.get("reason")
        rs = str(reason).strip() if reason is not None else None
        if not rs:
            return None, None, text
        return pred, rs, text
    except Exception:
        return None, None, text


def _fmt_classification_report(y_true: Any, y_pred: pd.Series) -> str:
    rep = classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=True)
    lines = ["              precision    recall  f1-score   support", ""]
    for label in [k for k in ("0", "1") if k in rep]:
        row = rep[label]
        lines.append(
            f"{label:>11} {row['precision']:>11.4f} {row['recall']:>9.4f} "
            f"{row['f1-score']:>10.4f} {int(row['support']):>9}"
        )
    lines.append("")
    ma = rep["macro avg"]
    lines.append(
        f"{'macro avg':>11} {ma['precision']:>11.4f} {ma['recall']:>9.4f} "
        f"{ma['f1-score']:>10.4f} {int(ma['support']):>9}"
    )
    return "\n".join(lines)


def main() -> int:
    m = _load_train_module()
    if hasattr(m, "_patch_accelerate_unwrap_model_compat"):
        m._patch_accelerate_unwrap_model_compat()

    if not Path(BASE_MODEL_DIR).exists():
        raise FileNotFoundError(BASE_MODEL_DIR)
    if not TEST_CSV.exists():
        raise FileNotFoundError(str(TEST_CSV))
    if not ADAPTER_ROOT_DIR.exists():
        raise FileNotFoundError(str(ADAPTER_ROOT_DIR))

    ckpt = _pick_checkpoint(ADAPTER_ROOT_DIR)
    st = _read_trainer_state_epoch_step(ckpt)
    if st is not None:
        print(f"checkpoint: {ckpt} (trainer_state epoch={st[0]:.4f}, global_step={st[1]})")
    else:
        print(f"checkpoint: {ckpt} (无 trainer_state.json，epoch 未知)")

    df = pd.read_csv(TEST_CSV)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing {TARGET_COL}")
    n = min(len(df), max(1, int(LIMIT)))
    df = df.iloc[:n].copy()
    y_true = df[TARGET_COL].astype(int).to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    QwenDualHead = m.QwenDualHeadForRidesharing
    base = QwenDualHead.from_pretrained(
        BASE_MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=dtype,
        ignore_mismatched_sizes=True,
    )
    base.config.use_cache = True

    from peft import PeftModel

    model = PeftModel.from_pretrained(base, str(ckpt))
    model.eval()
    model.to(device)

    preds_cls: list[int] = []
    probs: list[float] = []
    preds_gen: list[Optional[int]] = []
    reasons: list[Optional[str]] = []
    raws: list[str] = []

    for _, row in tqdm(df.iterrows(), total=n, desc="dual-head eval", unit="sample"):
        feats = m._row_to_features_dict(row)
        pc, pr = predict_class_head(model, tokenizer, feats, m, device, dtype)
        preds_cls.append(pc)
        probs.append(pr)

        if TRY_GENERATION:
            pg, rs, raw = generate_cot_json_greedy(model, tokenizer, feats, m, device)
            preds_gen.append(pg)
            reasons.append(rs)
            raws.append(raw)
        else:
            preds_gen.append(None)
            reasons.append(None)
            raws.append("")

    y_pred_cls_only = pd.Series(preds_cls, dtype=int)
    acc_cls = accuracy_score(y_true, y_pred_cls_only)
    f1_cls = f1_score(y_true, y_pred_cls_only, zero_division=0)
    try:
        auc = float(roc_auc_score(y_true, probs))
    except Exception:
        auc = None

    preds_final = [
        blend_cls_gen_pred(preds_cls[i], probs[i], preds_gen[i] if TRY_GENERATION else None)
        for i in range(n)
    ]
    y_pred_final = pd.Series(preds_final, dtype=int)
    acc_final = accuracy_score(y_true, y_pred_final)
    f1_final = f1_score(y_true, y_pred_final, zero_division=0)
    n_blended = sum(
        1
        for i in range(n)
        if TRY_GENERATION
        and preds_gen[i] is not None
        and float(probs[i]) < float(BLEND_LOW_CONF_THRESHOLD)
        and int(preds_gen[i]) != int(preds_cls[i])
    )

    report_cls_txt = _fmt_classification_report(y_true, y_pred_cls_only)
    report_final_txt = _fmt_classification_report(y_true, y_pred_final)

    gen_ok = sum(1 for p in preds_gen if p is not None)
    gen_match = sum(
        1 for i in range(n) if preds_gen[i] is not None and int(preds_gen[i]) == int(preds_cls[i])
    )

    print("--- classification report (cls only) ---")
    print(report_cls_txt)
    print(f"accuracy (cls only):  {acc_cls:.4f}")
    print(f"f1 (cls only):        {f1_cls:.4f}")
    print(f"auc_roc (cls prob):   {auc}")
    if TRY_GENERATION:
        print(f"generate parsed OK:   {gen_ok}/{n}")
        print(f"gen pred == cls pred: {gen_match}/{gen_ok if gen_ok else 1}")
        print(
            f"融合: cls_prob<{BLEND_LOW_CONF_THRESHOLD} 且 gen≠cls 时采用 gen；"
            f"本批 {n_blended} 条用 gen 覆盖 cls"
        )
        print("--- classification report (y_pred_final) ---")
        print(report_final_txt)
        print(f"accuracy (final):     {acc_final:.4f}")
        print(f"f1 (final):           {f1_final:.4f}")

    out = df.copy()
    out["y_true"] = y_true
    out["y_pred_cls"] = preds_cls
    out["cls_prob"] = probs
    if TRY_GENERATION:
        out["y_pred_gen"] = preds_gen
        out["reason_gen"] = reasons
        out["raw_gen"] = raws
        out["y_pred_final"] = preds_final
    else:
        out["y_pred_final"] = preds_cls
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote: {OUT_CSV.resolve()}")

    summary = {
        "base_model_dir": BASE_MODEL_DIR,
        "max_train_epoch_for_eval": MAX_TRAIN_EPOCH_FOR_EVAL,
        "blend_low_conf_threshold": float(BLEND_LOW_CONF_THRESHOLD),
        "n_blended_used_gen_over_cls": int(n_blended) if TRY_GENERATION else 0,
        "checkpoint_epoch": st[0] if st else None,
        "checkpoint_global_step": st[1] if st else None,
        "checkpoint": str(ckpt),
        "test_csv": str(TEST_CSV),
        "n_samples": int(n),
        "accuracy_cls": float(acc_cls),
        "f1_cls": float(f1_cls),
        "accuracy_final": float(acc_final),
        "f1_final": float(f1_final),
        "auc_roc": auc,
        "classification_report_cls": report_cls_txt,
        "classification_report_final": report_final_txt if TRY_GENERATION else report_cls_txt,
        "classification_report": report_cls_txt,
        "try_generation": TRY_GENERATION,
        "gen_parsed_ok": int(gen_ok),
        "gen_pred_match_cls": int(gen_match),
        "predictions_csv": str(OUT_CSV),
    }
    METRICS_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote: {METRICS_JSON.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
