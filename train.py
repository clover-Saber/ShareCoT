"""
Dual-head joint training for ridesharing:
- Head A: language modeling head for CoT reason generation.
- Head B: binary classification head for shared_trip_authorized (0/1).

Core idea:
1) Reuse Qwen hidden states.
2) Classification head 使用 **prompt 最后一个 token** 的 hidden（与推理时仅喂 prompt 一致），
   不再用「整段 prompt+target 的最后一个 token」。
3) Optimize a weighted sum of LM loss + classification BCE loss.
"""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ===== GPU（必须在 import torch 之前设置 CUDA_VISIBLE_DEVICES）=====
# 在此填写要用的物理卡号：单卡 "0"；多卡 "0,1"。若启动前已 export CUDA_VISIBLE_DEVICES，则以环境变量为准。
TRAIN_CUDA_VISIBLE_DEVICES = "0"
if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
    os.environ["CUDA_VISIBLE_DEVICES"] = TRAIN_CUDA_VISIBLE_DEVICES

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from transformers import (
    Qwen2Model,
    Qwen2PreTrainedModel,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model


TARGET_COL = "shared_trip_authorized"
PROJECT_DIR = Path(__file__).resolve().parent


def _patch_accelerate_unwrap_model_compat() -> None:
    """
    兼容旧版 accelerate：
    transformers 某些版本会调用 unwrap_model(..., keep_torch_compile=False)，
    但旧版 accelerate 的签名不接受该参数，导致 TypeError。
    """
    try:
        from accelerate import Accelerator  # type: ignore
    except Exception:
        return

    try:
        sig = inspect.signature(Accelerator.unwrap_model)
    except Exception:
        return

    if "keep_torch_compile" in sig.parameters:
        return

    orig = Accelerator.unwrap_model

    def _wrapped(self, model, *args, **kwargs):
        kwargs.pop("keep_torch_compile", None)
        return orig(self, model, *args, **kwargs)

    Accelerator.unwrap_model = _wrapped  # type: ignore[assignment]

# ===== 路径配置 =====
BASE_MODEL_DIR = "/home/yuhao/project/models/Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_BASE_CSV = PROJECT_DIR / "chicago_train.csv"
VAL_BASE_CSV = PROJECT_DIR / "chicago_val.csv"
TRAIN_COT_CSV = PROJECT_DIR / "train_with_cot.csv"
VAL_COT_CSV = PROJECT_DIR / "val_with_cot.csv"
OUT_DIR = PROJECT_DIR / "finetune/output_dual_head_x/qwen2.5-1.5b/lora/sft_dual_head"

# ===== 数据过滤 =====
USE_PASS_BOTH_ONLY = True
INCLUDE_NON_COT = True  # True: 无 CoT 的样本也训练（回退为仅 pred 监督）
MAX_TRAIN_ITEMS = 0  # 0 表示全部
MAX_VAL_ITEMS = 0

# ===== 训练超参 =====
SEED = 42
MAX_LENGTH = 512
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5.0
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRAD_ACC_STEPS = 8
LOGGING_STEPS = 10
SAVE_STEPS = 200
EVAL_STEPS = 200
USE_GRADIENT_CHECKPOINTING = True

# ===== 训练加速（不依赖 flash-attn）=====
# Ampere+ GPU 上通常可明显提速；若数值稳定性有顾虑可关闭。
ENABLE_TF32 = True
# 新版 PyTorch 的 fused AdamW，一般比普通 AdamW 更快。
ENABLE_FUSED_ADAMW = True
# 可选：torch.compile 可能进一步提速，但不同环境兼容性差异较大。
ENABLE_TORCH_COMPILE = False

# ===== 联合损失权重 =====
# 无 CoT 样本远多于有 CoT 时，适当加大 LM 权重，避免 CoT 梯度被淹没（可按验证集再调 3~5）。
LOSS_WEIGHT_LM = 3.0
LOSS_WEIGHT_CLS = 1.0
POS_WEIGHT = None  # 例如填 3.0 可提升正类权重

# ===== LoRA =====
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"  # peft 的快捷写法


SYSTEM_PROMPT = """\
你正在执行用户拼车授权意愿的二分类任务。
你将收到一段行程的自然语言描述（包含关键特征值）。
""".strip()

INSTRUCTION_WITH_REASON = (
    '完成二分类任务，并输出一行严格 JSON（STRICT JSON），且按以下顺序仅包含 2 个键：'
    '{"reason": "...", "pred": 0/1}。'
)

INSTRUCTION_PRED_ONLY = (
    '完成二分类任务，并输出一行严格 JSON（STRICT JSON），且仅包含 1 个键：{"pred": 0/1}。'
)


def _row_to_features_dict(row: pd.Series) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for k, v in row.items():
        if k == TARGET_COL:
            continue
        if pd.isna(v):
            d[k] = None
        elif hasattr(v, "item"):
            try:
                d[k] = v.item()
            except Exception:
                d[k] = v
        else:
            d[k] = v
    return d


def _build_user_prompt(features: dict[str, Any]) -> str:
    def g(k: str) -> Any:
        return features.get(k, None)

    user_choice_text = "价格敏感" if g("tip_paid") == 0 else "体验敏感"
    return (
        "这是一次网约车行程，请判断用户是否愿意授权拼车。\n"
        f"- 出发时间：{g('trip_start_timestamp')}\n"
        f"- 乘客{user_choice_text}\n"
        f"- 下车 POI 类别：{g('dropoff_poi_category')}\n"
        f"- 行程：{g('trip_seconds')} 秒，{g('trip_miles')} 英里\n"
        f"- 费用：车费={g('fare')}，附加费用={g('additional_charges')}\n"
        f"- 距市中心距离：上车点={g('pickup_to_center_distance_km')} km，下车点={g('dropoff_to_center_distance_km')} km\n"
        f"- 天气露点：{g('weather_dewpoint_2m')}°C\n"
    )


def _build_chat_prompt(user_prompt: str, instruction: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{instruction}\n\n{user_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _safe_parse_json_reason_pred(text: str) -> tuple[str, int] | None:
    s = (text or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
    except Exception:
        a, b = s.find("{"), s.rfind("}")
        if a == -1 or b == -1 or b <= a:
            return None
        try:
            obj = json.loads(s[a : b + 1])
        except Exception:
            return None
    reason = str(obj.get("reason", "")).strip()
    try:
        pred = int(obj.get("pred"))
    except Exception:
        return None
    if pred not in (0, 1):
        return None
    if not reason:
        return None
    return reason, pred


@dataclass
class TrainExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    class_labels: int


class RidesharingDualHeadDataset(Dataset):
    def __init__(
        self,
        *,
        base_csv_path: Path,
        cot_csv_path: Path,
        tokenizer: Any,
        max_length: int,
        pass_both_only: bool,
        include_non_cot: bool,
        max_items: int = 0,
    ) -> None:
        if not base_csv_path.exists():
            raise FileNotFoundError(f"找不到 base 数据：{base_csv_path}")
        if not cot_csv_path.exists():
            raise FileNotFoundError(f"找不到 CoT 数据：{cot_csv_path}")

        base_df = pd.read_csv(base_csv_path)
        cot_df = pd.read_csv(cot_csv_path)

        if TARGET_COL not in base_df.columns:
            raise ValueError(f"{base_csv_path} 缺少目标列: {TARGET_COL}")

        needed = {"row_index", "reason", "user_choice"}
        if pass_both_only:
            needed = needed | {"judge1_pass", "judge2_pass"}
        missing = needed - set(cot_df.columns)
        if missing:
            raise ValueError(f"{cot_csv_path} 缺少列: {sorted(missing)}")

        if pass_both_only:
            mask = (cot_df["judge1_pass"].astype(int) == 1) & (cot_df["judge2_pass"].astype(int) == 1)
            cot_df = cot_df[mask].copy()

        # row_index -> cot_reason（仅收集 reason 非空）
        cot_reason_map: dict[int, str] = {}
        for _, row in cot_df.iterrows():
            try:
                ridx = int(row["row_index"])
            except Exception:
                continue
            reason_raw = "" if pd.isna(row["reason"]) else str(row["reason"]).strip()
            if not reason_raw:
                continue
            parsed = _safe_parse_json_reason_pred(reason_raw)
            if parsed is not None:
                cot_reason, _ = parsed
            else:
                cot_reason = reason_raw
            if cot_reason:
                cot_reason_map[ridx] = cot_reason

        rows: list[dict[str, Any]] = []
        cot_used = 0
        non_cot_used = 0
        for ridx, base_row in base_df.reset_index(drop=True).iterrows():
            class_label = int(base_row[TARGET_COL])
            if class_label not in (0, 1):
                continue

            has_cot = ridx in cot_reason_map
            if (not has_cot) and (not include_non_cot):
                continue

            features = _row_to_features_dict(base_row)
            user_prompt = _build_user_prompt(features)
            if has_cot:
                cot_used += 1
                prompt_text = _build_chat_prompt(user_prompt, INSTRUCTION_WITH_REASON)
                target_text = json.dumps(
                    {"reason": cot_reason_map[ridx], "pred": class_label},
                    ensure_ascii=False,
                )
            else:
                non_cot_used += 1
                prompt_text = _build_chat_prompt(user_prompt, INSTRUCTION_PRED_ONLY)
                target_text = json.dumps({"pred": class_label}, ensure_ascii=False)

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            target_ids = tokenizer(target_text, add_special_tokens=False).input_ids

            # labels: prompt 部分全部置 -100，仅学习 target 部分
            input_ids = (prompt_ids + target_ids)[:max_length]
            labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
            attention_mask = [1] * len(input_ids)

            if len(input_ids) == 0:
                continue

            pl = len(prompt_ids)
            L = len(input_ids)
            # 截断后仍落在 prompt 内的最后一个位置（与「只喂 prompt 推理」对齐）
            num_prompt_toks = min(pl, L)
            cls_token_index = num_prompt_toks - 1

            rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "class_labels": class_label,
                    "cls_token_index": cls_token_index,
                }
            )

        if max_items > 0:
            rows = rows[:max_items]
        self.rows = rows
        self.cot_used = cot_used
        self.non_cot_used = non_cot_used

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class DualHeadDataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)
        input_ids, attention_mask, labels, class_labels, cls_idx = [], [], [], [], []
        for x in features:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)
            class_labels.append(int(x["class_labels"]))
            cls_idx.append(int(x["cls_token_index"]))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "class_labels": torch.tensor(class_labels, dtype=torch.float32),
            "cls_token_index": torch.tensor(cls_idx, dtype=torch.long),
        }


class QwenDualHeadForRidesharing(Qwen2PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.classification_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.post_init()

    # ---- PEFT / HF generation compatibility ----
    # PEFT(LoRA, task_type=CAUSAL_LM) 会在初始化时访问这些接口。
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        model_inputs.update(kwargs)
        return model_inputs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        cls_token_index: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # PEFT/Trainer 可能会在 kwargs 里注入 output_hidden_states / return_dict。
        # 这里统一处理，避免与显式参数重复导致 "multiple values"。
        model_kwargs = dict(kwargs)
        output_hidden_states = model_kwargs.pop("output_hidden_states", False)
        return_dict = model_kwargs.pop("return_dict", True)
        # 允许经 **kwargs 传入 cls_token_index（部分封装路径）
        if cls_token_index is None and "cls_token_index" in model_kwargs:
            cls_token_index = model_kwargs.pop("cls_token_index")
        else:
            model_kwargs.pop("cls_token_index", None)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **model_kwargs,
        )
        hidden_states = outputs.last_hidden_state  # [B, T, H]

        lm_logits = self.lm_head(hidden_states)  # [B, T, V]

        bsz = hidden_states.size(0)
        batch_idx = torch.arange(bsz, device=hidden_states.device)
        if cls_token_index is not None:
            # 训练/指定推理：用 prompt 最后一个 token（右 padding 时下标不变）
            idx = cls_token_index.to(device=hidden_states.device, dtype=torch.long)
            idx = torch.clamp(idx, min=0, max=hidden_states.size(1) - 1)
            cls_hidden = hidden_states[batch_idx, idx, :]
        elif attention_mask is None:
            last_idx = torch.full(
                (bsz,),
                hidden_states.size(1) - 1,
                dtype=torch.long,
                device=hidden_states.device,
            )
            cls_hidden = hidden_states[batch_idx, last_idx, :]
        else:
            # 无 cls_token_index：退化为最后一个非 pad（纯 prompt 推理时常用）
            last_idx = attention_mask.long().sum(dim=1) - 1
            last_idx = torch.clamp(last_idx, min=0)
            cls_hidden = hidden_states[batch_idx, last_idx, :]
        class_logits = self.classification_head(cls_hidden)  # [B, 1]

        return lm_logits, class_logits


class DualHeadTrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_weight_lm: float = 3.0,
        loss_weight_cls: float = 1.0,
        pos_weight: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_weight_lm = float(loss_weight_lm)
        self.loss_weight_cls = float(loss_weight_cls)
        self._pos_weight = None if pos_weight is None else float(pos_weight)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        class_labels = inputs.pop("class_labels").float()
        cls_token_index = inputs.pop("cls_token_index", None)

        lm_logits, class_logits = model(**inputs, cls_token_index=cls_token_index)

        # LM loss
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_lm = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Classification loss
        class_logits = class_logits.squeeze(-1)
        if self._pos_weight is not None:
            pw = torch.tensor([self._pos_weight], device=class_logits.device, dtype=class_logits.dtype)
            loss_cls = F.binary_cross_entropy_with_logits(class_logits, class_labels, pos_weight=pw)
        else:
            loss_cls = F.binary_cross_entropy_with_logits(class_logits, class_labels)

        total_loss = (self.loss_weight_lm * loss_lm) + (self.loss_weight_cls * loss_cls)

        if return_outputs:
            outputs = {
                "lm_logits": lm_logits.detach(),
                "class_logits": class_logits.detach(),
                "loss_lm": loss_lm.detach(),
                "loss_cls": loss_cls.detach(),
            }
            return total_loss, outputs
        return total_loss


def _build_training_args() -> TrainingArguments:
    """
    兼容不同 transformers 版本中的 TrainingArguments 参数命名差异。
    """
    sig = inspect.signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())

    kwargs: dict[str, Any] = {
        "output_dir": str(OUT_DIR),
        "do_train": True,
        "do_eval": True,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACC_STEPS,
        "learning_rate": LEARNING_RATE,
        "num_train_epochs": NUM_EPOCHS,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "logging_steps": LOGGING_STEPS,
        "save_steps": SAVE_STEPS,
        "eval_steps": EVAL_STEPS,
        "save_strategy": "steps",
        "bf16": torch.cuda.is_available(),
        "fp16": False,
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "report_to": [],
        "remove_unused_columns": False,
    }

    # 版本差异：evaluation_strategy vs eval_strategy
    if "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in supported:
        kwargs["eval_strategy"] = "steps"

    # 版本差异：overwrite_output_dir 可能不存在
    if "overwrite_output_dir" in supported:
        kwargs["overwrite_output_dir"] = True
    if "gradient_checkpointing" in supported:
        kwargs["gradient_checkpointing"] = USE_GRADIENT_CHECKPOINTING
    if "group_by_length" in supported:
        kwargs["group_by_length"] = True
    if ENABLE_FUSED_ADAMW and torch.cuda.is_available() and "optim" in supported:
        kwargs["optim"] = "adamw_torch_fused"

    filtered = {k: v for k, v in kwargs.items() if k in supported}
    return TrainingArguments(**filtered)


def main() -> int:
    set_seed(SEED)
    _patch_accelerate_unwrap_model_compat()
    print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if ENABLE_TF32:
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
    if not Path(BASE_MODEL_DIR).exists():
        raise FileNotFoundError(f"找不到基础模型: {BASE_MODEL_DIR}")
    if not TRAIN_BASE_CSV.exists():
        raise FileNotFoundError(f"找不到训练 base CSV: {TRAIN_BASE_CSV}")
    if not VAL_BASE_CSV.exists():
        raise FileNotFoundError(f"找不到验证 base CSV: {VAL_BASE_CSV}")
    if not TRAIN_COT_CSV.exists():
        raise FileNotFoundError(f"找不到训练 CoT CSV: {TRAIN_COT_CSV}")
    if not VAL_COT_CSV.exists():
        raise FileNotFoundError(f"找不到验证 CoT CSV: {VAL_COT_CSV}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = RidesharingDualHeadDataset(
        base_csv_path=TRAIN_BASE_CSV,
        cot_csv_path=TRAIN_COT_CSV,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        pass_both_only=USE_PASS_BOTH_ONLY,
        include_non_cot=INCLUDE_NON_COT,
        max_items=MAX_TRAIN_ITEMS,
    )
    val_ds = RidesharingDualHeadDataset(
        base_csv_path=VAL_BASE_CSV,
        cot_csv_path=VAL_COT_CSV,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        pass_both_only=USE_PASS_BOTH_ONLY,
        include_non_cot=INCLUDE_NON_COT,
        max_items=MAX_VAL_ITEMS,
    )

    if len(train_ds) == 0:
        raise ValueError("训练集为空，请检查 pass_both 过滤或 reason 字段。")
    if len(val_ds) == 0:
        raise ValueError("验证集为空，请检查 pass_both 过滤或 reason 字段。")

    print(f"train samples: {len(train_ds)}")
    print(f"  - with CoT: {train_ds.cot_used}, without CoT: {train_ds.non_cot_used}")
    print(f"val samples:   {len(val_ds)}")
    print(f"  - with CoT: {val_ds.cot_used}, without CoT: {val_ds.non_cot_used}")

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "ignore_mismatched_sizes": True,
    }
    # 优先启用 Flash Attention 2（若环境不支持会自动回退）
    if torch.cuda.is_available():
        load_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = QwenDualHeadForRidesharing.from_pretrained(
            BASE_MODEL_DIR,
            **load_kwargs,
        )
    except Exception as e:
        if load_kwargs.get("attn_implementation") == "flash_attention_2":
            print(f"[WARN] Flash Attention 2 不可用，回退默认注意力。原因: {e}")
            load_kwargs.pop("attn_implementation", None)
            model = QwenDualHeadForRidesharing.from_pretrained(
                BASE_MODEL_DIR,
                **load_kwargs,
            )
        else:
            raise

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        # 分类头为随机初始化的新层，必须参与训练；否则 loss_cls 几乎无法有效更新该层。
        modules_to_save=["classification_head"],
    )
    model = get_peft_model(model, lora_cfg)
    # 训练阶段关闭 KV cache，显著降低显存占用。
    if hasattr(model, "config"):
        model.config.use_cache = False
    if USE_GRADIENT_CHECKPOINTING:
        # 兼容不同 transformers/peft 版本
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[INFO] torch.compile 已启用。")
        except Exception as e:
            print(f"[WARN] torch.compile 启用失败，继续使用 eager 模式。原因: {e}")

    model.print_trainable_parameters()

    args = _build_training_args()

    collator = DualHeadDataCollator(pad_token_id=tokenizer.pad_token_id)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": collator,
        "loss_weight_lm": LOSS_WEIGHT_LM,
        "loss_weight_cls": LOSS_WEIGHT_CLS,
        "pos_weight": POS_WEIGHT,
    }
    # transformers 版本兼容：旧版 Trainer 用 tokenizer，新版改为 processing_class
    trainer_sig = inspect.signature(DualHeadTrainer.__init__)
    trainer_supported = set(trainer_sig.parameters.keys())
    if "tokenizer" in trainer_supported:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_supported:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = DualHeadTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(str(OUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUT_DIR / "final"))

    print("训练完成。")
    print(f"输出目录: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

