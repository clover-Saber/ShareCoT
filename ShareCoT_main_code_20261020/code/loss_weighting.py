"""Uncertainty weighting with explicit masking of absent rationale supervision.

The adaptive objective follows manuscript Eq. (6), including its one-half
coefficients. Both initial adaptive loss coefficients therefore equal 0.5.
Regularizers are applied only to observed tasks. No manual variance clipping.
"""
import math
import torch
from torch import nn


class LossWeighting(nn.Module):
    def __init__(self, mode='adaptive'):
        super().__init__()
        if mode not in ('adaptive', 'fixed_1', 'fixed_3', 'fixed_5'):
            raise ValueError(f'Unknown weighting mode: {mode}')
        self.mode = mode
        if mode == 'adaptive':
            self.log_var_cls = nn.Parameter(torch.zeros((), dtype=torch.float32))
            self.log_var_lm = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, cls_loss, lm_loss, has_rationale):
        if self.mode != 'adaptive':
            factor = float(self.mode.split('_')[1])
            return cls_loss + factor * lm_loss if has_rationale else cls_loss
        loss = 0.5 * (torch.exp(-self.log_var_cls) * cls_loss.float() + self.log_var_cls)
        if has_rationale:
            loss = loss + 0.5 * (torch.exp(-self.log_var_lm) * lm_loss.float() + self.log_var_lm)
        return loss

    def snapshot(self):
        if self.mode == 'adaptive':
            cls_var = self.log_var_cls.detach().float().item()
            lm_var = self.log_var_lm.detach().float().item()
            cls_weight, lm_weight = 0.5 * math.exp(-cls_var), 0.5 * math.exp(-lm_var)
        else:
            cls_var = lm_var = None
            cls_weight, lm_weight = 1., float(self.mode.split('_')[1])
        return {'mode': self.mode, 'log_var_cls': cls_var, 'log_var_lm': lm_var,
                'classification_weight': cls_weight, 'reasoning_weight': lm_weight,
                'reasoning_to_classification_ratio': lm_weight / cls_weight}
