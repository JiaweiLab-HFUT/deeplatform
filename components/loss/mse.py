import torch
from torch import nn
from core.register.registryhub import R

@R.loss.register('mse')
class MSELoss(nn.Module):
    def forward(self, y_pred, y_true, weights):
        loss = nn.functional.mse_loss(y_pred, y_true, reduction='none')
        # 每个样本的权重沿特征维广播，避免 [B, 1] * [B] 变成 [B, B]。
        weights = weights.to(device=loss.device, dtype=loss.dtype)
        weights = weights.reshape(-1, *([1] * (loss.ndim - 1)))
        return torch.mean(loss * weights)
