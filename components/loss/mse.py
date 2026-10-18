import torch
from torch import nn
from core.register.registryhub import R

@R.loss.register
def mse(y_pred, y_true, weights):
    criterion = nn.MSELoss(reduction='none')
    cut_off = y_true.shape[0]
    loss = criterion(y_pred[:cut_off], y_true)
    loss = torch.mean(loss * weights)
    return loss