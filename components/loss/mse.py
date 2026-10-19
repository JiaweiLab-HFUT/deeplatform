import torch
from torch import nn
from core.register.registryhub import R

@R.loss.register('mse')
class Mse:
    def __init__(self):
        self.criterion = nn.MSELoss(reduction='none')
    def __call__(self, y_pred, y_true):
        return self.criterion(y_pred, y_true)
