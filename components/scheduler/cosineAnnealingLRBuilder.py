from torch.optim.lr_scheduler import CosineAnnealingLR
from core.register.registryhub import R
@R.schedulers.register('cosineAnnealingLRBuilder')
class CosineAnnealingLRBuilder:
    def __init__(self, T_max = 20, eta_min = 1e-8):
        self.T_max = T_max
        self.eta_min = eta_min

    def build(self, optim, train_steps = None):
        return CosineAnnealingLR(optim, T_max=20, eta_min=1e-8)