from torch.optim.lr_scheduler import OneCycleLR
from core.register.registryhub import R

@R.schedulers.register('oneCycleLRBuilder')
class OneCycleLRBuilder:
    def __init__(self, train_epochs, max_lr, pct_start = 0.2):
        self.pct_start = pct_start
        self.train_epochs = train_epochs
        self.max_lr = max_lr

    def build(self, optim, train_steps = None):
        if train_steps is None:
            raise ValueError('train_steps cannot be None')
        return OneCycleLR(  optimizer=optim,
                            steps_per_epoch= train_steps,
                            pct_start=self.pct_start,
                            epochs=self.train_epochs,
                            max_lr=self.max_lr)