from core.register.registryhub import R
from torch import optim
@R.optimizers.register('adamBuilder')
class AdamBuilder:
    def __init__(self,model, lr, used_parameters:list[str] = None, weight_decay = 0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.used_parameters = used_parameters
        self.trained_parameters = []


    def build(self):
        if  self.used_parameters is None:
            self.trained_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        else:
            for name, p in self.model.named_parameters():
                for used_name in self.used_parameters:
                    if used_name in name:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False

            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    self.trained_parameters.append(p)
        return optim.Adam(self.trained_parameters, lr=self.lr, weight_decay=self.weight_decay)