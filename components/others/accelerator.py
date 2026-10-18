from accelerate import Accelerator
import torch

from core.register.registryhub import R


@R.runtime.register('accelerator')
def build_accelerator(device):
    return Accelerator(cpu=torch.device(device).type == 'cpu')
