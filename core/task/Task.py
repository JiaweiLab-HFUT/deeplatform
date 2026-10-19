import os
from abc import ABC, abstractmethod

from core.hook.hook_manager import HookManager


class Task(ABC):
    def __init__(self, context, hook_manager: HookManager):
        self.context = context
        self.hooks = hook_manager
        self.path = context.__experiment_path__
        self.checkpoint_path = self.path + '/checkpoint'
        os.makedirs(self.checkpoint_path, exist_ok=True)

    @abstractmethod
    def run(self): ...

