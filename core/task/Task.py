import os
from abc import ABC, abstractmethod

from core.hook.hook_registry import hook_emit


class Task(ABC):
    def __init__(self, context):
        self.context = context
        self.path = context.__experiment_path__
        self.checkpoint_path = self.path + '/checkpoint'
        os.makedirs(self.checkpoint_path, exist_ok=True)

    @abstractmethod
    def run(self): ...

