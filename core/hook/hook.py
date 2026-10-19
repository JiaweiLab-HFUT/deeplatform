from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.hook.hook_manager import HookManager


class Hook(ABC):
    """声明事件并实现处理函数；实例化不会自动绑定任何管理器。"""

    event: str = ''

    def __init__(self, priority: int = 0):
        if type(priority) is not int:
            raise TypeError('Hook.priority 必须是整数')
        self.priority = priority
        self._manager: HookManager | None = None

    @property
    def manager(self) -> HookManager:
        if self._manager is None:
            raise RuntimeError('Hook 尚未绑定到管理器，或已被解除绑定')
        return self._manager

    @abstractmethod
    def run(self, *args, **kwargs): ...
