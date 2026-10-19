"""一次实验内的 Hook 绑定与同步事件分发。"""

from collections import defaultdict
from itertools import count
import logging
from typing import Any

from core.hook.hook import Hook


log = logging.getLogger(__name__)


class HookManager:
    def __init__(self):
        self._hooks: dict[str, list[tuple[int, int, Hook]]] = defaultdict(list)
        self._tokens = count()

    def register(self, hook: Hook) -> int:
        """绑定实例并返回注销句柄；小优先级先执行，同优先级按绑定顺序。"""
        if not isinstance(hook, Hook):
            raise TypeError('HookManager 只能绑定 Hook 实例')
        if not isinstance(hook.event, str) or not hook.event.strip():
            raise ValueError('Hook.event 必须是非空字符串')
        if type(hook.priority) is not int:
            raise TypeError('Hook.priority 必须是整数')
        if hook._manager is not None:
            raise ValueError('同一个 Hook 实例只能绑定一次，请先解除原绑定')
        token = next(self._tokens)
        handlers = self._hooks[hook.event]
        handlers.append((hook.priority, token, hook))
        handlers.sort(key=lambda entry: entry[0])
        hook._manager = self
        return token

    def unregister(self, token: int) -> bool:
        for event, handlers in self._hooks.items():
            for index, (_, registered_token, hook) in enumerate(handlers):
                if registered_token == token:
                    handlers.pop(index)
                    hook._manager = None
                    if not handlers:
                        del self._hooks[event]
                    return True
        return False

    def emit(self, event: str, *args, **kwargs) -> list[Any]:
        """同步触发全部处理函数；异常记录后向外抛出。"""
        results = []
        for _, _, hook in list(self._hooks.get(event, ())):
            try:
                results.append(hook.run(*args, **kwargs))
            except Exception:
                log.exception('hook %r (%s) failed', event, type(hook).__name__)
                raise
        return results

    def clear(self) -> None:
        """解除全部绑定并释放实例引用。"""
        for handlers in self._hooks.values():
            for _, _, hook in handlers:
                hook._manager = None
        self._hooks.clear()
