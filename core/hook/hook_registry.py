import logging
from collections import defaultdict
from typing import Callable, Any

log = logging.getLogger(__name__)


class HookRegistry:
    def __init__(self):
        self._hooks: dict[str, list[tuple[int, Callable]]] = defaultdict(list)

    def hook(self, name: str, priority: int = 0):
        """装饰器方式注册"""
        def deco(fn: Callable):
            self.register_hook(name, priority, fn)
            return fn
        return deco

    def register_hook(self, name: str, priority: int, fn: Callable) -> None:
        if not callable(fn):
            raise TypeError(f"hook {name!r} handler must be callable")
        self._hooks[name].append((priority, fn))
        self._hooks[name].sort(key=lambda t: t[0])

    def unregister_hook(self, name: str, fn: Callable) -> bool:
        lst = self._hooks.get(name)
        if not lst:
            return False
        for i, (_, f) in enumerate(lst):
            # 处理绑定方法：比较 __func__ 和 __self__
            if f is fn or (
                getattr(f, "__self__", None) is getattr(fn, "__self__", None)
                and getattr(f, "__func__", None) is getattr(fn, "__func__", None)
            ):
                lst.pop(i)
                return True
        return False

    def hook_emit(self, name: str, *args, **kwargs) -> list[Any]:
        """按优先级执行，返回每个 hook 的返回值（异常时为 None）"""
        results = []
        for _, fn in list(self._hooks.get(name, ())):
            try:
                results.append(fn(*args, **kwargs))
            except Exception:
                log.exception("hook %r failed", name)
                results.append(None)
        return results


hook_registry = HookRegistry()


def hook_reg(name: str, priority: int = 0):
    return hook_registry.hook(name, priority)


def hook_emit(name: str, *args, **kwargs):
    return hook_registry.hook_emit(name, *args, **kwargs)