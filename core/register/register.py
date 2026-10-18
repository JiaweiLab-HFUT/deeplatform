# registry.py
from typing import Callable, Any


class Registry:
    def __init__(self, name):
        self.name = name
        self._store = {}

    def register(self, key=None)-> Callable:
        """既支持 @reg.register，也支持 @reg.register("name")"""
        def wrapper(obj : Callable[..., Any]):
            name = (key or obj.__name__).lower()
            if name in self._store:
                raise ValueError(f"[{self.name}] 重复注册: {name}")
            self._store[name] = obj
            return obj
        # 直接 @reg.register（无参数）时，key 就是被装饰的对象
        if callable(key):
            obj, key = key, None
            return wrapper(obj)
        elif key is None or isinstance(key, str):
            return wrapper
        else:
            raise ValueError(f"{key!r} should be str or None. ")

    def get(self, key):
        key = key.lower()
        if key not in self._store:
            raise KeyError(f"[{self.name}] 未注册: {key}，已有: {list(self._store)}")
        return self._store[key]

    def keys(self):
        return list(self._store)
