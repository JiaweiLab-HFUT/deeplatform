from core.register.register import Registry


class RegistryHub:
    def __init__(self):
        self._registries = {}

    def __getitem__(self, name: str) -> Registry:
        name = name.lower()
        return self._registries.setdefault(name, Registry(name))

    def __getattr__(self, name: str) -> Registry:
        # 保护：下划线开头（如 _registries）不能走 __getattr__，否则递归
        name = name.lower()
        if name.startswith("_"):
            raise AttributeError(name)
        return self[name]

    def all(self) -> list[Registry]:
        return list(self._registries.values())

R = RegistryHub()