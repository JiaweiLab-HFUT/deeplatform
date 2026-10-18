"""递归导入组件和任务模块，执行模块中的注册装饰器。"""

from importlib import import_module, invalidate_caches
from pathlib import Path
from types import ModuleType
from typing import Iterable


def _import_module(name: str) -> ModuleType:
    try:
        return import_module(name)
    except Exception as exc:
        raise ImportError(f"自动注册失败：无法导入模块 {name!r}: {exc}") from exc


def autodiscover_components(
    packages: Iterable[str] = ("components", "task"),
) -> list[str]:
    """按名称顺序递归导入指定包，返回导入的模块名。

    支持没有 __init__.py 的命名空间包；跳过以下划线或点开头的
    私有文件和目录，但会正常导入包的 __init__.py。
    导入只执行模块级代码，不创建组件实例或运行任务，因此模块应
    将脚本执行代码放到 if __name__ == "__main__" 保护下。
    重复调用复用 Python 的模块缓存，不会重新执行注册装饰器。
    导入失败立即报错，并保留原始异常，避免使用不完整的注册表。
    """
    invalidate_caches()
    imported = []
    seen = set()
    for package_name in packages:
        package = _import_module(package_name)
        if not hasattr(package, "__path__"):
            raise TypeError(f"扫描目标 {package_name!r} 必须是包或目录")

        module_names = {package_name}
        for package_path in package.__path__:
            root = Path(package_path)
            for path in root.rglob("*.py"):
                relative = path.relative_to(root)
                parts = relative.parts[:-1]
                if path.name != "__init__.py":
                    parts = (*parts, path.stem)
                if any(part.startswith(("_", ".")) for part in parts):
                    continue
                module_names.add(".".join((package_name, *parts)))

        for module_name in sorted(module_names):
            if module_name in seen:
                continue
            _import_module(module_name)
            seen.add(module_name)
            imported.append(module_name)
    return imported
