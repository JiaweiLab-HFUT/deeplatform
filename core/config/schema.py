"""将 YAML 字典规范化为组件描述，并检查名称与依赖关系。"""

from copy import deepcopy
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Any, Container, Iterator

import yaml


RUNTIME_NAMES = frozenset({'task', '__experiment_path__', '_memory', '_scope'})


class ConfigLoader(yaml.SafeLoader):
    """保留 safe_load 的安全约束，同时拒绝被 YAML 静默覆盖的重复键。"""

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    None, None, '配置键必须可哈希', key_node.start_mark,
                ) from exc
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    None, None, f'重复字段 {key!r}', key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


@dataclass(frozen=True)
class ComponentSpec:
    scope: str
    type: str
    name: str
    params: dict[str, Any]
    path: str


@dataclass(frozen=True)
class ExperimentSpec:
    mate: dict[str, Any]
    task: ComponentSpec
    components: tuple[ComponentSpec, ...]


def require_name(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{path} 必须是非空字符串')


def component_spec(value, scope, path, default_name, *, task=False):
    if not isinstance(value, dict):
        raise ValueError(f'{path} 必须是字典')
    allowed = {'type', 'params'} if task else {'type', 'name', 'params'}
    unknown = set(value) - allowed
    if unknown:
        fields = ', '.join(sorted(str(key) for key in unknown))
        raise ValueError(f'{path} 含未知字段：{fields}；允许字段：{", ".join(sorted(allowed))}')
    require_name(value.get('type'), f'{path}.type')
    name = value.get('name')
    if name is None:
        name = default_name if default_name is not None else value['type']
    require_name(name, f'{path}.name')
    params = value.get('params', {})
    if not isinstance(params, dict):
        raise ValueError(f'{path}.params 必须是字典，可以省略')
    if any(not isinstance(key, str) for key in params):
        raise ValueError(f'{path}.params 的参数名必须是字符串')
    if task:
        for injected in ('context', 'hook_manager'):
            if injected in params:
                raise ValueError(f'task.params.{injected} 由框架注入，不能在配置中提供')
    return ComponentSpec(scope.lower(), value['type'].lower(), name, deepcopy(params), path)


def parse_components(config):
    if not isinstance(config, dict):
        raise ValueError('组件配置必须是字典')
    scopes, owners, specs = set(), {}, []
    for scope, value in config.items():
        require_name(scope, '配置顶层名称')
        normalized = scope.lower()
        if normalized in scopes or normalized in ('mate', 'task'):
            raise ValueError(f'注册域 {scope!r} 与其他配置域重名（忽略大小写）')
        scopes.add(normalized)
        if isinstance(value, dict):
            entries = [(scope, value, scope)]
        elif isinstance(value, list):
            entries = [(f'{scope}[{i}]', item, None) for i, item in enumerate(value)]
        else:
            raise ValueError(f'{scope} 必须是组件字典或组件列表')
        for path, item, default in entries:
            spec = component_spec(item, scope, path, default)
            if spec.name in owners:
                raise ValueError(f'{path} 的组件名称 {spec.name!r} 与 {owners[spec.name]} 重复')
            if spec.name in RUNTIME_NAMES:
                raise ValueError(f'{path} 的组件名称 {spec.name!r} 与框架保留名称重复')
            owners[spec.name] = path
            specs.append(spec)
    return tuple(specs)


def references(value, path, active=None) -> Iterator[tuple[str, str]]:
    if isinstance(value, (dict, list, tuple)):
        active = set() if active is None else active
        identity = id(value)
        if identity in active:
            raise ValueError(f'{path} 存在循环配置结构')
        active.add(identity)
        try:
            children = value.items() if isinstance(value, dict) else enumerate(value)
            for key, child in children:
                child_path = f'{path}.{key}' if isinstance(value, dict) else f'{path}[{key}]'
                yield from references(child, child_path, active)
        finally:
            active.remove(identity)
    elif isinstance(value, str) and value.startswith('@') and len(value) > 1:
        yield path, value[1:]


def order_components(specs, inputs: Container[str]):
    by_name = {spec.name: spec for spec in specs}
    graph = {}
    for name in sorted(by_name):
        spec = by_name[name]
        dependencies = set()
        for path, reference in references(spec.params, f'{spec.path}.params'):
            if reference in by_name:
                dependencies.add(reference)
            elif reference not in inputs:
                raise ValueError(f'{path} 引用 @{reference}: 未定义')
        graph[name] = tuple(sorted(dependencies))
    try:
        names = tuple(TopologicalSorter(graph).static_order())
    except CycleError as exc:
        cycle = ' -> '.join(exc.args[1])
        raise ValueError(f'组件依赖成环（组件不能引用自身）：{cycle}') from exc
    return tuple(by_name[name] for name in names)


def validate_task_references(task, components, inputs):
    names = {spec.name for spec in components}
    for path, reference in references(task.params, 'task.params'):
        if reference not in names and reference not in inputs:
            raise ValueError(f'{path} 引用 @{reference}: 未定义')


def parse_config(config):
    if not isinstance(config, dict):
        raise ValueError('配置顶层必须是字典')
    for name in config:
        require_name(name, '配置顶层名称')
    mate = config.get('mate')
    if not isinstance(mate, dict):
        raise ValueError('mate 必须是字典')
    for name in mate:
        require_name(name, 'mate 中的名称')
    seed = mate.get('seed')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('mate.seed 必须是 0 到 2**32 - 1 之间的整数')
    require_name(mate.get('experiment'), 'mate.experiment')
    task = component_spec(config.get('task'), 'task', 'task', 'task', task=True)
    component_config = {key: value for key, value in config.items() if key not in ('mate', 'task')}
    components = parse_components(component_config)
    scopes = {scope.lower() for scope in component_config}
    for name in mate:
        if name.lower() in scopes | RUNTIME_NAMES:
            raise ValueError(f'mate.{name} 与注册域或框架保留名称冲突')
    for component in components:
        if component.name in mate:
            raise ValueError(f'{component.path} 的组件名称 {component.name!r} 与 mate.{component.name} 重复')
    inputs = set(mate) | {'task', '__experiment_path__'}
    ordered = order_components(components, inputs)
    validate_task_references(task, ordered, inputs)
    return ExperimentSpec(deepcopy(mate), task, ordered)
