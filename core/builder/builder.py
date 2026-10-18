"""先验证全部接口，再按依赖关系创建组件。"""

from dataclasses import dataclass
import inspect
from typing import Callable

from core.config.schema import (
    ComponentSpec, order_components, parse_components, validate_task_references,
)
from core.register.registryhub import R


def resolve_params(obj, context):
    if isinstance(obj, dict):
        return {key: resolve_params(value, context) for key, value in obj.items()}
    if isinstance(obj, list):
        return [resolve_params(value, context) for value in obj]
    if isinstance(obj, tuple):
        return tuple(resolve_params(value, context) for value in obj)
    if isinstance(obj, str) and obj.startswith('_') and len(obj) > 1:
        return context[obj[1:]]
    return obj


@dataclass(frozen=True)
class PlannedComponent:
    spec: ComponentSpec
    factory: Callable


@dataclass(frozen=True)
class BuildPlan:
    components: tuple[PlannedComponent, ...]
    task: PlannedComponent | None = None


class ComponentsBuilder:
    def __init__(self, context, register_hub=None):
        self.context = context
        self.register_hub = R if register_hub is None else register_hub

    def validate_scopes(self, scopes):
        for scope in scopes:
            if self.register_hub.get(scope) is None:
                raise ValueError(f'{scope}: 未注册的注册域 {scope!r}')

    def _validate_factory(self, spec, *, task=False):
        registry = self.register_hub.get(spec.scope)
        if registry is None:
            raise ValueError(f'{spec.path}: 未注册的注册域 {spec.scope!r}')
        try:
            factory = registry.get(spec.type)
        except KeyError as exc:
            raise ValueError(f'{spec.path}.type: {exc.args[0]}') from exc
        if not callable(factory):
            raise ValueError(f'{spec.path}.type: 注册对象必须可调用')
        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{spec.path}.type: 无法检查构造接口，请使用显式工厂包装该对象') from exc
        try:
            if task:
                signature.bind(self.context, **spec.params)
            else:
                signature.bind(**spec.params)
        except TypeError as exc:
            raise ValueError(f'{spec.path}.params 与 {spec.type!r} 的构造参数不匹配：{exc}') from exc
        return PlannedComponent(spec, factory)

    def plan(self, specs, *, task=None, inputs=None):
        """只检查接口和依赖，不执行任何组件构造函数。"""
        inputs = self.context if inputs is None else inputs
        ordered = order_components(specs, inputs)
        if task is not None:
            validate_task_references(task, ordered, inputs)
        components = tuple(self._validate_factory(spec) for spec in ordered)
        planned_task = self._validate_factory(task, task=True) if task is not None else None
        return BuildPlan(components, planned_task)

    def build_plan(self, plan):
        for component in plan.components:
            params = resolve_params(component.spec.params, self.context)
            self.context[component.spec.name] = component.factory(**params)

    def build_task(self, plan):
        if plan.task is None:
            raise ValueError('构建计划中没有 task')
        params = resolve_params(plan.task.spec.params, self.context)
        return plan.task.factory(self.context, **params)

    def build_all_components(self, cfg):
        """兼容直接传入组件字典的调用，不修改输入配置。"""
        specs = parse_components(cfg)
        self.validate_scopes(cfg)
        plan = self.plan(specs)
        self.build_plan(plan)
        return plan
