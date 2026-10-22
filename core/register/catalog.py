"""面向配置编辑器的注册目录与参数描述；查询不会创建组件实例。"""

import inspect
import math

from core.hook.hook import Hook
from core.register.autodiscover import autodiscover_components
from core.register.registryhub import R


class ComponentCatalogError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _hub(register_hub, discover):
    if discover:
        autodiscover_components()
    return R if register_hub is None else register_hub


def _name(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ComponentCatalogError('invalid_name', f'{field} 必须是非空字符串')
    return value.lower()


def _registry(hub, scope):
    scope = _name(scope, 'scope')
    registry = hub.get(scope)
    if registry is None:
        raise ComponentCatalogError('unknown_scope', f'未注册的注册域 {scope!r}')
    return registry


def _json_default(value, active=None):
    """只复制可直接往返 JSON 的默认值，避免把对象的 repr 当作可填写值。"""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) not in (list, dict):
        raise ValueError('默认值不能直接表示为 JSON')
    active = set() if active is None else active
    if id(value) in active:
        raise ValueError('默认值含循环引用')
    active.add(id(value))
    try:
        if type(value) is list:
            return [_json_default(item, active) for item in value]
        if any(type(key) is not str for key in value):
            raise ValueError('默认值的字典键必须是字符串')
        return {key: _json_default(item, active) for key, item in value.items()}
    finally:
        active.remove(id(value))


def _parameter(parameter):
    has_default = parameter.default is not inspect.Parameter.empty
    result = {
        'name': parameter.name,
        'kind': parameter.kind.name,
        'annotation': (None if parameter.annotation is inspect.Parameter.empty else
                       parameter.annotation if isinstance(parameter.annotation, str) else
                       inspect.formatannotation(parameter.annotation)),
        'required': not has_default,
        'has_default': has_default,
        'default_serializable': False,
        'configurable': parameter.kind is not inspect.Parameter.POSITIONAL_ONLY,
    }
    if has_default:
        try:
            result['default'] = _json_default(parameter.default)
            result['default_serializable'] = True
        except ValueError:
            try:
                result['default_repr'] = repr(parameter.default)
            except Exception:
                result['default_repr'] = f'<{type(parameter.default).__name__}>'
    return result


def _describe(scope, type_name, factory):
    result = {
        'scope': scope,
        'type': type_name,
        'module': getattr(factory, '__module__', None),
        'qualname': getattr(factory, '__qualname__', type(factory).__qualname__),
        'params': [],
        'required': [],
        'defaults': {},
        'injected_params': [],
        'accepts_extra_params': False,
        'variadic': {'args': None, 'kwargs': None},
        'configurable': True,
        'issues': [],
    }
    issues = result['issues']
    if not callable(factory):
        issues.append('注册对象必须可调用')
        result['configurable'] = False
        return result
    if scope == 'hook':
        if not inspect.isclass(factory) or not issubclass(factory, Hook) or inspect.isabstract(factory):
            issues.append('必须注册实现 run() 的 Hook 子类')
        elif not isinstance(factory.event, str) or not factory.event.strip():
            issues.append('Hook.event 必须是非空字符串')
    try:
        signature = inspect.signature(factory)
    except (ValueError, TypeError):
        issues.append('无法检查构造接口，请使用显式工厂包装该对象')
        result['configurable'] = False
        return result

    injected, context_value, manager_value = set(), object(), object()
    if scope == 'task':
        try:
            bound = signature.bind_partial(context_value, hook_manager=manager_value)
        except TypeError as exc:
            issues.append(f'任务接口不接受框架注入参数：{exc}')
        else:
            for name, value in bound.arguments.items():
                kind = signature.parameters[name].kind
                sources = []
                if value is context_value:
                    sources.append('context')
                elif value is manager_value:
                    sources.append('hook_manager')
                elif kind is inspect.Parameter.VAR_POSITIONAL:
                    sources.extend('context' for item in value if item is context_value)
                elif kind is inspect.Parameter.VAR_KEYWORD:
                    sources.extend(key for key, item in value.items() if item is manager_value)
                if sources:
                    injected.add(name)
                    result['injected_params'].extend(
                        {'name': name, 'source': source, 'kind': kind.name} for source in sources
                    )

    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            result['variadic']['args'] = parameter.name
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            result['variadic']['kwargs'] = parameter.name
            result['accepts_extra_params'] = True
            continue
        if parameter.name in injected:
            continue
        field = _parameter(parameter)
        if scope == 'task' and parameter.name in ('context', 'hook_manager'):
            field['configurable'] = False
            issues.append(f'task.params.{parameter.name} 由框架保留，不能在配置中提供')
        result['params'].append(field)
        if field['required']:
            result['required'].append(parameter.name)
        if field['configurable'] and field['default_serializable']:
            result['defaults'][parameter.name] = _json_default(field['default'])

    # 与构建器一样使用 Signature.bind 检查通过 YAML 关键字参数构造是否可行。
    # 占位对象只用于绑定检查，不调用工厂，不解析 @ 引用。
    required = {field['name']: object() for field in result['params']
                if field['required'] and field['configurable']}
    try:
        if scope == 'task':
            signature.bind(context_value, hook_manager=manager_value, **required)
        else:
            signature.bind(**required)
    except TypeError as exc:
        issues.append(f'无法通过配置 params 构造：{exc}')
    result['configurable'] = not issues
    return result


def list_scopes(*, discover=True, register_hub=None):
    """返回已注册组件的注册域名称，按名称排序；不包含空注册域。"""
    hub = _hub(register_hub, discover)
    return sorted(registry.name for registry in hub.all() if registry.keys())


def list_types(scope, *, discover=True, register_hub=None):
    """返回指定注册域的类型名称；查询未知名称不会创建注册域。"""
    hub = _hub(register_hub, discover)
    return sorted(_registry(hub, scope).keys())


def get_component_schema(scope, type_name, *, discover=True, register_hub=None):
    """描述实际构造接口，包括可填写参数、必填项、默认值及框架注入项。

    scope/type 忽略大小写。默认自动扫描 components 和 task，只导入模块，
    不创建实例。自定义注册表或已完成扫描时可设置 discover=False。
    返回值可直接用 json.dumps(..., allow_nan=False) 序列化。
    """
    hub = _hub(register_hub, discover)
    registry = _registry(hub, scope)
    type_name = _name(type_name, 'type')
    try:
        factory = registry.get(type_name)
    except KeyError as exc:
        raise ComponentCatalogError('unknown_type', f'{registry.name}: 未注册的组件类型 {type_name!r}') from exc
    return _describe(registry.name, type_name, factory)


def get_catalog(*, discover=True, register_hub=None):
    """一次导出完整目录，schema_version 为此描述格式的版本。"""
    hub = _hub(register_hub, discover)
    return {
        'schema_version': 1,
        'scopes': [
            {'scope': registry.name, 'types': [
                _describe(registry.name, key, registry.get(key)) for key in sorted(registry.keys())
            ]}
            for registry in sorted(hub.all(), key=lambda item: item.name) if registry.keys()
        ],
    }
