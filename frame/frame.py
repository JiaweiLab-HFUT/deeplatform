from pathlib import Path

import yaml

from core.builder.builder import ComponentsBuilder
from core.context.context import Context
from core.register.autodiscover import autodiscover_components
from core.register.registryhub import R
from utils.seed import set_seed


class Frame:
    def __init__(self):
        self.register_hub = R
        self.context = Context().bind('global')
        self.components_builder = ComponentsBuilder(self.context)

    def run(self, config_path):
        cfg = self.parse(config_path)
        autodiscover_components()
        self.build_mate_task(cfg)
        set_seed(self.context.seed)
        self.build_components(cfg)
        self.creat_experiment(config_path)
        task_class = self.get_task_cls()
        task = task_class(self.context)
        task.run()


    def parse(self, config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            raise ValueError("文件不存在",FileNotFoundError)
        except yaml.YAMLError:
            raise ValueError("YAML 格式错误:", yaml.YAMLError)
        self.check(config)
        return config

    def build_mate_task(self, cfg):
        mate_data = cfg["mate"]
        for key, value in mate_data.items():
            self.context[key] = value
        self.context['task'] =  cfg["task"]

    def build_components(self,cfg):
        components_cfg = {k: v for k, v in cfg.items() if k not in ('mate', 'task')}
        self.components_builder.build_all_components(components_cfg)

    def check(self, config):
        """在创建组件前校验结构、Context 名称和按构建顺序解析的引用。

        单个组件默认以注册域命名，列表组件默认以 type 命名，规则与
        ComponentsBuilder 一致。检查不修改配置，也不实例化组件。
        """
        def require_name(value, path):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{path} 必须是非空字符串")

        def check_component(component, path):
            if not isinstance(component, dict):
                raise ValueError(f"{path} 必须是字典")
            require_name(component.get('type'), f"{path}.type")
            if component.get('name') is not None:
                require_name(component['name'], f"{path}.name")
            params = component.get('params', {})
            if not isinstance(params, dict):
                raise ValueError(f"{path}.params 必须是字典，可以省略")
            if any(not isinstance(key, str) for key in params):
                raise ValueError(f"{path}.params 的参数名必须是字符串")

        if not isinstance(config, dict):
            raise ValueError("配置顶层必须是字典")
        for scope in config:
            require_name(scope, '配置顶层名称')
        mate = config.get('mate')
        if not isinstance(mate, dict):
            raise ValueError("mate 必须是字典")
        for name in mate:
            require_name(name, 'mate 中的名称')
        seed = mate.get('seed')
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("mate.seed 必须是 0 到 2**32 - 1 之间的整数")
        require_name(mate.get('experiment'), 'mate.experiment')
        check_component(config.get('task'), 'task')

        scopes = [name for name in config if name not in ('mate', 'task')]
        normalized_scopes = set()
        for scope in scopes:
            if scope.lower() in normalized_scopes or scope.lower() in ('mate', 'task'):
                raise ValueError(f"注册域 {scope!r} 与其他配置域重名（忽略大小写）")
            normalized_scopes.add(scope.lower())
        registry_names = normalized_scopes | {r.name.lower() for r in self.register_hub.all()}
        reserved = {'task', '__experiment_path__'}
        for name in mate:
            if name.lower() in registry_names or name.lower() in reserved:
                raise ValueError(f"mate.{name} 与注册域或框架保留名称冲突")

        # 所有组件共享 Context，所以也检查跨注册域重名和元数据覆盖。
        owners = {name: f'mate.{name}' for name in mate}
        owners.update({name: '框架保留名称' for name in reserved})
        components = []
        for scope in scopes:
            value = config[scope]
            if isinstance(value, dict):
                entries = [(scope, value, scope)]
            elif isinstance(value, list):
                entries = [(f'{scope}[{i}]', item, None) for i, item in enumerate(value)]
            else:
                raise ValueError(f"{scope} 必须是组件字典或组件列表")
            for path, component, default_name in entries:
                check_component(component, path)
                name = component.get('name')
                if name is None:
                    name = default_name if default_name is not None else component['type']
                if name in owners:
                    raise ValueError(f"{path} 的组件名称 {name!r} 与 {owners[name]} 重复")
                owners[name] = path
                components.append((path, name, component.get('params', {})))

        def references(value, path):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield from references(child, f'{path}.{key}')
            elif isinstance(value, (list, tuple)):
                for i, child in enumerate(value):
                    yield from references(child, f'{path}[{i}]')
            elif isinstance(value, str) and value.startswith('_') and len(value) > 1:
                yield path, value[1:]

        available = set(mate) | {'task'}
        for path, name, params in components:
            for reference_path, reference in references(params, f'{path}.params'):
                if reference not in available:
                    if reference in owners:
                        detail = '尚未构建，请将依赖组件放在前面；组件不能引用自身'
                    else:
                        detail = '未定义'
                    raise ValueError(f"{reference_path} 引用 _{reference}: {detail}")
            available.add(name)

    def get_task_cls(self):
        task_type = self.context["task"]['type']
        cls = self.register_hub['task'].get(task_type)
        return cls

    def creat_experiment(self, cfg_file):
        exp_fold = '/experiment'
        exp_name = self.context["experiment"]
        context['__experiment_path__'] = exp_fold + exp_name
        Path(exp_fold + exp_name).mkdir(exist_ok=False)
        #将cfg保存到实验目录



