from pathlib import Path
from datetime import datetime
import logging
import os
import shutil

import yaml
import torch

from core.builder.builder import BuildPlan, ComponentsBuilder
from core.config.schema import ConfigLoader, parse_config
from core.context.context import Context
from core.experiment.logging import experiment_logging
from core.register.autodiscover import autodiscover_components
from core.register.registryhub import R
from utils.seed import set_seed


class Frame:
    def __init__(self):
        self.register_hub = R
        self.context = Context().bind('global')
        self.components_builder = ComponentsBuilder(self.context)

    def run(self, config_path, device=None):
        cfg = self.prepare(config_path, device)
        return self._run(cfg)

    def prepare(self, config_path, device=None):
        """为一次运行创建唯一产物目录并保存配置。"""
        cfg = self.parse(config_path)
        if device is not None:
            cfg['mate']['device'] = device
        self.build_mate_task(cfg)
        self.creat_experiment(config_path, cfg)
        return cfg

    def run_prepared(self, directory):
        """后台子进程使用已准备的目录，不再创建第二个实验目录。"""
        directory = Path(directory).resolve()
        if (directory / 'status.json').exists():
            raise FileExistsError(f'实验目录已被运行使用，不能覆盖：{directory}')
        cfg = self.parse(directory / 'config.yaml')
        self.build_mate_task(cfg)
        self.context['__experiment_path__'] = str(directory)
        return self._run(cfg)

    def _run(self, cfg):
        metadata = {'seed': self.context.seed, 'device': cfg['mate'].get('device')}
        with experiment_logging(self.context.__experiment_path__, metadata):
            config_path = Path(self.context.__experiment_path__) / 'config.yaml'
            logging.info('Config: %s; seed=%s; device=%s', config_path,
                         self.context.seed, metadata['device'])
            if (metadata['device'] is not None
                    and torch.device(metadata['device']).type == 'cuda'
                    and not torch.cuda.is_available()):
                raise RuntimeError('CUDA 设备不可用，未启动训练；请检查 GPU 驱动及运行环境的设备访问。')
            autodiscover_components()
            plan = self.check(cfg)
            set_seed(self.context.seed)
            self.build_components(plan)
            task = self.components_builder.build_task(plan)
            task.run()
        return self.context.__experiment_path__


    def parse(self, config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.load(f, Loader=ConfigLoader)
        except FileNotFoundError as exc:
            raise ValueError(f'配置文件不存在：{config_path}') from exc
        except yaml.YAMLError as exc:
            raise ValueError(f'YAML 格式错误：{config_path}: {exc}') from exc
        self.check(config, validate_registry=False)
        return config

    def build_mate_task(self, cfg):
        mate_data = cfg["mate"]
        for key, value in mate_data.items():
            self.context[key] = value
        self.context['task'] =  cfg["task"]

    def build_components(self, config):
        self.components_builder.register_hub = self.register_hub
        plan = config if isinstance(config, BuildPlan) else self.check(config)
        self.components_builder.build_plan(plan)

    def check(self, config, *, validate_registry=True):
        """规范化配置；注册完成后还检查所有构造接口，不创建组件。"""
        spec = parse_config(config)
        if not validate_registry:
            return spec
        registry_names = {registry.name.lower() for registry in self.register_hub.all()}
        for name in spec.mate:
            if name.lower() in registry_names:
                raise ValueError(f'mate.{name} 与注册域或框架保留名称冲突')
        self.components_builder.register_hub = self.register_hub
        self.components_builder.validate_scopes(
            scope for scope in config if scope not in ('mate', 'task')
        )
        return self.components_builder.plan(
            spec.components, task=spec.task,
            inputs=set(spec.mate) | {'task', '__experiment_path__'},
        )

    def get_task_cls(self):
        task_type = self.context["task"]['type']
        cls = self.register_hub['task'].get(task_type)
        return cls

    def creat_experiment(self, cfg_file, config=None):
        root = Path(self.context['experiment_root']) if 'experiment_root' in self.context else (
            Path(__file__).resolve().parents[1] / 'experiment'
        )
        name = self.context.experiment.strip('/\\')
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('mate.experiment 必须是实验名称，不能包含目录路径')
        run_name = f'{name}_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}'
        directory = root.resolve() / run_name
        directory.mkdir(parents=True, exist_ok=False)
        self.context['__experiment_path__'] = str(directory)
        if config is None:
            shutil.copyfile(cfg_file, directory / 'config.yaml')
        else:
            (directory / 'config.yaml').write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8'
            )



