import yaml
from pathlib import Path
from typing import Any

from core.register.registryhub import R

def resolve_params(obj, context):
    if isinstance(obj, dict):
        return {k: resolve_params(v, context) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_params(v, context) for v in obj]
    if isinstance(obj, tuple):
        return tuple(resolve_params(v, context) for v in obj)
    if isinstance(obj, str) and obj.startswith("_") and len(obj) > 1:
        # "_xxx" -> context.xxx
        return context[obj[1:]]
    return obj

class ComponentsBuilder:
    def __init__(self, context):
        self.context = context
        self.register_hub = R

    def build_all_components(self, cfg):
        for register_scope, components in cfg.items():
            if isinstance(components, list):
                no_name_components_type = {}
                for component in components:
                    if component.get("name") is None:
                        if no_name_components_type.get(component.get("type")) is not None:
                            raise ValueError("Name is not specified")
                        no_name_components_type[component.get("type")] = 1
                        component['name'] = component.get("type")
                    self._build(register_scope, component.get("type"), component.get("name"), component.get("params"))

            elif isinstance(components, dict):
                component = components
                if component.get("name") is None:
                    component['name'] = register_scope
                self._build(register_scope, component.get("type"), component.get("name"), component.get("params"))

    def _build(self, register_scope, components_type, name, params: dict):
        cls = self.register_hub[register_scope].get(components_type)
        params = resolve_params({} if params is None else params, self.context)
        obj = cls(**params)
        self.context[name] = obj
