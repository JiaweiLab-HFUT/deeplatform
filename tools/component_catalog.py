"""导出配置表单需要的注册目录和构造参数。"""

import argparse
from contextlib import redirect_stdout
import json
import sys

from core.register.catalog import (
    ComponentCatalogError, get_catalog, get_component_schema, list_scopes, list_types,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description='查询配置表单所需的注册域、组件类型和参数')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('scopes', help='列出注册域')
    types = commands.add_parser('types', help='列出指定注册域的组件类型')
    types.add_argument('scope')
    params = commands.add_parser('params', help='查询组件参数、默认值和必填项')
    params.add_argument('scope')
    params.add_argument('type')
    commands.add_parser('all', help='导出完整目录')
    args = parser.parse_args(argv)
    try:
        # 扫描执行注册模块时产生的 Python 输出进入 stderr，stdout 留给 JSON。
        with redirect_stdout(sys.stderr):
            if args.command == 'scopes':
                result = list_scopes()
            elif args.command == 'types':
                result = list_types(args.scope)
            elif args.command == 'params':
                result = get_component_schema(args.scope, args.type)
            else:
                result = get_catalog()
    except (ComponentCatalogError, ImportError) as exc:
        error = {'code': exc.code if isinstance(exc, ComponentCatalogError) else 'discovery_failed',
                 'message': str(exc)}
        print(json.dumps({'error': error}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
