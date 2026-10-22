"""组件源码管理工具，通过 Python 函数或命令行安装源码；不执行上传的组件。"""

import argparse
import ast
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
from io import BytesIO
import json
import keyword
import os
from pathlib import Path
import sys
import tempfile
import tokenize


DEFAULT_MAX_UPLOAD_BYTES = 1024 * 1024


class ComponentUploadError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ComponentDefinition:
    scope: str
    type: str
    symbol: str
    module: str
    path: str


def _scope(value):
    value = value.lower()
    if not value.isidentifier() or value.startswith('_') or keyword.iskeyword(value) or value == 'mate':
        raise ComponentUploadError('invalid_scope', 'scope 必须是公开的 Python 标识符，且不能为 mate')
    return value


def _registration(decorator, aliases, symbol):
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if not isinstance(target, ast.Attribute) or target.attr != 'register':
        return None
    registry = target.value
    if isinstance(registry, ast.Attribute) and isinstance(registry.value, ast.Name) and registry.value.id in aliases:
        scope = registry.attr
    elif isinstance(registry, ast.Subscript) and isinstance(registry.value, ast.Name) and registry.value.id in aliases:
        if not isinstance(registry.slice, ast.Constant) or not isinstance(registry.slice.value, str):
            raise ComponentUploadError('invalid_registration', '注册域必须使用字符串常量')
        scope = registry.slice.value
    else:
        return None
    name = symbol
    if isinstance(decorator, ast.Call):
        if len(decorator.args) > 1 or len(decorator.keywords) > 1 or (decorator.args and decorator.keywords):
            raise ComponentUploadError('invalid_registration', 'register 只接受一个类型名称')
        argument = decorator.args[0] if decorator.args else None
        if decorator.keywords:
            if decorator.keywords[0].arg != 'key':
                raise ComponentUploadError('invalid_registration', 'register 的关键字参数必须是 key')
            argument = decorator.keywords[0].value
        if argument is not None:
            if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str) or not argument.value.strip():
                raise ComponentUploadError('invalid_registration', '注册类型名称必须是非空字符串常量')
            name = argument.value
    return _scope(scope), name.lower()


def inspect_component_source(content, relative_path):
    """提取模块顶层的标准 R 注册装饰器；编译检查不会执行源码。"""
    try:
        encoding, _ = tokenize.detect_encoding(BytesIO(content).readline)
        if encoding not in ('utf-8', 'utf-8-sig'):
            raise UnicodeError('源码编码声明必须为 UTF-8')
        source = content.decode('utf-8-sig')
        tree = ast.parse(source, filename=str(relative_path))
        compile(tree, str(relative_path), 'exec', dont_inherit=True)
    except (UnicodeError, SyntaxError, ValueError) as exc:
        raise ComponentUploadError('invalid_source', f'{relative_path}: Python 源码无效：{exc}') from exc
    aliases = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == 'core.register.registryhub' and node.level == 0
        for alias in node.names if alias.name == 'R'
    }
    parts = relative_path.with_suffix('').parts
    module = '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)
    definitions, seen = [], set()
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            continue
        for decorator in node.decorator_list:
            registration = _registration(decorator, aliases, node.name)
            if registration is None:
                continue
            if registration in seen:
                raise ComponentUploadError('duplicate_registration', f'{relative_path}: 重复注册 {registration}')
            seen.add(registration)
            definitions.append(ComponentDefinition(*registration, node.name, module, relative_path.as_posix()))
    return definitions


class ComponentStore:
    def __init__(self, project_root=None, max_upload_bytes=DEFAULT_MAX_UPLOAD_BYTES):
        self.project_root = (
            Path(__file__).resolve().parents[1] if project_root is None else Path(project_root).resolve()
        )
        if not all((self.project_root / folder).is_dir() for folder in ('components', 'task')):
            raise ValueError('project_root 必须包含 components 和 task 目录')
        if type(max_upload_bytes) is not int or max_upload_bytes <= 0:
            raise ValueError('max_upload_bytes 必须是正整数')
        self.max_upload_bytes = max_upload_bytes

    def list_components(self, scope=None):
        scope = _scope(scope) if scope is not None else None
        definitions, seen = [], {}
        for folder in ('components', 'task'):
            root = self.project_root / folder
            for path in sorted(root.rglob('*.py')):
                relative = path.relative_to(root)
                parts = relative.parts[:-1] if path.name == '__init__.py' else relative.with_suffix('').parts
                if any(part.startswith(('_', '.')) for part in parts):
                    continue
                for definition in inspect_component_source(path.read_bytes(), path.relative_to(self.project_root)):
                    key = (definition.scope, definition.type)
                    if key in seen:
                        raise ComponentUploadError('type_exists', f'{key} 在 {seen[key]} 和 {definition.path} 中重复')
                    seen[key] = definition.path
                    if scope is None or definition.scope == scope:
                        definitions.append(asdict(definition))
        return sorted(definitions, key=lambda item: (item['scope'], item['type']))

    def _directory(self, relative):
        current = self.project_root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ComponentUploadError('invalid_path', '上传目录不能经过符号链接')
            try:
                current.mkdir(exist_ok=True)
            except FileExistsError as exc:
                raise ComponentUploadError('path_exists', f'{relative} 与已有文件冲突') from exc
        return current

    @contextmanager
    def _write_lock(self):
        directory = self._directory(Path('components/uploaded'))
        descriptor = os.open(directory / '.upload.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, 'r+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def upload(self, scope, filename, content):
        scope = _scope(scope)
        if (not filename.endswith('.py') or '/' in filename or '\\' in filename
                or not filename[:-3].isidentifier() or filename.startswith('_')
                or keyword.iskeyword(filename[:-3])):
            raise ComponentUploadError('invalid_filename', '文件名必须是公开的 Python 模块名，例如 my_model.py')
        if not content:
            raise ComponentUploadError('empty_file', '组件文件不能为空')
        if len(content) > self.max_upload_bytes:
            raise ComponentUploadError('file_too_large', f'组件文件不能超过 {self.max_upload_bytes} 字节')
        directory = Path('components/uploaded') / scope
        relative = directory / filename
        definitions = inspect_component_source(content, relative)
        if not definitions:
            raise ComponentUploadError('missing_registration', '请在模块顶层使用 @R.<scope>.register 注册类或工厂函数')
        if any(definition.scope != scope for definition in definitions):
            raise ComponentUploadError('scope_mismatch', '源码中的注册域必须全部与指定的 scope 一致')
        target = self.project_root / relative
        with self._write_lock():
            if target.exists() or target.is_symlink():
                raise ComponentUploadError('file_exists', f'{relative} 已存在，不允许覆盖')
            existing = {(item['scope'], item['type']): item['path'] for item in self.list_components()}
            for definition in definitions:
                owner = existing.get((definition.scope, definition.type))
                if owner is not None:
                    raise ComponentUploadError('type_exists', f'{definition.scope}.{definition.type} 已在 {owner} 注册')
            parent = self._directory(directory)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(prefix='_upload_', suffix='.tmp', dir=parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.chmod(0o644)
                # 硬链接一次性发布完整文件，目标已存在时不会覆盖。
                os.link(temporary, target)
            except FileExistsError as exc:
                raise ComponentUploadError('file_exists', f'{relative} 已存在，不允许覆盖') from exc
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return {
            'path': relative.as_posix(),
            'sha256': hashlib.sha256(content).hexdigest(),
            'activation': 'next_experiment',
            'components': [asdict(definition) for definition in definitions],
        }


def upload_component(source_path, scope, *, filename=None, project_root=None,
                     max_upload_bytes=DEFAULT_MAX_UPLOAD_BYTES):
    """将本地 .py 文件复制安装到 components/uploaded/<scope>/。

    默认保留源文件名；返回目标相对路径、SHA-256 和注册声明。
    不执行源码、不修改源文件。校验或重名错误抛出 ComponentUploadError，
    文件读取错误抛出 OSError。新实验通过现有扫描器自动加载组件。
    """
    store = ComponentStore(project_root, max_upload_bytes)
    source = Path(source_path)
    with source.open('rb') as stream:
        content = stream.read(store.max_upload_bytes + 1)
    return store.upload(scope, source.name if filename is None else filename, content)


def list_components(scope=None, *, project_root=None):
    """静态查询 components 和 task 目录中的注册声明，不导入组件。"""
    return ComponentStore(project_root).list_components(scope)


def main(argv=None):
    parser = argparse.ArgumentParser(description='deeplatform 组件源码上传与查询')
    commands = parser.add_subparsers(dest='command', required=True)
    upload = commands.add_parser('upload', help='安装本地 Python 组件文件')
    upload.add_argument('source', type=Path, help='源 .py 文件路径')
    upload.add_argument('--scope', required=True, help='注册域，例如 model、hook、task')
    upload.add_argument('--filename', help='保存文件名，默认使用源文件名')
    upload.add_argument('--max-upload-bytes', type=int, default=DEFAULT_MAX_UPLOAD_BYTES)
    listing = commands.add_parser('list', help='查询组件注册声明')
    listing.add_argument('--scope', help='按注册域筛选')
    for command in (upload, listing):
        command.add_argument('--project-root', type=Path, help='项目目录，默认为当前代码所在项目')
    args = parser.parse_args(argv)
    try:
        if args.command == 'upload':
            result = upload_component(args.source, args.scope, filename=args.filename,
                                      project_root=args.project_root, max_upload_bytes=args.max_upload_bytes)
        else:
            items = list_components(args.scope, project_root=args.project_root)
            result = {'count': len(items), 'items': items}
    except (ValueError, OSError) as exc:
        error = {'code': exc.code if isinstance(exc, ComponentUploadError) else 'invalid_input',
                 'message': str(exc)}
        print(json.dumps({'error': error}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
