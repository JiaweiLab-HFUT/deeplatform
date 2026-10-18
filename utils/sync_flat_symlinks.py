from pathlib import Path
def sync_flat_symlinks(src_root: str, dst_dir: str, exclude_in_src: list[str]):
    """
    将 A/* 下的所有文件平铺软链接到 B，清理失效链接，并按最新修改时间解决重名冲突
    """
    src_path = Path(src_root).resolve()  # resolve() 会自动穿透 A 本身的软链接
    dst_path = Path(dst_dir).resolve()
    dst_path.mkdir(parents=True, exist_ok=True)

    # 1. 扫描 A 下所有子目录中的文件，若重名则保留修改时间（mtime）最新的文件
    expected_files = {}  # {文件名: 最新源文件的 Path 对象}
    for subdir in src_path.iterdir():
        if subdir.is_dir():  # 仅处理 A1, A2 等子目录
            if subdir.name in exclude_in_src:
                continue
            for item in subdir.iterdir():
                if item.is_file():
                    name = item.name
                    if name not in expected_files or item.stat().st_mtime > expected_files[name].stat().st_mtime:
                        expected_files[name] = item

    # 2. 清理 B 中已经失效、或已被移除的旧软链接
    for existing in dst_path.iterdir():
        if existing.is_symlink():
            if existing.name not in expected_files or not existing.exists():
                existing.unlink()

    # 3. 创建或更新软链接
    for name, target in expected_files.items():
        link_path = dst_path / name
        if link_path.is_symlink():
            # 如果已指向正确目标，跳过；如果指向了旧的同名文件，则替换
            if link_path.resolve() == target.resolve():
                continue
            link_path.unlink()
        link_path.symlink_to(target)