"""从配置文件启动训练：python main.py --config experiment/experiment_1/config.yaml。"""

import argparse
from pathlib import Path
import subprocess
import sys

from frame.frame import Frame


def launch_background(config_path, device=None):
    """先创建实验目录，再将子进程所有输出重定向到该目录。"""
    frame = Frame()
    frame.prepare(config_path, device)
    directory = Path(frame.context.__experiment_path__)
    with (directory / 'console.log').open('a', encoding='utf-8') as output:
        process = subprocess.Popen(
            [sys.executable, '-u', str(Path(__file__).resolve()),
             '--experiment-dir', str(directory)],
            cwd=Path(__file__).resolve().parent,
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return process, directory


def main():
    parser = argparse.ArgumentParser(description='启动 deeplatform 实验')
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).resolve().parent / 'experiment/experiment_1/config.yaml')
    parser.add_argument('--device', help='覆盖配置中的设备，例如 cpu 或 cuda:0')
    parser.add_argument('--background', action='store_true', help='后台运行，所有输出仍保存在本次实验目录')
    parser.add_argument('--experiment-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.experiment_dir is not None:
        if args.background or args.device is not None:
            parser.error('--experiment-dir 不能与 --background 或 --device 同时使用')
        Frame().run_prepared(args.experiment_dir)
    elif args.background:
        process, directory = launch_background(args.config, args.device)
        print(f'实验已在后台启动，PID={process.pid}\n实验目录：{directory}')
    else:
        Frame().run(args.config, device=args.device)


if __name__ == '__main__':
    main()
