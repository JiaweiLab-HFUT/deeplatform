"""从配置文件启动训练：python main.py --config experiment/experiment_1/config.yaml。"""

import argparse
from pathlib import Path

from frame.frame import Frame


def main():
    parser = argparse.ArgumentParser(description='启动 deeplatform 实验')
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).resolve().parent / 'experiment/experiment_1/config.yaml')
    parser.add_argument('--device', help='覆盖配置中的设备，例如 cpu 或 cuda:0')
    args = parser.parse_args()
    Frame().run(args.config, device=args.device)


if __name__ == '__main__':
    main()
