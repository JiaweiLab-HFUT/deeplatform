# deeplatform

使用 Miniconda 的 `pytorch` 环境，从项目根目录启动：

```bash
conda activate pytorch
python main.py --config experiment/experiment_1/config.yaml
```

示例配置使用 `cuda:0`。CUDA 不可用时会记录错误并停止，不会自动改用 CPU。
运行前可以使用 `nvidia-smi` 和 `torch.cuda.is_available()` 检查当前环境能否访问 GPU。

每次运行在 `experiment/<实验名称>_<时间>_<进程号>/` 下建立独立目录：

- `config.yaml`：本次运行的配置快照，包含命令行的设备覆盖。
- `train.log`：终端输出、进度、Python 日志和异常堆栈。
- `status.json`：进程号、设备、种子、开始和结束时间及运行状态。
- `metrics.jsonl`：每轮训练、验证和测试指标。
- `checkpoint/`：标签 Scaler 和训练过程中保存的最佳模型。

指标统一输出 Python `float`，没有样本的 seen/unseen 分组返回 `None`（JSON 中为 `null`）。

`mate.experiment_root` 可以设置实验输出的根目录。实验产物和数据软链接已在 `.gitignore` 中排除。

数据文件默认从 `/data/BatteryLife_Processed` 读取，并在项目数据目录下创建软链接。
电池寿命类别沿用本地 BatteryLife 原实现的区间定义，保存在
`components/datafactory/life_classes.json`。

运行测试：

```bash
python -m unittest discover -s tests -v
```
