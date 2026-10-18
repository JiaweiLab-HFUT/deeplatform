# deeplatform

使用 Miniconda 的 `pytorch` 环境，从项目根目录启动：

```bash
conda activate pytorch
python main.py --config experiment/experiment_1/config.yaml
```

后台运行使用同一个入口：

```bash
python main.py --config experiment/experiment_1/config.yaml --background
```

每次运行只创建一个实验目录。后台启动会先创建该目录，再启动复用该目录的子进程，
不需要额外的 shell 日志重定向，也不会在 `experiment/` 根目录生成日志文件。

示例配置使用 `cuda:0`。CUDA 不可用时会记录错误并停止，不会自动改用 CPU。
运行前可以使用 `nvidia-smi` 和 `torch.cuda.is_available()` 检查当前环境能否访问 GPU。

每次运行在 `experiment/<实验名称>_<时间>_<进程号>/` 下建立独立目录：

- `config.yaml`：本次运行的配置快照，包含命令行的设备覆盖。
- `train.log`：终端输出、进度、Python 日志和异常堆栈。
- `console.log`：后台运行时捕获进程的完整标准输出和标准错误，包含训练日志初始化前的启动错误。
- `status.json`：进程号、设备、种子、开始和结束时间及运行状态。
- `metrics.jsonl`：每轮训练、验证和测试指标。
- `checkpoint/`：标签 Scaler 和训练过程中保存的最佳模型。

指标统一输出 Python `float`，没有样本的 seen/unseen 分组返回 `None`（JSON 中为 `null`）。

`mate.experiment_root` 可以设置实验输出的根目录。实验产物和数据软链接已在 `.gitignore` 中排除。

`Frame` 创建目录并通过 Context 的 `__experiment_path__` 将路径交给任务；
`experiment_logging()` 在运行期间将 `print`、进度输出及 Python `logging` 同时写到终端和
目录内的 `train.log`，退出时记录状态并恢复原来的标准流与日志配置。
任务的每轮指标追加到同一目录内的 `metrics.jsonl`，Scaler 和模型写入其 `checkpoint/`。

数据文件默认从 `/data/BatteryLife_Processed` 读取，并在项目数据目录下创建软链接。
电池寿命类别沿用本地 BatteryLife 原实现的区间定义，保存在
`components/datafactory/life_classes.json`。

运行测试：

```bash
python -m unittest discover -s tests -v
```

配置会先解析成统一的 `ComponentSpec`，再生成 `BuildPlan`，最后创建实例。
配置规则如下：

- 组件描述只允许 `type`、`name`、`params`，拼写错误等未知字段会报错。
- `task` 只允许 `type`、`params`；Context 由框架作为第一个参数注入，
  `task.params` 的其余参数会实际传给任务构造函数。
- `mate` 保留为可扩展的业务元数据字典，其中名称不能与注册域、组件实例和框架保留名称冲突。
- 单个组件的默认名称是配置中的注册域名称；列表中默认名称是 `type`。
  实例名称区分大小写，所有注册域共享的实例名称必须唯一；注册域和类型查找忽略大小写。
- `params` 可以省略，默认为空字典；不能写成 `null`。
- `_name` 表示引用元数据或其他组件，支持嵌套字典、列表和元组。
  可以引用写在后面的组件；构建器按依赖拓扑排序，相同依赖图的构建顺序固定。
  未定义引用、自身引用和循环依赖会在创建实例之前报错。
- 自动注册后，框架检查全部注册域、组件类型及构造函数签名，包括必需参数、
  多余参数和关键字参数。`**kwargs` 工厂允许动态参数；参数值的业务范围仍由组件负责。
- YAML 重复键会报错。解析、校验和构建均不会向输入配置中写入默认名称等运行状态。

可以只检查配置并查看构建顺序，不启动训练：

```python
from core.register.autodiscover import autodiscover_components
from frame.frame import Frame

frame = Frame()
config = frame.parse("configs/matr.yaml")  # 结构、名称和依赖检查
autodiscover_components()
plan = frame.check(config)                 # 注册类型和构造参数检查
print([item.spec.name for item in plan.components])
```
