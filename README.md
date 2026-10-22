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

数据划分保存在 `datasets/batteryLife/data_split/`，每个 JSON 文件包含 `dataset` 名称以及
对应的 `train`、`val`、`test` 文件名列表，例如 `UL_PUR.json`：

```json
{
  "dataset": "UL_PUR",
  "train": [
    "UL-PUR_N10-NA7_18650_NCA_23C_0-100_0.5-0.5C_g.pkl",
    "UL-PUR_N15-NA10_18650_NCA_23C_0-100_0.5-0.5C_j.pkl"
  ],
  "val": [],
  "test": []
}
```

划分加载逻辑与 dataFactory 一起位于 `components/datafactory/BatteryLifeDataset.py`。
其中的 `BatteryLifeDataSplit()` 初始化时按文件名顺序加载所有 `.json`，
以文件内的 `dataset` 字段为键合并到实例的 `split_record` 中，dataFactory 通过数据集名称查询划分。
默认目录根据项目位置解析，不依赖启动时的工作目录；也可以通过
`BatteryLifeDataSplit(split_dir=...)` 指定其他目录。
新增划分时添加同结构的 JSON，再在配置中设置 `datafactory.params.dataset` 即可。
数据集名称区分大小写，不能跨文件重名；每个划分必须包含三个列表，允许列表为空，
同一数据集中不允许重复文件或 train/val/test 之间交叉。
`MIX_large.json` 保存完整的组合名单，修改基础数据集 JSON 不会自动更新组合名单。

## 通过 Python 或命令上传组件

上传工具位于 `tools/component_upload.py`，仅依赖 Python 标准库，不属于训练组件。
在项目根目录、Miniconda 的 `pytorch` 环境中执行：

```bash
python -m tools.component_upload upload examples/components/external_linear.py --scope model
```

也可以直接调用 Python 函数：

```python
from tools.component_upload import upload_component, list_components

result = upload_component("examples/components/external_linear.py", scope="model")
print(result["path"])  # components/uploaded/model/external_linear.py
print(list_components(scope="model"))
```

`upload_component(source_path, scope, *, filename=None, project_root=None, max_upload_bytes=1048576)`
复制源文件，默认保留文件名；可用 `filename` 指定保存文件名。
默认项目位置根据工具所在目录解析，不依赖工作目录；可用 `project_root` 指定其他项目。
命令行对应参数为 `--filename`、`--project-root` 和 `--max-upload-bytes`。
返回字典包含保存的相对路径 `path`、源码 `sha256`、注册声明列表 `components`
以及 `activation: next_experiment`。

查询已存在的组件：

```bash
python -m tools.component_upload list --scope model
```

查询省略 `--scope` 会列出所有注册域。命令成功时向标准输出写 JSON 并返回 0；
安装失败时向标准错误写入包含 `error.code`、`error.message` 的 JSON 并返回 1，
命令行参数格式错误返回 2。
Python 函数遇到组件校验或重名错误时抛出 `ComponentUploadError`，可通过 `.code` 区分原因；
文件读取失败抛出 `OSError`，不合法的工具初始化参数抛出 `ValueError`。

新上传的组件源码统一保存到 `components/uploaded/<scope>/<filename>`，
包括 `scope=task` 的任务组件。现有自动扫描器会识别其中的注册装饰器，
后续通过 `main.py` 启动实验时自动加载。例如上传上述示例后，在 YAML 中使用：

```yaml
model:
  type: ExternalLinear
  params:
    early_cycle_threshold: 100
    charge_discharge_length: 300
```

上传文件遵循现有组件接口，模块顶层显式导入注册表并使用标准装饰器：

```python
from core.register.registryhub import R

@R.model.register('MyModel')
class MyModel:
    ...
```

支持 `@R.model.register`、`@R.model.register()`、`@R.model.register('MyModel')` 和
`@R.model.register(key='MyModel')`，以及 R 的导入别名、`R['model']` 写法。
注册域和类型名称必须能从源码静态确定，不支持动态拼接名称或运行时手动调用 register。
一个文件可包含辅助代码和多个组件，但注册声明必须属于指定的同一注册域。

工具检查 UTF-8 Python 语法、注册声明和重名冲突，使用锁与原子发布避免并发覆盖及扫描到
未写完的源码。文件名必须是公开的 Python 模块名，不接受路径、私有模块或 `__init__.py`。
默认大小上限为 1 MiB，不覆盖已有文件或同域中的同名类型，类型名称查重忽略大小写。
上传和查询只做静态解析，不执行源码、安装依赖或启动训练；
组件接口与构造参数仍由实验启动时的现有检查负责。组件源码在后续实验自动扫描时执行，
因此应使用可信源码。源文件保持原样，已经运行的实验不进行热替换。

## 查询配置表单元数据

配置编辑业务可直接调用 `core/register/catalog.py` 的 Python 接口，逐级查询：

```python
from core.register.catalog import (
    list_scopes, list_types, get_component_schema, get_catalog,
)

scopes = list_scopes()                             # 注册域下拉选项
types = list_types("model")                        # 指定注册域的 type 下拉选项
schema = get_component_schema("model", "CPMLP")     # 选定类型的 params 字段
catalog = get_catalog()                            # 一次获取全部注册域、类型与参数
```

全部返回值可直接 JSON 序列化，便于业务后端将数据交给前端。注册域和类型名称返回小写，
查询忽略大小写，排序固定。未知注册域或类型抛出 `ComponentCatalogError`，其 `.code`
分别为 `unknown_scope` 或 `unknown_type`；查询不会凭空创建注册域。空注册域不会作为选项返回。

例如 `get_component_schema("schedulers", "oneCycleLRBuilder")` 的关键字段为：

```json
{
  "scope": "schedulers",
  "type": "onecyclelrbuilder",
  "required": ["train_epochs", "max_lr"],
  "defaults": {"pct_start": 0.2}
}
```

其 `params` 列表按构造函数参数顺序返回每个字段的完整描述：

| 字段 | 含义 |
| --- | --- |
| `name` | 写入 YAML `params` 的字段名 |
| `annotation` | 构造参数的 Python 类型标注字符串；未声明时为 `null`，不猜测类型 |
| `kind` | Python 参数种类，如 `POSITIONAL_OR_KEYWORD`、`KEYWORD_ONLY` |
| `required` | 是否没有构造默认值、必须由用户提供 |
| `has_default` | 是否声明了默认值，用于区分无默认值与默认值为 `null` |
| `default_serializable` | 默认值能否作为 JSON 数据直接返回并预填 |
| `default` | 可预填的默认值，仅在可直接表示为 JSON 时出现 |
| `default_repr` | 非 JSON 默认值的展示文本，仅供展示，不应写入配置 |
| `configurable` | 能否通过 YAML `params` 关键字参数填写此字段 |

`defaults` 汇总所有可预填字段，`required` 汇总必填字段名。例如默认值为 `0`、`false`、
`null` 时均原样保留；未声明默认值的字段不返回 `default`。
Path、tuple、自定义对象、非有限浮点数及循环容器等默认值只返回 `default_repr`，
仍然是选填字段，省略它们会使用构造函数原始默认值。返回的列表和字典不会与原始默认值共享。
现有 CPMLP 构造函数的七个参数均未声明默认值，因此查询时会列为必填，
不会从示例 YAML 中推测默认值。

组件描述还包含以下控制信息：

- `injected_params`：框架注入的参数及来源，不应出现在用户的 params 表单中。
  Task 的 Context 按位置注入，`hook_manager` 按关键字注入，继承的构造接口同样有效。
- `accepts_extra_params`：构造函数有 `**kwargs` 时为 `true`，可允许用户增加字段，
  这些动态字段无法从签名中穷举。`variadic` 记录 `*args`、`**kwargs` 的名称，
  它们不会被误列为必填的普通字段。
- `configurable`、`issues`：构造接口能否按当前框架规则通过配置调用，以及不能调用的原因。
  例如必须提供位置专用参数、Task 不接受注入参数、Hook 类型不符合约定等。
  这些类型仍会列出供诊断，表单应禁用 `configurable=false` 的类型或字段。

接口自动扫描 `components` 和 `task` 并读取真实注册对象的构造签名，支持类、继承的构造函数、
工厂函数及上传后的组件。查询不创建组件实例或启动训练；扫描导入模块时会执行注册装饰器。
`build()`、`forward()`、Hook 的 `run()` 参数不属于构造 params，不会混入表单。
类型标注仅用于展示，不求值字符串标注，不执行类型转换；组件值范围、依赖引用和业务约束
仍由现有配置检查及组件负责，不能仅凭签名推断。

已完成扫描的业务进程可传 `discover=False` 直接读取当前注册表；也可以通过
`register_hub=...` 指定独立注册表，此时通常同时设置 `discover=False`。
批量接口 `get_catalog()` 返回 `schema_version: 1` 和 `scopes` 列表，
每个注册域条目包含 `scope` 名称和 `types` 组件描述列表。

也可以通过命令获取同样的 JSON 数据：

```bash
python -m tools.component_catalog scopes
python -m tools.component_catalog types model
python -m tools.component_catalog params model CPMLP
python -m tools.component_catalog all > component_catalog.json
```

命令成功返回 0，未知名称或组件自动扫描失败时向标准错误输出 JSON 错误并返回 1。
自动扫描失败会报错，不会将部分注册结果伪装成完整目录。

运行测试：

```bash
python -m unittest discover -s tests -v
```

配置会先解析成统一的 `ComponentSpec`，再生成 `BuildPlan`，最后创建实例。
配置规则如下：

- 组件描述只允许 `type`、`name`、`params`，拼写错误等未知字段会报错。
- `task` 只允许 `type`、`params`；Context 由框架作为第一个参数注入，
  本次实验的 HookManager 通过 `hook_manager` 关键字参数注入。
  任务构造函数需要接受这两个参数；`task.params` 不能覆盖它们，其余参数会实际传给任务。
- `mate` 保留为可扩展的业务元数据字典，其中名称不能与注册域、组件实例和框架保留名称冲突。
- 单个组件的默认名称是配置中的注册域名称；列表中默认名称是 `type`。
  实例名称区分大小写，所有注册域共享的实例名称必须唯一；注册域和类型查找忽略大小写。
- `params` 可以省略，默认为空字典；不能写成 `null`。
- `@name` 表示引用元数据或其他组件，支持嵌套字典、列表和元组。
  YAML 中必须加引号，例如 `model: "@model"`、`train_epochs: "@epochs"`；
  `_name` 现在作为普通字符串传递。
  可以引用写在后面的组件；构建器按依赖拓扑排序，相同依赖图的构建顺序固定。
  未定义引用、自身引用和循环依赖会在创建实例之前报错。
- 自动注册后，框架检查全部注册域、组件类型及构造函数签名，包括必需参数、
  多余参数和关键字参数。`**kwargs` 工厂允许动态参数；参数值的业务范围仍由组件负责。
- YAML 重复键会报错。解析、校验和构建均不会向输入配置中写入默认名称等运行状态。

CPMLP 的超参数直接写在 `model.params` 中，构造函数逐项接收：

```yaml
model:
  type: CPMLP
  params:
    d_ff: 256
    d_model: 128
    charge_discharge_length: 300
    early_cycle_threshold: 100
    dropout: 0
    e_layers: 4
    d_layers: 2
```

这些参数也可以使用 `"@name"` 引用元数据。原来的 `configs: "@model_params"` 包装方式已移除。

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

Hook 类型和其他组件一样通过 `@R.hook.register` 注册，自动扫描只注册类型，不启用实例。
Hook 类继承 `core.hook.hook.Hook`，用类属性 `event` 声明事件，实现 `run()` 处理函数；
构造函数只保存参数，不自动绑定事件。只有配置中列出的 Hook 才会在本次实验启用。
例如，下面的配置运行示例任务并打印开始、结束信息：

```yaml
mate:
  seed: 42
  experiment: hook_demo
task:
  type: MyTask
hook:
  - type: PrintStart
    params:
      priority: 1
  - type: PrintEnd
```

每次运行在构建组件之前创建独立 `HookManager`，组件构建完成后按构建计划绑定 Hook，
优先级数字越小越先执行，同优先级按绑定顺序执行。绑定顺序由现有组件依赖排序决定，
不依赖 YAML 的排列；有先后要求时应显式设置优先级。
正常结束和异常退出都会清空管理器。复用同一个 Frame 运行时，Context 也会重新创建，
避免保留上次实验的组件。全局注册表始终只保存类型，不保存实验实例。

`Task.__init__(context, hook_manager)` 保存管理器为 `self.hooks`，不通过 Context 传递它：

```python
class CustomTask(Task):
    def __init__(self, context, hook_manager, label='example'):
        super().__init__(context, hook_manager)
        self.label = label

    def run(self):
        hooks = self.hooks
        hooks.emit('start')
        hooks.emit('end', self.label)
```

原来的全局 `hook_reg()` 和 `hook_emit()` 已移除，已有 Hook 需要改为上述类型注册与实例调用。
事件名称和参数沿用当前任务定义；处理函数异常记录后向外抛出，避免控制训练的判断失败被忽略。
手动绑定时，`register(hook)` 返回注销句柄，可以用 `unregister(token)` 精确移除该绑定。

HookManager 负责事件绑定与分发。需要停止训练时，Hook 可以直接使用任务传入的 Accelerator：

```python
@R.hook.register
class StopOnThreshold(Hook):
    event = 'epoch_over'

    def __init__(self, threshold, priority=0):
        super().__init__(priority)
        self.threshold = threshold

    def run(self, context, train_result, vali_result, test_result):
        if vali_result['mape'] <= self.threshold:
            context.accelerator.print('验证指标已达到目标')
            context.accelerator.set_trigger()
```

配置在 `hook` 中启用该类型，并通过 `params.threshold` 设置阈值即可。
当前 BatteryLifeRun 在每轮 `epoch_over` 事件后调用 `accelerator.check_trigger()`，
同步并消费各进程的触发信号，然后结束训练循环。因此其他事件中的 Hook 设置触发信号后，
也会完成当前 epoch 的训练、评估和保存，再退出循环并执行 `after_train`。
HookManager 不保存训练控制状态，也不提供停止接口。

注册时 Hook 仍会绑定管理器，注销或清理时解除绑定；同一实例不能同时绑定多个管理器。
