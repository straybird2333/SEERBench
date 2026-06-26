# Benchmark 评测框架

这是一个用于评测视觉语言模型在多个任务上性能的统一评测框架。

## 支持的模型

- Qwen VL (qwen-vl-max, qwen-vl-plus, qwen2.5-vl-*, qwen3-vl-*, etc.)
- OpenAI GPT-4V/GPT-4o (gpt-4o, gpt-4o-mini, gpt-4-turbo, etc.)
- SenseNova-SI (本地部署模型)

## 支持的评测任务

1. **camera_wearer** - 相机佩戴者识别
2. **object_correspondence** - 跨视角物体对应
3. **object_prediction** - 遮挡物体预测（支持多选题）
4. **object_relative_position** - 物体相对位置（支持多选题）
5. **camera_relative_position** - 相机相对位置
6. **relative_distance** - 相对距离
7. **camera_wearer_type2** - 相机佩戴者识别 Type2
8. **noise_collaboration** - 噪声协作

## 数据源

评测数据位于仓库根目录的 `Benchmark_annotations/`。也可以用环境变量覆盖：

```bash
export SEERBENCH_DATA_DIR=/path/to/Benchmark_annotations
export SEERBENCH_IMAGE_ROOT=/path/to/egoexo4d_val
```

数据文件格式：
- `task_1.json` - Camera Wearer
- `task_2.json` - Object Correspondence
- `task_3.json` - Object Prediction
- `task_4.json` - Object Relative Position
- `task_5.json` - Camera Relative Position
- `task_6.json` - Relative Distance
- `task_7.json` - Camera Wearer Type2
- `task_9.json` - Noise Collaboration

## 快速开始

### 1. 安装依赖

```bash
cd Eval/Eval_final
pip install -r requirements.txt
```

### 2. 配置 API 密钥

根据使用的模型提供商设置环境变量：

```bash
# Qwen
export DASHSCOPE_API_KEY=your_key_here

# OpenAI
export OPENAI_API_KEY=your_key_here
```

### 3. 运行评测

```bash
# 评测单个任务
python evaluate.py --provider qwen --model qwen-vl-max-latest --task camera_wearer

# 评测所有任务
python evaluate.py --provider qwen --task all

# 限制样本数量进行测试
python evaluate.py --provider qwen --task camera_wearer --limit 5
```

## 使用方法

### 基本命令

```bash
python evaluate.py --provider <provider> --task <task> [选项]
```

### 必需参数

- `--provider`: 模型提供商 (`qwen`, `openai`, `sensenova_si`)
- `--task`: 评测任务名称或 `all`（所有任务）

### 可选参数

- `--model`: 模型名称或路径（默认使用配置中的默认模型）
- `--output_dir`: 结果输出目录（默认：`./results`）
- `--limit`: 限制评测样本数量（用于快速测试）
- `--base_url`: API 基础 URL（用于代理，仅 OpenAI）
- `--reasoning_effort`: OpenAI 推理模式强度（仅 OpenAI）
- `--enable_thinking`: 启用 Qwen 推理/思考模式（仅 Qwen）
- `--model_type`: SenseNova-SI 模型类型（仅 sensenova_si）
- `--save_input`: 保存模型输入示例（默认：True）
- `--no_save_input`: 不保存模型输入示例
- `--random_baseline`: 运行随机基准评测（不调用模型，仅用于计算随机猜测的正确率）
  - 单选题：随机选择一个选项
  - 多选题：随机选择多个选项（1到所有选项数量的随机子集）

### 使用示例

```bash
# 使用 Qwen 模型
python evaluate.py --provider qwen --model qwen-vl-max-latest --task camera_wearer

# 使用 OpenAI 模型
python evaluate.py --provider openai --model gpt-4o --task object_correspondence

# 使用 SenseNova-SI 本地模型
python evaluate.py --provider sensenova_si --model /path/to/model --task object_prediction

# 评测所有任务
python evaluate.py --provider qwen --task all

# 测试模式（只评测5个样本）
python evaluate.py --provider qwen --task camera_wearer --limit 5

# 运行随机基准评测（不调用模型，计算随机猜测的正确率）
python evaluate.py --random_baseline --task camera_wearer

# 运行所有任务的随机基准评测
python evaluate.py --random_baseline --task all
```

## 输出结果

评测结果保存在 `results/` 目录下，文件名格式：
```
{task}_{provider}_{model}_{timestamp}.json
```

结果文件包含：
- `summary`: 评测摘要（准确率、样本数等）
- `results`: 每个样本的详细结果

## 目录结构

```
Eval_final/
├── config.py                 # 配置文件
├── data_sources.yaml         # 数据源配置
├── evaluate.py              # 主评测脚本
├── evaluators.py            # 评测函数
├── data_loader.py           # 数据加载器
├── data_adapters/          # 数据适配器
│   ├── __init__.py
│   ├── base.py
│   └── benchmark_task_adapter.py
├── models/                  # 模型接口
│   ├── __init__.py
│   ├── base.py
│   ├── qwen.py
│   ├── openai_gpt.py
│   └── sensenova_si.py
├── prompts.py               # 提示词模板
├── utils.py                 # 工具函数
├── requirements.txt         # 依赖列表
└── results/                 # 结果输出目录
```

## 随机基准评测

随机基准评测用于计算随机猜测的正确率，作为模型性能的基准线。

### 使用方法

```bash
# 评测单个任务的随机基准
python evaluate.py --random_baseline --task camera_wearer

# 评测所有任务的随机基准
python evaluate.py --random_baseline --task all

# 限制样本数量进行快速测试
python evaluate.py --random_baseline --task view_movement --limit 10
```

### 随机选择策略

- **单选题**：从所有选项中随机选择一个
- **多选题**（`view_movement`, `object_movement`）：随机选择1到所有选项数量的子集

### 输出结果

随机基准评测的结果保存在 `results/random_baseline/` 目录下，文件名格式：
```
{task}_random_baseline_{timestamp}.json
```

结果文件包含：
- `summary`: 评测摘要（准确率、样本数等）
- `results`: 每个样本的详细结果（随机选择的答案、是否正确等）

## 注意事项

1. **图像路径**：确保数据文件中的图像路径正确，系统会自动尝试添加数据根目录前缀
2. **多选题支持**：`view_movement` 和 `object_movement` 支持多选题答案
3. **标注任务**：部分任务需要 PIL 库来绘制标注（点、边界框、遮挡区域）
4. **路径映射**：支持通过 `SEERBENCH_ROOT`、`SEERBENCH_DATA_DIR`、`SEERBENCH_IMAGE_ROOT` 覆盖仓库、标注和图像路径
5. **随机基准**：随机基准评测不调用任何模型，仅用于计算随机猜测的正确率作为基准线

## 故障排除

### 数据文件找不到

检查 `data_sources.yaml` 中的路径配置，确保数据文件存在。

### API 密钥错误

确保设置了正确的环境变量：
```bash
echo $DASHSCOPE_API_KEY  # Qwen
echo $OPENAI_API_KEY     # OpenAI
```

### PIL 库缺失

```bash
pip install Pillow
```

## 更新日志

### v1.0.0
- 初始版本
- 支持从 Benchmark_annotations 目录读取数据
- 支持 8 个评测任务
- 支持 Qwen、OpenAI、SenseNova-SI 三种模型提供商
