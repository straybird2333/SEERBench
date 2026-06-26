# 本地开源模型评测 (Eval_opensource)

使用**本地已下载模型**运行与 `Eval_final` 相同的评测任务与数据，不依赖在线 API。

## 目录结构

```
Eval_opensource/
├── evaluate.py
├── config_opensource.py
├── local_models/
│   ├── __init__.py
│   ├── base.py
│   ├── local_hf_vlm.py
│   ├── spatial_ssrl_vlm.py
│   ├── sensenova_si_vlm.py
│   ├── rynnbrain_vlm.py
│   └── mimo_embodied_vlm.py
├── requirements.txt
├── Dockerfile
├── README.md
└── results/
```

## 环境配置（Conda 推荐）

```bash
conda create -n eval_opensource python=3.10 -y
conda activate eval_opensource
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

cd Eval/Eval_opensource
pip install -r requirements.txt
pip install -r ../Eval_final/requirements.txt
pip install qwen-vl-utils
python evaluate.py --help
```

## Docker（可选）

```bash
docker run -it --gpus all \
  -v /path/to/SEERBench:/workspace \
  -e LOCAL_MODEL_ROOT=/workspace/models \
  -e RYNNBRAIN_2B_MODEL_PATH=/workspace/models/DAMO_Academy_RynnBrain-2B \
  -e RYNNBRAIN_8B_MODEL_PATH=/workspace/models/DAMO_Academy_RynnBrain-8B \
  -e MIMO_EMBODIED_MODEL_PATH=/workspace/models/XiaomiMiMo_MiMo-Embodied-7B \
  -w /workspace/Eval/Eval_opensource \
  pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel bash
```

## 支持模型（节选）

| 别名 | 说明 |
|------|------|
| `qwen2.5-vl-7b` 等 | 见 `config_opensource.py` 中 `LOCAL_MODEL_ALIASES` |
| `spatial-ssrl` | `SPATIAL_SSRL_MODEL_PATH` |
| `sensenova-si` | `SENSENOVA_SI_MODEL_PATH` |
| **`rynnbrain-2b`** | 默认 `LOCAL_MODEL_ROOT/DAMO_Academy_RynnBrain-2B`，`RYNNBRAIN_2B_MODEL_PATH` |
| **`rynnbrain-8b`** | 默认 `LOCAL_MODEL_ROOT/DAMO_Academy_RynnBrain-8B`，`RYNNBRAIN_8B_MODEL_PATH` |
| **`mimo-embodied-7b`** / **`mimo-embodied`** | 默认 `LOCAL_MODEL_ROOT/XiaomiMiMo_MiMo-Embodied-7B`，`MIMO_EMBODIED_MODEL_PATH` |
| 目录名 `XiaomiMiMo_MiMo-Embodied-7B` 或该目录绝对路径 | 同上，自动使用 `MiMoEmbodiedVLM` |

## MiMo-Embodied-7B（小米）

权重目录示例：`/path/to/models/XiaomiMiMo_MiMo-Embodied-7B`。官方示例使用 `AutoModelForImageTextToText` + `AutoProcessor`，`trust_remote_code=True`，图像字段为 `{"type": "image", "path": "<绝对路径>"}`（与 `LocalHFVLM` 里 Qwen2.5-VL 的 `url` 写法不同），评测使用 `local_models/mimo_embodied_vlm.py`。

### 环境

```bash
conda activate eval_opensource
pip install -U "transformers>=4.51.3"
```

（checkpoint `config.json` 标注 `transformers_version` 为 4.51.3；若与 RynnBrain 共用环境已装 4.57.x，通常也可尝试。）

### 运行

```bash
python evaluate.py --model mimo-embodied-7b --task camera_wearer_type2
python evaluate.py --model mimo-embodied-7b --task camera_wearer --limit 5
export MIMO_EMBODIED_MODEL_PATH=/your/path/to/MiMo-Embodied-7B
python evaluate.py --model mimo-embodied-7b --task all
```

## RynnBrain-2B / 8B（DAMO Academy）

[RynnBrain](https://github.com/alibaba-damo-academy/RynnBrain) 系列；**2B**：[HF](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-2B)；**8B**：[HF](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-8B)。封装：`local_models/rynnbrain_vlm.py`。

### 环境

```bash
pip install -U "transformers==4.57.1"
```

### 运行

```bash
python evaluate.py --model rynnbrain-2b --task camera_wearer_type2
python evaluate.py --model rynnbrain-8b --task camera_wearer_type2
```

## 路径环境变量

| 变量 | 含义 |
|------|------|
| `LOCAL_MODEL_ROOT` | 通用模型根目录 |
| `RYNNBRAIN_2B_MODEL_PATH` | RynnBrain-2B |
| `RYNNBRAIN_8B_MODEL_PATH` | RynnBrain-8B |
| `MIMO_EMBODIED_MODEL_PATH` | MiMo-Embodied-7B |
| `SPATIAL_SSRL_MODEL_PATH` | Spatial-SSRL |
| `SENSENOVA_SI_MODEL_PATH` | SenseNova-SI |
| `EVAL_OUTPUT_DIR` | 评测输出根目录 |

## 任务与参数

与 `Eval_final` 一致；`python evaluate.py --help` 查看 `--task`、`--limit`、`--max_new_tokens` 等。

## 指定 GPU

`evaluate.py` 支持 `--gpu`（或 `--cuda-device`），在加载 PyTorch 前设置 `CUDA_VISIBLE_DEVICES`。若已在 shell 中 `export CUDA_VISIBLE_DEVICES`，则不会覆盖。

```bash
python evaluate.py --gpu 0 --model rynnbrain-2b --task camera_wearer_type2 --limit 2
python evaluate.py --gpu 0,1 --model rynnbrain-8b --task all
```

## 结果

输出在 `results/local/`，文件名形如 `{task}_local_{model_name}_{timestamp}.json`。
