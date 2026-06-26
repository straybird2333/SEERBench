# SEERBench

SEERBench evaluates vision-language models on multi-view ego-exo understanding and collaboration.

This repository currently provides the evaluation code. The benchmark data will be released separately.

## Data

Baidu Netdisk: `TODO: add link here`

Expected layout after downloading and extracting the data:

```text
SEERBench/
  Benchmark_annotations/
  data/
    egoexo4d_val/
```

If the data is stored elsewhere:

```bash
export SEERBENCH_DATA_DIR=/path/to/Benchmark_annotations
export SEERBENCH_IMAGE_ROOT=/path/to/egoexo4d_val
```

## Installation

```bash
pip install -r Eval/Eval_final/requirements.txt
```

For local open-source models:

```bash
pip install -r Eval/Eval_opensource/requirements.txt
```

## Evaluation

Random baseline:

```bash
cd Eval/Eval_final
python evaluate.py --random_baseline --task camera_wearer
```

API model:

```bash
cd Eval/Eval_final
python evaluate.py --provider openai --model gpt-4o --task camera_wearer
```

Local model:

```bash
cd Eval/Eval_opensource
export LOCAL_MODEL_ROOT=/path/to/models
python evaluate.py --model qwen2.5-vl-7b --task camera_wearer
```

Use `--task all` to evaluate all tasks.

## Tasks

SEERBench includes camera wearer recognition, visibility, camera relative position, relative distance, object relation, object correspondence, object prediction, view movement, object movement, view selection, and noise collaboration tasks.

