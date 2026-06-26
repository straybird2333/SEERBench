# SEERBench

SEERBench is a benchmark for evaluating multi-view ego-exo visual understanding and collaboration in vision-language models.

This repository currently publishes the evaluation code and usage documentation. Benchmark annotations and image assets will be released separately through an external download link.

## Repository Layout

```text
Eval/
  Eval_final/        # API-based evaluation framework
  Eval_opensource/   # Local open-source VLM evaluation framework
docs/
  data_organization.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Eval/Eval_final/requirements.txt
```

For local Hugging Face models:

```bash
pip install -r Eval/Eval_opensource/requirements.txt
```

## Data Placement

After downloading the released data package, place it as:

```text
SEERBench/
  Benchmark_annotations/
    Camera_Wearer.json
    ...
  data/
    egoexo4d_val/
      <scene>/<view>/<frame>.jpg
```

You can also keep data outside the repository and set:

```bash
export SEERBENCH_DATA_DIR=/path/to/Benchmark_annotations
export SEERBENCH_IMAGE_ROOT=/path/to/egoexo4d_val
```

## Run Evaluation

Random baseline:

```bash
cd Eval/Eval_final
python evaluate.py --random_baseline --task camera_wearer --limit 5
```

API model:

```bash
cd Eval/Eval_final
export OPENAI_API_KEY=your_key
python evaluate.py --provider openai --model gpt-4o --task camera_wearer
```

Local model:

```bash
cd Eval/Eval_opensource
export LOCAL_MODEL_ROOT=/path/to/local/models
python evaluate.py --model qwen2.5-vl-7b --task camera_wearer --limit 5
```

## Supported Tasks

- `camera_wearer`
- `camera_wearer_type2`
- `ego_2_exo_visibility`
- `camera_relative_position`
- `relative_distance`
- `object_relative_position`
- `object_correspondence`
- `object_prediction`
- `view_movement_1` to `view_movement_6`
- `object_movement`
- `object_movement_1`
- `view_selection`
- `view_selection_2`
- `view_selection_3`
- `noise_collaboration`

## License

License information will be added before the full public release.

