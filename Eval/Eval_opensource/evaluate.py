#!/usr/bin/env python3
"""
本地开源模型评测脚本

现有所有任务（--task 可指定下列任一项或 all）:
  基础: camera_wearer, camera_wearer_type2, ego_2_exo_visibility, camera_relative_position,
        relative_distance, object_relative_position, object_correspondence, object_prediction
  View_Movement: view_movement_1, view_movement_2, view_movement_3, view_movement_4,
                 view_movement_5, view_movement_6
  Object_Movement: object_movement, object_movement_1
  View_Selection: view_selection, view_selection_2, view_selection_3
  Noise: noise_collaboration（Noise_Collaboration.json，不定项）

使用已下载的模型（默认目录由 LOCAL_MODEL_ROOT 指定）进行评测。
与 Eval_final 保持一致：
- 数据：同一 DataLoader、data_sources.yaml、Benchmark_annotations
- 评测逻辑与 prompt：直接复用 Eval_final 的 TASK_EVALUATORS 与 get_prompts，无本地覆盖
- 随机基线、PIL 任务列表、多选题列表：均使用 Eval_final 的 config（ALL_TASKS、MULTIPLE_CHOICE_TASKS）
仅将在线 API 替换为本地模型推理（LocalHFVLM）。

支持模型（示例，按目录名或别名）:
  - Qwen2.5-VL-7B: Qwen_Qwen2.5-VL-7B-Instruct 或 --model qwen2.5-vl-7b
  - Qwen3-VL-8B: Qwen_Qwen3-VL-8B-Instruct 或 --model qwen3-vl-8b
  - RynnBrain-2B / 8B: rynnbrain-2b、rynnbrain-8b（见 README）
  - MiMo-Embodied-7B: mimo-embodied-7b / mimo-embodied / 目录名 XiaomiMiMo_MiMo-Embodied-7B
  - 其他见 config.LOCAL_MODEL_ALIASES，也可直接写目录名或绝对路径

用法:
  # 指定任务与模型（别名或目录名）
  python evaluate.py --model qwen2.5-vl-7b --task camera_wearer_type2

  # 使用模型目录名
  python evaluate.py --model Qwen_Qwen2.5-VL-7B-Instruct --task view_movement_1

  # 限制样本数（测试用）
  python evaluate.py --model qwen2.5-vl-7b --task camera_wearer --limit 5

  # 评测所有任务
  python evaluate.py --model Qwen_Qwen2.5-VL-7B-Instruct --task all

  # 指定 GPU（物理卡号，写入 CUDA_VISIBLE_DEVICES；未传则不在此脚本内改环境变量）
  python evaluate.py --gpu 0 --model rynnbrain-2b --task camera_wearer_type2 --limit 2
"""

import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict


def _apply_gpu_from_argv_early() -> None:
    """
    在导入 Eval_final / torch 之前解析 --gpu，设置 CUDA_VISIBLE_DEVICES。
    若环境变量已存在且非空，则不覆盖（便于 shell: CUDA_VISIBLE_DEVICES=1 python ...）。
    """
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing is not None and str(existing).strip() != "":
        return
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] in ("--gpu", "--cuda-device") and i + 1 < len(argv):
            val = str(argv[i + 1]).strip()
            # 避免把下一个 flag（如 --task）当成设备号
            if val and not val.startswith("-"):
                os.environ["CUDA_VISIBLE_DEVICES"] = val
            return
        i += 1


_apply_gpu_from_argv_early()

# 复用 Eval_final 的评测逻辑与数据加载
EVAL_FINAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Eval_final"))
if EVAL_FINAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_FINAL_DIR)

from config import ALL_TASKS, MULTIPLE_CHOICE_TASKS
from data_loader import DataLoader
from evaluators import TASK_EVALUATORS, evaluate_random_baseline
from utils import HAS_PIL

# 本目录配置与模型（独立模块名避免与 Eval_final 的 config 冲突）
_OPENSOURCE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _OPENSOURCE_ROOT not in sys.path:
    sys.path.insert(0, _OPENSOURCE_ROOT)
from config_opensource import (
    get_output_dir as get_opensource_output_dir,
    resolve_model_path,
    SPATIAL_SSRL_MODEL_PATH,
    SENSENOVA_SI_MODEL_PATH,
    RYNNBRAIN_2B_MODEL_PATH,
    RYNNBRAIN_8B_MODEL_PATH,
    MIMO_EMBODIED_MODEL_PATH,
)
from local_models import (
    LocalHFVLM,
    SpatialSSRLVLM,
    SenseNovaSIVLM,
    RynnBrainVLM,
    MiMoEmbodiedVLM,
)


def create_local_model(model_name: str, **kwargs):
    """根据模型名或路径创建本地 VLM 实例。"""
    if model_name == "spatial-ssrl":
        return SpatialSSRLVLM(model_path=SPATIAL_SSRL_MODEL_PATH, **kwargs)
    if model_name == "sensenova-si":
        return SenseNovaSIVLM(model_path=SENSENOVA_SI_MODEL_PATH, **kwargs)
    if model_name == "rynnbrain-2b":
        return RynnBrainVLM(model_path=RYNNBRAIN_2B_MODEL_PATH, **kwargs)
    if model_name == "rynnbrain-8b":
        return RynnBrainVLM(model_path=RYNNBRAIN_8B_MODEL_PATH, **kwargs)
    if model_name in ("mimo-embodied-7b", "mimo-embodied"):
        return MiMoEmbodiedVLM(model_path=MIMO_EMBODIED_MODEL_PATH, **kwargs)
    path = resolve_model_path(model_name)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"本地模型目录不存在: {path}")
    # 目录名 XiaomiMiMo_MiMo-Embodied-7B 须走专用封装（image+path），不能走 LocalHFVLM 的 url 格式
    if os.path.basename(os.path.normpath(path)) == "XiaomiMiMo_MiMo-Embodied-7B":
        return MiMoEmbodiedVLM(model_path=os.path.abspath(path), **kwargs)
    return LocalHFVLM(model_path=path, **kwargs)


def run_random_baseline_local(
    tasks: List[str],
    output_dir: str,
    limit: int = None,
) -> List[Dict]:
    """运行随机基准（不调用模型），与 Eval_final 使用相同逻辑与 MULTIPLE_CHOICE_TASKS。"""
    data_loader = DataLoader()
    os.makedirs(output_dir, exist_ok=True)
    all_summaries = []

    for task in tasks:
        print(f"\n开始随机基准评测任务: {task}")
        try:
            data = data_loader.load_unified_data(task)
        except Exception as e:
            print(f"错误: 加载任务 {task} 的数据失败: {e}")
            import traceback
            traceback.print_exc()
            continue
        if not data:
            print(f"警告: 任务 {task} 没有数据，跳过")
            continue
        if limit:
            data = data[:limit]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"{task}_random_baseline_{timestamp}.json")
        allow_multiple = task in MULTIPLE_CHOICE_TASKS
        try:
            summary = evaluate_random_baseline(
                data=data, task=task, output_path=output_path, allow_multiple=allow_multiple
            )
            all_summaries.append(summary)
        except Exception as e:
            print(f"错误: 随机基准评测任务 {task} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    return all_summaries


def run_evaluation(
    model_name: str,
    tasks: List[str],
    output_dir: str,
    limit: int = None,
    index: int = None,
    save_input: bool = True,
    parallel: int = None,
    **model_kwargs,
) -> List[Dict]:
    """使用本地模型运行评测。"""
    model = create_local_model(model_name, **model_kwargs)
    print(f"已加载本地模型: {model}")

    data_loader = DataLoader()
    os.makedirs(output_dir, exist_ok=True)
    all_summaries = []

    for task in tasks:
        print(f"\n开始评测任务: {task}")
        # 检查任务是否需要 PIL（与 Eval_final 一致）
        if task in ['camera_wearer', 'object_correspondence', 'object_prediction',
                    'object_relative_position', 'relative_distance']:
            if not HAS_PIL:
                print(f"警告: 任务 {task} 需要 PIL 库，跳过")
                continue

        # 加载统一格式的数据
        try:
            data = data_loader.load_unified_data(task)
        except Exception as e:
            print(f"错误: 加载任务 {task} 的数据失败: {e}")
            import traceback
            traceback.print_exc()
            continue
        if not data:
            print(f"警告: 任务 {task} 没有数据，跳过")
            continue

        # 如果指定了 index，只评测特定样本
        if index is not None:
            if index < 0 or index >= len(data):
                print(f"错误: 索引 {index} 超出范围 [0, {len(data)-1}]")
                continue
            data = [data[index]]
            print(f"评测单个样本: index={index}")
        # 限制样本数量
        elif limit:
            data = data[:limit]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_safe = os.path.basename(model.model_name).replace("/", "_").replace("\\", "_")
        if index is not None:
            output_path = os.path.join(
                output_dir, f"{task}_local_{model_name_safe}_idx{index}_{timestamp}.json"
            )
        else:
            output_path = os.path.join(
                output_dir, f"{task}_local_{model_name_safe}_{timestamp}.json"
            )

        # 运行评测（与 Eval_final 相同：同一 evaluator，同一 prompt/数据流）
        evaluator = TASK_EVALUATORS.get(task)
        if evaluator:
            try:
                summary = evaluator(model, data, output_path, save_input=save_input, parallel=parallel)
                all_summaries.append(summary)
            except Exception as e:
                print(f"错误: 评测任务 {task} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        else:
            print(f"未知任务: {task}")
    return all_summaries


def main():
    parser = argparse.ArgumentParser(
        description="本地开源 VLM 评测（与 Eval_final 任务与数据一致）"
    )
    parser.add_argument(
        "--random_baseline",
        action="store_true",
        help="仅运行随机基准，不加载模型",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型名（别名/目录名/绝对路径），见 config.LOCAL_MODEL_ALIASES",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=ALL_TASKS + ["all"],
        help="评测任务",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="结果输出目录（默认: ./results）",
    )
    parser.add_argument("--limit", type=int, default=None, help="限制样本数（测试用）")
    parser.add_argument("--index", type=int, default=None, help="只评测指定索引的样本")
    parser.add_argument(
        "--save_input",
        action="store_true",
        default=True,
        help="保存首条输入示例",
    )
    parser.add_argument("--no_save_input", dest="save_input", action="store_false")
    parser.add_argument("--parallel", type=int, default=None, help="并行请求数")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="生成最大 token 数",
    )
    parser.add_argument(
        "--gpu",
        "--cuda-device",
        dest="gpu",
        type=str,
        default=None,
        metavar="ID",
        help=(
            "使用的物理 GPU 编号，写入 CUDA_VISIBLE_DEVICES（如 0 或 0,1）。"
            "已在进程环境中设置 CUDA_VISIBLE_DEVICES 时不会覆盖。"
            "实际生效在程序最开头解析，须紧跟在 python evaluate.py 之后尽早出现。"
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir or get_opensource_output_dir()
    output_dir = os.path.abspath(output_dir)
    tasks = ALL_TASKS if args.task == "all" else [args.task]

    if args.random_baseline:
        output_dir = os.path.join(output_dir, "random_baseline")
        summaries = run_random_baseline_local(tasks=tasks, output_dir=output_dir, limit=args.limit)
    else:
        if not args.model:
            parser.error("--model 为必填（或使用 --random_baseline）")
        output_dir = os.path.join(output_dir, "local")
        summaries = run_evaluation(
            model_name=args.model,
            tasks=tasks,
            output_dir=output_dir,
            limit=args.limit,
            index=args.index,
            save_input=args.save_input,
            parallel=args.parallel,
            max_new_tokens=args.max_new_tokens,
        )

    print(f"\n{'='*60}")
    print("评测总结")
    print(f"{'='*60}")
    for s in summaries:
        print(f"\n任务: {s['task']}")
        print(f"  模型: {s.get('model', 'N/A')} ({s.get('provider', 'N/A')})")
        print(f"  样本数: {s['total']}")
        if "accuracy" in s:
            print(f"  准确率: {s['accuracy']:.2%}")
        if "partial_rate" in s:
            print(f"  部分正确率: {s['partial_rate']:.2%}")
        if "f1_score" in s:
            print(f"  F1: {s['f1_score']:.2%}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
