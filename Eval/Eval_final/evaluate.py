#!/usr/bin/env python3
"""
通用视觉语言模型评测脚本（使用统一数据格式）

支持的模型:
- Qwen VL (qwen-vl-max, qwen-vl-plus, qwen2.5-vl-*, qwen3-vl-*, etc.)
- OpenAI GPT-4V/GPT-4o (gpt-4o, gpt-4o-mini, gpt-4-turbo, etc.)
- OpenRouter (各种模型，如 anthropic/claude-3.5-sonnet)
- SenseNova-SI (本地部署模型)

支持的评测任务（每个文件独立评测）:
- camera_wearer: Camera_Wearer.json - 单选题
- camera_wearer_type2: Camera_Wearer_Type2.json - 单选题
- ego_2_exo_visibility: Ego2Exo_Visibility.json - 单选题
- camera_relative_position: Camera_Relative_Position.json - 单选题
- relative_distance: Relative_Distance.json - 单选题
- object_relative_position: Object_Relative_Position.json - 单选题
- object_correspondence: Object_Correspondence.json - 单选题
- object_prediction: Object_Prediction.json - 单选题
- view_movement_1: View_Movement_1.json - 多选题（移动后哪些物体不可见）
- view_movement_2: View_Movement_2.json - 多选题（移动后哪些物体不可见）
- view_movement_3: View_Movement_3.json - 单选题（移动后物体的时钟方向）
- view_movement_4: View_Movement_4.json - 单选题（移动后能看到多少物体）
- view_movement_5: View_Movement_5.json - 单选题（移动后exo相机的时钟方向）
- view_movement_6: View_Movement_6.json - 单选题（移动后最近的物体）
- object_movement: Object_Movement.json - 单选题（物体移动后的时钟方向）
- object_movement_1: Object_Movement_1.json - 多选题（物体移动后哪些exo能看到）
- view_selection: View_Selection.json - 单选题（结合ego判断哪些exo有帮助）
- view_selection_2: View_Selection_2.json - 单选题（判断是否需要多视角协作）
- view_selection_3: View_Selection_3.json - 单选题（选择最有帮助的exo视角）

使用方法:
    # 使用 Qwen 模型
    python evaluate.py --provider qwen --model qwen-vl-max-latest --task camera_wearer
    
    # 使用 OpenAI 模型
    python evaluate.py --provider openai --model gpt-4o --task view_movement_1
    
    # 使用 OpenRouter 模型
    python evaluate.py --provider openrouter --model anthropic/claude-3.5-sonnet --task object_movement_1

    # 使用智谱 GLM-4.6V 模型
    python evaluate.py --provider glm --model glm-4.6v --task camera_wearer
    python evaluate.py --provider glm --model glm-4.6v --task all --enable_thinking
    
    # 运行所有任务
    python evaluate.py --provider qwen --task all
"""

import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ALL_TASKS,
    DEFAULT_MODELS, SUPPORTED_PROVIDERS,
    get_output_dir, MAX_PARALLEL_REQUESTS, PARALLEL_BATCH_SIZE,
    MULTIPLE_CHOICE_TASKS,
)
from data_loader import DataLoader
from evaluators import TASK_EVALUATORS, evaluate_random_baseline
from models import QwenVLM, OpenAIVLM, OpenRouterVLM, GLMVLM
try:
    from models import SenseNovaSIVLM
    HAS_SENSENOVA_SI = True
except (ImportError, SyntaxError) as e:
    HAS_SENSENOVA_SI = False
    SenseNovaSIVLM = None
from utils import HAS_PIL


# ==================== 模型创建 ====================

def create_model(provider: str, model_name: str = None, **kwargs):
    """
    创建模型实例
    
    Args:
        provider: 模型提供商 ('qwen', 'openai', 'openrouter', 'sensenova_si')
        model_name: 模型名称或路径
        **kwargs: 其他配置参数
    
    Returns:
        模型实例
    """
    if provider == 'qwen':
        model_name = model_name or DEFAULT_MODELS['qwen']
        return QwenVLM(model_name=model_name, **kwargs)
    elif provider == 'openai':
        model_name = model_name or DEFAULT_MODELS['openai']
        return OpenAIVLM(model_name=model_name, **kwargs)
    elif provider == 'openrouter':
        model_name = model_name or DEFAULT_MODELS['openrouter']
        return OpenRouterVLM(model_name=model_name, **kwargs)
    elif provider == 'sensenova_si':
        if not HAS_SENSENOVA_SI or SenseNovaSIVLM is None:
            raise ImportError(
                "SenseNova-SI 代码未找到。\n"
                "请确保 SenseNova-SI 目录存在，或通过环境变量设置:\n"
                "  export SENSENOVA_SI_PATH=/path/to/SenseNova-SI"
            )
        model_name = model_name or DEFAULT_MODELS['sensenova_si']
        # 提取 model_type 参数
        model_type = kwargs.pop('model_type', 'auto')
        return SenseNovaSIVLM(
            model_path=model_name,
            model_type=model_type,
            **kwargs
        )
    elif provider == 'glm':
        model_name = model_name or DEFAULT_MODELS['glm']
        return GLMVLM(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported: {SUPPORTED_PROVIDERS}")


# ==================== 评测运行 ====================

def run_random_baseline(
    tasks: List[str],
    output_dir: str,
    limit: int = None,
) -> List[Dict]:
    """
    运行随机基准评测（不调用模型）
    
    Args:
        tasks: 任务列表
        output_dir: 输出目录
        limit: 限制样本数量
    
    Returns:
        评测结果摘要列表
    """
    # 创建数据加载器
    data_loader = DataLoader()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    all_summaries = []
    
    for task in tasks:
        print(f"\n开始随机基准评测任务: {task}")
        
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
        
        # 限制样本数量
        if limit:
            data = data[:limit]
        
        # 生成输出文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'{task}_random_baseline_{timestamp}.json')
        
        # 判断是否为多选题（与 config.MULTIPLE_CHOICE_TASKS 一致）
        allow_multiple = task in MULTIPLE_CHOICE_TASKS
        
        # 运行随机基准评测
        try:
            summary = evaluate_random_baseline(
                data=data,
                task=task,
                output_path=output_path,
                allow_multiple=allow_multiple
            )
            all_summaries.append(summary)
        except Exception as e:
            print(f"错误: 随机基准评测任务 {task} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_summaries


def run_evaluation(
    provider: str,
    model_name: str,
    tasks: List[str],
    output_dir: str,
    limit: int = None,
    index: int = None,
    save_input: bool = True,
    parallel: int = None,
    **model_kwargs
) -> List[Dict]:
    """
    运行评测
    
    Args:
        provider: 模型提供商
        model_name: 模型名称
        tasks: 任务列表
        output_dir: 输出目录
        limit: 限制样本数量
        index: 评测特定样本的索引（从0开始）
        save_input: 是否保存输入示例
        parallel: 并行请求数（None表示串行）
        **model_kwargs: 模型配置参数
    
    Returns:
        评测结果摘要列表
    """
    # 创建模型
    model = create_model(provider, model_name, **model_kwargs)
    print(f"已加载模型: {model}")
    
    # 创建数据加载器
    data_loader = DataLoader()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    all_summaries = []
    
    for task in tasks:
        print(f"\n开始评测任务: {task}")
        
        # 检查任务是否需要 PIL
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
        
        # 如果指定了index，只评测特定样本
        if index is not None:
            if index < 0 or index >= len(data):
                print(f"错误: 索引 {index} 超出范围 [0, {len(data)-1}]")
                continue
            data = [data[index]]
            print(f"评测单个样本: index={index}")
        # 限制样本数量
        elif limit:
            data = data[:limit]
        
        # 生成输出文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 处理模型名称（如果是路径，提取最后一部分）
        model_name_safe = os.path.basename(model_name) if os.path.sep in str(model_name) else model_name
        model_name_safe = model_name_safe.replace('/', '_').replace('\\', '_')
        
        # 如果是单个样本评测，在文件名中添加索引
        if index is not None:
            output_path = os.path.join(output_dir, f'{task}_{provider}_{model_name_safe}_idx{index}_{timestamp}.json')
        else:
            output_path = os.path.join(output_dir, f'{task}_{provider}_{model_name_safe}_{timestamp}.json')
        
        # 运行评测
        evaluator = TASK_EVALUATORS.get(task)
        if evaluator:
            try:
                # 如果启用并行评测
                if parallel and parallel > 1:
                    from evaluators import evaluate_single_choice_task_parallel
                    # 并行评测需要特殊处理
                    summary = evaluator(model, data, output_path, save_input=save_input, parallel=parallel)
                else:
                    summary = evaluator(model, data, output_path, save_input=save_input)
                all_summaries.append(summary)
            except Exception as e:
                print(f"错误: 评测任务 {task} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        else:
            print(f"未知任务: {task}")
    
    return all_summaries


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='通用视觉语言模型评测脚本（使用统一数据格式）')
    parser.add_argument('--random_baseline', action='store_true',
                        help='运行随机基准评测（不调用模型，仅用于计算随机猜测的正确率）')
    parser.add_argument('--provider', type=str, default=None,
                        choices=SUPPORTED_PROVIDERS,
                        help='模型提供商 (qwen, openai, openrouter, sensenova_si)，如果使用 --random_baseline 则不需要')
    parser.add_argument('--model', type=str, default=None,
                        help='模型名称或路径，如果使用 --random_baseline 则不需要')
    parser.add_argument('--model_type', type=str, default='auto',
                        choices=['auto', 'qwen', 'internvl', 'bagel'],
                        help='SenseNova-SI 模型类型 (仅 sensenova_si 使用)')
    parser.add_argument('--task', type=str, required=True,
                        choices=ALL_TASKS + ['all'],
                        help='评测任务')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (默认: ./results)')
    parser.add_argument('--limit', type=int, default=None,
                        help='限制评测样本数量（用于测试）')
    parser.add_argument('--index', type=int, default=None,
                        help='评测特定样本的索引（从0开始计数）')
    parser.add_argument('--base_url', type=str, default=None,
                        help='API 基础 URL（用于代理，仅 OpenAI/OpenRouter）')
    parser.add_argument('--reasoning_effort', type=str, default=None,
                        choices=['none', 'minimal', 'low', 'medium', 'high', 'xhigh'],
                        help='OpenAI 推理模式强度 (仅 OpenAI 使用)')
    parser.add_argument('--enable_thinking', action='store_true',
                        help='启用 Qwen 推理/思考模式 (仅 Qwen 使用)')
    parser.add_argument('--data_sources_config', type=str, default=None,
                        help='数据源配置文件路径（可选）')
    parser.add_argument('--save_input', action='store_true', default=True,
                        help='保存模型输入示例到本地（默认：True）')
    parser.add_argument('--no_save_input', dest='save_input', action='store_false',
                        help='不保存模型输入示例')
    parser.add_argument('--parallel', type=int, default=None,
                        help='并行请求数（用于加速评测，默认不启用并行）')
    
    args = parser.parse_args()
    
    # 处理随机基准评测
    if args.random_baseline:
        # 随机基准评测不需要provider和model
        if args.provider or args.model:
            print("警告: --random_baseline 模式下不需要 --provider 和 --model 参数，将忽略这些参数")
        
        # 处理输出目录
        if args.output_dir is None:
            args.output_dir = get_output_dir()
        args.output_dir = os.path.join(args.output_dir, 'random_baseline')
        
        # 确定任务列表
        if args.task == 'all':
            tasks = ALL_TASKS
        else:
            tasks = [args.task]
        
        # 运行随机基准评测
        summaries = run_random_baseline(
            tasks=tasks,
            output_dir=args.output_dir,
            limit=args.limit
        )
    else:
        # 正常模型评测
        if not args.provider:
            parser.error("--provider 是必需的（除非使用 --random_baseline）")
        
        # 处理输出目录
        if args.output_dir is None:
            args.output_dir = get_output_dir()  # 默认使用 ./results
        # 如果用户指定了自定义路径，直接使用（不进行路径映射）

        # 根据模型提供商与推理模式分目录
        if args.provider == 'qwen':
            sub_dir = 'qwen/thinking' if args.enable_thinking else 'qwen/normal'
            args.output_dir = os.path.join(args.output_dir, sub_dir)
        elif args.provider == 'openai':
            sub_dir = 'openai/high' if args.reasoning_effort == 'high' else 'openai/normal'
            args.output_dir = os.path.join(args.output_dir, sub_dir)
        elif args.provider == 'openrouter':
            # 从模型名中提取提供商（如 anthropic/claude-3.5-sonnet -> anthropic）
            model_provider = args.model.split('/')[0] if args.model and '/' in args.model else 'default'
            args.output_dir = os.path.join(args.output_dir, 'openrouter', model_provider)
        elif args.provider == 'glm':
            sub_dir = 'glm/thinking' if getattr(args, 'enable_thinking', False) else 'glm/normal'
            args.output_dir = os.path.join(args.output_dir, sub_dir)

        # 确定任务列表
        if args.task == 'all':
            tasks = ALL_TASKS
        else:
            tasks = [args.task]
        
        # 构建模型参数
        model_kwargs = {}
        if args.base_url and args.provider == 'openai':
            model_kwargs['base_url'] = args.base_url
        if args.provider == 'sensenova_si':
            model_kwargs['model_type'] = args.model_type
        if args.reasoning_effort and args.provider == 'openai':
            model_kwargs['reasoning_effort'] = args.reasoning_effort
        if args.enable_thinking and args.provider in ('qwen', 'glm'):
            model_kwargs['enable_thinking'] = True
        
        # 运行评测
        summaries = run_evaluation(
            provider=args.provider,
            model_name=args.model,
            tasks=tasks,
            output_dir=args.output_dir,
            limit=args.limit,
            index=args.index,
            save_input=args.save_input,
            parallel=args.parallel,
            **model_kwargs
        )
    
    # 打印总结
    print(f"\n{'='*60}")
    print("评测总结")
    print(f"{'='*60}")
    for summary in summaries:
        print(f"\n任务: {summary['task']}")
        print(f"  模型: {summary['model']} ({summary['provider']})")
        print(f"  样本数: {summary['total']}")
        if 'accuracy' in summary:
            print(f"  准确率: {summary['accuracy']:.2%}")
        if 'partial_rate' in summary:
            print(f"  部分正确率: {summary['partial_rate']:.2%}")
        if 'f1_score' in summary:
            print(f"  F1分数: {summary['f1_score']:.2%}")
    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()
