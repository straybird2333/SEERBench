"""
评测函数模块
使用统一数据格式进行评测

现有所有任务（与 TASK_EVALUATORS 一致，--task 可指定下列任一项或 all）:
  基础: camera_wearer, camera_wearer_type2, ego_2_exo_visibility, camera_relative_position,
        relative_distance, object_relative_position, object_correspondence, object_prediction
  View_Movement: view_movement_1, view_movement_2, view_movement_3, view_movement_4,
                 view_movement_5, view_movement_6
  Object_Movement: object_movement, object_movement_1
  View_Selection: view_selection, view_selection_2, view_selection_3
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import random
from datetime import datetime
from typing import List, Dict, Callable, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_adapters import UnifiedDataItem
from prompts import get_prompts
from utils import (
    extract_answer, format_options, prepare_annotated_image,
    image_to_base64, HAS_PIL, draw_annotations_on_image,
    build_camera_mapping, map_camera_names_in_text, map_options_camera_names
)
from config import REQUEST_DELAY, MAX_PARALLEL_REQUESTS, MULTIPLE_CHOICE_TASKS


def process_single_item(
    model,
    item: UnifiedDataItem,
    task: str,
    system_prompt: str,
    user_prompt_template: str,
    get_images_fn: Callable,
    idx: int,
    total: int,
) -> Tuple[int, Dict]:
    """
    处理单个评测项（用于并行评测）
    
    Returns:
        (索引, 结果字典)
    """
    # 准备输入
    images = get_images_fn(item)
    question = item.question
    options = item.options
    ground_truth = item.answer
    
    options_str = format_options(options)
    user_prompt = user_prompt_template.format(
        question=question,
        options=options_str
    )
    
    # 调用模型
    api_response = model.call(
        images=images,
        question=user_prompt,
        system_prompt=system_prompt
    )
    
    # 提取答案
    valid_options = list(options.keys())
    answer_text = api_response['answer'] if api_response['success'] else ""
    allow_multiple = task in MULTIPLE_CHOICE_TASKS
    extracted_answer = extract_answer(answer_text, valid_options, allow_multiple=allow_multiple)
    
    # 判断正确性
    if allow_multiple:
        ground_truth_set = set([a.strip() for a in str(ground_truth).split(',')])
        extracted_set = set([a.strip() for a in str(extracted_answer).split(',')])
        is_correct = ground_truth_set == extracted_set
    else:
        is_correct = extracted_answer == ground_truth
    
    result = {
        'id': item.id,
        'question': question,
        'ground_truth': ground_truth,
        'raw_response': api_response['raw_response'],
        'model_response': answer_text,
        'extracted_answer': extracted_answer,
        'is_correct': is_correct,
        'usage': api_response.get('usage'),
        'error': api_response.get('error')
    }
    
    return idx, result, is_correct


def evaluate_single_choice_task_parallel(
    model,
    data: List[UnifiedDataItem],
    task: str,
    output_path: str,
    system_prompt: str,
    user_prompt_template: str,
    get_images_fn: Callable,
    save_input: bool = True,
    parallel: int = 5,
) -> Dict:
    """
    并行版本的单选题评测函数
    
    Args:
        model: 模型实例
        data: 统一格式的数据项列表
        task: 任务名称
        output_path: 输出文件路径
        system_prompt: 系统提示词
        user_prompt_template: 用户提示词模板
        get_images_fn: 获取图像列表的函数
        save_input: 是否保存输入示例
        parallel: 并行请求数
    """
    results = [None] * len(data)
    correct = 0
    total = len(data)
    
    print(f"\n{'='*60}")
    print(f"评测任务: {task} (并行模式, {parallel} 线程)")
    print(f"模型: {model.model_name} ({model.provider})")
    print(f"样本数量: {total}")
    print(f"{'='*60}\n")
    
    # 保存第一个样本的输入内容
    if save_input and len(data) > 0:
        item = data[0]
        images = get_images_fn(item)
        options_str = format_options(item.options)
        user_prompt = user_prompt_template.format(
            question=item.question,
            options=options_str
        )
        output_dir = os.path.dirname(output_path)
        save_model_input(task, item, images, system_prompt, user_prompt, output_dir)
    
    # 并行处理
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {}
        for idx, item in enumerate(data):
            future = executor.submit(
                process_single_item,
                model, item, task,
                system_prompt, user_prompt_template,
                get_images_fn, idx, total
            )
            futures[future] = idx
        
        # 收集结果
        completed = 0
        for future in as_completed(futures):
            try:
                idx, result, is_correct = future.result()
                results[idx] = result
                if is_correct:
                    correct += 1
                completed += 1
                print(f"[{completed}/{total}] {result['id']}: GT={result['ground_truth']} | Pred={result['extracted_answer']} | {'✓' if is_correct else '✗'}", flush=True)
            except Exception as e:
                idx = futures[future]
                print(f"[{completed+1}/{total}] 处理出错: {e}", flush=True)
                results[idx] = {
                    'id': data[idx].id,
                    'error': str(e),
                    'is_correct': False
                }
                completed += 1
    
    # 过滤空结果
    results = [r for r in results if r is not None]
    
    # 计算指标
    accuracy = correct / total if total > 0 else 0
    
    summary = {
        'task': task,
        'model': model.model_name,
        'provider': model.provider,
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'parallel': parallel,
        'timestamp': datetime.now().isoformat()
    }
    
    # 保存结果
    output = {'summary': summary, 'results': results}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"评测完成!")
    print(f"准确率: {accuracy:.2%} ({correct}/{total})")
    print(f"结果已保存至: {output_path}")
    print(f"{'='*60}\n")
    
    return summary


def save_model_input(
    task: str,
    item: UnifiedDataItem,
    images: List[str],
    system_prompt: str,
    user_prompt: str,
    output_dir: str
):
    """
    保存完整的模型输入内容到本地文件
    
    Args:
        task: 任务名称
        item: 数据项
        images: 图像列表（base64编码，已标注，实际发送给模型的图像）
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        output_dir: 输出目录
    """
    import base64
    from PIL import Image
    import io
    
    # 创建输入示例目录
    input_examples_dir = os.path.join(output_dir, 'input_examples')
    os.makedirs(input_examples_dir, exist_ok=True)
    
    # 创建图像保存目录
    images_dir = os.path.join(input_examples_dir, f'{task}_images')
    os.makedirs(images_dir, exist_ok=True)
    
    # 保存所有输入的图像（实际发送给模型的图像）
    saved_image_files = []
    image_info_list = []
    
    # 获取视图名称列表（按顺序）
    view_names = list(item.images.keys())
    
    for idx, image_base64 in enumerate(images):
        try:
            # 从base64解码图像
            if image_base64.startswith('data:image'):
                # 提取base64数据
                header, data = image_base64.split(',', 1)
                image_data = base64.b64decode(data)
            else:
                # 直接是base64字符串
                image_data = base64.b64decode(image_base64)
            
            # 转换为PIL Image
            img = Image.open(io.BytesIO(image_data))
            
            # 确定视图名称（如果有对应的视图）
            if idx < len(view_names):
                view_name = view_names[idx]
            else:
                view_name = f"image_{idx}"
            
            # 保存图像
            image_filename = f'{task}_input_image_{idx:02d}_{view_name}.jpg'
            image_filepath = os.path.join(images_dir, image_filename)
            img.save(image_filepath, 'JPEG', quality=95)
            
            saved_image_files.append(image_filename)
            
            # 获取该视图的标注信息
            annotations = item.get_annotations_for_view(view_name) if view_name in item.images else []
            
            image_info_list.append({
                'index': idx,
                'view_name': view_name,
                'filename': image_filename,
                'original_path': item.images.get(view_name, ''),
                'has_annotations': len(annotations) > 0,
                'annotation_count': len(annotations),
                'annotations': annotations,
                'image_size': {'width': img.width, 'height': img.height}
            })
            
        except Exception as e:
            print(f"  警告: 保存第 {idx+1} 张图像失败: {e}")
            image_info_list.append({
                'index': idx,
                'view_name': f'image_{idx}',
                'error': str(e)
            })
    
    # 构建输入内容
    input_content = {
        'task': task,
        'sample_id': item.id,
        'question': item.question,
        'options': item.options,
        'ground_truth': item.answer,
        'image_count': len(images),
        'images': image_info_list,
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'metadata': item.metadata,
        'note': '这些图像是实际发送给模型的输入。如果任务需要标注（camera_wearer, object_correspondence, object_prediction），图像已经根据标注信息进行了标注（点、边界框、遮挡区域等）。图像保存在 input_examples/{task}_images/ 目录下。'
    }
    
    # 保存到JSON文件
    input_file = os.path.join(input_examples_dir, f'{task}_input_example.json')
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(input_content, f, ensure_ascii=False, indent=2)
    
    print(f"  模型输入示例已保存至: {input_file}")
    print(f"  输入图像已保存至: {images_dir}/ ({len(saved_image_files)} 张)")


def evaluate_single_choice_task(
    model,
    data: List[UnifiedDataItem],
    task: str,
    output_path: str,
    system_prompt: str,
    user_prompt_template: str,
    get_images_fn,
    save_input: bool = True,
    parallel: int = None,
) -> Dict:
    """
    通用单选题评测函数（使用统一数据格式）
    
    Args:
        model: 模型实例
        data: 统一格式的数据项列表
        task: 任务名称
        output_path: 输出文件路径
        system_prompt: 系统提示词
        user_prompt_template: 用户提示词模板
        get_images_fn: 获取图像列表的函数
        save_input: 是否保存输入示例
        parallel: 并行请求数（None或1表示串行）
    """
    # 如果启用并行评测，使用并行版本
    if parallel and parallel > 1:
        return evaluate_single_choice_task_parallel(
            model, data, task, output_path,
            system_prompt, user_prompt_template,
            get_images_fn, save_input, parallel
        )
    results = []
    correct = 0
    total = len(data)
    
    print(f"\n{'='*60}")
    print(f"评测任务: {task}")
    print(f"模型: {model.model_name} ({model.provider})")
    print(f"样本数量: {total}")
    print(f"{'='*60}\n")
    
    # 保存第一个样本的输入内容
    input_saved = False
    
    for idx, item in enumerate(data):
        print(f"[{idx+1}/{total}] Processing: {item.id}")
        
        # 准备输入
        images = get_images_fn(item)
        question = item.question
        options = item.options
        ground_truth = item.answer
        
        options_str = format_options(options)
        user_prompt = user_prompt_template.format(
            question=question,
            options=options_str
        )
        
        # 保存第一个样本的输入内容
        if save_input and not input_saved and idx == 0:
            output_dir = os.path.dirname(output_path)
            save_model_input(task, item, images, system_prompt, user_prompt, output_dir)
            input_saved = True
        
        # 调用模型
        api_response = model.call(
            images=images,
            question=user_prompt,
            system_prompt=system_prompt
        )
        
        # 提取答案
        valid_options = list(options.keys())
        answer_text = api_response['answer'] if api_response['success'] else ""
        # 对于多选题任务，允许提取多个答案
        allow_multiple = task in MULTIPLE_CHOICE_TASKS
        extracted_answer = extract_answer(answer_text, valid_options, allow_multiple=allow_multiple)
        
        # 判断正确性（支持多选题）
        if allow_multiple:
            # 多选题：比较答案集合
            ground_truth_set = set([a.strip() for a in str(ground_truth).split(',')])
            extracted_set = set([a.strip() for a in str(extracted_answer).split(',')])
            is_correct = ground_truth_set == extracted_set
        else:
            # 单选题：直接比较
            is_correct = extracted_answer == ground_truth
        if is_correct:
            correct += 1
        
        # 记录结果
        result = {
            'id': item.id,
            'question': question,
            'ground_truth': ground_truth,
            'raw_response': api_response['raw_response'],
            'model_response': answer_text,
            'extracted_answer': extracted_answer,
            'is_correct': is_correct,
            'usage': api_response.get('usage'),
            'error': api_response.get('error')
        }
        results.append(result)
        
        print(f"  GT: {ground_truth} | Pred: {extracted_answer} | {'✓' if is_correct else '✗'}")
        
        time.sleep(REQUEST_DELAY)
    
    # 计算指标
    accuracy = correct / total if total > 0 else 0
    
    summary = {
        'task': task,
        'model': model.model_name,
        'provider': model.provider,
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }
    
    # 保存结果
    output = {'summary': summary, 'results': results}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"评测完成!")
    print(f"准确率: {accuracy:.2%} ({correct}/{total})")
    print(f"结果已保存至: {output_path}")
    print(f"{'='*60}\n")
    
    return summary


def evaluate_view_selection(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True) -> Dict:
    """评测视角选择任务（单选题）"""
    system_prompt, user_prompt_template = get_prompts('view_selection')
    
    results = []
    correct = 0
    total = len(data)
    
    print(f"\n{'='*60}")
    print(f"评测任务: view_selection (单选题)")
    print(f"模型: {model.model_name} ({model.provider})")
    print(f"样本数量: {total}")
    print(f"{'='*60}\n")
    
    # 保存第一个样本的输入内容
    input_saved = False
    
    for idx, item in enumerate(data):
        print(f"[{idx+1}/{total}] Processing: {item.id}")
        
        # 准备输入：ego图像 + candidate图像（按顺序）
        images = []
        if 'ego' in item.images:
            images.append(image_to_base64(item.images['ego']))
        
        # 按照 available_exo_views 的顺序添加候选图像，确保 exo-1, exo-2 等对应正确
        # 如果没有 available_exo_views，则按字典顺序
        available_exo_views = item.metadata.get('available_exo_views', [])
        if available_exo_views:
            # 按照 available_exo_views 的顺序添加
            for view_name in available_exo_views:
                if view_name in item.images:
                    images.append(image_to_base64(item.images[view_name]))
        else:
            # 如果没有指定顺序，按字典顺序添加（排除ego）
            for view_name, img_path in sorted(item.images.items()):
                if view_name != 'ego':
                    images.append(image_to_base64(img_path))
        
        options_str = format_options(item.options)
        user_prompt = user_prompt_template.format(
            question=item.question,
            options=options_str
        )
        
        # 保存第一个样本的输入内容
        if save_input and not input_saved and idx == 0:
            output_dir = os.path.dirname(output_path)
            save_model_input('view_selection', item, images, system_prompt, user_prompt, output_dir)
            input_saved = True
        
        # 调用模型
        api_response = model.call(
            images=images,
            question=user_prompt,
            system_prompt=system_prompt
        )
        
        # 提取答案
        valid_options = list(item.options.keys())
        answer_text = api_response['answer'] if api_response['success'] else ""
        extracted_answer = extract_answer(answer_text, valid_options, allow_multiple=False)
        
        # 判断正确性（单选题）
        ground_truth = item.answer
        is_correct = extracted_answer == ground_truth
        if is_correct:
            correct += 1
        
        # 记录结果
        result = {
            'id': item.id,
            'question': item.question,
            'ground_truth': ground_truth,
            'raw_response': api_response['raw_response'],
            'model_response': answer_text,
            'extracted_answer': extracted_answer,
            'is_correct': is_correct,
            'usage': api_response.get('usage')
        }
        results.append(result)
        
        status = '✓' if is_correct else '✗'
        print(f"  GT: {ground_truth} | Pred: {extracted_answer} | {status}")
        
        time.sleep(REQUEST_DELAY)
    
    # 计算指标
    accuracy = correct / total if total > 0 else 0
    summary = {
        'task': 'view_selection',
        'model': model.model_name,
        'provider': model.provider,
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }
    
    output = {'summary': summary, 'results': results}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"评测完成!")
    print(f"准确率: {accuracy:.2%} ({correct}/{total})")
    print(f"结果已保存至: {output_path}")
    print(f"{'='*60}\n")
    
    return summary


def evaluate_when_to_collaborate(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """评测何时需要协作任务（单选题）"""
    system_prompt, user_prompt_template = get_prompts('when_to_collaborate')

    def get_images(item: UnifiedDataItem):
        """获取 5 张图像：ego + exo(按原顺序)"""
        images = []
        if 'ego' in item.images:
            images.append(image_to_base64(item.images['ego']))
        for view_name, img_path in item.images.items():
            if view_name == 'ego':
                continue
            images.append(image_to_base64(img_path))
        return images

    return evaluate_single_choice_task(
        model, data, 'when_to_collaborate', output_path,
        system_prompt, user_prompt_template,
        get_images, save_input, parallel
    )


def evaluate_camera_wearer(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """评测相机佩戴者识别任务"""
    system_prompt, user_prompt_template = get_prompts('camera_wearer')
    
    def get_images(item: UnifiedDataItem):
        """获取带标注的图像列表"""
        images = []
        for view_name, img_path in item.images.items():
            annotations = item.get_annotations_for_view(view_name)
            images.append(prepare_annotated_image(img_path, annotations))
        return images
    
    return evaluate_single_choice_task(
        model, data, 'camera_wearer', output_path,
        system_prompt, user_prompt_template,
        get_images, save_input, parallel
    )


def _process_camera_wearer_type2_item(args):
    """处理单个 camera_wearer_type2 样本（用于并行处理）"""
    idx, item, model, system_prompt, user_prompt_template = args
    
    option_cameras = list(item.options.values())
    camera_to_exo_mapping, _, _ = build_camera_mapping(list(item.images.keys()))
    
    images = []
    if 'ego' in item.images:
        annotations = item.get_annotations_for_view('ego')
        images.append(prepare_annotated_image(item.images['ego'], annotations))
    
    for cam_name in option_cameras:
        if cam_name in item.images:
            annotations = item.get_annotations_for_view(cam_name)
            images.append(prepare_annotated_image(item.images[cam_name], annotations))
    
    mapped_options = map_options_camera_names(item.options, camera_to_exo_mapping)
    options_str = format_options(mapped_options)
    user_prompt = user_prompt_template.format(question=item.question, options=options_str)
    
    api_response = model.call(images=images, question=user_prompt, system_prompt=system_prompt)
    
    valid_options = list(mapped_options.keys())
    answer_text = api_response['answer'] if api_response['success'] else ""
    extracted_answer = extract_answer(answer_text, valid_options, allow_multiple=False)
    
    ground_truth = item.answer
    is_correct = extracted_answer == ground_truth
    
    result = {
        'id': item.id,
        'question': item.question,
        'options': mapped_options,
        'ground_truth': ground_truth,
        'predicted': extracted_answer,
        'is_correct': is_correct,
        'model_response': answer_text,
        'api_success': api_response['success'],
        'camera_mapping': camera_to_exo_mapping,
        'camera_order': option_cameras,
    }
    
    return idx, result, is_correct


def evaluate_camera_wearer_type2(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """
    评测相机佩戴者识别任务 Type2（ego视图中有bbox标注，选择对应的exo视图）
    支持并行处理
    """
    system_prompt, user_prompt_template = get_prompts('camera_wearer_type2')
    total = len(data)
    
    # 保存输入示例
    if save_input and data:
        item = data[0]
        option_cameras = list(item.options.values())
        camera_to_exo_mapping, _, _ = build_camera_mapping(list(item.images.keys()))
        
        images = []
        if 'ego' in item.images:
            annotations = item.get_annotations_for_view('ego')
            images.append(prepare_annotated_image(item.images['ego'], annotations))
        for cam_name in option_cameras:
            if cam_name in item.images:
                annotations = item.get_annotations_for_view(cam_name)
                images.append(prepare_annotated_image(item.images[cam_name], annotations))
        
        mapped_options = map_options_camera_names(item.options, camera_to_exo_mapping)
        options_str = format_options(mapped_options)
        user_prompt = user_prompt_template.format(question=item.question, options=options_str)
        output_dir = os.path.dirname(output_path)
        save_model_input('camera_wearer_type2', item, images, system_prompt, user_prompt, output_dir)
    
    # 并行处理
    if parallel and parallel > 1:
        print(f"\n{'='*60}")
        print(f"评测任务: camera_wearer_type2 (并行模式, {parallel} 线程)")
        print(f"模型: {model.model_name} ({model.provider})")
        print(f"样本数量: {total}")
        print(f"{'='*60}\n")
        
        results = [None] * total
        correct = 0
        
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {}
            for idx, item in enumerate(data):
                future = executor.submit(
                    _process_camera_wearer_type2_item,
                    (idx, item, model, system_prompt, user_prompt_template)
                )
                futures[future] = idx
            
            completed = 0
            for future in as_completed(futures):
                try:
                    idx, result, is_correct = future.result()
                    results[idx] = result
                    if is_correct:
                        correct += 1
                    completed += 1
                    print(f"[{completed}/{total}] ID: {result['id']} | GT: {result['ground_truth']} | Pred: {result['predicted']} | {'✓' if is_correct else '✗'}", flush=True)
                except Exception as e:
                    idx = futures[future]
                    print(f"[{completed+1}/{total}] 处理出错 (idx={idx}): {e}", flush=True)
                    results[idx] = {
                        'id': data[idx].id,
                        'error': str(e),
                        'is_correct': False,
                        'ground_truth': data[idx].answer,
                        'predicted': '',
                    }
                    completed += 1
    else:
        # 串行处理
        print(f"\n{'='*60}")
        print(f"评测任务: camera_wearer_type2")
        print(f"模型: {model.model_name} ({model.provider})")
        print(f"样本数量: {total}")
        print(f"{'='*60}\n")
        
        results = []
        correct = 0
        
        for idx, item in enumerate(data):
            print(f"[{idx+1}/{total}] Processing: {item.id}")
            _, result, is_correct = _process_camera_wearer_type2_item(
                (idx, item, model, system_prompt, user_prompt_template)
            )
            results.append(result)
            if is_correct:
                correct += 1
            print(f"  GT: {result['ground_truth']} | Pred: {result['predicted']} | {'✓' if is_correct else '✗'}")
            time.sleep(REQUEST_DELAY)
    
    accuracy = correct / total if total > 0 else 0
    
    summary = {
        'task': 'camera_wearer_type2',
        'model': model.model_name,
        'provider': model.provider,
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'parallel': parallel if parallel and parallel > 1 else None,
        'timestamp': datetime.now().isoformat()
    }
    
    output = {'summary': summary, 'results': results}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"评测完成!")
    print(f"准确率: {accuracy:.2%} ({correct}/{total})")
    print(f"结果已保存至: {output_path}")
    print(f"{'='*60}\n")
    
    return summary


def evaluate_object_correspondence(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """评测跨视角物体对应任务"""
    system_prompt, user_prompt_template = get_prompts('object_correspondence')
    
    def get_images(item: UnifiedDataItem):
        """获取带标注的图像列表"""
        images = []
        for view_name, img_path in item.images.items():
            annotations = item.get_annotations_for_view(view_name)
            if annotations:
                images.append(prepare_annotated_image(img_path, annotations))
            else:
                images.append(prepare_annotated_image(img_path, []))
        return images
    
    return evaluate_single_choice_task(
        model, data, 'object_correspondence', output_path,
        system_prompt, user_prompt_template,
        get_images, save_input, parallel
    )


def evaluate_object_prediction(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """评测遮挡物体预测任务"""
    system_prompt, user_prompt_template = get_prompts('object_prediction')
    
    def get_images(item: UnifiedDataItem):
        """获取用于预测的图像列表：ego 图像带 mask，其它视角作为参考图像"""
        images = []
        for view_name, img_path in item.images.items():
            annotations = item.get_annotations_for_view(view_name)
            # ego 视角通常带有 mask 标注；其它视角一般没有标注，此时返回原图
            images.append(prepare_annotated_image(img_path, annotations))
        return images
    
    return evaluate_single_choice_task(
        model, data, 'object_prediction', output_path,
        system_prompt, user_prompt_template,
        get_images, save_input, parallel
    )


def evaluate_object_relative_position(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """评测物体相对位置任务（ego/exo 中标记物体相对观察者的方位）"""
    system_prompt, user_prompt_template = get_prompts('object_relative_position')

    def get_images(item: UnifiedDataItem):
        """获取带标注的图像列表：ego + exo，exo 上有标记点"""
        images = []
        for view_name, img_path in item.images.items():
            annotations = item.get_annotations_for_view(view_name)
            images.append(prepare_annotated_image(img_path, annotations))
        return images

    return evaluate_single_choice_task(
        model, data, 'object_relative_position', output_path,
        system_prompt, user_prompt_template,
        get_images, save_input, parallel
    )


def evaluate_relative_distance(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """评测相对距离任务（确定标记物体/人相对于ego相机佩戴者的相对距离）"""
    system_prompt, user_prompt_template = get_prompts('relative_distance')
    
    def get_images(item: UnifiedDataItem):
        """获取带标注的图像列表：ego + exo，exo上有标记点"""
        images = []
        for view_name, img_path in item.images.items():
            annotations = item.get_annotations_for_view(view_name)
            images.append(prepare_annotated_image(img_path, annotations))
        return images
    
    return evaluate_single_choice_task(
        model, data, 'relative_distance', output_path,
        system_prompt, user_prompt_template,
        get_images, save_input, parallel
    )


def _process_camera_relative_position_item(args):
    """处理单个 camera_relative_position 样本（用于并行处理）"""
    idx, item, model, system_prompt, user_prompt_template = args
    
    camera_to_exo_mapping, _, camera_list = build_camera_mapping(list(item.images.keys()))
    
    images = []
    if 'ego' in item.images:
        images.append(image_to_base64(item.images['ego']))
    for cam_name in camera_list:
        if cam_name in item.images:
            images.append(image_to_base64(item.images[cam_name]))
    
    question = map_camera_names_in_text(item.question, camera_to_exo_mapping)
    mapped_options = map_options_camera_names(item.options, camera_to_exo_mapping)
    options_str = format_options(mapped_options)
    user_prompt = user_prompt_template.format(question=question, options=options_str)
    
    api_response = model.call(images=images, question=user_prompt, system_prompt=system_prompt)
    
    valid_options = list(mapped_options.keys())
    answer_text = api_response['answer'] if api_response['success'] else ""
    extracted_answer = extract_answer(answer_text, valid_options, allow_multiple=False)
    
    ground_truth = item.answer
    is_correct = extracted_answer == ground_truth
    
    result = {
        'id': item.id,
        'question': question,
        'ground_truth': ground_truth,
        'raw_response': api_response['raw_response'],
        'model_response': answer_text,
        'extracted_answer': extracted_answer,
        'is_correct': is_correct,
        'usage': api_response.get('usage'),
        'error': api_response.get('error'),
        'camera_mapping': camera_to_exo_mapping,
        'camera_order': camera_list,
    }
    
    return idx, result, is_correct


def evaluate_camera_relative_position(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """
    评测相机相对位置任务
    支持并行处理
    
    映射逻辑：cam01 -> exo01, cam02 -> exo02, ...
    问题和选项中的相机名都会被映射
    """
    system_prompt, user_prompt_template = get_prompts('camera_relative_position')
    total = len(data)
    
    # 保存输入示例
    if save_input and data:
        item = data[0]
        camera_to_exo_mapping, _, camera_list = build_camera_mapping(list(item.images.keys()))
        images = []
        if 'ego' in item.images:
            images.append(image_to_base64(item.images['ego']))
        for cam_name in camera_list:
            if cam_name in item.images:
                images.append(image_to_base64(item.images[cam_name]))
        question = map_camera_names_in_text(item.question, camera_to_exo_mapping)
        mapped_options = map_options_camera_names(item.options, camera_to_exo_mapping)
        options_str = format_options(mapped_options)
        user_prompt = user_prompt_template.format(question=question, options=options_str)
        output_dir = os.path.dirname(output_path)
        save_model_input('camera_relative_position', item, images, system_prompt, user_prompt, output_dir)
    
    # 并行处理
    if parallel and parallel > 1:
        print(f"\n{'='*60}")
        print(f"评测任务: camera_relative_position (并行模式, {parallel} 线程)")
        print(f"模型: {model.model_name} ({model.provider})")
        print(f"样本数量: {total}")
        print(f"{'='*60}\n")
        
        results = [None] * total
        correct = 0
        
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {}
            for idx, item in enumerate(data):
                future = executor.submit(
                    _process_camera_relative_position_item,
                    (idx, item, model, system_prompt, user_prompt_template)
                )
                futures[future] = idx
            
            completed = 0
            for future in as_completed(futures):
                try:
                    idx, result, is_correct = future.result()
                    results[idx] = result
                    if is_correct:
                        correct += 1
                    completed += 1
                    print(f"[{completed}/{total}] ID: {result['id']} | GT: {result['ground_truth']} | Pred: {result['extracted_answer']} | {'✓' if is_correct else '✗'}", flush=True)
                except Exception as e:
                    idx = futures[future]
                    print(f"[{completed+1}/{total}] 处理出错 (idx={idx}): {e}", flush=True)
                    results[idx] = {
                        'id': data[idx].id,
                        'error': str(e),
                        'is_correct': False,
                        'ground_truth': data[idx].answer,
                        'extracted_answer': '',
                    }
                    completed += 1
    else:
        # 串行处理
        print(f"\n{'='*60}")
        print(f"评测任务: camera_relative_position")
        print(f"模型: {model.model_name} ({model.provider})")
        print(f"样本数量: {total}")
        print(f"{'='*60}\n")
        
        results = []
        correct = 0
        
        for idx, item in enumerate(data):
            print(f"[{idx+1}/{total}] Processing: {item.id}")
            _, result, is_correct = _process_camera_relative_position_item(
                (idx, item, model, system_prompt, user_prompt_template)
            )
            results.append(result)
            if is_correct:
                correct += 1
            print(f"  GT: {result['ground_truth']} | Pred: {result['extracted_answer']} | {'✓' if is_correct else '✗'}")
            time.sleep(REQUEST_DELAY)
    
    accuracy = correct / total if total > 0 else 0
    
    summary = {
        'task': 'camera_relative_position',
        'model': model.model_name,
        'provider': model.provider,
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'parallel': parallel if parallel and parallel > 1 else None,
        'timestamp': datetime.now().isoformat()
    }
    
    output = {'summary': summary, 'results': results}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"评测完成!")
    print(f"准确率: {accuracy:.2%} ({correct}/{total})")
    print(f"结果已保存至: {output_path}")
    print(f"{'='*60}\n")
    
    return summary


def evaluate_noise_collaboration(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """
    评测噪声协作任务（不定项选择：题目中“选最好的”=单选，“选所有的”=多选）
    
    映射逻辑说明:
    ============================================
    原始数据示例:
    - 图像顺序: ego (带噪声), cam01, cam02, cam04
    - 问题: "Please select the clearest exocentric (exo) viewpoint..."
    - 选项: A: "cam01", B: "cam02", C: "cam04", D: "None of them..."
    - 正确答案: B
    
    映射后:
    - 图像顺序: ego (带噪声), cam01, cam02, cam04 (保持不变，使用带噪声的图像)
    - 映射关系（保留原始编号）:
      * cam01 → exo01
      * cam02 → exo02
      * cam04 → exo04
    - 问题更新: 保持不变（问题中不包含相机名）
    - 选项更新:
      * A: "exo01" (对应原来的cam01)
      * B: "exo02" (对应原来的cam02)
      * C: "exo04" (对应原来的cam04)
      * D: "None of them..." (保持不变)
    - 正确答案: B (保持不变)
    
    关键点:
    1. 使用带噪声的图像（image_paths），而不是原始图像（image_paths_original）
    2. 图像顺序保持不变，只进行名称映射
    3. 选项中的相机名都被替换为对应的exoXX名称
    4. 答案选项（A/B/C/D）保持不变，确保评测的正确性
    5. 处理相机编号缺失的情况（如只有cam01, cam02, cam04，缺少cam03）
    ============================================
    
    Args:
        model: 模型实例
        data: 统一格式的数据项列表
        output_path: 输出文件路径
        save_input: 是否保存输入示例
        parallel: 并行数（当前不支持，保留用于接口一致性）
    """
    system_prompt, user_prompt_template = get_prompts('noise_collaboration')
    
    results = []
    correct = 0
    total = len(data)
    skipped = 0  # 因图像文件缺失而跳过的样本数
    
    print(f"\n{'='*60}")
    print(f"评测任务: noise_collaboration（不定项：按题目选单/多选）")
    print(f"模型: {model.model_name} ({model.provider})")
    print(f"样本数量: {total}")
    print(f"{'='*60}\n")
    
    # 保存第一个样本的输入内容
    input_saved = False
    
    for idx, item in enumerate(data):
        print(f"[{idx+1}/{total}] Processing: {item.id}")
        
        # 使用统一的映射函数（保留原始编号：cam01 -> exo01）
        camera_to_exo_mapping, exo_to_camera_mapping, camera_list = build_camera_mapping(list(item.images.keys()))
        
        if idx == 0:  # 第一个样本打印映射关系
            print(f"  [DEBUG] 相机映射关系:")
            for cam_name in camera_list:
                if cam_name in camera_to_exo_mapping:
                    print(f"    {cam_name} → {camera_to_exo_mapping[cam_name]}")
            # 打印噪声信息
            noise_notes = item.metadata.get('image_noise_notes', {})
            if noise_notes:
                print(f"  [DEBUG] 噪声信息:")
                for view_name, noise_note in noise_notes.items():
                    print(f"    {view_name}: {noise_note}")
        
        # 准备图像列表：ego (带噪声) + exo相机（按原始顺序）
        # 若标注中某张噪声图在磁盘上不存在则跳过该样本，避免评测中断
        images = []
        try:
            if 'ego' in item.images:
                images.append(image_to_base64(item.images['ego']))
            for cam_name in camera_list:
                if cam_name in item.images:
                    images.append(image_to_base64(item.images[cam_name]))
        except ValueError as e:
            if 'Image file not found' in str(e):
                print(f"  跳过: 图像文件缺失 ({e})")
                skipped += 1
                continue
            raise
        
        # 问题保持不变（问题中不包含相机名）
        question = item.question
        
        # 使用统一的选项映射函数
        mapped_options = map_options_camera_names(item.options, camera_to_exo_mapping)
        
        options_str = format_options(mapped_options)
        user_prompt = user_prompt_template.format(
            question=question,
            options=options_str
        )
        
        # 保存第一个样本的输入内容
        if save_input and not input_saved and idx == 0:
            output_dir = os.path.dirname(output_path)
            save_model_input('noise_collaboration', item, images, system_prompt, user_prompt, output_dir)
            input_saved = True
        
        # 调用模型
        api_response = model.call(
            images=images,
            question=user_prompt,
            system_prompt=system_prompt
        )
        
        # 提取答案（不定项：题目“选最好的”=单选，“选所有的”=多选，按 GT 是否含逗号判断）
        valid_options = list(mapped_options.keys())
        answer_text = api_response['answer'] if api_response['success'] else ""
        ground_truth = item.answer
        is_multiple = ',' in str(ground_truth).strip()
        extracted_answer = extract_answer(answer_text, valid_options, allow_multiple=is_multiple)
        
        # 判断正确性
        if is_multiple:
            ground_truth_set = set(a.strip() for a in str(ground_truth).split(','))
            extracted_set = set(a.strip() for a in str(extracted_answer).split(',')) if extracted_answer else set()
            is_correct = ground_truth_set == extracted_set
        else:
            is_correct = extracted_answer == ground_truth
        if is_correct:
            correct += 1
        
        # 记录结果（包含映射信息用于调试）
        result = {
            'id': item.id,
            'question': question,
            'ground_truth': ground_truth,
            'raw_response': api_response['raw_response'],
            'model_response': answer_text,
            'extracted_answer': extracted_answer,
            'is_correct': is_correct,
            'usage': api_response.get('usage'),
            'error': api_response.get('error'),
            'camera_mapping': camera_to_exo_mapping,  # 保存映射关系用于调试
            'camera_order': camera_list,  # 保存相机顺序
            'noise_notes': item.metadata.get('image_noise_notes', {}),  # 保存噪声信息
        }
        results.append(result)
        
        print(f"  GT: {ground_truth} | Pred: {extracted_answer} | {'✓' if is_correct else '✗'}")
        
        time.sleep(REQUEST_DELAY)
    
    # 计算指标（仅对实际参与评测的样本计准确率）
    evaluated = total - skipped
    accuracy = correct / evaluated if evaluated > 0 else 0
    
    summary = {
        'task': 'noise_collaboration',
        'model': model.model_name,
        'provider': model.provider,
        'total': total,
        'skipped': skipped,
        'evaluated': evaluated,
        'correct': correct,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }
    
    output = {'summary': summary, 'results': results}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"评测完成!")
    if skipped > 0:
        print(f"跳过样本（图像缺失）: {skipped}")
    print(f"准确率: {accuracy:.2%} ({correct}/{evaluated})")
    print(f"结果已保存至: {output_path}")
    print(f"{'='*60}\n")
    
    return summary


def evaluate_ego_2_exo_visibility(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """评测Ego-2-Exo可见性任务（单选题）"""
    system_prompt, user_prompt_template = get_prompts('ego_2_exo_visibility')
    
    def get_images(item: UnifiedDataItem):
        """获取图像列表：ego + exo"""
        images = []
        if 'ego' in item.images:
            images.append(image_to_base64(item.images['ego']))
        for view_name, img_path in item.images.items():
            if view_name != 'ego':
                images.append(image_to_base64(img_path))
        return images
    
    return evaluate_single_choice_task(
        model, data, 'ego_2_exo_visibility', output_path,
        system_prompt, user_prompt_template,
        get_images, save_input, parallel
    )


def evaluate_view_movement(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """评测视角移动任务（支持多选题）"""
    system_prompt, user_prompt_template = get_prompts('view_movement')
    
    def get_images(item: UnifiedDataItem):
        """获取图像列表：ego + exo"""
        images = []
        if 'ego' in item.images:
            images.append(image_to_base64(item.images['ego']))
        for view_name, img_path in item.images.items():
            if view_name != 'ego':
                images.append(image_to_base64(img_path))
        return images
    
    return evaluate_single_choice_task(
        model, data, 'view_movement', output_path,
        system_prompt, user_prompt_template,
        get_images, save_input, parallel
    )


def evaluate_object_movement(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """
    评测物体移动任务（单选题：物体移动后的时钟方向）
    
    Object_Movement.json - 选项值是位置描述（如 "12 o'clock"），不需要相机名映射
    支持并行处理
    """
    system_prompt, user_prompt_template = get_prompts('object_movement')
    
    def get_images(item: UnifiedDataItem):
        images = []
        if 'ego' in item.images:
            images.append(image_to_base64(item.images['ego']))
        for view_name, img_path in item.images.items():
            if view_name != 'ego':
                images.append(image_to_base64(img_path))
        return images
    
    return evaluate_single_choice_task(
        model, data, 'object_movement', output_path,
        system_prompt, user_prompt_template,
        get_images, save_input, parallel
    )


def evaluate_random_baseline(
    data: List[UnifiedDataItem],
    task: str,
    output_path: str,
    allow_multiple: bool = False,
) -> Dict:
    """
    随机基准评测函数（不调用模型，仅用于计算随机猜测的正确率）
    
    Args:
        data: 统一格式的数据项列表
        task: 任务名称
        output_path: 输出文件路径
        allow_multiple: 是否允许多选题
    
    Returns:
        评测结果摘要
    """
    import random
    
    results = []
    correct = 0
    total = len(data)
    
    print(f"\n{'='*60}")
    print(f"随机基准评测任务: {task}")
    print(f"模型: Random Baseline (随机选择)")
    print(f"样本数量: {total}")
    print(f"题型: {'多选题' if allow_multiple else '单选题'}")
    print(f"{'='*60}\n")
    
    # 设置随机种子以确保可复现性（可选）
    random.seed(42)
    
    for idx, item in enumerate(data):
        print(f"[{idx+1}/{total}] Processing: {item.id}")
        
        options = item.options
        ground_truth = item.answer
        valid_options = list(options.keys())
        
        # 随机选择答案
        if allow_multiple:
            # 多选题：随机选择0到所有选项数量的子集
            # 首先确定选择多少个选项（随机选择1到所有选项数量）
            num_selections = random.randint(1, len(valid_options))
            # 随机选择选项
            selected_options = random.sample(valid_options, num_selections)
            random_answer = ','.join(sorted(selected_options))
        else:
            # 单选题：随机选择一个选项
            random_answer = random.choice(valid_options)
        
        # 判断正确性
        if allow_multiple:
            # 多选题：比较答案集合
            ground_truth_set = set([a.strip() for a in str(ground_truth).split(',')])
            random_set = set([a.strip() for a in str(random_answer).split(',')])
            is_correct = ground_truth_set == random_set
        else:
            # 单选题：直接比较
            is_correct = random_answer == ground_truth
        
        if is_correct:
            correct += 1
        
        # 记录结果
        result = {
            'id': item.id,
            'question': item.question,
            'ground_truth': ground_truth,
            'random_answer': random_answer,
            'is_correct': is_correct,
        }
        results.append(result)
        
        print(f"  GT: {ground_truth} | Random: {random_answer} | {'✓' if is_correct else '✗'}")
    
    # 计算指标
    accuracy = correct / total if total > 0 else 0
    
    summary = {
        'task': task,
        'model': 'random_baseline',
        'provider': 'random',
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat(),
        'note': '随机基准评测：不调用任何模型，仅用于计算随机猜测的正确率'
    }
    
    # 保存结果
    output = {'summary': summary, 'results': results}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"随机基准评测完成!")
    print(f"准确率: {accuracy:.2%} ({correct}/{total})")
    print(f"结果已保存至: {output_path}")
    print(f"{'='*60}\n")
    
    return summary


# ==================== View_Movement 系列评测函数 ====================

def evaluate_view_movement_1(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """评测 View_Movement_1 任务（多选题：移动后哪些物体不可见）"""
    system_prompt, user_prompt_template = get_prompts('view_movement_1')
    
    def get_images(item: UnifiedDataItem):
        images = []
        if 'ego' in item.images:
            images.append(image_to_base64(item.images['ego']))
        for view_name, img_path in item.images.items():
            if view_name != 'ego':
                images.append(image_to_base64(img_path))
        return images
    
    return evaluate_single_choice_task(
        model, data, 'view_movement_1', output_path,
        system_prompt, user_prompt_template,
        get_images, save_input, parallel
    )


def evaluate_view_movement_2(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """评测 View_Movement_2 任务（多选题：移动后哪些物体不可见）"""
    system_prompt, user_prompt_template = get_prompts('view_movement_2')
    
    def get_images(item: UnifiedDataItem):
        images = []
        if 'ego' in item.images:
            images.append(image_to_base64(item.images['ego']))
        for view_name, img_path in item.images.items():
            if view_name != 'ego':
                images.append(image_to_base64(img_path))
        return images
    
    return evaluate_single_choice_task(
        model, data, 'view_movement_2', output_path,
        system_prompt, user_prompt_template,
        get_images, save_input, parallel
    )


def evaluate_view_movement_3(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """评测 View_Movement_3 任务（单选题：移动后物体的时钟方向）"""
    system_prompt, user_prompt_template = get_prompts('view_movement_3')
    
    def get_images(item: UnifiedDataItem):
        images = []
        if 'ego' in item.images:
            images.append(image_to_base64(item.images['ego']))
        for view_name, img_path in item.images.items():
            if view_name != 'ego':
                images.append(image_to_base64(img_path))
        return images
    
    return evaluate_single_choice_task(
        model, data, 'view_movement_3', output_path,
        system_prompt, user_prompt_template,
        get_images, save_input, parallel
    )


def evaluate_view_movement_4(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """评测 View_Movement_4 任务（单选题：移动后能看到多少物体）"""
    system_prompt, user_prompt_template = get_prompts('view_movement_4')
    
    def get_images(item: UnifiedDataItem):
        images = []
        if 'ego' in item.images:
            images.append(image_to_base64(item.images['ego']))
        for view_name, img_path in item.images.items():
            if view_name != 'ego':
                images.append(image_to_base64(img_path))
        return images
    
    return evaluate_single_choice_task(
        model, data, 'view_movement_4', output_path,
        system_prompt, user_prompt_template,
        get_images, save_input, parallel
    )


def evaluate_view_movement_5(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """评测 View_Movement_5 任务（单选题：移动后 exo 相机的时钟方向）"""
    system_prompt, user_prompt_template = get_prompts('view_movement_5')
    
    def get_images(item: UnifiedDataItem):
        images = []
        if 'ego' in item.images:
            images.append(image_to_base64(item.images['ego']))
        for view_name, img_path in item.images.items():
            if view_name != 'ego':
                images.append(image_to_base64(img_path))
        return images
    
    return evaluate_single_choice_task(
        model, data, 'view_movement_5', output_path,
        system_prompt, user_prompt_template,
        get_images, save_input, parallel
    )


def evaluate_view_movement_6(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """评测 View_Movement_6 任务（单选题：移动后最近的物体）"""
    system_prompt, user_prompt_template = get_prompts('view_movement_6')
    
    def get_images(item: UnifiedDataItem):
        images = []
        if 'ego' in item.images:
            images.append(image_to_base64(item.images['ego']))
        for view_name, img_path in item.images.items():
            if view_name != 'ego':
                images.append(image_to_base64(img_path))
        return images
    
    return evaluate_single_choice_task(
        model, data, 'view_movement_6', output_path,
        system_prompt, user_prompt_template,
        get_images, save_input, parallel
    )


# ==================== Object_Movement_1 评测函数 ====================

def _process_object_movement_1_item(args):
    """处理单个 object_movement_1 样本（用于并行处理）"""
    idx, item, model, system_prompt, user_prompt_template = args
    
    images = []
    if 'ego' in item.images:
        images.append(image_to_base64(item.images['ego']))
    
    for view_name, img_path in item.images.items():
        if view_name != 'ego':
            images.append(image_to_base64(img_path))
    
    first_option_value = list(item.options.values())[0] if item.options else ""
    needs_camera_mapping = first_option_value.startswith('cam')
    
    if needs_camera_mapping:
        camera_to_exo_mapping, _, _ = build_camera_mapping(list(item.images.keys()))
        mapped_options = map_options_camera_names(item.options, camera_to_exo_mapping)
    else:
        mapped_options = item.options
    
    question = item.question
    options_str = format_options(mapped_options)
    user_prompt = user_prompt_template.format(
        question=question,
        options=options_str
    )
    
    api_response = model.call(
        images=images,
        question=user_prompt,
        system_prompt=system_prompt
    )
    
    valid_options = list(mapped_options.keys())
    answer_text = api_response['answer'] if api_response['success'] else ""
    extracted_answer = extract_answer(answer_text, valid_options, allow_multiple=True)
    
    ground_truth = item.answer
    ground_truth_set = set([a.strip() for a in str(ground_truth).split(',')])
    extracted_set = set([a.strip() for a in str(extracted_answer).split(',')])
    is_correct = ground_truth_set == extracted_set
    
    result = {
        'id': item.id,
        'question': question,
        'ground_truth': ground_truth,
        'raw_response': api_response['raw_response'],
        'model_response': answer_text,
        'extracted_answer': extracted_answer,
        'is_correct': is_correct,
        'usage': api_response.get('usage'),
        'error': api_response.get('error')
    }
    
    return idx, result, is_correct


def evaluate_object_movement_1(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """
    评测 Object_Movement_1 任务（多选题：物体移动后哪些 exo 能看到）
    
    选项值是相机名（如 "cam01"），需要映射到 exoXX
    支持并行处理
    """
    system_prompt, user_prompt_template = get_prompts('object_movement_1')
    
    total = len(data)
    
    # 保存输入示例
    if save_input and data:
        item = data[0]
        images = []
        if 'ego' in item.images:
            images.append(image_to_base64(item.images['ego']))
        for view_name, img_path in item.images.items():
            if view_name != 'ego':
                images.append(image_to_base64(img_path))
        
        first_option_value = list(item.options.values())[0] if item.options else ""
        needs_camera_mapping = first_option_value.startswith('cam')
        if needs_camera_mapping:
            camera_to_exo_mapping, _, _ = build_camera_mapping(list(item.images.keys()))
            mapped_options = map_options_camera_names(item.options, camera_to_exo_mapping)
        else:
            mapped_options = item.options
        
        options_str = format_options(mapped_options)
        user_prompt = user_prompt_template.format(question=item.question, options=options_str)
        output_dir = os.path.dirname(output_path)
        save_model_input('object_movement_1', item, images, system_prompt, user_prompt, output_dir)
    
    # 并行处理
    if parallel and parallel > 1:
        print(f"\n{'='*60}")
        print(f"评测任务: object_movement_1 (多选题, 并行模式, {parallel} 线程)")
        print(f"模型: {model.model_name} ({model.provider})")
        print(f"样本数量: {total}")
        print(f"{'='*60}\n")
        
        results = [None] * total
        correct = 0
        
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {}
            for idx, item in enumerate(data):
                future = executor.submit(
                    _process_object_movement_1_item,
                    (idx, item, model, system_prompt, user_prompt_template)
                )
                futures[future] = idx
            
            completed = 0
            for future in as_completed(futures):
                try:
                    idx, result, is_correct = future.result()
                    results[idx] = result
                    if is_correct:
                        correct += 1
                    completed += 1
                    print(f"[{completed}/{total}] ID: {result['id']} | GT: {result['ground_truth']} | Pred: {result['extracted_answer']} | {'✓' if is_correct else '✗'}", flush=True)
                except Exception as e:
                    idx = futures[future]
                    print(f"[{completed+1}/{total}] 处理出错 (idx={idx}): {e}", flush=True)
                    results[idx] = {
                        'id': data[idx].id,
                        'error': str(e),
                        'is_correct': False,
                        'ground_truth': data[idx].answer,
                        'extracted_answer': '',
                    }
                    completed += 1
    else:
        # 串行处理
        print(f"\n{'='*60}")
        print(f"评测任务: object_movement_1 (多选题)")
        print(f"模型: {model.model_name} ({model.provider})")
        print(f"样本数量: {total}")
        print(f"{'='*60}\n")
        
        results = []
        correct = 0
        
        for idx, item in enumerate(data):
            print(f"[{idx+1}/{total}] Processing: {item.id}")
            
            _, result, is_correct = _process_object_movement_1_item(
                (idx, item, model, system_prompt, user_prompt_template)
            )
            
            results.append(result)
            if is_correct:
                correct += 1
            
            print(f"  GT: {result['ground_truth']} | Pred: {result['extracted_answer']} | {'✓' if is_correct else '✗'}")
            time.sleep(REQUEST_DELAY)
    
    accuracy = correct / total if total > 0 else 0
    
    summary = {
        'task': 'object_movement_1',
        'model': model.model_name,
        'provider': model.provider,
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'parallel': parallel if parallel and parallel > 1 else None,
        'timestamp': datetime.now().isoformat()
    }
    
    output = {'summary': summary, 'results': results}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"评测完成!")
    print(f"准确率: {accuracy:.2%} ({correct}/{total})")
    print(f"结果已保存至: {output_path}")
    print(f"{'='*60}\n")
    
    return summary


def evaluate_view_selection(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """
    评测 View_Selection 任务（单选题：结合 ego 视角判断哪些 exo 有帮助）
    支持并行处理
    """
    system_prompt, user_prompt_template = get_prompts('view_selection')
    
    def get_images(item: UnifiedDataItem):
        """获取图像列表：ego + 所有 exo 视角"""
        images = []
        # 先添加 ego
        if 'ego' in item.images:
            images.append(image_to_base64(item.images['ego']))
        # 再添加其他视角（按顺序）
        for view_name, img_path in item.images.items():
            if view_name != 'ego':
                images.append(image_to_base64(img_path))
        return images
    
    return evaluate_single_choice_task(
        model, data, 'view_selection', output_path,
        system_prompt, user_prompt_template,
        get_images, save_input, parallel
    )


def evaluate_view_selection_2(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """
    评测 View_Selection_2 任务（单选题：判断是否需要多视角协作）
    支持并行处理
    """
    system_prompt, user_prompt_template = get_prompts('view_selection_2')
    
    def get_images(item: UnifiedDataItem):
        """获取图像列表：ego + 所有 exo 视角"""
        images = []
        # 先添加 ego
        if 'ego' in item.images:
            images.append(image_to_base64(item.images['ego']))
        # 再添加其他视角（按顺序）
        for view_name, img_path in item.images.items():
            if view_name != 'ego':
                images.append(image_to_base64(img_path))
        return images
    
    return evaluate_single_choice_task(
        model, data, 'view_selection_2', output_path,
        system_prompt, user_prompt_template,
        get_images, save_input, parallel
    )


def evaluate_view_selection_3(model, data: List[UnifiedDataItem], output_path: str, save_input: bool = True, parallel: int = None) -> Dict:
    """
    评测 View_Selection_3 任务（单选题：选择最有帮助的单个exo视角）
    支持并行处理
    """
    system_prompt, user_prompt_template = get_prompts('view_selection_3')
    
    def get_images(item: UnifiedDataItem):
        """获取图像列表：ego + 所有 exo 视角"""
        images = []
        if 'ego' in item.images:
            images.append(image_to_base64(item.images['ego']))
        for view_name, img_path in item.images.items():
            if view_name != 'ego':
                images.append(image_to_base64(img_path))
        return images
    
    return evaluate_single_choice_task(
        model, data, 'view_selection_3', output_path,
        system_prompt, user_prompt_template,
        get_images, save_input, parallel
    )


# 任务评测函数映射
TASK_EVALUATORS = {
    # 基础任务
    'camera_wearer': evaluate_camera_wearer,
    'camera_wearer_type2': evaluate_camera_wearer_type2,
    'ego_2_exo_visibility': evaluate_ego_2_exo_visibility,
    'camera_relative_position': evaluate_camera_relative_position,
    'relative_distance': evaluate_relative_distance,
    'object_relative_position': evaluate_object_relative_position,
    'object_correspondence': evaluate_object_correspondence,
    'object_prediction': evaluate_object_prediction,
    
    # View_Movement 系列
    'view_movement_1': evaluate_view_movement_1,
    'view_movement_2': evaluate_view_movement_2,
    'view_movement_3': evaluate_view_movement_3,
    'view_movement_4': evaluate_view_movement_4,
    'view_movement_5': evaluate_view_movement_5,
    'view_movement_6': evaluate_view_movement_6,
    
    # Object_Movement 系列
    'object_movement': evaluate_object_movement,
    'object_movement_1': evaluate_object_movement_1,
    
    # View_Selection 任务
    'view_selection': evaluate_view_selection,
    'view_selection_2': evaluate_view_selection_2,
    'view_selection_3': evaluate_view_selection_3,
    # 噪声协作（不定项）
    'noise_collaboration': evaluate_noise_collaboration,
}
