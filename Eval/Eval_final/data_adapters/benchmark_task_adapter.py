"""
Benchmark Task 数据适配器
处理 Benchmark_annotations 目录下的 TaskName.json 文件
"""
import json
import os
from typing import List, Dict, Any
from .base import BaseAdapter, UnifiedDataItem
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import map_path


class BenchmarkTaskAdapter(BaseAdapter):
    """Benchmark Task 数据适配器，支持从 TaskName.json 文件加载数据"""
    
    # 任务ID到基础任务名称的映射（用于数据解析）
    TASK_ID_TO_BASE_NAME = {
        1: 'camera_wearer',
        2: 'ego_2_exo_visibility',
        3: 'view_movement',  # view_movement_1~6 都使用 task3 字段
        4: 'object_movement',  # object_movement 和 object_movement_1 都使用 task4 字段
        5: 'camera_relative_position',
        6: 'relative_distance',
        7: 'camera_wearer_type2',  # ego中标记相机，选择对应exo
        10: 'object_relative_position',
        11: 'object_correspondence',
        12: 'object_prediction',
        13: 'view_selection',  # 结合ego判断哪些exo有帮助
        15: 'view_selection_2',  # 判断是否需要协作
        16: 'view_selection_3',  # 选择最有帮助的exo视角
        9: 'noise_collaboration',  # 噪声协作（不定项）
    }
    
    # 任务名称到基础名称的映射（用于确定如何解析数据）
    TASK_TO_BASE_NAME = {
        'camera_wearer': 'camera_wearer',
        'camera_wearer_type2': 'camera_wearer_type2',
        'ego_2_exo_visibility': 'ego_2_exo_visibility',
        'camera_relative_position': 'camera_relative_position',
        'relative_distance': 'relative_distance',
        'object_relative_position': 'object_relative_position',
        'object_correspondence': 'object_correspondence',
        'object_prediction': 'object_prediction',
        # View_Movement 系列
        'view_movement_1': 'view_movement',
        'view_movement_2': 'view_movement',
        'view_movement_3': 'view_movement',
        'view_movement_4': 'view_movement',
        'view_movement_5': 'view_movement',
        'view_movement_6': 'view_movement',
        # Object_Movement 系列
        'object_movement': 'object_movement',
        'object_movement_1': 'object_movement',
        # View_Selection 任务
        'view_selection': 'view_selection',
        'view_selection_2': 'view_selection_2',
        'view_selection_3': 'view_selection_3',
        'noise_collaboration': 'noise_collaboration',
    }
    
    def load_raw_data(self) -> List[Dict]:
        """加载原始数据，返回数据项列表"""
        if os.path.exists(self.file_path):
            actual_path = self.file_path
        else:
            actual_path = map_path(self.file_path)
            if not os.path.exists(actual_path):
                raise FileNotFoundError(f"Data file not found: {self.file_path} (mapped to: {actual_path})")
        
        with open(actual_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError(f"Expected list, got {type(data)}")
        
        return data
    
    def convert_to_unified(self, raw_item: Dict, task_name: str) -> UnifiedDataItem:
        """转换为统一格式"""
        # 获取基础任务名称（用于确定如何解析数据）
        if task_name in self.TASK_TO_BASE_NAME:
            base_task_name = self.TASK_TO_BASE_NAME[task_name]
        else:
            base_task_name = task_name
        
        # 验证 task_id（如果存在）
        task_id = raw_item.get('task_id')
        if task_id is not None and task_id in self.TASK_ID_TO_BASE_NAME:
            expected_base = self.TASK_ID_TO_BASE_NAME[task_id]
            if base_task_name != expected_base:
                raise ValueError(f"Task base name mismatch: expected {expected_base}, got {base_task_name}")
        
        # Resolve image paths against the released dataset layout.
        _project_root = os.environ.get(
            "SEERBENCH_ROOT",
            os.path.dirname(os.path.dirname(os.path.abspath(self.file_path))),
        )
        data_root = os.environ.get(
            "SEERBENCH_IMAGE_ROOT",
            os.path.join(_project_root, "data", "egoexo4d_val"),
        )

        def _resolve_image_path(img_path: str) -> str:
            if not img_path:
                return img_path
            if os.path.isabs(img_path):
                mapped = map_path(img_path)
                if os.path.exists(mapped):
                    return mapped
                legacy_prefix = "/public/lfy/SU_benchmark/"
                if img_path.startswith(legacy_prefix):
                    return os.path.join(_project_root, img_path[len(legacy_prefix):])
                return img_path
            if img_path.startswith(("Task-1/", "data/")):
                return os.path.join(_project_root, img_path)
            return os.path.join(data_root, img_path)

        images = {}
        if 'image_paths' in raw_item:
            image_paths = raw_item.get('image_paths', {})
            for view_name, img_path in image_paths.items():
                images[view_name] = _resolve_image_path(img_path)
        else:
            ego_path = raw_item.get('ego_path')
            exo_paths = raw_item.get('exo_paths', [])
            
            if ego_path:
                images['ego'] = _resolve_image_path(ego_path)
            
            for exo_path in exo_paths:
                if exo_path:
                    parts = exo_path.split('/')
                    cam_name = None
                    for part in parts:
                        if part.startswith('cam'):
                            cam_name = part
                            break
                    
                    if cam_name:
                        images[cam_name] = _resolve_image_path(exo_path)
        
        annotations = raw_item.get('annotations', {})
        
        question = ""
        options = {}
        answer = ""
        
        # 根据基础任务类型提取数据
        if base_task_name in ['camera_wearer', 'camera_wearer_type2', 'camera_relative_position', 'relative_distance', 
                              'object_relative_position', 'object_correspondence', 'object_prediction', 'view_selection', 'view_selection_2', 'view_selection_3', 'noise_collaboration']:
            qa_pair = raw_item.get('qa_pair', {})
            question = qa_pair.get('question', '')
            options = qa_pair.get('options', {})
            answer = qa_pair.get('answer', '')
            if not question:
                question = raw_item.get('question', '')
            if not options:
                options = raw_item.get('options', {})
            if not answer:
                answer = raw_item.get('answer', '')
        elif base_task_name == 'ego_2_exo_visibility':
            task2 = raw_item.get('task2', {})
            if task2:
                question = task2.get('question', '')
                answer = task2.get('answer', '')
            else:
                qa_pair = raw_item.get('qa_pair', {})
                question = qa_pair.get('question', '') or raw_item.get('question', '')
                answer = qa_pair.get('answer', '') or raw_item.get('answer', '')
            options = raw_item.get('options', {})
        elif base_task_name == 'view_movement':
            task3 = raw_item.get('task3', {})
            if task3:
                question = task3.get('question', '')
                answer = task3.get('answer', '')
            else:
                question = raw_item.get('question', '')
                answer = raw_item.get('answer', '')
            options = raw_item.get('options', {})
            if isinstance(answer, list):
                answer = ','.join(sorted(answer))
        elif base_task_name == 'object_movement':
            task4 = raw_item.get('task4', {})
            if task4:
                question = task4.get('question', '')
                answer_list = task4.get('answer', [])
            else:
                question = raw_item.get('question', '')
                answer_list = raw_item.get('answer', [])
            options = raw_item.get('options', {})
            if isinstance(answer_list, list):
                answer = ','.join(sorted(answer_list))
            else:
                answer = str(answer_list)
        
        item_id = raw_item.get('id', '')
        if not item_id:
            scene_id = raw_item.get('scene_id', '')
            ego_view = raw_item.get('ego_view', '')
            frame = raw_item.get('frame', '')
            if scene_id and ego_view and frame is not None:
                item_id = f"{scene_id}_{ego_view}_{frame}_{task_name}"
            else:
                item_id = f"item_{len(images)}"
        
        # 处理列表格式的答案（多选题）
        if isinstance(answer, list):
            answer = ','.join(answer)
        
        metadata = {
            'scene_id': raw_item.get('scene_id'),
            'ego_view': raw_item.get('ego_view'),
            'frame': raw_item.get('frame'),
            'task_id': task_id,
            'image_noise_notes': raw_item.get('image_noise_notes', {}),
        }
        
        return UnifiedDataItem(
            id=item_id,
            task=task_name,
            question=question,
            options=options,
            answer=answer,
            images=images,
            annotations=annotations,
            metadata=metadata
        )
    
    def get_supported_tasks(self) -> List[str]:
        """返回支持的任务列表"""
        return list(self.TASK_TO_BASE_NAME.keys())
