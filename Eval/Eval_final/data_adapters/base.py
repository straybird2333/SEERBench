"""
数据适配器基类和统一数据格式定义
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class UnifiedDataItem:
    """
    统一的数据格式，所有数据源转换后的标准格式
    """
    id: str
    task: str
    question: str
    options: Dict[str, str]
    answer: str
    images: Dict[str, str] = field(default_factory=dict)  # 图像路径字典，key为视图名称
    annotations: Dict[str, List[Dict]] = field(default_factory=dict)  # 标注信息，key为视图名称
    metadata: Dict[str, Any] = field(default_factory=dict)  # 保留原始元数据
    
    def get_image_list(self, ordered_keys: Optional[List[str]] = None) -> List[str]:
        """
        获取有序的图像路径列表
        
        Args:
            ordered_keys: 指定图像的顺序，如果为None则使用字典顺序
        
        Returns:
            图像路径列表
        """
        if ordered_keys:
            return [self.images[key] for key in ordered_keys if key in self.images]
        return list(self.images.values())
    
    def get_annotations_for_view(self, view_name: str) -> List[Dict]:
        """获取指定视图的标注"""
        return self.annotations.get(view_name, [])
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        验证数据项的完整性
        
        Returns:
            (is_valid, error_message)
        """
        if not self.id:
            return False, "Missing id"
        if not self.task:
            return False, "Missing task"
        if not self.question:
            return False, "Missing question"
        if not self.options:
            return False, "Missing options"
        if not self.answer:
            return False, "Missing answer"
        # 对于多选题，答案可能是列表或逗号分隔的多个选项
        if isinstance(self.answer, list):
            answer_parts = [a.strip() for a in self.answer]
        else:
            answer_parts = [a.strip() for a in str(self.answer).split(',')]
        valid_parts = [a for a in answer_parts if a in self.options]
        if not valid_parts:
            return False, f"Answer '{self.answer}' not in options"
        if not self.images:
            return False, "Missing images"
        return True, None


class BaseAdapter(ABC):
    """
    数据适配器基类
    所有数据源适配器都需要继承此类并实现相应方法
    """
    
    def __init__(self, file_path: str):
        """
        初始化适配器
        
        Args:
            file_path: 数据文件路径
        """
        self.file_path = file_path
        self._raw_data = None
    
    @abstractmethod
    def load_raw_data(self) -> Any:
        """
        加载原始数据文件
        
        Returns:
            原始数据（格式取决于具体数据源）
        """
        pass
    
    @abstractmethod
    def convert_to_unified(self, raw_item: Any, task_name: str) -> UnifiedDataItem:
        """
        将原始数据项转换为统一格式
        
        Args:
            raw_item: 原始数据项
            task_name: 任务名称
        
        Returns:
            统一格式的数据项
        """
        pass
    
    def validate_unified(self, item: UnifiedDataItem) -> Tuple[bool, Optional[str]]:
        """
        验证统一格式的数据项
        
        Args:
            item: 统一格式的数据项
        
        Returns:
            (is_valid, error_message)
        """
        return item.validate()
    
    def load_and_convert(self, task_name: str) -> List[UnifiedDataItem]:
        """
        加载原始数据并转换为统一格式
        
        Args:
            task_name: 任务名称
        
        Returns:
            统一格式的数据项列表
        """
        raw_data = self.load_raw_data()
        unified_items = []
        
        for raw_item in raw_data:
            try:
                unified_item = self.convert_to_unified(raw_item, task_name)
                is_valid, error_msg = self.validate_unified(unified_item)
                if not is_valid:
                    print(f"Warning: Invalid item {unified_item.id}: {error_msg}")
                    continue
                unified_items.append(unified_item)
            except Exception as e:
                print(f"Error converting item: {e}")
                continue
        
        return unified_items
    
    def get_supported_tasks(self) -> List[str]:
        """
        获取此适配器支持的任务列表
        
        Returns:
            任务名称列表
        """
        # 子类可以重写此方法
        return []
