"""
统一数据加载器
支持配置驱动的数据源管理和统一数据格式转换
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from typing import List, Dict, Optional, Type
from data_adapters import (
    BaseAdapter,
    UnifiedDataItem,
    BenchmarkTaskAdapter,
)
from config import map_path, get_data_dir_benchmark, TASK_TO_FILE


class DataLoader:
    """统一数据加载器"""
    
    # 适配器类映射
    ADAPTER_CLASSES = {
        'BenchmarkTaskAdapter': BenchmarkTaskAdapter,
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化数据加载器
        
        Args:
            config_path: 数据源配置文件路径，如果为None则使用默认路径
        """
        if config_path is None:
            # 使用相对于当前文件的路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, 'data_sources.yaml')
        
        # 先检查原路径是否存在，如果不存在再尝试映射路径
        if os.path.exists(config_path):
            self.config_path = config_path
        else:
            self.config_path = map_path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _validate_config(self):
        """验证配置文件格式"""
        if 'data_sources' not in self.config:
            raise ValueError("Config file must contain 'data_sources' section")
        if 'task_to_source' not in self.config:
            raise ValueError("Config file must contain 'task_to_source' section")
    
    def _get_adapter_class(self, adapter_name: str) -> Type[BaseAdapter]:
        """获取适配器类"""
        if adapter_name not in self.ADAPTER_CLASSES:
            raise ValueError(f"Unknown adapter: {adapter_name}")
        return self.ADAPTER_CLASSES[adapter_name]
    
    def _get_data_source_config(self, source_name: str) -> Dict:
        """获取数据源配置"""
        data_sources = self.config.get('data_sources', {})
        if source_name not in data_sources:
            raise ValueError(f"Data source '{source_name}' not found in config")
        return data_sources[source_name]
    
    def load_unified_data(self, task_name: str, data_source: Optional[str] = None) -> List[UnifiedDataItem]:
        """
        加载统一格式的数据
        
        Args:
            task_name: 任务名称
            data_source: 数据源名称（可选，如果不指定则从配置中查找）
        
        Returns:
            统一格式的数据项列表
        """
        # 如果没有指定数据源，从配置中查找
        if data_source is None:
            task_to_source = self.config.get('task_to_source', {})
            if task_name not in task_to_source:
                raise ValueError(f"Task '{task_name}' not found in task_to_source mapping")
            data_source = task_to_source[task_name]
        
        # 获取数据源配置
        source_config = self._get_data_source_config(data_source)
        
        # 验证任务是否在数据源支持的任务列表中
        supported_tasks = source_config.get('tasks', [])
        if task_name not in supported_tasks:
            raise ValueError(f"Task '{task_name}' not supported by data source '{data_source}'")
        
        # 获取基础目录路径和适配器名称
        base_path = source_config.get('file_path')
        adapter_name = source_config.get('adapter')
        
        if not base_path:
            raise ValueError(f"Data source '{data_source}' missing 'file_path'")
        if not adapter_name:
            raise ValueError(f"Data source '{data_source}' missing 'adapter'")
        if not os.path.isabs(base_path):
            base_path = os.path.abspath(os.path.join(os.path.dirname(self.config_path), base_path))
        
        # 获取任务对应的单个文件
        if task_name not in TASK_TO_FILE:
            raise ValueError(f"Unknown task: {task_name}")
        
        file_name = TASK_TO_FILE[task_name]
        file_path = os.path.join(base_path, file_name)
        
        # 检查文件是否存在
        actual_path = file_path
        if not os.path.exists(actual_path):
            actual_path = map_path(file_path)
        
        if not os.path.exists(actual_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        # 创建适配器实例并加载数据
        adapter_class = self._get_adapter_class(adapter_name)
        adapter = adapter_class(file_path)
        unified_items = adapter.load_and_convert(task_name)
        
        print(f"任务 {task_name} 加载 {file_name}: {len(unified_items)} 条数据")
        return unified_items
    
    def get_available_tasks(self) -> List[str]:
        """获取所有可用的任务列表"""
        return list(self.config.get('task_to_source', {}).keys())
    
    def get_data_source_info(self, source_name: str) -> Dict:
        """获取数据源信息"""
        source_config = self._get_data_source_config(source_name)
        return {
            'name': source_name,
            'file_path': source_config.get('file_path'),
            'adapter': source_config.get('adapter'),
            'tasks': source_config.get('tasks', []),
            'description': source_config.get('description', ''),
        }
    
    def list_data_sources(self) -> List[Dict]:
        """列出所有数据源"""
        data_sources = self.config.get('data_sources', {})
        return [
            self.get_data_source_info(name)
            for name in data_sources.keys()
        ]
