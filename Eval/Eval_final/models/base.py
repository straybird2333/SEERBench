"""
视觉语言模型的抽象基类
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseVLM(ABC):
    """视觉语言模型的抽象基类"""
    
    def __init__(self, model_name: str, **kwargs):
        """
        初始化模型
        
        Args:
            model_name: 模型名称
            **kwargs: 其他配置参数
        """
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def call(
        self,
        images: List[str],
        question: str,
        system_prompt: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用模型进行推理
        
        Args:
            images: 图像列表（可以是路径或 base64 编码）
            question: 用户问题/提示
            system_prompt: 系统提示词
            **kwargs: 其他参数
        
        Returns:
            包含以下字段的字典:
            - success: bool, 是否成功
            - answer: str, 模型回答文本
            - raw_response: Any, 原始响应
            - usage: Dict, token 使用量（如果有）
            - error: str, 错误信息（如果有）
        """
        pass
    
    @property
    @abstractmethod
    def provider(self) -> str:
        """返回模型提供商名称"""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name}, provider={self.provider})"
