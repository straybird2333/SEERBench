"""
视觉语言模型的抽象基类（与 Eval_final 接口一致）
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseVLM(ABC):
    """视觉语言模型的抽象基类"""

    def __init__(self, model_name: str, **kwargs):
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
        调用模型进行推理。
        images: 图像列表（路径或 base64 data URL）
        question: 用户问题
        system_prompt: 系统提示词
        返回: { success, answer, raw_response, usage?, error? }
        """
        pass

    @property
    @abstractmethod
    def provider(self) -> str:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name}, provider={self.provider})"
