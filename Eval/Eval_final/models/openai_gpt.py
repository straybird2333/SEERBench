"""
OpenAI GPT-4V/GPT-4o 模型实现
"""
import os
import time
import base64
from typing import List, Dict, Any

from .base import BaseVLM

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class OpenAIVLM(BaseVLM):
    """OpenAI GPT-4V/GPT-4o 视觉语言模型"""
    
    # 可用的模型列表
    AVAILABLE_MODELS = [
        # GPT-4 系列
        'gpt-4o',
        'gpt-4o-mini',
        'gpt-4-turbo',
        'gpt-4-vision-preview',
        'gpt-4o-2024-05-13',
        'gpt-4o-2024-08-06',
        'gpt-4o-2024-11-20',
        'gpt-4o-mini-2024-07-18',
        # GPT-5 系列 (使用 max_completion_tokens)
        'gpt-5',
        'gpt-5.2-2025-12-11',
        # o1/o3 系列
        'o1',
        'o1-mini',
        'o1-preview',
        'o3-mini',
    ]
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: str = None,
        base_url: str = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        temperature: float = 0.1,
        max_tokens: int = 50,
        reasoning_effort: str = None,
        **kwargs
    ):
        """
        初始化 OpenAI GPT 模型
        
        Args:
            model_name: 模型名称
            api_key: API Key，如果为 None 则从环境变量读取
            base_url: API 基础 URL（用于代理或自定义端点）
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            reasoning_effort: 推理模式强度 ('none', 'minimal', 'low', 'medium', 'high', 'xhigh')
        """
        if not HAS_OPENAI:
            raise ImportError("openai is required. Install with: pip install openai")
        
        super().__init__(model_name, **kwargs)
        
        # 设置 API Key
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is required. Set it via environment variable or pass api_key parameter."
            )
        
        # 初始化客户端
        client_kwargs = {'api_key': self.api_key}
        if base_url:
            client_kwargs['base_url'] = base_url
        self.client = OpenAI(**client_kwargs)
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
    
    @property
    def provider(self) -> str:
        return "OpenAI"
    
    def _prepare_image_content(self, image: str) -> Dict:
        """
        准备图像内容格式
        
        Args:
            image: base64 编码的图像或 URL
        
        Returns:
            OpenAI 格式的图像内容
        """
        # 检查是否已经是 data URL 格式
        if image.startswith('data:image'):
            return {
                "type": "image_url",
                "image_url": {
                    "url": image,
                    "detail": "high"
                }
            }
        # 检查是否是 http URL
        elif image.startswith('http://') or image.startswith('https://'):
            return {
                "type": "image_url",
                "image_url": {
                    "url": image,
                    "detail": "high"
                }
            }
        # 检查是否是 file:// URL
        elif image.startswith('file://'):
            # 读取本地文件并转为 base64
            file_path = image[7:]  # 移除 'file://' 前缀
            return self._file_to_image_content(file_path)
        # 假设是本地文件路径
        elif os.path.exists(image):
            return self._file_to_image_content(image)
        else:
            # 假设是纯 base64 字符串
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}",
                    "detail": "high"
                }
            }
    
    def _file_to_image_content(self, file_path: str) -> Dict:
        """将本地文件转为图像内容"""
        import mimetypes
        mime_type = mimetypes.guess_type(file_path)[0] or 'image/jpeg'
        
        with open(file_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_data}",
                "detail": "high"
            }
        }
    
    def _is_new_model(self) -> bool:
        """检查是否是需要使用 max_completion_tokens 的新模型"""
        new_model_patterns = ['gpt-5', 'o1', 'o3', 'o4']
        return any(pattern in self.model_name.lower() for pattern in new_model_patterns)
    
    def _supports_temperature(self) -> bool:
        """检查模型是否支持自定义 temperature 参数"""
        # o1/o3/o4 系列和 gpt-5 系列不支持自定义 temperature
        # 这些模型只支持默认 temperature=1
        no_temp_patterns = ['o1', 'o3', 'o4', 'gpt-5']
        if any(pattern in self.model_name.lower() for pattern in no_temp_patterns):
            return False
        return True
    
    def call(
        self,
        images: List[str],
        question: str,
        system_prompt: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用 OpenAI GPT-4V API
        
        Args:
            images: 图像列表（base64 编码、URL 或本地路径）
            question: 用户问题
            system_prompt: 系统提示词
        
        Returns:
            包含响应信息的字典
        """
        # 构建消息内容
        content = []
        for img in images:
            content.append(self._prepare_image_content(img))
        content.append({"type": "text", "text": question})
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        
        # 获取参数
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        reasoning_effort = kwargs.get('reasoning_effort', self.reasoning_effort)
        
        # 构建 API 调用参数
        api_kwargs = {
            'model': self.model_name,
            'messages': messages,
        }
        
        # 只在支持 temperature 的模型上添加该参数
        if self._supports_temperature():
            api_kwargs['temperature'] = temperature
        
        # 新模型使用 max_completion_tokens，旧模型使用 max_tokens
        if self._is_new_model():
            api_kwargs['max_completion_tokens'] = max_tokens
        else:
            api_kwargs['max_tokens'] = max_tokens
        
        # 添加推理模式参数（如果提供）
        if reasoning_effort:
            valid_efforts = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh']
            if reasoning_effort.lower() in valid_efforts:
                api_kwargs['reasoning_effort'] = reasoning_effort.lower()
            else:
                print(f"警告: 无效的 reasoning_effort 值 '{reasoning_effort}'，有效值: {valid_efforts}")
        
        error_msg = ""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**api_kwargs)
                
                answer_text = response.choices[0].message.content
                
                return {
                    'success': True,
                    'answer': answer_text.strip() if answer_text else "",
                    'raw_response': {
                        'content': answer_text,
                        'finish_reason': response.choices[0].finish_reason,
                        'model': response.model
                    },
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    } if response.usage else None,
                    'request_id': response.id,
                    'error': None
                }
                
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                print(f"  OpenAI 请求异常 (尝试 {attempt + 1}/{self.max_retries}): {error_msg}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))
        
        return {
            'success': False,
            'answer': None,
            'raw_response': None,
            'usage': None,
            'request_id': None,
            'error': error_msg
        }
