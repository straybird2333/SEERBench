"""
OpenRouter 模型实现
支持通过 OpenRouter API 调用各种模型（Claude, Gemini, Llama, etc.）
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


# 在初始化客户端之前强制设置
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"



class OpenRouterVLM(BaseVLM):
    """OpenRouter 视觉语言模型，支持多种后端模型"""
    
    # 常用的 OpenRouter 模型列表
    AVAILABLE_MODELS = [
        # Anthropic Claude 系列
        'anthropic/claude-sonnet-4.5',
        'anthropic/claude-3.5-sonnet:beta',
        'anthropic/claude-3-opus',
        'anthropic/claude-3-sonnet',
        'anthropic/claude-3-haiku',
        # Google Gemini 系列
        'google/gemini-3.0-flash-exp:free',
        'google/gemini-pro-1.5',
        'google/gemini-pro-vision',
        'google/gemini-3-flash-preview',
        'google/gemini-3-pro-preview',
        # 'google/gemini-flash-3',
        # Meta Llama 系列
        'meta-llama/llama-3.2-90b-vision-instruct',
        'meta-llama/llama-3.2-11b-vision-instruct',
        # OpenAI 系列
        'openai/gpt-4o',
        'openai/gpt-4o-mini',
        'openai/gpt-4-turbo',
        # Qwen 系列
        'qwen/qwen-2-vl-72b-instruct',
        'qwen/qwen-2-vl-7b-instruct',
        # 其他
        'mistralai/pixtral-12b',
    ]
    
    def __init__(
        self,
        model_name: str = "anthropic/claude-3.5-sonnet",
        api_key: str = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        temperature: float = 0.0,
        max_tokens: int = 10000,
        **kwargs
    ):
        """
        初始化 OpenRouter 模型
        
        Args:
            model_name: 模型名称（OpenRouter 格式，如 "anthropic/claude-3.5-sonnet"）
            api_key: API Key，如果为 None 则从环境变量 OPENROUTER_API_KEY 读取
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
            temperature: 温度参数
            max_tokens: 最大生成 token 数
        """
        if not HAS_OPENAI:
            raise ImportError("openai is required. Install with: pip install openai")
        
        super().__init__(model_name, **kwargs)
        
        # 设置 API Key
        self.api_key = api_key or os.environ.get('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required. Set it via environment variable or pass api_key parameter."
            )
        
        # 初始化 OpenAI 客户端，使用 OpenRouter 端点
        # 增加超时时间以应对网络不稳定
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=120.0,  # 120秒超时
            max_retries=0   # 我们自己处理重试
        )
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @property
    def provider(self) -> str:
        return "OpenRouter"
    
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
            file_path = image[7:]
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
    
    def call(
        self,
        images: List[str],
        question: str,
        system_prompt: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用 OpenRouter API
        
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
        
        # 构建 API 调用参数
        api_kwargs = {
            'model': self.model_name,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'extra_headers': {
                'HTTP-Referer': 'https://github.com/evaluation-benchmark',
                'X-Title': 'VLM Evaluation Benchmark'
            }
        }
        
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
                print(f"  OpenRouter 请求异常 (尝试 {attempt + 1}/{self.max_retries}): {error_msg}")
            
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
