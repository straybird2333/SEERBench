"""
Qwen VL 模型实现
"""
import os
import time
from typing import List, Dict, Any

from .base import BaseVLM

try:
    import dashscope
    from dashscope import MultiModalConversation
    HAS_DASHSCOPE = True
except ImportError:
    HAS_DASHSCOPE = False


class QwenVLM(BaseVLM):
    """Qwen 视觉语言模型"""
    
    # 可用的模型列表
    AVAILABLE_MODELS = [
        'qwen-vl-plus',
        'qwen-vl-max',
        'qwen-vl-max-latest',
        'qwen2-vl-72b-instruct',
        'qwen2-vl-7b-instruct',
        'qwen2-vl-2b-instruct',
        'qwen2.5-vl-72b-instruct',
        'qwen2.5-vl-32b-instruct',
        'qwen2.5-vl-7b-instruct',
        'qwen2.5-vl-3b-instruct',
        'qwen3-vl-max',
        'qwen3-vl-plus',
        'qwen3-vl-flash',
        'qwen3-vl-32b-instruct',
    ]
    
    def __init__(
        self,
        model_name: str = "qwen-vl-max-latest",
        api_key: str = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        temperature: float = 0.1,
        top_p: float = 0.01,
        max_tokens: int = 50,
        enable_thinking: bool = False,
        **kwargs
    ):
        """
        初始化 Qwen VL 模型
        
        Args:
            model_name: 模型名称
            api_key: API Key，如果为 None 则从环境变量读取
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
            temperature: 温度参数
            top_p: top_p 参数
            max_tokens: 最大生成 token 数
            enable_thinking: 是否启用推理/思考模式
        """
        if not HAS_DASHSCOPE:
            raise ImportError("dashscope is required. Install with: pip install dashscope")
        
        super().__init__(model_name, **kwargs)
        
        # 设置 API Key
        self.api_key = api_key or os.environ.get('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY is required. Set it via environment variable or pass api_key parameter."
            )
        dashscope.api_key = self.api_key
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
    
    @property
    def provider(self) -> str:
        return "Alibaba/Qwen"
    
    def call(
        self,
        images: List[str],
        question: str,
        system_prompt: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用 Qwen VL API
        
        Args:
            images: 图像列表（base64 编码或 file:// URL）
            question: 用户问题
            system_prompt: 系统提示词
        
        Returns:
            包含响应信息的字典
        """
        # 构建消息内容
        content = []
        for img in images:
            content.append({'image': img})
        content.append({'text': question})
        
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': content})
        
        # 获取参数
        temperature = kwargs.get('temperature', self.temperature)
        top_p = kwargs.get('top_p', self.top_p)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        enable_thinking = kwargs.get('enable_thinking', self.enable_thinking)
        
        # 构建 API 调用参数
        api_kwargs = {
            'model': self.model_name,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
        }
        
        # 添加推理模式参数（如果启用）
        if enable_thinking:
            # Qwen 通过 extra_body 传递 enable_thinking
            api_kwargs['extra_body'] = {'enable_thinking': True}
        
        for attempt in range(self.max_retries):
            try:
                response = MultiModalConversation.call(**api_kwargs)
                
                if response.status_code == 200:
                    raw_output = response.output.choices[0]['message']['content']
                    answer_text = raw_output[0]['text'] if raw_output else ""
                    
                    return {
                        'success': True,
                        'answer': answer_text.strip(),
                        'raw_response': raw_output,
                        'usage': getattr(response, 'usage', None),
                        'request_id': getattr(response, 'request_id', None),
                        'error': None
                    }
                else:
                    error_msg = f"API error: {response.code} - {response.message}"
                    print(f"  Qwen API 错误 (尝试 {attempt + 1}/{self.max_retries}): {error_msg}")
                    
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                print(f"  Qwen 请求异常 (尝试 {attempt + 1}/{self.max_retries}): {error_msg}")
            
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
