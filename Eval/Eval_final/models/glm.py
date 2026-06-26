"""
智谱 GLM-4.6V 视觉语言模型实现
使用智谱开放平台 API：https://open.bigmodel.cn
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


class GLMVLM(BaseVLM):
    """智谱 GLM-4.6V 视觉语言模型（OpenAI 兼容 API）"""

    AVAILABLE_MODELS = [
        'glm-4.6v',
        'glm-4.6v-flashx',
        'glm-4.6v-flash',
        'glm-4v',
        'glm-4v-plus',
    ]

    def __init__(
        self,
        model_name: str = "glm-4.6v",
        api_key: str = None,
        base_url: str = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        temperature: float = 0.1,
        max_tokens: int = 10000,
        enable_thinking: bool = False,
        **kwargs
    ):
        """
        初始化智谱 GLM 视觉模型

        Args:
            model_name: 模型名称，如 glm-4.6v
            api_key: API Key，若为 None 则从环境变量 ZHIPU_API_KEY 读取
            base_url: API 基础 URL，默认使用智谱官方
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            enable_thinking: 是否开启深度思考模式
        """
        if not HAS_OPENAI:
            raise ImportError("openai is required. Install with: pip install openai")

        super().__init__(model_name, **kwargs)

        self.api_key = api_key or os.environ.get('ZHIPU_API_KEY')
        if not self.api_key:
            raise ValueError(
                "ZHIPU_API_KEY is required. Set it via environment variable or pass api_key parameter."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url or "https://open.bigmodel.cn/api/paas/v4",
            timeout=120.0,
            max_retries=0,
        )

        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking

    @property
    def provider(self) -> str:
        return "Zhipu/GLM"

    def _prepare_image_content(self, image: str) -> Dict:
        """将单张图像转为 API 所需的 image_url 格式。"""
        if image.startswith('data:image'):
            return {"type": "image_url", "image_url": {"url": image}}
        if image.startswith('http://') or image.startswith('https://'):
            return {"type": "image_url", "image_url": {"url": image}}
        if image.startswith('file://'):
            return self._file_to_image_content(image[7:])
        if os.path.exists(image):
            return self._file_to_image_content(image)
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}

    def _file_to_image_content(self, file_path: str) -> Dict:
        """本地文件转为 base64 data URL。"""
        import mimetypes
        mime_type = mimetypes.guess_type(file_path)[0] or 'image/jpeg'
        with open(file_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
        }

    def call(
        self,
        images: List[str],
        question: str,
        system_prompt: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用智谱 GLM-4.6V API（多图 + 文本）。

        Args:
            images: 图像列表（路径、URL 或 base64）
            question: 用户问题
            system_prompt: 系统提示词

        Returns:
            包含 success, answer, raw_response, usage, error 的字典
        """
        content = []
        for img in images:
            content.append(self._prepare_image_content(img))
        content.append({"type": "text", "text": question})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        enable_thinking = kwargs.get('enable_thinking', self.enable_thinking)

        api_kwargs = {
            'model': self.model_name,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        if enable_thinking:
            api_kwargs['extra_body'] = {'thinking': {'type': 'enabled'}}

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
                        'finish_reason': getattr(response.choices[0], 'finish_reason', None),
                        'model': getattr(response, 'model', self.model_name),
                    },
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens,
                    } if response.usage else None,
                    'request_id': getattr(response, 'id', None),
                    'error': None,
                }
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                print(f"  GLM 请求异常 (尝试 {attempt + 1}/{self.max_retries}): {error_msg}")
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))

        return {
            'success': False,
            'answer': None,
            'raw_response': None,
            'usage': None,
            'request_id': None,
            'error': error_msg,
        }
