"""
Spatial-SSRL 本地/HuggingFace 模型封装
基于 Qwen2.5-VL，支持 internlm/Spatial-SSRL-7B 或本地权重目录。
推理格式与官方 demo 一致（可选 <think> 与 \\boxed{} 的 format_prompt）。
"""
import os
import sys
import base64
import tempfile
import traceback
from typing import List, Dict, Any, Optional

from .base import BaseVLM


def _images_to_paths(images: List[str]) -> tuple:
    """将 base64 data URL 或路径转为本地文件路径。返回 (paths, temp_files)。"""
    paths = []
    temp_files = []
    for img in images:
        if img.startswith("data:image"):
            header, data = img.split(",", 1)
            raw = base64.b64decode(data)
            ext = ".jpg"
            if "png" in header:
                ext = ".png"
            fd, path = tempfile.mkstemp(suffix=ext)
            os.close(fd)
            with open(path, "wb") as f:
                f.write(raw)
            paths.append(path)
            temp_files.append(path)
        elif img.startswith("file://"):
            paths.append(img[7:].lstrip("/"))
        elif os.path.isfile(img):
            paths.append(os.path.abspath(img))
        else:
            paths.append(img)
    return paths, temp_files


# 与 Spatial-SSRL 官方 demo 一致的格式提示（可选）
DEFAULT_FORMAT_PROMPT = (
    "\n You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags. "
    "The final answer MUST BE put in \\boxed{}."
)


class SpatialSSRLVLM(BaseVLM):
    """Spatial-SSRL 视觉语言模型（Qwen2.5-VL 架构）。"""

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        torch_dtype: Optional[str] = "auto",
        max_new_tokens: int = 64,
        use_format_prompt: bool = True,
        **kwargs
    ):
        super().__init__(model_path, **kwargs)
        self.model_path = model_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.use_format_prompt = use_format_prompt
        self._model = None
        self._processor = None
        self._load_model()

    def _load_model(self):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(self.model_path)

    @property
    def provider(self) -> str:
        return "Spatial-SSRL"

    def call(
        self,
        images: List[str],
        question: str,
        system_prompt: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        paths, temp_files = _images_to_paths(images)
        try:
            return self._call_impl(paths, question, system_prompt, **kwargs)
        finally:
            for p in temp_files:
                try:
                    os.unlink(p)
                except Exception:
                    pass

    def _call_impl(
        self,
        image_paths: List[str],
        question: str,
        system_prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            return {
                "success": False,
                "answer": "",
                "raw_response": None,
                "usage": None,
                "error": "Spatial-SSRL 需要 qwen_vl_utils。请安装: pip install qwen-vl-utils",
            }
        import torch
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        text = question
        if self.use_format_prompt:
            text = text + DEFAULT_FORMAT_PROMPT
        if system_prompt:
            text = (system_prompt.strip() + "\n\n" + text).strip()
        content = []
        for p in image_paths:
            path = p if os.path.isabs(p) else os.path.abspath(p)
            content.append({"type": "image", "image": path})
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        try:
            text_prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            device = next(self._model.parameters()).device
            inputs = inputs.to(device)
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            answer_text = output_text[0] if output_text else ""
            return {
                "success": True,
                "answer": answer_text.strip(),
                "raw_response": answer_text,
                "usage": None,
                "error": None,
            }
        except Exception as e:
            print(f"\n[Spatial-SSRL ERROR] 推理失败: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return {
                "success": False,
                "answer": "",
                "raw_response": None,
                "usage": None,
                "error": f"Spatial-SSRL 推理失败: {e}",
            }
