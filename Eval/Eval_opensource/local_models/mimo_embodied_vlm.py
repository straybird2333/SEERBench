"""
小米 MiMo-Embodied 本地模型封装
与官方示例一致：AutoModelForImageTextToText + AutoProcessor，trust_remote_code，
用户消息中图像为 {"type": "image", "path": "<绝对路径>"}（非 RynnBrain 的 image/url 字段）。
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


class MiMoEmbodiedVLM(BaseVLM):
    """MiMo-Embodied 系列（Qwen2.5-VL 系权重，自定义 chat / processor）。"""

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 64,
        **kwargs,
    ):
        super().__init__(model_path, **kwargs)
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._device = None
        self._load_model()

    def _load_model(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self._model.to(self._device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

    @property
    def provider(self) -> str:
        return "MiMo-Embodied"

    def call(
        self,
        images: List[str],
        question: str,
        system_prompt: str = "",
        **kwargs,
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
        **kwargs,
    ) -> Dict[str, Any]:
        import torch
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)

        conversation = []
        if system_prompt:
            conversation.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )
        content = []
        for p in image_paths:
            path = p if os.path.isabs(p) else os.path.abspath(p)
            content.append({"type": "image", "path": path})
        content.append({"type": "text", "text": question})
        conversation.append({"role": "user", "content": content})

        try:
            model_inputs = self._processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            model_inputs = model_inputs.to(self._device)

            with torch.inference_mode():
                output_ids = self._model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            prompt_length = model_inputs["input_ids"].size(1)
            gen_ids = output_ids[:, prompt_length:]
            response = self._processor.decode(
                gen_ids[0], skip_special_tokens=True
            )
            return {
                "success": True,
                "answer": response.strip(),
                "raw_response": response,
                "usage": None,
                "error": None,
            }
        except Exception as e:
            print(f"\n[MiMo-Embodied ERROR] 推理失败: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return {
                "success": False,
                "answer": "",
                "raw_response": None,
                "usage": None,
                "error": f"MiMo-Embodied 推理失败: {e}",
            }
