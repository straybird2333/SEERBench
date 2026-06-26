"""
SenseNova-SI 本地模型封装
基于 Qwen3-VL，支持 sensenova/SenseNova-SI-1.1-Qwen3-VL-8B 或本地权重目录。
推理流程与官方 sensenova_si 包一致（AutoModelForImageTextToText + apply_chat_template）。
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


class SenseNovaSIVLM(BaseVLM):
    """SenseNova-SI 视觉语言模型（Qwen3-VL 架构）。"""

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        torch_dtype: Optional[str] = "auto",
        max_new_tokens: int = 64,
        **kwargs
    ):
        super().__init__(model_path, **kwargs)
        self.model_path = model_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._load_model()

    def _load_model(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            dtype=self.torch_dtype,
        )
        self._processor = AutoProcessor.from_pretrained(self.model_path)

    @property
    def provider(self) -> str:
        return "SenseNova-SI"

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
        import torch
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

        content = []
        for p in image_paths:
            path = p if os.path.isabs(p) else os.path.abspath(p)
            content.append({"type": "image", "image": path})
        content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": content})

        try:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self._model.device)

            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
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
            print(f"\n[SenseNova-SI ERROR] 推理失败: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return {
                "success": False,
                "answer": "",
                "raw_response": None,
                "usage": None,
                "error": f"SenseNova-SI 推理失败: {e}",
            }
