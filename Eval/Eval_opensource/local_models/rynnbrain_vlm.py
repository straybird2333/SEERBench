"""
RynnBrain（DAMO Academy）本地模型封装
与官方 cookbooks 一致：AutoModelForImageTextToText + AutoProcessor，trust_remote_code，
消息中图像字段为 {"type": "image", "image": <路径>}。
适用于 Alibaba-DAMO-Academy/RynnBrain-2B、RynnBrain-8B 等基于 Qwen3-VL 的权重目录。
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


class RynnBrainVLM(BaseVLM):
    """RynnBrain 系列视觉语言模型（Qwen3-VL 系，HF AutoModelForImageTextToText）。"""

    def __init__(
        self,
        model_path: str,
        device_map: Optional[str] = None,
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
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        # device_map="auto" 等依赖 accelerate；未安装时改为整模 .to(device)，避免 ValueError
        td = torch.float16 if self.torch_dtype == "float16" else torch.bfloat16
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_kw = dict(
            pretrained_model_name_or_path=self.model_path,
            torch_dtype=td,
            trust_remote_code=True,
        )
        use_map = bool(self.device_map) and str(self.device_map).lower() not in ("none", "")
        if use_map:
            try:
                import accelerate  # noqa: F401
            except ImportError:
                use_map = False
        if use_map:
            load_kw["device_map"] = self.device_map
            self._model = AutoModelForImageTextToText.from_pretrained(**load_kw)
        else:
            self._model = AutoModelForImageTextToText.from_pretrained(**load_kw)
            self._model.to(self._device)

        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

    @property
    def provider(self) -> str:
        return "RynnBrain"

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
            print(f"\n[RynnBrain ERROR] 推理失败: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return {
                "success": False,
                "answer": "",
                "raw_response": None,
                "usage": None,
                "error": f"RynnBrain 推理失败: {e}",
            }
