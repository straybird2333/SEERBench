"""
本地 HuggingFace 视觉语言模型封装
支持从 LOCAL_MODEL_ROOT 加载的 Qwen2.5-VL、Qwen3-VL 等。
"""
import os
import re
import tempfile
import base64
from typing import List, Dict, Any, Optional

from .base import BaseVLM

# 根据 config.json 的 model_type 或 architectures 选择加载方式
def _detect_model_type(model_path: str) -> str:
    import json
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return "qwen2_5_vl"  # 默认尝试 Qwen2.5-VL
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    model_type = config.get("model_type", "").lower()
    arch = config.get("architectures", [])
    if "qwen3_vl" in model_type or "Qwen3VL" in str(arch):
        return "qwen3_vl"
    if "qwen2_5_vl" in model_type or "Qwen2_5_VL" in str(arch):
        return "qwen2_5_vl"
    if "qwen2_vl" in model_type or "Qwen2VL" in str(arch):
        return "qwen2_vl"
    if "internvl" in model_type or "InternVL" in str(arch):
        return "internvl"
    if "llava_next" in model_type or "LlavaNext" in str(arch):
        return "llava_next"
    if "llava_onevision" in model_type or "LlavaOnevision" in str(arch):
        return "llava_onevision"
    if "deepseek_vl" in model_type or "DeepseekVL" in str(arch):
        return "deepseek_vl"
    if model_type == "multi_modality" and "deepseek" in model_path.lower():
        return "deepseek_vl"
    # 默认按 Qwen2.5-VL 尝试（当前仓库多为 Qwen 系列）
    return "qwen2_5_vl"


def _is_deepseek_vl_legacy_config(model_path: str) -> bool:
    """旧版 DeepSeek-VL：config 中 vision_config.model_type 为 'vision'，无法被 DeepseekVLConfig 解析。"""
    import json
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return False
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    vc = config.get("vision_config") or {}
    return vc.get("model_type") == "vision" or "MultiModalityCausalLM" in config.get("architectures", [])


def _patch_deepseek_vl_siglip_dpr() -> bool:
    """
    修补已安装的 deepseek_vl 包内 siglip_vit.py：将 dpr = [x.item() for x in torch.linspace(...)]
    改为纯 Python 计算，避免 meta 张量上调用 .item() 报错。
    """
    import re
    try:
        import deepseek_vl
        pkg_dir = os.path.dirname(os.path.abspath(deepseek_vl.__file__))
        siglip_path = os.path.join(pkg_dir, "models", "siglip_vit.py")
    except Exception:
        return False
    if not os.path.isfile(siglip_path):
        return False
    # 匹配 dpr = [ x.item() for x in torch.linspace(0, drop_path_rate, depth) ] 及多行、行尾注释（仅到行尾，避免 DOTALL 时 #.* 吃掉后续代码）
    buggy = re.compile(
        r"dpr\s*=\s*\[\s*x\.item\(\)\s+for\s+x\s+in\s+torch\.linspace\s*\(\s*0\s*,\s*drop_path_rate\s*,\s*depth\s*\)\s*\]\s*(?:#[^\n]*)?",
    )
    safe = "dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]"
    with open(siglip_path, "r", encoding="utf-8") as f:
        content = f.read()
    if not buggy.search(content) or safe in content:
        return False
    content = buggy.sub(safe, content)
    with open(siglip_path, "w", encoding="utf-8") as f:
        f.write(content)
    # 若已加载过该模块，需移除以便后续加载使用修补后代码
    for key in list(__import__("sys").modules):
        if "deepseek_vl" in key and "siglip" in key:
            del __import__("sys").modules[key]
    return True


def _patch_internvl_vision_init(model_path: str) -> bool:
    """
    修补 InternVL 的 modeling_intern_vit.py：将 dpr = [x.item() for x in torch.linspace(...)]
    改为纯 Python 计算，避免在 meta 张量上调用 .item() 报错。
    会尝试修补模型目录或 HF cache 中的文件。
    """
    import re
    # 匹配 dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
    buggy = re.compile(
        r"dpr\s*=\s*\[x\.item\(\)\s+for\s+x\s+in\s+torch\.linspace\s*\([^)]*\)\s*\]"
    )
    safe = "n_dpr = getattr(config, 'num_hidden_layers', 24); dpr = [config.drop_path_rate * i / max(n_dpr - 1, 1) for i in range(n_dpr)]"
    patched = False
    # 1) 模型目录（本地加载时可能从这里读）
    if model_path and os.path.isdir(model_path):
        vit_path = os.path.join(model_path, "modeling_intern_vit.py")
        if os.path.isfile(vit_path):
            with open(vit_path, "r", encoding="utf-8") as f:
                content = f.read()
            if buggy.search(content) and safe not in content:
                content = buggy.sub(safe, content)
                with open(vit_path, "w", encoding="utf-8") as f:
                    f.write(content)
                patched = True
    # 2) HF cache（transformers_modules）
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    modules_dir = os.path.join(hf_home, "modules", "transformers_modules")
    if os.path.isdir(modules_dir):
        for name in os.listdir(modules_dir):
            vit_path = os.path.join(modules_dir, name, "modeling_intern_vit.py")
            if not os.path.isfile(vit_path):
                continue
            with open(vit_path, "r", encoding="utf-8") as f:
                content = f.read()
            if buggy.search(content) and safe not in content:
                content = buggy.sub(safe, content)
                with open(vit_path, "w", encoding="utf-8") as f:
                    f.write(content)
                patched = True
    if patched:
        import sys
        # 清除已加载的 InternVL 相关模块，迫使从已修补文件重新加载
        for key in list(sys.modules):
            if "transformers_modules" in key and ("modeling_intern_vit" in key or "internvl" in key.lower()):
                del sys.modules[key]
    return patched


def _patch_internvl_chat_tied_weights(model_path: str) -> bool:
    """
    修补 InternVL 的 modeling_internvl_chat.py：为兼容新版 transformers，
    在 __init__ 中确保存在 all_tied_weights_keys 且为 dict（新版需 .keys()/.update()）。
    """
    import re
    # 若之前误补成 set()，改为 {}
    set_to_dict = re.compile(r"self\.all_tied_weights_keys\s*=\s*set\(\)")
    # 在 super().__init__(config) 后插入 all_tied_weights_keys 初始化（仅补主模型类）
    pattern = re.compile(
        r"(\s+)super\(\).__init__\(config\)\n(?!\s+if not hasattr\(self, 'all_tied_weights_keys'\))"
    )
    insertion = (
        r"\1super().__init__(config)\n"
        r"\1if not hasattr(self, 'all_tied_weights_keys'):\n"
        r"\1    self.all_tied_weights_keys = {}\n"
    )
    patched = False

    def do_patch(file_path: str) -> bool:
        if not os.path.isfile(file_path):
            return False
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        changed = False
        # 将 set() 改为 {}（兼容之前误补或旧缓存）
        if set_to_dict.search(content):
            content = set_to_dict.sub("self.all_tied_weights_keys = {}", content)
            changed = True
        if "all_tied_weights_keys" not in content and pattern.search(content):
            content = pattern.sub(insertion, content, count=1)
            changed = True
        if not changed:
            return False
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True

    if model_path and os.path.isdir(model_path):
        chat_path = os.path.join(model_path, "modeling_internvl_chat.py")
        if do_patch(chat_path):
            patched = True

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    modules_dir = os.path.join(hf_home, "modules", "transformers_modules")
    if os.path.isdir(modules_dir):
        for name in os.listdir(modules_dir):
            chat_path = os.path.join(modules_dir, name, "modeling_internvl_chat.py")
            if do_patch(chat_path):
                patched = True

    if patched:
        import sys
        for key in list(sys.modules):
            if "transformers_modules" in key and "internvl" in key.lower():
                del sys.modules[key]
    return patched


def _images_to_paths(images: List[str]) -> List[str]:
    """将 base64 data URL 或路径转为本地临时文件路径（供 processor 使用）。"""
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


class LocalHFVLM(BaseVLM):
    """本地 HuggingFace VLM（当前实现 Qwen2.5-VL / Qwen3-VL）。"""

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        torch_dtype: Optional[str] = None,
        max_new_tokens: int = 64,
        **kwargs
    ):
        super().__init__(model_path, **kwargs)
        self.model_path = os.path.abspath(model_path)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._model_type = _detect_model_type(self.model_path)
        self._load_model()

    def _load_model(self):
        import torch
        # Qwen3-VL：使用官方 Qwen3VLForConditionalGeneration；先加载到 CPU 再 dispatch，避免 device_map=auto 时逐 param materialize 极慢
        if self._model_type == "qwen3_vl":
            try:
                from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            except ImportError:
                raise ImportError(
                    "Qwen3-VL 需要较新版本 transformers。请升级: pip install -U transformers"
                ) from None
            dtype = torch.bfloat16
            if self.torch_dtype == "float16":
                dtype = torch.float16
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map=None,
                torch_dtype=dtype,
                attn_implementation="sdpa",
                low_cpu_mem_usage=False,
            )
            if self.device_map in ("auto", "balanced", "balanced_low_0") or self.device_map is not None:
                try:
                    from accelerate import infer_auto_device_map
                except ImportError:
                    from accelerate.utils import infer_auto_device_map
                device_map = infer_auto_device_map(
                    self._model,
                    max_memory=None,
                    no_split_module_classes=getattr(
                        self._model.config, "no_split_module_classes", None
                    )
                    or getattr(self._model, "_no_split_modules", None)
                    or [],
                )
                # 单卡时整模型 .to(device) 比 dispatch_model 逐模块迁移更快，GPU 显存会较快打满
                devices = set(device_map.values()) if isinstance(device_map, dict) else set()
                if len(devices) == 1:
                    self._model = self._model.to(next(iter(devices)))
                else:
                    from accelerate import dispatch_model
                    self._model = dispatch_model(self._model, device_map=device_map)
            elif self.device_map is not None:
                self._model = self._model.to(self.device_map)
            self._processor = AutoProcessor.from_pretrained(self.model_path)
            return
        if self._model_type in ("qwen2_5_vl", "qwen2_vl"):
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            dtype = torch.bfloat16
            if self.torch_dtype == "float16":
                dtype = torch.float16
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map=self.device_map,
                torch_dtype=dtype,
                attn_implementation="sdpa",
            )
            self._processor = AutoProcessor.from_pretrained(self.model_path)
            return
        if self._model_type == "internvl":
            try:
                from transformers import AutoModel, AutoProcessor
                from transformers import modeling_utils as _mu
                # 修补 modeling_intern_vit（meta 张量）
                _patch_internvl_vision_init(self.model_path)
                _patch_internvl_chat_tied_weights(self.model_path)
                # 在 _finalize_model_loading 入口统一把 all_tied_weights_keys 从 set 改为 dict，
                # 避免后续 mark_tied_weights_as_initialized / _move_missing_keys 里 .keys() 报错
                _orig_finalize = _mu.PreTrainedModel._finalize_model_loading
                @staticmethod
                def _finalize_with_fix(model, load_config, loading_info):
                    v = getattr(model, "all_tied_weights_keys", None)
                    if isinstance(v, set):
                        model.all_tied_weights_keys = {}
                    return _orig_finalize(model, load_config, loading_info)
                _mu.PreTrainedModel._finalize_model_loading = _finalize_with_fix
                try:
                    dtype = torch.bfloat16
                    if self.torch_dtype == "float16":
                        dtype = torch.float16
                    self._model = AutoModel.from_pretrained(
                        self.model_path,
                        device_map=None,
                        dtype=dtype,
                        trust_remote_code=True,
                        low_cpu_mem_usage=False,
                    )
                    if self.device_map in ("auto", "balanced", "balanced_low_0"):
                        from accelerate import dispatch_model
                        try:
                            from accelerate import infer_auto_device_map
                        except ImportError:
                            from accelerate.utils import infer_auto_device_map
                        # dispatch_model 需要 dict 形态的 device_map，不能直接传字符串 "auto"
                        device_map = infer_auto_device_map(
                            self._model,
                            max_memory=None,
                            no_split_module_classes=getattr(
                                self._model.config, "no_split_module_classes", None
                            )
                            or getattr(self._model, "_no_split_modules", None)
                            or [],
                        )
                        self._model = dispatch_model(self._model, device_map=device_map)
                    elif self.device_map is not None:
                        self._model = self._model.to(self.device_map)
                    self._processor = AutoProcessor.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                    )
                finally:
                    _mu.PreTrainedModel._finalize_model_loading = _orig_finalize
                return
            except Exception as e:
                raise RuntimeError(
                    f"InternVL 等自定义架构需在对应环境中加载。错误: {e}"
                ) from e
        if self._model_type == "llava_next":
            try:
                from transformers import LlavaNextVideoForConditionalGeneration, AutoProcessor
                dtype = torch.bfloat16
                if self.torch_dtype == "float16":
                    dtype = torch.float16
                self._model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                    self.model_path,
                    device_map=self.device_map,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                )
                self._processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                )
                return
            except Exception as e:
                raise RuntimeError(
                    f"LLaVA-NeXT 加载失败。错误: {e}"
                ) from e
        if self._model_type == "llava_onevision":
            try:
                from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
                dtype = torch.bfloat16
                if self.torch_dtype == "float16":
                    dtype = torch.float16
                self._model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                    self.model_path,
                    device_map=self.device_map,
                    torch_dtype=dtype,
                )
                self._processor = AutoProcessor.from_pretrained(self.model_path)
                return
            except Exception as e:
                raise RuntimeError(
                    f"LLaVA-OneVision 加载失败。错误: {e}"
                ) from e
        if self._model_type == "deepseek_vl":
            try:
                from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM
                dtype = torch.bfloat16
                if self.torch_dtype == "float16":
                    dtype = torch.float16
                self._deepseek_vl_legacy = False
                # 旧版 checkpoint（multi_modality）：须先导入 deepseek_vl 以向 transformers 注册 config，再用 AutoModelForCausalLM 加载
                if _is_deepseek_vl_legacy_config(self.model_path):
                    try:
                        _patch_deepseek_vl_siglip_dpr()  # 修补 siglip_vit 中 dpr 的 .item()，避免 meta 张量报错
                        import deepseek_vl.models  # 注册 "multi_modality" 等，使 AutoConfig 能解析 config.json
                        from deepseek_vl.models.modeling_vlm import MultiModalityCausalLM
                        # 在 __init__ 中注入 all_tied_weights_keys，否则 _finalize_model_loading 里 .update() 会报错
                        _orig_init = MultiModalityCausalLM.__init__
                        def _patched_init(self, config, *args, **kwargs):
                            _orig_init(self, config, *args, **kwargs)
                            if not hasattr(self, "all_tied_weights_keys"):
                                self.all_tied_weights_keys = {}
                        MultiModalityCausalLM.__init__ = _patched_init
                    except ImportError as ie:
                        raise RuntimeError(
                            "旧版 DeepSeek-VL (multi_modality) 需安装官方包以加载。请执行: pip install git+https://github.com/deepseek-ai/DeepSeek-VL.git"
                        ) from ie
                    # 先用 device_map=None 加载，避免 infer_auto_device_map 访问 model.all_tied_weights_keys（旧版无此属性）
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        device_map=None,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                        low_cpu_mem_usage=False,
                    )
                    if self.device_map in ("auto", "balanced", "balanced_low_0") or self.device_map is not None:
                        from accelerate import dispatch_model
                        try:
                            from accelerate import infer_auto_device_map
                        except ImportError:
                            from accelerate.utils import infer_auto_device_map
                        # 旧版 MultiModalityCausalLM 无 all_tied_weights_keys，infer_auto_device_map 会访问，先补上
                        if not hasattr(self._model, "all_tied_weights_keys"):
                            self._model.all_tied_weights_keys = getattr(
                                self._model, "_tied_weights_keys", None
                            ) or {}
                        device_map = infer_auto_device_map(
                            self._model,
                            max_memory=None,
                            no_split_module_classes=getattr(
                                self._model.config, "no_split_module_classes", None
                            )
                            or getattr(self._model, "_no_split_modules", None)
                            or [],
                        )
                        self._model = dispatch_model(self._model, device_map=device_map)
                    elif self.device_map is not None:
                        self._model = self._model.to(self.device_map)
                    from deepseek_vl.models import VLChatProcessor
                    self._processor = VLChatProcessor.from_pretrained(self.model_path)
                    self._deepseek_vl_legacy = True
                else:
                    from transformers import DeepseekVLForConditionalGeneration
                    self._model = DeepseekVLForConditionalGeneration.from_pretrained(
                        self.model_path,
                        device_map=self.device_map,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                        attn_implementation="sdpa",
                    )
                    self._processor = AutoProcessor.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                    )
                return
            except Exception as e:
                raise RuntimeError(
                    f"DeepSeek-VL 加载失败。错误: {e}"
                ) from e
        raise ValueError(
            f"不支持的本地模型类型: {self._model_type}，路径: {self.model_path}"
        )

    @property
    def provider(self) -> str:
        return "LocalHF"

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
        # Qwen2.5-VL apply_chat_template 要求每条 message["content"] 为内容块列表，不能为字符串
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        content = []
        for p in image_paths:
            # Qwen2.5-VL processor 只接受: http(s) URL、本地文件路径、或 base64；不接受 file://
            path = p if os.path.isabs(p) else os.path.abspath(p)
            content.append({"type": "image", "url": path})
        content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": content})

        if self._model_type in ("qwen2_5_vl", "qwen2_vl", "qwen3_vl"):
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self._model.device)
            with torch.no_grad():
                out_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            input_len = inputs["input_ids"].shape[1]
            gen_ids = out_ids[:, input_len:]
            answer_text = self._processor.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            answer_text = answer_text[0] if answer_text else ""
            return {
                "success": True,
                "answer": answer_text.strip(),
                "raw_response": answer_text,
                "usage": None,
                "error": None,
            }
        if self._model_type == "internvl":
            return self._call_impl_internvl(
                image_paths, question, system_prompt, max_new_tokens, **kwargs
            )
        if self._model_type == "llava_next":
            # LLaVA-NeXT：与 Qwen 类似，apply_chat_template + generate（processor 可能接受 image url/path）
            # 序列过长会报 "Token indices sequence length is longer than the specified maximum"（如 11140 > 10250），需截断
            try:
                cfg = self._model.config
                model_max = getattr(
                    getattr(cfg, "text_config", None) or cfg,
                    "model_max_length",
                    None,
                ) or getattr(cfg, "max_position_embeddings", 10250)
                max_input_len = max(256, model_max - max_new_tokens)
                apply_kw = dict(
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                try:
                    inputs = self._processor.apply_chat_template(
                        messages,
                        max_length=max_input_len,
                        truncation=True,
                        **apply_kw,
                    )
                except TypeError:
                    inputs = self._processor.apply_chat_template(messages, **apply_kw)
                    if isinstance(inputs, dict) and inputs.get("input_ids") is not None:
                        seq_len = inputs["input_ids"].shape[1]
                        if seq_len > max_input_len:
                            inputs["input_ids"] = inputs["input_ids"][:, :max_input_len].contiguous()
                            if "attention_mask" in inputs:
                                inputs["attention_mask"] = inputs["attention_mask"][:, :max_input_len].contiguous()
                    elif hasattr(inputs, "input_ids") and inputs.input_ids.shape[1] > max_input_len:
                        inputs.input_ids = inputs.input_ids[:, :max_input_len].contiguous()
                        if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
                            inputs.attention_mask = inputs.attention_mask[:, :max_input_len].contiguous()
                if hasattr(inputs, "to"):
                    inputs = inputs.to(self._model.device)
                else:
                    inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
                with torch.no_grad():
                    out_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                input_len = inputs["input_ids"].shape[1]
                gen_ids = out_ids[:, input_len:]
                answer_text = self._processor.batch_decode(
                    gen_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                answer_text = answer_text[0] if answer_text else ""
                return {
                    "success": True,
                    "answer": answer_text.strip(),
                    "raw_response": answer_text,
                    "usage": None,
                    "error": None,
                }
            except Exception as e:
                return {
                    "success": False,
                    "answer": "",
                    "raw_response": None,
                    "usage": None,
                    "error": f"LLaVA-NeXT 推理失败: {e}",
                }
        if self._model_type == "llava_onevision":
            # LLaVA-OneVision：apply_chat_template + generate，与 Qwen 相同消息格式
            try:
                inputs = self._processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self._model.device)
                with torch.no_grad():
                    out_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                input_len = inputs["input_ids"].shape[1]
                gen_ids = out_ids[:, input_len:]
                answer_text = self._processor.batch_decode(
                    gen_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                answer_text = answer_text[0] if answer_text else ""
                return {
                    "success": True,
                    "answer": answer_text.strip(),
                    "raw_response": answer_text,
                    "usage": None,
                    "error": None,
                }
            except Exception as e:
                return {
                    "success": False,
                    "answer": "",
                    "raw_response": None,
                    "usage": None,
                    "error": f"LLaVA-OneVision 推理失败: {e}",
                }
        if self._model_type == "deepseek_vl":
            if getattr(self, "_deepseek_vl_legacy", False):
                return self._call_impl_deepseek_vl_legacy(
                    image_paths, question, system_prompt, max_new_tokens, **kwargs
                )
            # 新版 DeepSeek-VL：apply_chat_template + generate
            try:
                inputs = self._processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self._model.device)
                with torch.no_grad():
                    out_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                input_len = inputs["input_ids"].shape[1]
                gen_ids = out_ids[:, input_len:]
                answer_text = self._processor.batch_decode(
                    gen_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                answer_text = answer_text[0] if answer_text else ""
                return {
                    "success": True,
                    "answer": answer_text.strip(),
                    "raw_response": answer_text,
                    "usage": None,
                    "error": None,
                }
            except Exception as e:
                return {
                    "success": False,
                    "answer": "",
                    "raw_response": None,
                    "usage": None,
                    "error": f"DeepSeek-VL 推理失败: {e}",
                }
        return {
            "success": False,
            "answer": "",
            "raw_response": None,
            "usage": None,
            "error": f"未实现的模型类型: {self._model_type}",
        }

    def _call_impl_deepseek_vl_legacy(
        self,
        image_paths: List[str],
        question: str,
        system_prompt: str,
        max_new_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """旧版 DeepSeek-VL (MultiModalityCausalLM)：使用官方 VLChatProcessor + prepare_inputs_embeds + language_model.generate。"""
        import torch
        try:
            from deepseek_vl.utils.io import load_pil_images
        except ImportError:
            return {
                "success": False,
                "answer": "",
                "raw_response": None,
                "usage": None,
                "error": "旧版 DeepSeek-VL 推理需安装官方包: pip install git+https://github.com/deepseek-ai/DeepSeek-VL.git",
            }
        # 官方格式：每张图一个 <image_placeholder>，images 为路径列表
        placeholders = " ".join(["<image_placeholder>"] * len(image_paths))
        content = f"{placeholders} {question}".strip()
        conversation = [
            {"role": "User", "content": content, "images": list(image_paths)},
            {"role": "Assistant", "content": ""},
        ]
        pil_images = load_pil_images(conversation)
        prepare_inputs = self._processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
        ).to(self._model.device)
        with torch.no_grad():
            inputs_embeds = self._model.prepare_inputs_embeds(**prepare_inputs)
            tokenizer = self._processor.tokenizer
            outputs = self._model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        answer_text = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return {
            "success": True,
            "answer": answer_text.strip(),
            "raw_response": answer_text,
            "usage": None,
            "error": None,
        }

    def _call_impl_internvl(
        self,
        image_paths: List[str],
        question: str,
        system_prompt: str,
        max_new_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """InternVL 推理：使用 model.chat() 构建含 <IMG_CONTEXT> 的 prompt，避免 assert selected.sum() != 0。"""
        import torch
        from PIL import Image
        pil_images = []
        for p in image_paths:
            path = p if os.path.isabs(p) else os.path.abspath(p)
            try:
                pil_images.append(Image.open(path).convert("RGB"))
            except Exception as e:
                return {
                    "success": False,
                    "answer": "",
                    "raw_response": None,
                    "usage": None,
                    "error": f"无法打开图像 {path}: {e}",
                }
        device = next(self._model.parameters()).device

        # 单独构造 pixel_values 张量：多种方式尝试，避免依赖单一 attribute 名
        def _extract_pv(obj, depth=0):
            if depth > 2:
                return None
            if isinstance(obj, torch.Tensor) and obj.dim() >= 3:
                return obj
            pv = None
            if isinstance(obj, dict):
                pv = obj.get("pixel_values")
                if pv is None:
                    pv = obj.get("pixel_value")
            elif hasattr(obj, "get") and callable(obj.get):
                pv = obj.get("pixel_values")
                if pv is None:
                    pv = obj.get("pixel_value")
            elif hasattr(obj, "pixel_values"):
                pv = getattr(obj, "pixel_values", None)
            if pv is not None and not isinstance(pv, torch.Tensor):
                pv = _extract_pv(pv, depth + 1)
            return pv

        pixel_values_tensor = None
        proc = self._processor
        if pil_images:
            # 1) 用 processor 直接处理图像（多数 VLM 支持 images=）
            try:
                out = proc(images=pil_images, return_tensors="pt")
                pv = _extract_pv(out)
                if isinstance(pv, torch.Tensor):
                    pixel_values_tensor = pv
            except Exception:
                pass
            # 2) 从 processor 子组件取 image_processor / feature_extractor 等
            if pixel_values_tensor is None:
                for attr in ("image_processor", "image_processor_2", "feature_extractor", "vision_processor", "image_processor_1"):
                    if hasattr(proc, attr):
                        try:
                            ip = getattr(proc, attr)
                            if callable(ip):
                                out = ip(images=pil_images, return_tensors="pt")
                                pv = _extract_pv(out)
                                if isinstance(pv, torch.Tensor):
                                    pixel_values_tensor = pv
                                    break
                        except Exception:
                            continue
            # 3) 从模型目录单独加载 AutoImageProcessor（不依赖 composite processor 的 attribute）
            if pixel_values_tensor is None:
                try:
                    from transformers import AutoImageProcessor
                    ip = AutoImageProcessor.from_pretrained(self.model_path, trust_remote_code=True)
                    out = ip(images=pil_images, return_tensors="pt")
                    pv = _extract_pv(out)
                    if isinstance(pv, torch.Tensor):
                        pixel_values_tensor = pv
                except Exception:
                    pass
            # 4) 按 preprocessor_config.json 或 InternVL 默认（448, ImageNet mean/std）手动预处理，保证必有张量
            if pixel_values_tensor is None and pil_images:
                import json
                import numpy as np
                size, mean, std = 448, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                preprocess_path = os.path.join(self.model_path, "preprocessor_config.json")
                if os.path.isfile(preprocess_path):
                    try:
                        with open(preprocess_path, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                        size = int(cfg.get("size", cfg.get("crop_size", 448)))
                        mean = list(cfg.get("image_mean", mean))
                        std = list(cfg.get("image_std", std))
                    except Exception:
                        pass
                resample = getattr(Image, "Resampling", Image).LANCZOS
                tensors = []
                for img in pil_images:
                    img = img.resize((size, size), resample)
                    arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                    arr = (arr - torch.tensor(mean, dtype=arr.dtype).view(3, 1, 1)) / torch.tensor(std, dtype=arr.dtype).view(3, 1, 1)
                    tensors.append(arr)
                pixel_values_tensor = torch.stack(tensors)
        if pixel_values_tensor is None:
            return {
                "success": False,
                "answer": "",
                "raw_response": None,
                "usage": None,
                "error": "InternVL 无法从 processor 获取 pixel_values 张量（已尝试 processor(images=)、image_processor、AutoImageProcessor、preprocessor_config 手动预处理）",
            }
        if pixel_values_tensor.dim() == 3:
            pixel_values_tensor = pixel_values_tensor.unsqueeze(0)
        # 与模型权重 dtype 一致，避免 "Input type (float) and bias type (c10::BFloat16)" 报错
        model_dtype = next(self._model.parameters()).dtype
        pixel_values_tensor = pixel_values_tensor.to(device=device, dtype=model_dtype)

        # 使用 model.chat() 由模型内部构建含 <IMG_CONTEXT> 的 prompt，避免 input_ids 中无占位符导致 assert 失败
        # chat() 内部会做 generation_config['eos_token_id'] = ...，必须传可变 dict，不能传 GenerationConfig
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}
        question_with_image = question if "<image>" in question else "<image>\n" + question
        try:
            with torch.no_grad():
                response = self._model.chat(
                    tokenizer,
                    pixel_values_tensor,
                    question_with_image,
                    generation_config,
                    num_patches_list=[pixel_values_tensor.shape[0]],
                )
        except Exception as e:
            return {
                "success": False,
                "answer": "",
                "raw_response": None,
                "usage": None,
                "error": f"InternVL chat 调用失败: {e}",
            }
        if system_prompt and response.strip().lower().startswith(system_prompt.strip().lower()):
            response = response[len(system_prompt.strip()):].strip()
        return {
            "success": True,
            "answer": response.strip(),
            "raw_response": response,
            "usage": None,
            "error": None,
        }
