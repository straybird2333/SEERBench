"""
SenseNova-SI 本地模型实现

参考 SenseNova-SI/example.py 的使用方式
"""
import os
import sys
from typing import List, Dict, Any

from .base import BaseVLM

PROJECT_ROOT = os.environ.get(
    "SEERBENCH_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)

# 路径映射：兼容 benchmark 构建阶段路径
PATH_MAPPING = {
    '/public/lfy/SU_benchmark': PROJECT_ROOT,
    '/workspace': PROJECT_ROOT,
}

def map_path_for_docker(path: str) -> str:
    """
    将路径映射到 Docker 容器中的路径
    
    Args:
        path: 原始路径
    
    Returns:
        映射后的路径（如果映射路径存在则使用，否则返回原路径）
    """
    for old_prefix, new_prefix in PATH_MAPPING.items():
        if path.startswith(old_prefix):
            mapped_path = path.replace(old_prefix, new_prefix, 1)
            # 如果映射后的路径存在，使用它
            if os.path.exists(mapped_path):
                return mapped_path
            # 如果原路径存在，使用原路径
            if os.path.exists(path):
                return path
            # 都不存在，返回映射后的路径（让调用者处理错误）
            return mapped_path
    return path

# SenseNova-SI 路径（可以通过环境变量覆盖）
SENSENOVA_SI_PATH = os.environ.get(
    'SENSENOVA_SI_PATH',
    os.path.join(PROJECT_ROOT, "Eval", "SenseNova-SI"),
)

# 尝试多个可能的路径
POSSIBLE_PATHS = [
    SENSENOVA_SI_PATH,
    os.path.join(PROJECT_ROOT, "Eval", "SenseNova-SI"),
    os.path.join(os.path.dirname(__file__), "../../SenseNova-SI"),
    os.path.join(os.path.dirname(__file__), "../../../Eval/SenseNova-SI"),
]

# 查找 SenseNova-SI 目录
def _find_sensenova_si_path():
    """查找 SenseNova-SI 目录"""
    for path in POSSIBLE_PATHS:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and os.path.exists(os.path.join(abs_path, 'sensenova_si')):
            return abs_path
    return None

# 找到 SenseNova-SI 路径
FOUND_SENSENOVA_SI_PATH = _find_sensenova_si_path()
if FOUND_SENSENOVA_SI_PATH:
    if FOUND_SENSENOVA_SI_PATH not in sys.path:
        sys.path.insert(0, FOUND_SENSENOVA_SI_PATH)

# 尝试导入（延迟导入，在类初始化时再检查）
try:
    from sensenova_si import get_model
    HAS_SENSENOVA_SI = True
except ImportError:
    HAS_SENSENOVA_SI = False
    get_model = None


class SenseNovaSIVLM(BaseVLM):
    """SenseNova-SI 本地部署模型"""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "auto",
        generation_config: Dict[str, Any] = None,
        device_map: str = "auto",
        dtype: str = "auto",
        **kwargs
    ):
        """
        初始化 SenseNova-SI 模型
        
        参考 SenseNova-SI/example.py 的使用方式：
        ```python
        from sensenova_si import get_model
        model = get_model(model_path, model_type=model_type)
        response = model.generate(question, images=image_paths)
        ```
        
        Args:
            model_path: 模型路径（本地路径或 HuggingFace 模型名，如 "sensenova/SenseNova-SI-1.3-InternVL3-8B"）
            model_type: 模型类型 ('qwen', 'internvl', 'bagel', 'auto')
            generation_config: 生成配置字典
            device_map: 设备映射（传递给模型，如果支持）
            dtype: 数据类型（传递给模型，如果支持）
        """
        # 检查是否可以导入
        if not HAS_SENSENOVA_SI or get_model is None:
            # 再次尝试导入，尝试多个路径
            get_model_func = None
            last_error = None
            
            # 尝试使用找到的路径和所有可能的路径
            paths_to_try = []
            if FOUND_SENSENOVA_SI_PATH:
                paths_to_try.append(FOUND_SENSENOVA_SI_PATH)
            paths_to_try.extend([p for p in POSSIBLE_PATHS if p not in paths_to_try])
            
            for path in paths_to_try:
                abs_path = os.path.abspath(path)
                if not os.path.exists(abs_path):
                    continue
                
                try:
                    if abs_path not in sys.path:
                        sys.path.insert(0, abs_path)
                    from sensenova_si import get_model as _get_model
                    get_model_func = _get_model
                    globals()['get_model'] = _get_model
                    print(f"✓ 成功从路径导入 SenseNova-SI: {abs_path}")
                    break
                except ImportError as e:
                    last_error = e
                    continue
            
            if get_model_func is None:
                error_msg = str(last_error) if last_error else "Unknown error"
                # 检查是否是缺少依赖
                missing_deps = []
                if 'torch' in error_msg.lower() or 'No module named \'torch\'' in error_msg:
                    missing_deps.append('torch')
                if 'transformers' in error_msg.lower():
                    missing_deps.append('transformers')
                if 'PIL' in error_msg or 'Pillow' in error_msg:
                    missing_deps.append('Pillow')
                if 'flash_attn' in error_msg.lower() or 'flash-attn' in error_msg.lower():
                    missing_deps.append('flash-attn')
                
                dep_hint = ""
                if missing_deps:
                    dep_hint = f"\n缺少依赖: {', '.join(missing_deps)}\n"
                    if 'flash-attn' in missing_deps:
                        dep_hint += f"\n注意: flash-attn 需要特殊安装，建议使用 SenseNova-SI 的环境。\n"
                        dep_hint += f"如果使用 uv 管理环境，请激活: source {FOUND_SENSENOVA_SI_PATH if FOUND_SENSENOVA_SI_PATH else SENSENOVA_SI_PATH}/.venv/bin/activate\n"
                        dep_hint += f"或者参考 SenseNova-SI 的安装说明安装依赖。\n"
                    else:
                        dep_hint += f"请安装: pip install {' '.join(missing_deps)}\n"
                    
                    if FOUND_SENSENOVA_SI_PATH:
                        venv_path = os.path.join(FOUND_SENSENOVA_SI_PATH, '.venv', 'bin', 'activate')
                        if os.path.exists(venv_path):
                            dep_hint += f"\n推荐: 激活 SenseNova-SI 的环境:\n"
                            dep_hint += f"  source {venv_path}\n"
                            dep_hint += f"然后再运行评测脚本。\n"
                
                # 检查是否找到了路径
                path_info = ""
                if FOUND_SENSENOVA_SI_PATH:
                    path_info = f"✓ 已找到 SenseNova-SI 代码路径: {FOUND_SENSENOVA_SI_PATH}\n"
                else:
                    searched_paths = "\n".join([f"  - {os.path.abspath(p)}" for p in paths_to_try[:5]])
                    path_info = f"已尝试的路径:\n{searched_paths}\n"
                
                raise ImportError(
                    f"SenseNova-SI 导入失败（依赖问题）。\n"
                    f"{path_info}"
                    f"错误信息: {error_msg}\n"
                    f"{dep_hint}"
                )
        
        super().__init__(model_path, **kwargs)
        
        self.model_type = model_type
        self.device_map = device_map
        self.dtype = dtype
        
        # 加载模型（参考 example.py 第 58 行）
        print(f"正在加载 SenseNova-SI 模型: {model_path} (type: {model_type})")
        self.model = get_model(
            model_path=model_path,
            model_type=model_type
        )
        
        # 设置生成配置
        if generation_config:
            if hasattr(self.model, 'default_generation_config'):
                self.model.default_generation_config.update(generation_config)
    
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
        """
        调用 SenseNova-SI 模型
        
        参考 example.py 第 78 行和第 85 行的使用方式：
        ```python
        response = model.generate(question, images=image_paths)
        ```
        
        Args:
            images: 图像列表（base64 编码的 data URL 或本地路径）
            question: 用户问题
            system_prompt: 系统提示词（如果模型支持）
        
        Returns:
            包含响应信息的字典
        """
        # 处理图像路径
        # SenseNova-SI 需要本地文件路径（参考 example.py，images 是文件路径列表）
        image_paths = []
        temp_files = []
        
        try:
            import tempfile
            import base64
            
            for img in images:
                if img.startswith('data:image'):
                    # 提取 base64 数据并保存为临时文件
                    header, data = img.split(',', 1)
                    image_data = base64.b64decode(data)
                    # 根据 MIME 类型确定文件扩展名
                    if 'png' in header:
                        suffix = '.png'
                    elif 'jpeg' in header or 'jpg' in header:
                        suffix = '.jpg'
                    else:
                        suffix = '.jpg'
                    
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    temp_file.write(image_data)
                    temp_file.close()
                    image_paths.append(temp_file.name)
                    temp_files.append(temp_file.name)
                elif img.startswith('file://'):
                    # 移除 file:// 前缀并映射路径
                    local_path = img[7:]
                    mapped_path = map_path_for_docker(local_path)
                    image_paths.append(mapped_path)
                elif os.path.exists(img):
                    # 映射路径后使用
                    mapped_path = map_path_for_docker(img)
                    image_paths.append(mapped_path)
                else:
                    # 尝试映射路径
                    mapped_path = map_path_for_docker(img)
                    if os.path.exists(mapped_path):
                        image_paths.append(mapped_path)
                    else:
                        print(f"警告: 无法处理的图像格式或路径不存在: {img[:50]}... (映射后: {mapped_path[:50]}...)")
                        continue
            
            # 合并系统提示词和问题（如果提供）
            if system_prompt:
                full_question = f"{system_prompt}\n\n{question}"
            else:
                full_question = question
            
            # 调用模型生成（参考 example.py 第 78/85 行）
            # model.generate(question, images=image_paths)
            response_text = self.model.generate(
                question=full_question,
                images=image_paths,
                **kwargs
            )
            
            # 处理返回结果（根据 example.py，返回的是字符串）
            if isinstance(response_text, list):
                response_text = response_text[0] if response_text else ""
            elif not isinstance(response_text, str):
                response_text = str(response_text)
            
            return {
                'success': True,
                'answer': response_text.strip(),
                'raw_response': response_text,
                'usage': None,  # SenseNova-SI 不提供 token 使用量
                'request_id': None,
                'error': None
            }
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            import traceback
            print(f"  SenseNova-SI 请求异常: {error_msg}")
            print(f"  详细错误: {traceback.format_exc()}")
            return {
                'success': False,
                'answer': None,
                'raw_response': None,
                'usage': None,
                'request_id': None,
                'error': error_msg
            }
        
        finally:
            # 清理临时文件
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
