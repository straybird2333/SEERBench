"""
本地开源模型评测配置
复用 Eval_final 的任务定义，使用本地模型目录
"""
import os

# 项目根目录：从本文件位置推导 (Eval/Eval_opensource/config_opensource.py -> 上两级)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ==================== 本地模型根目录 ====================
# Docker 中可通过环境变量覆盖，例如: -e LOCAL_MODEL_ROOT=/workspace/models
LOCAL_MODEL_ROOT = os.environ.get(
    "LOCAL_MODEL_ROOT",
    os.path.join(_PROJECT_ROOT, "fjh", "Model"),
)
OUTPUT_DIR = os.environ.get("EVAL_OUTPUT_DIR", "./results")

def get_model_root():
    """获取本地模型根目录"""
    return LOCAL_MODEL_ROOT

def get_output_dir():
    """获取输出目录"""
    return OUTPUT_DIR

# ==================== 任务列表（与 Eval_final 一致）====================
ALL_TASKS = [
    'camera_wearer',
    'camera_wearer_type2',
    'ego_2_exo_visibility',
    'camera_relative_position',
    'relative_distance',
    'object_relative_position',
    'object_correspondence',
    'object_prediction',
    'view_movement_1',
    'view_movement_2',
    'view_movement_3',
    'view_movement_4',
    'view_movement_5',
    'view_movement_6',
    'object_movement',
    'object_movement_1',
    'view_selection',
    'view_selection_2',
    'view_selection_3',
    'noise_collaboration',  # Noise_Collaboration.json - 不定项：选最好/选全部
]

# Spatial-SSRL 模型路径（可为 HF 模型名或本地权重目录）
# 环境变量 SPATIAL_SSRL_MODEL_PATH 可覆盖，例如: -e SPATIAL_SSRL_MODEL_PATH=/path/to/Spatial-SSRL-7B
SPATIAL_SSRL_MODEL_PATH = os.environ.get(
    "SPATIAL_SSRL_MODEL_PATH",
    os.path.join(_PROJECT_ROOT, "fjh", "Models", "internlm_Spatial-SSRL-7B"),
)

# SenseNova-SI 模型路径
# 环境变量 SENSENOVA_SI_MODEL_PATH 可覆盖
SENSENOVA_SI_MODEL_PATH = os.environ.get(
    "SENSENOVA_SI_MODEL_PATH",
    os.path.join(_PROJECT_ROOT, "fjh", "Models", "sensenova_SenseNova-SI-1.1-Qwen3-VL-8B"),
)

# RynnBrain-2B（DAMO Academy，基于 Qwen3-VL）；默认与 LOCAL_MODEL_ROOT 下目录名一致
RYNNBRAIN_2B_MODEL_PATH = os.environ.get(
    "RYNNBRAIN_2B_MODEL_PATH",
    os.path.join(LOCAL_MODEL_ROOT, "DAMO_Academy_RynnBrain-2B"),
)

# RynnBrain-8B（同系列，基于 Qwen3-VL-8B-Instruct）
RYNNBRAIN_8B_MODEL_PATH = os.environ.get(
    "RYNNBRAIN_8B_MODEL_PATH",
    os.path.join(LOCAL_MODEL_ROOT, "DAMO_Academy_RynnBrain-8B"),
)

# 小米 MiMo-Embodied-7B（AutoModelForImageTextToText + 官方 path 图像字段）
MIMO_EMBODIED_MODEL_PATH = os.environ.get(
    "MIMO_EMBODIED_MODEL_PATH",
    os.path.join(LOCAL_MODEL_ROOT, "XiaomiMiMo_MiMo-Embodied-7B"),
)

# 本地模型目录名到显示名的映射（便于 --model 参数使用简短名）
# 用户也可直接传完整路径或目录名
LOCAL_MODEL_ALIASES = {
    'qwen2.5-vl-7b': 'Qwen_Qwen2.5-VL-7B-Instruct',
    'qwen3-vl-8b': 'Qwen_Qwen3-VL-8B-Instruct',
    'qwen3-vl-8b-thinking': 'Qwen_Qwen3-VL-8B-Thinking',
    'internvl3-8b': 'OpenGVLab_InternVL3-8B',
    'internvl3-14b': 'OpenGVLab_InternVL3-14B',
    'llava-next-7b': 'llava-hf_LLaVA-NeXT-Video-7B-hf',
    'llava-qwen2-7b': 'llava-hf_llava-onevision-qwen2-7b-si-hf',
    'deepseek-vl-7b': 'deepseek-ai_deepseek-vl-7b-chat',
    'glm4v-9b': 'zai-org_GLM-4.1V-9B-Base',
    'glm4v-9b-thinking': 'zai-org_GLM-4.1V-9B-Thinking',
    # Spatial-SSRL：使用 SPATIAL_SSRL_MODEL_PATH，不映射到 LOCAL_MODEL_ROOT
    'spatial-ssrl': None,
    # SenseNova-SI：使用 SENSENOVA_SI_MODEL_PATH，不映射到 LOCAL_MODEL_ROOT
    'sensenova-si': None,
    # RynnBrain-2B / 8B：使用对应 RYNNBRAIN_*_MODEL_PATH（见 evaluate.py 分发）
    'rynnbrain-2b': None,
    'rynnbrain-8b': None,
    # MiMo-Embodied：使用 MIMO_EMBODIED_MODEL_PATH
    'mimo-embodied-7b': None,
    'mimo-embodied': None,
}

def resolve_model_path(model_name: str) -> str:
    """
    将用户指定的模型名解析为本地绝对路径或 HF 模型 id。
    model_name 可以是：别名、目录名、或绝对路径。
    spatial-ssrl 使用 SPATIAL_SSRL_MODEL_PATH（可为 HF id 或本地权重目录）。
    """
    # Spatial-SSRL：不映射到 LOCAL_MODEL_ROOT，直接使用 SPATIAL_SSRL_MODEL_PATH
    if model_name == "spatial-ssrl":
        return SPATIAL_SSRL_MODEL_PATH
    # SenseNova-SI：不映射到 LOCAL_MODEL_ROOT，直接使用 SENSENOVA_SI_MODEL_PATH
    if model_name == "sensenova-si":
        return SENSENOVA_SI_MODEL_PATH
    if model_name == "rynnbrain-2b":
        return RYNNBRAIN_2B_MODEL_PATH
    if model_name == "rynnbrain-8b":
        return RYNNBRAIN_8B_MODEL_PATH
    if model_name in ("mimo-embodied-7b", "mimo-embodied"):
        return MIMO_EMBODIED_MODEL_PATH
    root = get_model_root()
    if not root or not os.path.isdir(root):
        return model_name
    # 已是绝对路径且存在
    if os.path.isabs(model_name) and os.path.isdir(model_name):
        return model_name
    # 别名
    if model_name in LOCAL_MODEL_ALIASES:
        alias_path = LOCAL_MODEL_ALIASES[model_name]
        if alias_path is None:
            return model_name  # 如 spatial-ssrl 已在上面处理
        path = os.path.join(root, alias_path)
        if os.path.isdir(path):
            return path
        return path
    # 相对目录名（在 MODEL_ROOT 下）
    path = os.path.join(root, model_name)
    if os.path.isdir(path):
        return path
    return path
