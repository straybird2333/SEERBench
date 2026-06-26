"""
配置文件
"""
import os

# ==================== 路径映射配置 ====================
PROJECT_ROOT = os.environ.get(
    "SEERBENCH_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)

# Compatibility with paths used during benchmark construction.
PATH_MAPPING = {
    '/public/lfy/SU_benchmark': PROJECT_ROOT,
    '/workspace': PROJECT_ROOT,
}

def map_path(path: str) -> str:
    """
    将路径映射到 Docker 容器中的路径
    
    Args:
        path: 原始路径
    
    Returns:
        映射后的路径
    """
    for old_prefix, new_prefix in PATH_MAPPING.items():
        if path.startswith(old_prefix):
            return path.replace(old_prefix, new_prefix, 1)
    return path

# ==================== 数据目录 ====================
# 新数据目录：Benchmark_annotations
DATA_DIR_BENCHMARK = os.environ.get(
    "SEERBENCH_DATA_DIR",
    os.path.join(PROJECT_ROOT, "Benchmark_annotations"),
)
OUTPUT_DIR = "./results"  # 默认输出目录（当前目录下的 results）

def get_data_dir_benchmark():
    """获取Benchmark数据目录（自动映射）"""
    mapped = map_path(DATA_DIR_BENCHMARK)
    if os.path.exists(mapped):
        return mapped
    return DATA_DIR_BENCHMARK

def get_output_dir():
    """获取输出目录（相对路径，不进行映射）"""
    return OUTPUT_DIR

# ==================== 任务列表 ====================
# 所有支持的任务（每个文件作为独立任务）
ALL_TASKS = [
    # 单文件任务
    'camera_wearer',              # Camera_Wearer.json - 单选题
    'camera_wearer_type2',        # Camera_Wearer_Type2.json - 单选题：ego中标记相机，选择对应exo
    'ego_2_exo_visibility',       # Ego2Exo_Visibility.json - 单选题
    'camera_relative_position',   # Camera_Relative_Position.json - 单选题
    'relative_distance',          # Relative_Distance.json - 单选题
    'object_relative_position',   # Object_Relative_Position.json - 单选题
    'object_correspondence',      # Object_Correspondence.json - 单选题
    'object_prediction',          # Object_Prediction.json - 单选题
    
    # View_Movement 系列（每个文件独立）
    'view_movement_1',            # View_Movement_1.json - 多选题：移动后哪些物体不可见
    'view_movement_2',            # View_Movement_2.json - 多选题：移动后哪些物体不可见
    'view_movement_3',            # View_Movement_3.json - 单选题：移动后物体的时钟方向
    'view_movement_4',            # View_Movement_4.json - 单选题：移动后能看到多少物体
    'view_movement_5',            # View_Movement_5.json - 单选题：移动后exo相机的时钟方向
    'view_movement_6',            # View_Movement_6.json - 单选题：移动后最近的物体
    
    # Object_Movement 系列（每个文件独立）
    'object_movement',            # Object_Movement.json - 单选题：物体移动后的时钟方向
    'object_movement_1',          # Object_Movement_1.json - 多选题：物体移动后哪些exo能看到
    
    # View_Selection 任务
    'view_selection',             # View_Selection.json - 单选题：结合ego判断哪些exo有帮助
    'view_selection_2',           # View_Selection_2.json - 单选题：判断是否需要协作
    'view_selection_3',           # View_Selection_3.json - 单选题：选择最有帮助的exo视角
    'noise_collaboration',        # Noise_Collaboration.json - 不定项：选最好/选全部
]

# 任务名称到文件名的映射（每个任务对应单个文件）
TASK_TO_FILE = {
    'camera_wearer': 'Camera_Wearer.json',
    'camera_wearer_type2': 'Camera_Wearer_Type2.json',
    'ego_2_exo_visibility': 'Ego2Exo_Visibility.json',
    'camera_relative_position': 'Camera_Relative_Position.json',
    'relative_distance': 'Relative_Distance.json',
    'object_relative_position': 'Object_Relative_Position.json',
    'object_correspondence': 'Object_Correspondence.json',
    'object_prediction': 'Object_Prediction.json',
    'view_movement_1': 'View_Movement_1.json',
    'view_movement_2': 'View_Movement_2.json',
    'view_movement_3': 'View_Movement_3.json',
    'view_movement_4': 'View_Movement_4.json',
    'view_movement_5': 'View_Movement_5.json',
    'view_movement_6': 'View_Movement_6.json',
    'object_movement': 'Object_Movement.json',
    'object_movement_1': 'Object_Movement_1.json',
    'view_selection': 'View_Selection.json',
    'view_selection_2': 'View_Selection_2.json',
    'view_selection_3': 'View_Selection_3.json',
    'noise_collaboration': 'Noise_Collaboration.json',
}

# 多选题任务列表
MULTIPLE_CHOICE_TASKS = [
    'view_movement_1',   # 哪些物体移动后不可见
    'view_movement_2',   # 哪些物体移动后不可见
    'object_movement_1', # 物体移动后哪些exo能看到
    'view_selection',    # 哪些exo视角可以帮助
]

# ==================== API 配置 ====================
MAX_RETRIES = 3
RETRY_DELAY = 2  # 秒
REQUEST_DELAY = 0.5  # 请求之间的间隔

# ==================== 评测配置 ====================
# 相机相对位置任务：是否打乱相机顺序（用于防止模型记忆固定顺序）
SHUFFLE_CAMERA_ORDER = True

# ==================== 模型配置 ====================
DEFAULT_MODELS = {
    'qwen': 'qwen-vl-max-latest',
    'openai': 'gpt-4o',
    'openrouter': 'anthropic/claude-3.5-sonnet',
    'sensenova_si': 'sensenova/SenseNova-SI-1.3-InternVL3-8B',  # 默认模型路径（HuggingFace 模型名）
    'glm': 'glm-4.6v',
}

# ==================== 支持的模型提供商 ====================
SUPPORTED_PROVIDERS = ['qwen', 'openai', 'openrouter', 'sensenova_si', 'glm']

# ==================== 并行评测配置 ====================
MAX_PARALLEL_REQUESTS = 5  # 默认并行请求数
PARALLEL_BATCH_SIZE = 10   # 每批次处理的样本数
