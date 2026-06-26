"""
工具函数模块
"""
import os
import re
import io
import base64
from typing import List, Dict, Tuple, Optional

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ==================== 路径映射 ====================
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
        映射后的路径（如果映射路径存在则使用，否则返回原路径）
    """
    for old_prefix, new_prefix in PATH_MAPPING.items():
        if path.startswith(old_prefix):
            mapped_path = path.replace(old_prefix, new_prefix, 1)
            # 如果映射后的路径存在，使用它；否则尝试原路径
            if os.path.exists(mapped_path):
                return mapped_path
            # 如果原路径存在，使用原路径
            if os.path.exists(path):
                return path
            # 都不存在，返回映射后的路径（让调用者处理错误）
            return mapped_path
    return path


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """将十六进制颜色转换为 RGB 元组"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def get_image_mime_type(image_path: str) -> str:
    """根据文件扩展名获取 MIME 类型"""
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp'
    }
    return mime_types.get(ext, 'image/jpeg')


def image_to_base64(image_path: str, apply_path_mapping: bool = True) -> str:
    """
    将本地图片转换为 base64 data URL
    
    Args:
        image_path: 图像路径
        apply_path_mapping: 是否应用路径映射（用于 Docker 容器）
    
    Returns:
        base64 data URL
    """
    # 应用路径映射
    if apply_path_mapping:
        actual_path = map_path(image_path)
    else:
        actual_path = image_path
    
    if not os.path.exists(actual_path):
        raise ValueError(f'Image file not found: {image_path} (mapped to: {actual_path})')
    
    mime_type = get_image_mime_type(actual_path)
    with open(actual_path, 'rb') as f:
        image_data = f.read()
    
    base64_data = base64.b64encode(image_data).decode('utf-8')
    return f"data:{mime_type};base64,{base64_data}"


def pil_image_to_base64(img: Image.Image, format: str = 'JPEG') -> str:
    """将 PIL Image 转换为 base64 data URL"""
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    base64_data = base64.b64encode(buffer.read()).decode('utf-8')
    mime_type = 'image/jpeg' if format.upper() == 'JPEG' else 'image/png'
    return f"data:{mime_type};base64,{base64_data}"


def draw_annotations_on_image(image_path: str, annotations: List[Dict], apply_path_mapping: bool = True) -> Image.Image:
    """
    在图像上绘制标注（点、bbox、mask）
    
    Args:
        image_path: 图像路径
        annotations: 标注列表
        apply_path_mapping: 是否应用路径映射（用于 Docker 容器）
    
    Returns:
        绘制了标注的 PIL Image 对象
    """
    if not HAS_PIL:
        raise ImportError("PIL is required for drawing annotations. Install with: pip install Pillow")
    
    # 应用路径映射
    if apply_path_mapping:
        actual_path = map_path(image_path)
    else:
        actual_path = image_path
    
    img = Image.open(actual_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for ann in annotations:
        ann_type = ann.get('type', 'point')
        x = ann.get('x', 0) * width
        y = ann.get('y', 0) * height
        label = ann.get('label', '')
        color = ann.get('color', '#ff0000')
        
        # 解析颜色
        try:
            rgb_color = hex_to_rgb(color)
        except:
            rgb_color = (255, 0, 0)
        
        if ann_type == 'point':
            # 绘制点（圆形）
            radius = 8
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                        fill=rgb_color, outline=(255, 255, 255))
            # 绘制标签
            if label:
                draw.text((x + radius + 5, y - radius), label, fill=(255, 255, 0), font=font)
        
        elif ann_type == 'bbox':
            # 绘制边界框
            w = ann.get('width', 0.1) * width
            h = ann.get('height', 0.1) * height
            draw.rectangle([x, y, x + w, y + h], outline=rgb_color, width=3)
            # 绘制标签
            if label:
                text_bbox = draw.textbbox((x, y - 25), label, font=font)
                draw.rectangle(text_bbox, fill=rgb_color)
                draw.text((x, y - 25), label, fill=(255, 255, 255), font=font)
        
        elif ann_type == 'mask':
            # 绘制遮挡区域（完全不透明的遮罩）
            w = ann.get('width', 0.1) * width
            h = ann.get('height', 0.1) * height
            mask_color = hex_to_rgb(color) if color != '#ffffff' else (128, 128, 128)
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            # alpha 设置为 255，表示完全不透明
            overlay_draw.rectangle([x, y, x + w, y + h], fill=(*mask_color, 255))
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(img)
    
    return img


def prepare_annotated_image(image_path: str, annotations: List[Dict], apply_path_mapping: bool = True) -> str:
    """
    准备带标注的图像，返回 base64 编码
    
    Args:
        image_path: 原始图像路径
        annotations: 标注列表
        apply_path_mapping: 是否应用路径映射（用于 Docker 容器）
    
    Returns:
        base64 编码的图像 URL
    """
    if not annotations:
        return image_to_base64(image_path, apply_path_mapping=apply_path_mapping)
    
    if not HAS_PIL:
        print(f"警告: PIL 未安装，无法绘制标注，使用原图")
        return image_to_base64(image_path, apply_path_mapping=apply_path_mapping)
    
    annotated_img = draw_annotations_on_image(image_path, annotations, apply_path_mapping=apply_path_mapping)
    return pil_image_to_base64(annotated_img)


def extract_answer(response: str, valid_options: List[str], allow_multiple: bool = True) -> str:
    """
    从模型回答中提取答案选项
    
    Args:
        response: 模型的原始回答
        valid_options: 有效的选项列表，如 ['A', 'B', 'C', 'D']
        allow_multiple: 是否允许多选答案
    
    Returns:
        提取的答案（单个或多个，逗号分隔）
    """
    if not response:
        return ""
    
    response_clean = response.upper().strip()
    
    # 1. 完全匹配单个选项
    for opt in valid_options:
        if response_clean == opt:
            return opt
    
    # 2. 匹配逗号分隔的多选格式 (如 "A, B, C" 或 "A,B,C")
    comma_pattern = r'^([A-D](?:\s*,\s*[A-D])+)$'
    comma_match = re.match(comma_pattern, response_clean.replace(' ', '').replace(',', ', '))
    if comma_match:
        letters = [l.strip() for l in response_clean.split(',')]
        valid_letters = [l for l in letters if l in valid_options]
        if valid_letters:
            if allow_multiple:
                return ','.join(sorted(set(valid_letters)))
            else:
                return valid_letters[0]
    
    # 3. 对于单选题，查找结论性语句中的答案
    if not allow_multiple:
        # 查找 "answer is X", "answer: X", "therefore X", "conclusion: X" 等模式
        conclusion_patterns = [
            r'(?:THE\s+)?ANSWER\s*(?:IS|:)\s*\**\s*([A-D])\b',
            r'(?:THEREFORE|THUS|SO|HENCE)\s*,?\s*\**\s*([A-D])\b',
            r'CONCLUSION\s*:\s*\**\s*([A-D])\b',
            r'I\s+(?:CHOOSE|SELECT|PICK)\s+\**\s*([A-D])\b',
            r'\*\*([A-D])\*\*',  # **A** 格式
            r'^([A-D])\.',  # 以 "A." 开头
            r'^([A-D])\s*$',  # 纯字母结尾
        ]
        for pattern in conclusion_patterns:
            match = re.search(pattern, response_clean)
            if match and match.group(1) in valid_options:
                return match.group(1)
        
        # 查找最后一个独立出现的选项字母（结论通常在最后）
        last_match = None
        for match in re.finditer(r'(?:^|[^A-Z])([A-D])(?:[^A-Z]|$)', response_clean):
            if match.group(1) in valid_options:
                last_match = match.group(1)
        if last_match:
            return last_match
    
    # 4. 查找所有出现的选项字母
    found = []
    for opt in valid_options:
        pattern = r'(?:^|[^A-Z])(' + opt + r')(?:[^A-Z]|$)'
        if re.search(pattern, response_clean):
            found.append(opt)
    
    if found:
        if allow_multiple:
            return ','.join(sorted(found))
        else:
            return found[-1]  # 单选返回最后一个（可能是结论）
    
    # 5. 更宽松的匹配
    found = []
    for opt in valid_options:
        if opt in response_clean:
            found.append(opt)
    
    if found:
        if allow_multiple:
            return ','.join(sorted(found))
        else:
            return found[-1]
    
    # 6. 使用正则表达式查找所有独立的大写字母
    pattern = r'\b([A-Z])\b'
    matches = re.findall(pattern, response_clean)
    valid_matches = [m for m in matches if m in valid_options]
    
    if valid_matches:
        if allow_multiple:
            return ','.join(sorted(set(valid_matches)))
        else:
            return valid_matches[-1]  # 单选返回最后一个
    
    return ""


def format_options(options: Dict[str, str]) -> str:
    """格式化选项为字符串"""
    return '\n'.join([f"{k}: {v}" for k, v in options.items()])


# ==================== 相机名称映射 ====================

# 相机名称映射配置
CAMERA_NAME_MAPPING = {
    # cam01 -> exo01, cam02 -> exo02, ...
    'cam': 'exo',
    # aria01 视图保持为 ego
    'aria': 'ego',
}

def build_camera_mapping(view_names: List[str]) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    """
    构建相机名称映射关系
    
    将 cam01, cam02, cam03 等映射为 exo01, exo02, exo03
    处理编号不连续的情况（如只有 cam01, cam03, cam04）
    
    Args:
        view_names: 视图名称列表（包括 ego）
    
    Returns:
        (camera_to_exo, exo_to_camera, camera_list)
        - camera_to_exo: 原始相机名到新映射名的映射 {"cam01": "exo01", ...}
        - exo_to_camera: 新映射名到原始相机名的映射 {"exo01": "cam01", ...}
        - camera_list: 按编号排序的相机名列表 ["cam01", "cam03", ...]
    """
    # 提取所有 exo 相机（排除 ego/aria 视角），按数字排序
    exo_cameras = []
    for view_name in view_names:
        if view_name == 'ego' or view_name.startswith('aria'):
            continue
        if view_name.startswith('cam'):
            match = re.match(r'cam(\d+)', view_name)
            if match:
                cam_num = int(match.group(1))
                exo_cameras.append((cam_num, view_name))
    
    # 按相机编号排序
    exo_cameras.sort(key=lambda x: x[0])
    camera_list = [cam_name for _, cam_name in exo_cameras]
    
    # 构建映射关系：保留原始编号，如 cam01 -> exo01, cam03 -> exo03
    camera_to_exo = {}
    exo_to_camera = {}
    for cam_num, cam_name in exo_cameras:
        # 保留原始编号格式：cam01 -> exo01
        exo_name = f"exo{cam_num:02d}"
        camera_to_exo[cam_name] = exo_name
        exo_to_camera[exo_name] = cam_name
    
    return camera_to_exo, exo_to_camera, camera_list


def map_camera_names_in_text(text: str, camera_to_exo: Dict[str, str]) -> str:
    """
    将文本中的相机名称替换为新的映射名称
    
    Args:
        text: 原始文本
        camera_to_exo: 相机名称映射字典
    
    Returns:
        替换后的文本
    """
    result = text
    # 按照相机名长度降序排列，避免部分替换问题（如cam01被cam0替换）
    sorted_cameras = sorted(camera_to_exo.keys(), key=len, reverse=True)
    for original_cam in sorted_cameras:
        new_exo = camera_to_exo[original_cam]
        # 使用单词边界确保完整匹配
        pattern = r'\b' + re.escape(original_cam) + r'\b'
        result = re.sub(pattern, new_exo, result)
    return result


def map_options_camera_names(options: Dict[str, str], camera_to_exo: Dict[str, str]) -> Dict[str, str]:
    """
    将选项中的相机名称替换为新的映射名称
    
    Args:
        options: 原始选项字典
        camera_to_exo: 相机名称映射字典
    
    Returns:
        映射后的选项字典
    """
    mapped_options = {}
    for opt_key, opt_value in options.items():
        mapped_options[opt_key] = map_camera_names_in_text(opt_value, camera_to_exo)
    return mapped_options
