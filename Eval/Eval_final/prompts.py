"""
评测任务的提示词模板
每个文件作为独立任务，有专门的 prompt

题型说明：
- 单选题 (Single Choice): 只能选择一个答案，回答格式为单个字母如 "A"
- 多选题 (Multiple Choice): 可以选择多个答案，回答格式为逗号分隔的字母如 "A, B, C"

任务题型汇总：
┌─────────────────────────────┬──────────┬───────┬────────┐
│ 任务名称                     │ 题型     │ 样本数 │ 选项数  │
├─────────────────────────────┼──────────┼───────┼────────┤
│ camera_wearer               │ 单选题   │ 99    │ 3-4    │
│ camera_wearer_type2         │ 单选题   │ 100   │ 4      │
│ ego_2_exo_visibility        │ 单选题   │ 80    │ 3      │
│ camera_relative_position    │ 单选题   │ 1512  │ 3-4    │
│ relative_distance           │ 单选题   │ 95    │ 4      │
│ object_relative_position    │ 单选题   │ 108   │ 4      │
│ object_correspondence       │ 单选题   │ 20    │ 3-4    │
│ object_prediction           │ 单选题   │ 26    │ 3-4    │
├─────────────────────────────┼──────────┼───────┼────────┤
│ view_movement_1             │ 多选题   │ 60    │ 4      │
│ view_movement_2             │ 多选题   │ 60    │ 4      │
│ view_movement_3             │ 单选题   │ 30    │ 4      │
│ view_movement_4             │ 单选题   │ 40    │ 4      │
│ view_movement_5             │ 单选题   │ 10    │ 4      │
│ view_movement_6             │ 单选题   │ 50    │ 4      │
├─────────────────────────────┼──────────┼───────┼────────┤
│ object_movement             │ 单选题   │ 40    │ 4      │
│ object_movement_1           │ 多选题   │ 60    │ 4      │
│ noise_collaboration         │ 不定项   │ -     │ 4      │ 按题：选最好=单选，选全部=多选
└─────────────────────────────┴──────────┴───────┴────────┘
"""

# ==================== 通用格式强调 ====================
SINGLE_CHOICE_FORMAT = """
========================================
CRITICAL: OUTPUT FORMAT REQUIREMENTS
========================================
This is a SINGLE CHOICE question. You MUST select exactly ONE answer.

CORRECT OUTPUT: A
CORRECT OUTPUT: B
CORRECT OUTPUT: C
CORRECT OUTPUT: D

WRONG (DO NOT DO THIS): A, B
WRONG (DO NOT DO THIS): The answer is A
WRONG (DO NOT DO THIS): Based on my analysis, A
WRONG (DO NOT DO THIS): I think A because...

STRICT RULES:
1. Your ENTIRE response must be exactly ONE letter
2. Do NOT output multiple letters or commas
3. Do NOT write any explanation or reasoning
4. Do NOT write any words, sentences, or punctuation
5. ONLY output: A or B or C or D
========================================"""

MULTIPLE_CHOICE_FORMAT = """
========================================
CRITICAL: OUTPUT FORMAT REQUIREMENTS
========================================
This is a MULTIPLE CHOICE question. You may select one or more answers.

CORRECT OUTPUT: A
CORRECT OUTPUT: A, B
CORRECT OUTPUT: B, C, D
CORRECT OUTPUT: A, B, C, D

WRONG (DO NOT DO THIS): The answers are A and B
WRONG (DO NOT DO THIS): Based on analysis, I choose A, B
WRONG (DO NOT DO THIS): A and B

STRICT RULES:
1. Output ONLY letter(s), separated by commas if multiple
2. Do NOT write any explanation or reasoning
3. Do NOT write any words or sentences
4. ONLY output letters and commas, nothing else
========================================"""


# ==================== Camera_Wearer ====================
CAMERA_WEARER_SYSTEM_PROMPT = f"""You are a visual assistant. Identify correspondences between cameras/people shown in ego and exo views.

Pay attention to the marked points with labels (A, B, C, etc.) in the images.
{SINGLE_CHOICE_FORMAT}"""

CAMERA_WEARER_USER_PROMPT = """Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:"""


# ==================== Camera_Wearer_Type2 ====================
CAMERA_WEARER_TYPE2_SYSTEM_PROMPT = f"""You are a visual assistant. Given an egocentric (ego) view image with a marked camera (bounding box), identify which exocentric (exo) view corresponds to that marked camera.

Image mapping:
- Image 1: ego-view (first-person perspective, may have a bounding box marking a camera)
- Image 2+: exo camera views in order (exo01, exo02, exo03, ...)
{SINGLE_CHOICE_FORMAT}"""

CAMERA_WEARER_TYPE2_USER_PROMPT = """Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:"""


# ==================== Ego2Exo_Visibility ====================
EGO_2_EXO_VISIBILITY_SYSTEM_PROMPT = f"""You are a visual assistant. Determine if an object visible in the egocentric (ego) view is also visible in the exocentric (exo) view.

Image mapping:
- Image 1: ego-view (first-person perspective)
- Image 2: exo-view (third-person perspective)
{SINGLE_CHOICE_FORMAT}"""

EGO_2_EXO_VISIBILITY_USER_PROMPT = """Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:"""


# ==================== Camera_Relative_Position ====================
CAMERA_RELATIVE_POSITION_SYSTEM_PROMPT = f"""You are a visual assistant. Given ego-view and multiple exo-view images, determine the spatial arrangement (clockwise or counterclockwise order) of exo cameras relative to the ego view.

Image mapping:
- Image 1: ego-view (first-person perspective)
- Image 2+: exo cameras in order (exo01, exo02, exo03, ...)

Note: The exo cameras are labeled with their original numbering (exo01, exo02, etc.). The question may ask about the order starting from a specific exo camera.
{SINGLE_CHOICE_FORMAT}"""

CAMERA_RELATIVE_POSITION_USER_PROMPT = """Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:"""


# ==================== Relative_Distance ====================
RELATIVE_DISTANCE_SYSTEM_PROMPT = f"""You are a visual assistant. Given ego-view and multiple exo-view images, determine the relative distance of marked objects/persons to the ego camera wearer.

Image mapping:
- Image 1: ego-view (first-person perspective)
- Image 2+: exo cameras in order (exo01, exo02, exo03, ...) with marked objects

Pay attention to the marked points with labels (A, B, C, D, etc.) in the exo-view images. Each label corresponds to an object or person. Determine which marked object/person is closest or farthest from the ego camera wearer based on the spatial relationships visible in the images.
{SINGLE_CHOICE_FORMAT}"""

RELATIVE_DISTANCE_USER_PROMPT = """Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:"""


# ==================== Object_Relative_Position ====================
OBJECT_RELATIVE_POSITION_SYSTEM_PROMPT = f"""You are a visual assistant. Given ego-view and exo-view images, the exo image has a marked object (point). Answer the multiple-choice question: what is the position of that marked object in the exo perspective relative to the ego perspective observer. Assuming ego perspective as the 12 o'clock direction. (e.g., ahead/behind, left/right, using clock positions).

Image mapping:
- Image 1: ego-view (first-person perspective)
- Image 2: exo-view (third-person perspective, with marked object)
{SINGLE_CHOICE_FORMAT}"""

OBJECT_RELATIVE_POSITION_USER_PROMPT = """Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:"""


# ==================== Object_Correspondence ====================
OBJECT_CORRESPONDENCE_SYSTEM_PROMPT = f"""You are a visual assistant. Match objects across different viewpoints.

One image shows a marked object (query), another shows candidate objects labeled A, B, C, etc. Identify which candidate corresponds to the query object.
{SINGLE_CHOICE_FORMAT}"""

OBJECT_CORRESPONDENCE_USER_PROMPT = """Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:"""


# ==================== Object_Prediction ====================
OBJECT_PREDICTION_SYSTEM_PROMPT = f"""You are a visual assistant. Predict what object is hidden under the masked/occluded region.

Use contextual clues from the surrounding scene to infer the hidden object.
{SINGLE_CHOICE_FORMAT}"""

OBJECT_PREDICTION_USER_PROMPT = """Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:"""


# ==================== View_Movement_1 ====================
VIEW_MOVEMENT_1_SYSTEM_PROMPT = f"""You are a visual assistant. Answer questions about which objects would NOT be visible after the ego camera wearer moves (turns, walks, etc.).

Image mapping:
- Image 1: ego-view (first-person perspective)
- Image 2+: exo-view(s) (third-person perspective, for reference)

The question asks which objects would NOT be visible in the ego view after a specific movement. You need to select ALL objects that would become invisible.
{MULTIPLE_CHOICE_FORMAT}"""

VIEW_MOVEMENT_1_USER_PROMPT = """Question: {question}

Options:
{options}

RESPOND WITH LETTER(S) ONLY (comma-separated if multiple):"""


# ==================== View_Movement_2 ====================
VIEW_MOVEMENT_2_SYSTEM_PROMPT = f"""You are a visual assistant. Answer questions about which objects would NOT be visible after the ego camera wearer moves (turns, walks, etc.).

Image mapping:
- Image 1: ego-view (first-person perspective)
- Image 2+: exo-view(s) (third-person perspective, for reference)

The question asks which objects would NOT be visible in the ego view after a specific movement. You need to select ALL objects that would become invisible.
{MULTIPLE_CHOICE_FORMAT}"""

VIEW_MOVEMENT_2_USER_PROMPT = """Question: {question}

Options:
{options}

RESPOND WITH LETTER(S) ONLY (comma-separated if multiple):"""


# ==================== View_Movement_3 ====================
VIEW_MOVEMENT_3_SYSTEM_PROMPT = f"""You are a visual assistant. Answer questions about the relative clock-direction of an object after the ego camera wearer moves.

Image mapping:
- Image 1: ego-view (first-person perspective)
- Image 2: exo-view (third-person perspective, for reference)

After the ego camera wearer performs a movement (e.g., turns left/right by a certain angle), determine the clock-direction of the specified object relative to the wearer's new facing direction.

12 o'clock = the wearer's facing direction after movement.
{SINGLE_CHOICE_FORMAT}"""

VIEW_MOVEMENT_3_USER_PROMPT = """Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:"""


# ==================== View_Movement_4 ====================
VIEW_MOVEMENT_4_SYSTEM_PROMPT = f"""You are a visual assistant. Answer questions about how many objects of a certain type can be seen after the ego camera wearer moves.

Image mapping:
- Image 1: ego-view (first-person perspective)
- Image 2: exo-view (third-person perspective, for reference)

After the ego camera wearer performs a movement (e.g., turns left/right by a certain angle), count how many objects of the specified type would be visible in the ego view.
{SINGLE_CHOICE_FORMAT}"""

VIEW_MOVEMENT_4_USER_PROMPT = """Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:"""


# ==================== View_Movement_5 ====================
VIEW_MOVEMENT_5_SYSTEM_PROMPT = f"""You are a visual assistant. Answer questions about the relative clock-direction of an exo camera after the ego camera wearer rotates.

Image mapping:
- Image 1: ego-view (first-person perspective)
- Image 2: exo-view (from the exo camera in question)

After the ego camera wearer rotates by a certain angle, determine the clock-direction of the exo camera relative to the observer's new facing direction.

12 o'clock = the observer's new facing direction after rotation.
{SINGLE_CHOICE_FORMAT}"""

VIEW_MOVEMENT_5_USER_PROMPT = """Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:"""


# ==================== View_Movement_6 ====================
VIEW_MOVEMENT_6_SYSTEM_PROMPT = f"""You are a visual assistant. Answer questions about which object would be closest to the ego camera wearer after a movement.

Image mapping:
- Image 1: ego-view (first-person perspective)
- Image 2: exo-view (third-person perspective, for reference)

After the ego camera wearer performs a movement (e.g., walks forward, turns and takes steps), determine which of the listed objects would be closest to the wearer.
{SINGLE_CHOICE_FORMAT}"""

VIEW_MOVEMENT_6_USER_PROMPT = """Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:"""


# ==================== Object_Movement ====================
OBJECT_MOVEMENT_SYSTEM_PROMPT = f"""You are a visual assistant. Answer questions about an object's relative clock-direction after it is hypothetically moved to a new position.

Image mapping:
- Image 1: ego-view (first-person perspective)
- Image 2+: exo-view(s) (third-person perspective, for reference)

The question describes a hypothetical movement of an object. Determine what clock-direction that object would be at relative to the ego observer after the movement.

12 o'clock = the observer's facing direction.
{SINGLE_CHOICE_FORMAT}"""

OBJECT_MOVEMENT_USER_PROMPT = """Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:"""


# ==================== Object_Movement_1 ====================
OBJECT_MOVEMENT_1_SYSTEM_PROMPT = f"""You are a visual assistant. Answer questions about which exo views can still see an object after it is hypothetically moved to a new position.

Image mapping:
- Image 1: ego-view (first-person perspective)
- Image 2+: exo-views in order (exo01, exo02, exo03, ...)

The question describes a hypothetical movement of an object. Determine which exo view(s) can still see the object after it is moved. You need to select ALL exo views that can see the object.
{MULTIPLE_CHOICE_FORMAT}"""

OBJECT_MOVEMENT_1_USER_PROMPT = """Question: {question}

Options:
{options}

RESPOND WITH LETTER(S) ONLY (comma-separated if multiple):"""


# ==================== Noise_Collaboration ====================
# 题型：不定项选择 (Variable: 有的题目选“最好的”，有的选“所有的”)
# 数据：Noise_Collaboration.json；题目中 "select the clearest" = 单选，"select all relevant" = 多选
# 内容：ego 或 exo 有噪声时，根据问题选最清晰视角或所有相关视角
NOISE_COLLABORATION_SYSTEM_PROMPT = f"""You are a visual assistant. The egocentric (ego) view or some exocentric (exo) views may contain noise (e.g., motion blur, overexposure, underexposure) that makes it harder to answer the question.

Your task depends on the question wording:
- If the question asks for the CLEAREST or BEST single viewpoint (e.g., "select the clearest viewpoint"), choose exactly ONE option.
- If the question asks for ALL RELEVANT viewpoints (e.g., "select all relevant viewpoints"), choose ALL options that are relevant.

Image mapping:
- Image 1: ego-view (first-person perspective, may contain noise)
- Image 2+: exo-1, exo-2, exo-3, ... (candidate exo cameras in order; some may contain noise)

Output format: one letter for single-choice questions (e.g., A), or comma-separated letters for multiple (e.g., A, B or A, B, D). Do not add any explanation.
{MULTIPLE_CHOICE_FORMAT}"""

NOISE_COLLABORATION_USER_PROMPT = """Question: {question}

Options:
{options}

Respond with only the letter(s) of your answer (one letter, or comma-separated if the question asks for all relevant):"""


# ==================== View_Selection ====================
# 题型：多选题 (Multiple Choice)
# 样本数：18，选项数：4
# 内容：结合 ego 视角，判断哪些 exo 视角可以帮助确定某个目标
VIEW_SELECTION_SYSTEM_PROMPT = f"""You are a visual assistant. Given an egocentric (ego) view and multiple exocentric (exo) views, determine which exo view(s) can help achieve a specific goal (e.g., determine object position, identify a person, etc.).

Image mapping:
- Image 1: ego-view (first-person perspective)
- Image 2+: exo-views in order (exo-1, exo-2, exo-3, exo-4, ...)

Each option represents one exo view. You need to select ALL exo views that can provide helpful information for the specified task.
{MULTIPLE_CHOICE_FORMAT}"""

VIEW_SELECTION_USER_PROMPT = """Question: {question}

Options:
{options}

RESPOND WITH LETTER(S) ONLY (comma-separated if multiple):"""


# ==================== View_Selection_2 ====================
# 题型：单选题 (Single Choice)
# 样本数：53，选项数：2
# 内容：判断是否需要多视角协作来回答问题
VIEW_SELECTION_2_SYSTEM_PROMPT = f"""You are a visual assistant. Given an egocentric (ego) view and multiple exocentric (exo) views, determine whether multi-view collaboration is needed to answer a specific question.

Image mapping:
- Image 1: ego-view (first-person perspective)
- Image 2+: exo-views (third-person perspectives)

The question asks whether you need collaboration from exo views to answer a question about the ego view scene. Consider:
- Can the question be answered using only the ego view?
- Or do you need additional information from exo views?

If the question involves selecting which viewpoints are helpful:
- "y" means the viewpoint provides relevant information
- "n" means the viewpoint is irrelevant
{SINGLE_CHOICE_FORMAT}"""

VIEW_SELECTION_2_USER_PROMPT = """Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:"""


# ==================== View_Selection_3 ====================
# 题型：单选题 (Single Choice)，与 View_Selection 同结构但答案为单选
VIEW_SELECTION_3_SYSTEM_PROMPT = f"""You are a visual assistant. Given an egocentric (ego) view and multiple exocentric (exo) views, determine which exo view can help achieve a specific goal (e.g., determine object position, identify a person, etc.).

Image mapping:
- Image 1: ego-view (first-person perspective)
- Image 2+: exo-views in order (exo-1, exo-2, exo-3, exo-4, ...)

Select exactly ONE exo view that best provides the needed information.
{SINGLE_CHOICE_FORMAT}"""

VIEW_SELECTION_3_USER_PROMPT = """Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:"""


# ==================== 提示词获取函数 ====================
def get_prompts(task: str) -> tuple:
    """
    获取指定任务的提示词模板
    
    Args:
        task: 任务名称
    
    Returns:
        (system_prompt, user_prompt_template) 元组
    
    题型说明：
    - 单选题: camera_wearer, camera_wearer_type2, ego_2_exo_visibility, camera_relative_position,
              relative_distance, object_relative_position, object_correspondence,
              object_prediction, view_movement_3, view_movement_4, view_movement_5,
              view_movement_6, object_movement
    - 多选题: view_movement_1, view_movement_2, object_movement_1
    - 不定项(按题): noise_collaboration（题目“选最好的”=单选，“选所有的”=多选）
    """
    prompts = {
        # 基础任务（全部单选题）
        'camera_wearer': (CAMERA_WEARER_SYSTEM_PROMPT, CAMERA_WEARER_USER_PROMPT),
        'camera_wearer_type2': (CAMERA_WEARER_TYPE2_SYSTEM_PROMPT, CAMERA_WEARER_TYPE2_USER_PROMPT),
        'ego_2_exo_visibility': (EGO_2_EXO_VISIBILITY_SYSTEM_PROMPT, EGO_2_EXO_VISIBILITY_USER_PROMPT),
        'camera_relative_position': (CAMERA_RELATIVE_POSITION_SYSTEM_PROMPT, CAMERA_RELATIVE_POSITION_USER_PROMPT),
        'relative_distance': (RELATIVE_DISTANCE_SYSTEM_PROMPT, RELATIVE_DISTANCE_USER_PROMPT),
        'object_relative_position': (OBJECT_RELATIVE_POSITION_SYSTEM_PROMPT, OBJECT_RELATIVE_POSITION_USER_PROMPT),
        'object_correspondence': (OBJECT_CORRESPONDENCE_SYSTEM_PROMPT, OBJECT_CORRESPONDENCE_USER_PROMPT),
        'object_prediction': (OBJECT_PREDICTION_SYSTEM_PROMPT, OBJECT_PREDICTION_USER_PROMPT),
        
        # View_Movement 系列（混合：1/2是多选，3/4/5/6是单选）
        'view_movement_1': (VIEW_MOVEMENT_1_SYSTEM_PROMPT, VIEW_MOVEMENT_1_USER_PROMPT),  # 多选
        'view_movement_2': (VIEW_MOVEMENT_2_SYSTEM_PROMPT, VIEW_MOVEMENT_2_USER_PROMPT),  # 多选
        'view_movement_3': (VIEW_MOVEMENT_3_SYSTEM_PROMPT, VIEW_MOVEMENT_3_USER_PROMPT),  # 单选
        'view_movement_4': (VIEW_MOVEMENT_4_SYSTEM_PROMPT, VIEW_MOVEMENT_4_USER_PROMPT),  # 单选
        'view_movement_5': (VIEW_MOVEMENT_5_SYSTEM_PROMPT, VIEW_MOVEMENT_5_USER_PROMPT),  # 单选
        'view_movement_6': (VIEW_MOVEMENT_6_SYSTEM_PROMPT, VIEW_MOVEMENT_6_USER_PROMPT),  # 单选
        
        # Object_Movement 系列（混合：基础是单选，_1是多选）
        'object_movement': (OBJECT_MOVEMENT_SYSTEM_PROMPT, OBJECT_MOVEMENT_USER_PROMPT),    # 单选
        'object_movement_1': (OBJECT_MOVEMENT_1_SYSTEM_PROMPT, OBJECT_MOVEMENT_1_USER_PROMPT),  # 多选
        
        # View_Selection 任务
        'view_selection': (VIEW_SELECTION_SYSTEM_PROMPT, VIEW_SELECTION_USER_PROMPT),  # 多选
        'view_selection_2': (VIEW_SELECTION_2_SYSTEM_PROMPT, VIEW_SELECTION_2_USER_PROMPT),  # 单选
        'view_selection_3': (VIEW_SELECTION_3_SYSTEM_PROMPT, VIEW_SELECTION_3_USER_PROMPT),  # 单选
        
        # Noise_Collaboration（不定项：题目中“选最好的”=单选，“选所有的”=多选）
        'noise_collaboration': (NOISE_COLLABORATION_SYSTEM_PROMPT, NOISE_COLLABORATION_USER_PROMPT),
    }
    
    if task not in prompts:
        raise ValueError(f"Unknown task: {task}")
    
    return prompts[task]
