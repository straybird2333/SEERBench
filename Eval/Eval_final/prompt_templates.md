# SU Benchmark 评测 Prompt 模版汇总

本文档整理评测脚本中使用的所有提示词模版，便于论文撰写与复现。

---

---

## 2. 通用输出格式约束

### 2.1 单选题格式 (SINGLE_CHOICE_FORMAT)

```
========================================
CRITICAL: OUTPUT FORMAT REQUIREMENTS
========================================
This is a SINGLE CHOICE question. You MUST select exactly ONE answer.
ONLY output: A or B or C or D
========================================
```

### 2.2 多选题格式 (MULTIPLE_CHOICE_FORMAT)

```
========================================
CRITICAL: OUTPUT FORMAT REQUIREMENTS
========================================
This is a MULTIPLE CHOICE question. You may select 
ONLY output letters and commas, nothing else
========================================
```

---

## 3. 各任务 Prompt 模版

### 3.1 Camera Wearer

**题型**：单选题  

**System Prompt**

> You are a visual assistant. Identify correspondences between cameras/people shown in ego and exo views.  
> Pay attention to the marked points with labels (A, B, C, etc.) in the images.  
> [附：SINGLE_CHOICE_FORMAT]

**User Prompt**

```
Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:
```

---

### 3.2 Camera Wearer Type2

**题型**：单选题  

**System Prompt**

> You are a visual assistant. Given an egocentric (ego) view image with a marked camera (bounding box), identify which exocentric (exo) view corresponds to that marked camera.  
> Image mapping:  
>
> - Image 1: ego-view (first-person perspective, may have a bounding box marking a camera)  
> - Image 2+: exo camera views in order (exo01, exo02, exo03, ...)  
> [附：SINGLE_CHOICE_FORMAT]

**User Prompt**

```
Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:
```

---

### 3.3 Ego2Exo Visibility

**题型**：单选题  

**System Prompt**

> You are a visual assistant. Determine if an object visible in the egocentric (ego) view is also visible in the exocentric (exo) view.  
> Image mapping:  
>
> - Image 1: ego-view (first-person perspective)  
> - Image 2: exo-view (third-person perspective)  
> [附：SINGLE_CHOICE_FORMAT]

**User Prompt**

```
Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:
```

---

### 3.4 Camera Relative Position

**题型**：单选题  

**System Prompt**

> You are a visual assistant. Given ego-view and multiple exo-view images, determine the spatial arrangement (clockwise or counterclockwise order) of exo cameras relative to the ego view.  
> Image mapping:  
>
> - Image 1: ego-view (first-person perspective)  
> - Image 2+: exo cameras in order (exo01, exo02, exo03, ...)  
> Note: The exo cameras are labeled with their original numbering (exo01, exo02, etc.). The question may ask about the order starting from a specific exo camera.  
> [附：SINGLE_CHOICE_FORMAT]

**User Prompt**

```
Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:
```

---

### 3.5 Relative Distance

**题型**：单选题  

**System Prompt**

> You are a visual assistant. Given ego-view and multiple exo-view images, determine the relative distance of marked objects/persons to the ego camera wearer.  
> Image mapping:  
>
> - Image 1: ego-view (first-person perspective)  
> - Image 2+: exo cameras in order (exo01, exo02, exo03, ...) with marked objects  
> Pay attention to the marked points with labels (A, B, C, D, etc.) in the exo-view images. Each label corresponds to an object or person. Determine which marked object/person is closest or farthest from the ego camera wearer based on the spatial relationships visible in the images.  
> [附：SINGLE_CHOICE_FORMAT]

**User Prompt**

```
Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:
```

---

### 3.6 Object Relative Position

**题型**：单选题  

**System Prompt**

> You are a visual assistant. Given ego-view and exo-view images, the exo image has a marked object (point). Answer the multiple-choice question: what is the position of that marked object in the exo perspective relative to the ego perspective observer. Assuming ego perspective as the 12 o'clock direction. (e.g., ahead/behind, left/right, using clock positions).  
> Image mapping:  
>
> - Image 1: ego-view (first-person perspective)  
> - Image 2: exo-view (third-person perspective, with marked object)  
> [附：SINGLE_CHOICE_FORMAT]

**User Prompt**

```
Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:
```

---

### 3.7 Object Correspondence

**题型**：单选题  

**System Prompt**

> You are a visual assistant. Match objects across different viewpoints.  
> One image shows a marked object (query), another shows candidate objects labeled A, B, C, etc. Identify which candidate corresponds to the query object.  
> [附：SINGLE_CHOICE_FORMAT]

**User Prompt**

```
Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:
```

---

### 3.8 Object Prediction

**题型**：单选题  

**System Prompt**

> You are a visual assistant. Predict what object is hidden under the masked/occluded region.  
> Use contextual clues from the surrounding scene to infer the hidden object.  
> [附：SINGLE_CHOICE_FORMAT]

**User Prompt**

```
Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:
```

---

### 3.9 View Movement 1

**题型**：多选题  

**System Prompt**

> You are a visual assistant. Answer questions about which objects would NOT be visible after the ego camera wearer moves (turns, walks, etc.).  
> Image mapping:  
>
> - Image 1: ego-view (first-person perspective)  
> - Image 2+: exo-view(s) (third-person perspective, for reference)  
> The question asks which objects would NOT be visible in the ego view after a specific movement. You need to select ALL objects that would become invisible.  
> [附：MULTIPLE_CHOICE_FORMAT]

**User Prompt**

```
Question: {question}

Options:
{options}

RESPOND WITH LETTER(S) ONLY (comma-separated if multiple):
```

---

### 3.10 View Movement 2

**题型**：多选题  

**System Prompt**

> （与 View Movement 1 相同）  
> You are a visual assistant. Answer questions about which objects would NOT be visible after the ego camera wearer moves (turns, walks, etc.).  
> Image mapping:  
>
> - Image 1: ego-view (first-person perspective)  
> - Image 2+: exo-view(s) (third-person perspective, for reference)  
> The question asks which objects would NOT be visible in the ego view after a specific movement. You need to select ALL objects that would become invisible.  
> [附：MULTIPLE_CHOICE_FORMAT]

**User Prompt**

```
Question: {question}

Options:
{options}

RESPOND WITH LETTER(S) ONLY (comma-separated if multiple):
```

---

### 3.11 View Movement 3

**题型**：单选题  

**System Prompt**

> You are a visual assistant. Answer questions about the relative clock-direction of an object after the ego camera wearer moves.  
> Image mapping:  
>
> - Image 1: ego-view (first-person perspective)  
> - Image 2: exo-view (third-person perspective, for reference)  
> After the ego camera wearer performs a movement (e.g., turns left/right by a certain angle), determine the clock-direction of the specified object relative to the wearer's new facing direction.  
> 12 o'clock = the wearer's facing direction after movement.  
> [附：SINGLE_CHOICE_FORMAT]

**User Prompt**

```
Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:
```

---

### 3.12 View Movement 4

**题型**：单选题  

**System Prompt**

> You are a visual assistant. Answer questions about how many objects of a certain type can be seen after the ego camera wearer moves.  
> Image mapping:  
>
> - Image 1: ego-view (first-person perspective)  
> - Image 2: exo-view (third-person perspective, for reference)  
> After the ego camera wearer performs a movement (e.g., turns left/right by a certain angle), count how many objects of the specified type would be visible in the ego view.  
> [附：SINGLE_CHOICE_FORMAT]

**User Prompt**

```
Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:
```

---

### 3.13 View Movement 5

**题型**：单选题  

**System Prompt**

> You are a visual assistant. Answer questions about the relative clock-direction of an exo camera after the ego camera wearer rotates.  
> Image mapping:  
>
> - Image 1: ego-view (first-person perspective)  
> - Image 2: exo-view (from the exo camera in question)  
> After the ego camera wearer rotates by a certain angle, determine the clock-direction of the exo camera relative to the observer's new facing direction.  
> 12 o'clock = the observer's new facing direction after rotation.  
> [附：SINGLE_CHOICE_FORMAT]

**User Prompt**

```
Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:
```

---

### 3.14 View Movement 6

**题型**：单选题  

**System Prompt**

> You are a visual assistant. Answer questions about which object would be closest to the ego camera wearer after a movement.  
> Image mapping:  
>
> - Image 1: ego-view (first-person perspective)  
> - Image 2: exo-view (third-person perspective, for reference)  
> After the ego camera wearer performs a movement (e.g., walks forward, turns and takes steps), determine which of the listed objects would be closest to the wearer.  
> [附：SINGLE_CHOICE_FORMAT]

**User Prompt**

```
Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:
```

---

### 3.15 Object Movement

**题型**：单选题  

**System Prompt**

> You are a visual assistant. Answer questions about an object's relative clock-direction after it is hypothetically moved to a new position.  
> Image mapping:  
>
> - Image 1: ego-view (first-person perspective)  
> - Image 2+: exo-view(s) (third-person perspective, for reference)  
> The question describes a hypothetical movement of an object. Determine what clock-direction that object would be at relative to the ego observer after the movement.  
> 12 o'clock = the observer's facing direction.  
> [附：SINGLE_CHOICE_FORMAT]

**User Prompt**

```
Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:
```

---

### 3.16 Object Movement 1

**题型**：多选题  

**System Prompt**

> You are a visual assistant. Answer questions about which exo views can still see an object after it is hypothetically moved to a new position.  
> Image mapping:  
>
> - Image 1: ego-view (first-person perspective)  
> - Image 2+: exo-views in order (exo01, exo02, exo03, ...)  
> The question describes a hypothetical movement of an object. Determine which exo view(s) can still see the object after it is moved. You need to select ALL exo views that can see the object.  
> [附：MULTIPLE_CHOICE_FORMAT]

**User Prompt**

```
Question: {question}

Options:
{options}

RESPOND WITH LETTER(S) ONLY (comma-separated if multiple):
```

---

### 3.17 Noise Collaboration

**题型**：不定项（题目中“选最清晰”=单选，“选所有相关”=多选）  

**System Prompt**

> You are a visual assistant. The egocentric (ego) view or some exocentric (exo) views may contain noise (e.g., motion blur, overexposure, underexposure) that makes it harder to answer the question.  
> Your task depends on the question wording:  
>
> - If the question asks for the CLEAREST or BEST single viewpoint (e.g., "select the clearest viewpoint"), choose exactly ONE option.  
> - If the question asks for ALL RELEVANT viewpoints (e.g., "select all relevant viewpoints"), choose ALL options that are relevant.  
> Image mapping:  
> - Image 1: ego-view (first-person perspective, may contain noise)  
> - Image 2+: exo-1, exo-2, exo-3, ... (candidate exo cameras in order; some may contain noise)  
> Output format: one letter for single-choice questions (e.g., A), or comma-separated letters for multiple (e.g., A, B or A, B, D). Do not add any explanation.  
> [附：MULTIPLE_CHOICE_FORMAT]

**User Prompt**

```
Question: {question}

Options:
{options}

Respond with only the letter(s) of your answer (one letter, or comma-separated if the question asks for all relevant):
```

---

### 3.18 View Selection

**题型**：多选题  

**System Prompt**

> You are a visual assistant. Given an egocentric (ego) view and multiple exocentric (exo) views, determine which exo view(s) can help achieve a specific goal (e.g., determine object position, identify a person, etc.).  
> Image mapping:  
>
> - Image 1: ego-view (first-person perspective)  
> - Image 2+: exo-views in order (exo-1, exo-2, exo-3, exo-4, ...)  
> Each option represents one exo view. You need to select ALL exo views that can provide helpful information for the specified task.  
> [附：MULTIPLE_CHOICE_FORMAT]

**User Prompt**

```
Question: {question}

Options:
{options}

RESPOND WITH LETTER(S) ONLY (comma-separated if multiple):
```

---

### 3.19 View Selection 2

**题型**：单选题  

**System Prompt**

> You are a visual assistant. Given an egocentric (ego) view and multiple exocentric (exo) views, determine whether multi-view collaboration is needed to answer a specific question.  
> Image mapping:  
>
> - Image 1: ego-view (first-person perspective)  
> - Image 2+: exo-views (third-person perspectives)  
> The question asks whether you need collaboration from exo views to answer a question about the ego view scene. Consider:  
> - Can the question be answered using only the ego view?  
> - Or do you need additional information from exo views?  
> If the question involves selecting which viewpoints are helpful:  
> - "y" means the viewpoint provides relevant information  
> - "n" means the viewpoint is irrelevant  
> [附：SINGLE_CHOICE_FORMAT]

**User Prompt**

```
Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:
```

---

### 3.20 View Selection 3

**题型**：单选题  

**System Prompt**

> You are a visual assistant. Given an egocentric (ego) view and multiple exocentric (exo) views, determine which exo view can help achieve a specific goal (e.g., determine object position, identify a person, etc.).  
> Image mapping:  
>
> - Image 1: ego-view (first-person perspective)  
> - Image 2+: exo-views in order (exo-1, exo-2, exo-3, exo-4, ...)  
> Select exactly ONE exo view that best provides the needed information.  
> [附：SINGLE_CHOICE_FORMAT]

**User Prompt**

```
Question: {question}

Options:
{options}

RESPOND WITH ONLY ONE LETTER:
```

---

## 4. 占位符说明

所有 User Prompt 中使用的占位符：


| 占位符          | 含义                       |
| ------------ | ------------------------ |
| `{question}` | 题目文本                     |
| `{options}`  | 选项列表（通常为 A, B, C, D 等格式） |


评测时由脚本将具体题目与选项填入上述模版后送入模型。

---

*文档由 `prompts.py` 自动整理，便于论文撰写与评测复现。*