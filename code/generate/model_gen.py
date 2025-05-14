import json
import os
import torch
import jsonlines
import sys
from pathlib import Path
import re
from tqdm import tqdm

# Adding path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 添加课堂对话分类标签系统
DIALOGUE_LABEL_SYSTEM = {
    1: {"name": "基础知识", "description": "参照教科书或教师以前教过的知识，可以判断出正确或错误的答案。"},
    2: {"name": "个人信息", "description": "说话人生活中的事件，不被认为是其他参与者知道的；个人对神情或艺术作品等的想象性反应；发言人对个人关系或情况的个人看法。"},
    3: {"name": "分析", "description": "将一个整体抽象地分离其组成部分，以研究这些部分及其关系；它涉及推理，使知识变得明朗和易于理解。"},
    4: {"name": "归纳", "description": "通过对详细事实进行推理而形成一般概念的过程；它涉及到归纳推理和思想的发展，目的是对信息进行以外的问题作出回应。"},
    5: {"name": "推断与迁移", "description": "对可能性的考虑，超越了目前的知识水平；但基于理论或事实依据。"},
    6: {"name": "回应与拓展", "description": "这里的问题涉及到别人之前的回答被动态地用来吸收；可以通过评论来实现，明确强调之前的回应，并在此基础上发展。"},
    7: {"name": "认同", "description": "对陈述的明确接受或同意。"},
    8: {"name": "质疑", "description": "怀疑、完全/部分不同意，质疑或拒绝一个陈述，包括一个简单的\"no\"回答，当它表示拒绝一个想法，而不是回答一个问题。"},
    9: {"name": "指导", "description": "根据学生的学习速度和认知水平提供帮助和支持；老师对如何组织学习活动出明确的指导，并要求其他人做出相应的反应。"}
}

class ModelGenerator:
    """
    Base class for generating model outputs for educational evaluation tasks.
    Standardized implementation that can be used as a benchmark for different models.
    """
    def __init__(self, task_path, model_path, model_name, device="0", is_few_shot=False, 
                 few_shot_path=None, is_vllm=False, tensor_parallel_size=1, 
                 gpu_memory_utilization=0.9, output_file=None):
        """
        Args:
            task_path: Path to the task data file
            model_path: Path to the model weights
            model_name: Name of the model
            device: GPU device to use
            is_few_shot: Whether to use few-shot examples
            few_shot_path: Path to few-shot examples
            is_vllm: Whether to use vLLM for acceleration
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
            output_file: File path to save outputs
        """
        # Convert task_path to absolute path if it's not already
        if not os.path.isabs(task_path):
            task_path = os.path.abspath(task_path)

        # Ensure the file exists
        if not os.path.exists(task_path):
            raise FileNotFoundError(f"Task file not found: {task_path}")
            
        self.task_path = task_path
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.is_few_shot = is_few_shot
        self.few_shot_path = few_shot_path
        
        if is_few_shot and few_shot_path is None:
            raise ValueError("Few-shot path must be provided when is_few_shot=True")
            
        self.is_vllm = is_vllm
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        
        self.output_file = output_file
        
        # Determine task type based on file path
        self.task_type = self._determine_task_type()
        
    def _determine_task_type(self):
        """
        Determine the task type from the file name using standardized rules
        """
        # Determine task type based on filename
        filename = os.path.basename(self.task_path)
        filepath = self.task_path.lower()  # 转为小写以便于匹配
        task_type = "unknown" # Default task type
        
        # 先检查文件路径中是否包含Ethics目录，这是优先级最高的
        if "/Ethics/" in filepath or "\\Ethics\\" in filepath:
            task_type = "multiple_choice"
            print(f"Determined task type for {filename} (based on ethics directory): {task_type}")
            return task_type
        
        # 然后按照文件名匹配规则处理其他类型
        if "Junior" in filename or "Primary" in filename or "Senior" in filename:
            task_type = "multiple_choice"
        elif "Writing" in filename:
            task_type = "essay"
        elif "Reading" in filename or "Poetry" in filename:
            task_type = "short_answer"
        elif "Essay" in filename:
            task_type = "essay_grading"
        # 新增题目生成任务类型
        elif "Question" in filename:
            task_type = "question_generation"
        # 新增教学设计任务类型
        elif "Teaching" in filename:
            task_type = "teaching_design"
        # 新增课堂对话分类任务类型
        elif "Classroom" in filename or "Dialogue" in filename:
            task_type = "conversation_classification"
        else:
            # 伦理相关文件名识别
            if "Ethics" in filename:
                task_type = "multiple_choice"
            
        # Debug print to confirm task type
        print(f"Determined task type for {filename}: {task_type}")
        if task_type == "unknown":
            print(f"Warning: Could not determine task type for file {filename}. Please check filename conventions.")
            
        return task_type
        
    def model_init_vllm(self):
        """Initialize model using vLLM for faster inference"""
        try:
            from vllm import LLM, SamplingParams
            model = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=True,
                dtype="half",
                enforce_eager=True
            )
            tokenizer = model.get_tokenizer()
            return model, tokenizer
        except ImportError:
            print("vLLM not installed. Please install with 'pip install vllm'")
            raise
    
    def model_init(self):
        """
        Initialize model and tokenizer - each model subclass must implement this
        """
        raise NotImplementedError("Subclasses must implement model_init")
    
    def load_few_shot_examples(self):
        """
        加载few-shot示例，根据任务类型从few_shot目录读取对应的示例
        
        Returns:
            str: 格式化的few-shot示例文本
        """
        if not self.is_few_shot or not self.few_shot_path:
            return ""
            
        if not os.path.exists(self.few_shot_path):
            print(f"Few-shot example file not found: {self.few_shot_path}")
            return ""
            
        # 读取few-shot示例文件
        try:
            with jsonlines.open(self.few_shot_path) as f:
                examples = list(f)[:3]  # 默认使用前3个示例
                
            if not examples:
                print(f"No examples found in {self.few_shot_path}")
                return ""
                
            examples_text = "以下是几个示例：\n\n"
            
            # 根据不同的任务类型构造不同的示例格式
            for i, ex in enumerate(examples, 1):
                if self.task_type == "multiple_choice":
                    options = ex.get("options", [])
                    if options:
                        options_text = "\n".join([f"{opt['id']}. {opt['content']}" for opt in options])
                        examples_text += f"示例{i}:\n问题: {ex['ques_content']}\n{options_text}\n答案: {ex['ques_answer']}\n\n"
                    else:
                        examples_text += f"示例{i}:\n问题: {ex['ques_content']}\n答案: {ex['ques_answer']}\n\n"
                        
                elif self.task_type == "short_answer":
                    examples_text += f"示例{i}:\n问题: {ex['ques_content']}\n答案: {ex['ques_answer']}\n\n"
                    
                elif self.task_type == "essay":
                    # 限制作文长度避免prompt过长
                    answer_preview = ex.get('ques_answer', '')[:300]
                    if len(ex.get('ques_answer', '')) > 300:
                        answer_preview += "..."
                    examples_text += f"示例{i}:\n题目: {ex['ques_content']}\n作文: {answer_preview}\n\n"
                    
                elif self.task_type == "essay_grading":
                    essay_title = ex.get("question", "")
                    essay_content = ex.get("ques_answer", "")    
                    score = ex.get("score", "")
                    
                    examples_text += f"示例{i}:\n作文题目: {essay_title}\n作文内容: {essay_content}\n评分: {score}\n\n"
                    
                elif self.task_type == "question_generation":
                    # 题目生成任务
                    grade = ex.get("grade", "")
                    knowledge_point = ex.get("knowledge_point", "")
                    task_description = ex.get("task_description", "")
                    answer = ex.get("answer", "")
                    
                    examples_text += f"示例{i}:\n年级: {grade}\n知识点: {knowledge_point}\n任务描述: {task_description}\n生成的题目: {answer}\n\n"
                    
                elif self.task_type == "teaching_design":
                    # 教学设计任务
                    grade = ex.get("grade", "")
                    subject = ex.get("subject", "")
                    topic = ex.get("topic", "")
                    teaching_design_requirements = ex.get("teaching_design_requirements", "")
                    answer = ex.get("answer", "")
                    
                    examples_text += f"示例{i}:\n年级: {grade}\n学科: {subject}\n主题: {topic}\n教学设计要求: {teaching_design_requirements}\n教学设计: {answer}\n\n"
                    
                elif self.task_type == "conversation_classification":
                    # 对话分类任务
                    dialogue = ex.get("dialogue", "")
                    label = ex.get("label", "")
                    
                    examples_text += f"示例{i}:\n对话内容: {dialogue}\n类别: {label}\n\n"
                    
                else:
                    # 默认格式
                    examples_text += f"示例{i}:\n问题: {ex.get('ques_content', '')}\n答案: {ex.get('ques_answer', '')}\n\n"
            
            examples_text += "现在，请回答下面的问题：\n"
            return examples_text
            
        except Exception as e:
            print(f"Error loading few-shot examples: {e}")
            return ""
    
    def truncate_prompt(self, prompt, tokenizer, max_length=2048):
        """Truncate prompt to fit within model context window"""
        encoded = tokenizer.encode(prompt)
        if len(encoded) > max_length:
            print(f"Input too long, truncating to {max_length} tokens")
            half_length = max_length // 2
            # Keep beginning and end, cut out middle
            tokens = encoded[:half_length] + encoded[-half_length:]
            prompt = tokenizer.decode(tokens)
        return prompt
    
    def prepare_prompts(self, max_items=500, offset=0):
        """
        Prepare standardized prompts for all models based on task type
        Returns: list of prompts, list of answers, list of questions, list of subjects
        
        Args:
            max_items: Maximum number of items to process in a single batch
            offset: Starting offset for data processing (for batch processing)
        """
        questions = []
        prompts = []
        answers = []
        subjects = []
        
        # 加载few-shot示例
        few_shot_examples = ""
        if self.is_few_shot and self.few_shot_path:
            print(f"Loading few-shot examples from {self.few_shot_path}")
            try:
                few_shot_examples = self.load_few_shot_examples()
                print(f"Loaded few-shot examples: {len(few_shot_examples.split())} words")
            except Exception as e:
                print(f"Error loading few-shot examples: {e}")
                few_shot_examples = ""
        
        # Ensure task_path exists
        if not os.path.exists(self.task_path):
            raise FileNotFoundError(f"Task file not found: {self.task_path}\nCurrent working directory: {os.getcwd()}")
        
        print(f"Opening task file: {self.task_path}")
        
        # 计算文件中的数据条数
        item_count = 0
        with jsonlines.open(self.task_path) as f:
            for _ in f:
                item_count += 1
        
        print(f"Total items in file: {item_count}")
        
        if offset >= item_count:
            print(f"Warning: Offset {offset} exceeds total items {item_count}")
            return [], [], [], []
            
        # 根据offset和max_items计算实际要处理的数据范围
        if max_items is None:
            # If max_items is None, process all items from the offset
            end_pos = item_count
        else:
            # Otherwise, calculate the end position based on max_items
            end_pos = min(offset + max_items, item_count)

        print(f"Processing items from {offset} to {end_pos-1} (total: {end_pos-offset})")
        
        # 读取数据
        processed_count = 0
        current_pos = 0
        with jsonlines.open(self.task_path) as f:
            for item in f:
                # 跳过offset之前的数据
                if current_pos < offset:
                    current_pos += 1
                    continue
                    
                # 如果达到了结束位置，停止处理
                if current_pos >= end_pos:
                    break
                    
                current_pos += 1
                
                subject = item.get("subject", "")
                subjects.append(subject)
                
                # 特别处理课堂对话分类任务
                if self.task_type == "conversation_classification":
                    dialogue = item.get("dialogue", "")
                    label = item.get("label", "")
                    
                    # 保存问题和答案，用于结果保存
                    question = f"对话内容：{dialogue}"
                    questions.append(question)
                    answers.append(label)
                    
                    # 构造系统提示词和用户提示词
                    label_descriptions = []
                    for label_num, label_info in DIALOGUE_LABEL_SYSTEM.items():
                        description = f"{label_num}. {label_info['name']}: {label_info['description']}"
                        label_descriptions.append(description)
                    
                    system_prompt = "你是一个专业的教育对话分类器。请根据以下9种类别对给定的对话内容进行分类：\n"
                    system_prompt += "\n".join(label_descriptions)
                    system_prompt += "\n请只返回分类的数字标签（1-9）。"
                    
                    user_prompt = f"对话内容：{dialogue}\n\n请问这段对话属于哪个类别？只需返回分类的数字标签（1-9）。"
                    
                    # 添加few-shot示例
                    if few_shot_examples:
                        user_prompt = few_shot_examples + "\n" + user_prompt
                    
                    # 组合提示词
                    prompt = f"{system_prompt}\n\n{user_prompt}"
                    prompts.append(prompt)
                
                # 处理选择题
                elif self.task_type == "multiple_choice":
                    # Find question content in either format
                    question_content = item.get("ques_content", item.get("question", ""))
                    answer = item.get("ques_answer", item.get("answer", ""))
                    
                    # Check for options in different formats
                    options = item.get("options", item.get("choices", []))
                    
                    # Format options to string
                    if isinstance(options, list):
                        if options and isinstance(options[0], dict):
                            # Dict-style options
                            options_text = "\n".join([f"{opt.get('id', chr(65+i))}. {opt.get('content', opt.get('text', ''))}" 
                                                    for i, opt in enumerate(options)])
                        else:
                            # Simple list options
                            options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
                    elif isinstance(options, dict):
                        # Dictionary options
                        options_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
                    else:
                        options_text = ""
                    
                    # Build question text
                    if options_text:
                        question_with_options = f"{question_content}\n\n{options_text}"
                    else:
                        question_with_options = question_content
                    
                    questions.append(question_with_options)
                    answers.append(answer)
                    
                    # 构造提示词
                    prompt = f"请回答下面的选择题，直接给出选项字母即可（如A、B、C或D）:\n\n{question_with_options}"
                    
                    # 添加few-shot示例
                    if few_shot_examples:
                        prompt = few_shot_examples + "\n" + prompt
                    
                    prompts.append(prompt)
                
                # 处理简答题
                elif self.task_type == "short_answer":
                    question_content = item.get("ques_content", item.get("question", ""))
                    answer = item.get("ques_answer", item.get("answer", ""))
                    
                    questions.append(question_content)
                    answers.append(answer)
                    
                    # 构造提示词
                    prompt = f"请回答下面的问题：\n\n{question_content}"
                    
                    # 添加few-shot示例
                    if few_shot_examples:
                        prompt = few_shot_examples + "\n" + prompt
                    
                    prompts.append(prompt)
                
                # 处理作文题
                elif self.task_type == "essay":
                    question_content = item.get("ques_content", item.get("question", ""))
                    # 作文题可能没有标准答案
                    answer = item.get("ques_answer", item.get("answer", ""))
                    
                    questions.append(question_content)
                    answers.append(answer)
                    
                    # 构造提示词
                    prompt = f"请根据以下题目写一篇作文：\n\n{question_content}"
                    
                    # 添加few-shot示例
                    if few_shot_examples:
                        prompt = few_shot_examples + "\n" + prompt
                    
                    prompts.append(prompt)
                
                # 处理作文评分
                elif self.task_type == "essay_grading":
                    # 使用question字段作为作文题目，ques_answer字段作为学生作答内容，score字段作为参考分数
                    essay_title = item.get("question", "")
                    essay_content = item.get("ques_answer", item.get("essay", ""))
                    # 获取参考分数，如果存在，否则为空字符串
                    score = item.get("score", "")
                    
                    # 组合题目和内容作为问题
                    question_info = f"作文题目：{essay_title}\n\n作文内容：\n{essay_content}"
                    questions.append(question_info)
                    answers.append(score)
                    
                    # 构造提示词
                    prompt = f"请对下面的作文进行评分（满分100分）不需要解释理由：\n\n作文题目：{essay_title}\n\n作文内容：\n{essay_content}"
                    
                    # 添加few-shot示例
                    if few_shot_examples:
                        prompt = few_shot_examples + "\n" + prompt
                    
                    prompts.append(prompt)
                
                # 处理题目生成任务
                elif self.task_type == "question_generation":
                    task_description = item.get("task_description", "")
                    knowledge_point = item.get("knowledge_point", "")
                    grade = item.get("grade", "")
                    
                    # 组合问题信息
                    question_info = f"年级: {grade}\n知识点: {knowledge_point}\n任务: {task_description}"
                    questions.append(question_info)
                    
                    # 获取标准答案（如果有）
                    answer = item.get("answer", "")
                    answers.append(answer)
                    
                    # 构造提示词
                    prompt = f"请根据以下要求，设计一道高质量的教育评估题目：\n\n年级: {grade}\n知识点: {knowledge_point}\n任务描述: {task_description}"
                    
                    # 根据任务描述添加特定指导
                    if "选择题" in task_description:
                        prompt += "\n\n请设计一道选择题，包括题干和选项，注明正确答案。题目应符合教育标准，难度适中，选项设计合理。"
                    elif "填空题" in task_description:
                        prompt += "\n\n请设计一道填空题，将需要填写的地方用下划线或括号标注，并提供参考答案。"
                    elif "计算题" in task_description or "解答题" in task_description:
                        prompt += "\n\n请设计一道解答题，提供题目描述和完整的参考答案，包括解题步骤。"
                    elif "简答题" in task_description or "论述题" in task_description:
                        prompt += "\n\n请设计一道简答题，提供问题和评分要点，确保题目能够测试学生的理解能力和表达能力。"
                    
                    # 添加few-shot示例
                    if few_shot_examples:
                        prompt = few_shot_examples + "\n" + prompt
                    
                    prompts.append(prompt)
                
                # 处理教学设计任务
                elif self.task_type == "teaching_design":
                    grade = item.get("grade", "")
                    subject = item.get("subject", "")
                    topic = item.get("topic", "")
                    teaching_design_requirements = item.get("teaching_design_requirements", "")
                    
                    # 组合问题信息
                    question_info = f"年级: {grade}\n学科: {subject}\n主题: {topic}\n教学设计要求: {teaching_design_requirements}"
                    questions.append(question_info)
                    
                    # 获取标准答案（如果有）
                    answer = item.get("answer", "")
                    answers.append(answer)
                    
                    # 构造提示词
                    prompt = f"请根据以下要求，设计一个详细的教学方案：\n\n年级: {grade}\n学科: {subject}\n主题: {topic}\n教学设计要求: {teaching_design_requirements}"
                    
                    # 添加few-shot示例
                    if few_shot_examples:
                        prompt = few_shot_examples + "\n" + prompt
                    
                    prompts.append(prompt)
                
                # 其他未知任务类型（使用默认简单提示词格式）
                else:
                    question_content = item.get("ques_content", item.get("question", item.get("input", "")))
                    answer = item.get("ques_answer", item.get("answer", item.get("reference_answer", "")))
                    
                    questions.append(question_content)
                    answers.append(answer)
                    
                    prompt = f"请回答以下问题：\n\n{question_content}"
                    
                    # 添加few-shot示例
                    if few_shot_examples:
                        prompt = few_shot_examples + "\n" + prompt
                    
                    prompts.append(prompt)
                
                processed_count += 1
        
        print(f"Prepared {processed_count} prompts")
        return prompts, answers, questions, subjects 

    def extract_choice_answer(self, text):
        """
        从模型输出文本中提取选择题答案
        
        Args:
            text: 模型输出文本
            
        Returns:
            str: 提取的选择题答案（一个字母）
        """
        # 先尝试直接提取单个字母答案
        choices = re.findall(r'[A-Da-d]', text)
        if choices:
            return choices[0].upper()
            
        # 尝试匹配"答案(是)X"模式
        answer_match = re.search(r'答案(?:是|为)?\s*[：:]?\s*([A-Da-d])', text)
        if answer_match:
            return answer_match.group(1).upper()
            
        # 尝试匹配"选X"模式
        choice_match = re.search(r'选\s*([A-Da-d])', text)
        if choice_match:
            return choice_match.group(1).upper()
            
        # 尝试匹配英文表述
        if re.search(r'[Tt]he answer is\s*([A-Da-d])', text):
            return re.search(r'[Tt]he answer is\s*([A-Da-d])', text).group(1).upper()
            
        if re.search(r'[Cc]hoose\s*([A-Da-d])', text):
            return re.search(r'[Cc]hoose\s*([A-Da-d])', text).group(1).upper()
            
        # 如果上述方法都失败，返回第一行的第一个大写字母
        lines = text.strip().split('\n')
        for line in lines:
            for char in line:
                if char in "ABCDabcd":
                    return char.upper()
                    
        # 如果还是没有找到，返回原文
        return text
        
    def cleanup_essay_output(self, text):
        """
        清理作文输出，移除提示词相关内容
        
        Args:
            text: 模型输出文本
            
        Returns:
            str: 清理后的文本
        """
        # 移除常见的前缀
        prefixes_to_remove = [
            "我根据题目，写了一篇作文：", 
            "以下是我根据题目写的作文：", 
            "下面是我根据题目写的作文：",
            "根据题目，我写了以下作文：",
            "以下是作文：",
            "下面是作文：",
            "作文：",
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                
        return text
        
    def extract_score(self, text):
        """
        从评分结果中提取分数
        
        Args:
            text: 评分文本
            
        Returns:
            str: 提取的分数
        """
        # 尝试提取数字分数
        score_patterns = [
            r'(\d{1,3})分',  # 中文分数表示
            r'(\d{1,3})\s*[/／]\s*100',  # 分数/100形式
            r'分数[：:]\s*(\d{1,3})',  # "分数："形式
            r'评分[：:]\s*(\d{1,3})',  # "评分："形式
            r'得分[：:]\s*(\d{1,3})',  # "得分："形式
            r'[sS]core[：:]\s*(\d{1,3})',  # "Score："形式
            r'(\d{1,3})\s*(?:points|marks)',  # "85 points"形式
            r'(\d{1,3})',  # 直接返回的分数
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, text)
            if match:
                score = match.group(1)
                try:
                    score_num = int(score)
                    # 确保分数在合理范围内
                    if 0 <= score_num <= 100:
                        return score
                except ValueError:
                    pass
        
        # 如果没有找到有效分数，尝试提取文本中的第一个数字
        numbers = re.findall(r'\d+', text)
        if numbers:
            first_num = int(numbers[0])
            if 0 <= first_num <= 100:
                return numbers[0]
                
        # 如果仍然没有找到分数，返回原文
        return text
        
    def post_process_output(self, output_text, task_type):
        """
        处理模型输出，根据任务类型进行后处理
        
        Args:
            output_text: 模型原始输出文本
            task_type: 任务类型
            
        Returns:
            str: 处理后的输出
        """
        # 确保输出文本是字符串
        if not isinstance(output_text, str):
            return str(output_text)
            
        # 去除首尾空白
        cleaned_text = output_text.strip()
        
        # 根据任务类型进行处理
        if task_type == "multiple_choice":
            # 选择题提取选项字母
            return self.extract_choice_answer(cleaned_text)
            
        elif task_type == "essay":
            # 作文清理
            return self.cleanup_essay_output(cleaned_text)
            
        elif task_type == "essay_grading":
            # 作文评分提取分数
            return self.extract_score(cleaned_text)
            
        elif task_type == "conversation_classification":
            # 对话分类提取类别编号
            digits = re.findall(r'\d', cleaned_text)
            if digits:
                # 获取第一个1-9的数字
                for digit in digits:
                    if '1' <= digit <= '9':
                        return digit
            
            # 查找类别名称对应的编号
            for label_num, label_info in DIALOGUE_LABEL_SYSTEM.items():
                if label_info["name"] in cleaned_text:
                    return str(label_num)
                    
            return cleaned_text
            
        elif task_type in ["question_generation", "teaching_design"]:
            # 清理生成的题目或教学设计
            return self.cleanup_generation_output(cleaned_text)
            
        else:
            # 其他类型直接返回清理后的文本
            return cleaned_text
            
    def cleanup_generation_output(self, text):
        """
        清理生成的内容，移除提示词相关内容
        
        Args:
            text: 模型输出文本
            
        Returns:
            str: 清理后的文本
        """
        # 移除常见的前缀
        prefixes_to_remove = [
            "以下是我设计的", 
            "下面是我设计的", 
            "我设计了",
            "以下是设计的",
            "这是我的设计：",
            "设计如下：",
            "请根据以下要求",
            "好的，",
            "根据要求，",
            "好的",
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                
        return text
        
    def generate_output(self, tokenizer, model, batch_size=1, max_items=None, offset=0):
        """
        使用模型生成输出
        
        Args:
            tokenizer: 分词器对象
            model: 模型对象
            batch_size: 批处理大小
            max_items: 最大处理项目数
            offset: 处理起始位置
            
        Returns:
            tuple: (保存的结果文件路径, 完成的项目数)
        """
        print(f"Generating outputs for {self.task_path} with model {self.model_name}")
        
        # 准备提示词
        prompts, answers, questions, subjects = self.prepare_prompts(max_items, offset)
        
        if not prompts:
            print("No prompts generated, exiting.")
            return None, 0
            
        print(f"Generating responses for {len(prompts)} prompts")
        outputs = []
        
        # 使用tqdm显示进度
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch_prompts = prompts[i:i+batch_size]
            
            # 因为不同模型有不同的生成方法，所以这里提供一个通用的接口
            # 实际模型需要自己实现具体的生成逻辑
            try:
                if self.is_vllm:
                    # 使用vLLM进行生成
                    from vllm import SamplingParams
                    
                    # 配置采样参数
                    sampling_params = SamplingParams(
                        temperature=0.7,
                        top_p=0.95,
                        max_tokens=512
                    )
                    
                    # 批量生成
                    result_outputs = model.generate(batch_prompts, sampling_params)
                    
                    # 提取生成的文本
                    for j, output in enumerate(result_outputs):
                        output_text = output.outputs[0].text
                        current_index = i + j
                        
                        # 后处理输出
                        processed_output = self.post_process_output(output_text, self.task_type)
                        outputs.append(processed_output)
                else:
                    # 使用普通模型进行生成
                    # 每个具体模型类需要自己实现生成逻辑
                    for j, prompt in enumerate(batch_prompts):
                        # 截断过长的提示词
                        truncated_prompt = self.truncate_prompt(prompt, tokenizer)
                        
                        # 编码输入
                        inputs = tokenizer(truncated_prompt, return_tensors="pt").to(model.device)
                        
                        # 生成输出（每个具体模型需要自己实现）
                        with torch.no_grad():
                            output_ids = model.generate(
                                **inputs, 
                                max_new_tokens=512,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.95
                            )
                        
                        # 解码输出
                        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                        
                        # 去除原始提示词部分
                        if truncated_prompt in output_text:
                            output_text = output_text.replace(truncated_prompt, "").strip()
                        
                        # 后处理输出
                        processed_output = self.post_process_output(output_text, self.task_type)
                        outputs.append(processed_output)
                        
            except Exception as e:
                print(f"Error during generation for batch starting at index {i}: {e}")
                # 对于失败的批次，填充空白结果
                outputs.extend([""] * len(batch_prompts))
        
        print(f"Generated {len(outputs)} outputs")
        
        # 保存结果
        output_file = self.save_results(questions, outputs, answers, subjects)
        
        return output_file, len(outputs)
    
    def infer_category_from_filename(self, filename):
        """
        从文件名推断评估分类
        
        Args:
            filename: 文件名
            
        Returns:
            str: 推断的分类名称
        """
        filename = filename.lower()
        
        # 从目录结构和文件名推断分类
        if "/memory/" in filename.lower() or "\\memory\\" in filename.lower() or "知识" in filename or "knowledge" in filename.lower():
            return "memory_recall"
        elif "/understanding/" in filename.lower() or "\\understanding\\" in filename.lower() or "理解" in filename or "reading" in filename.lower():
            return "reading_comprehension"
        elif "/application/" in filename.lower() or "\\application\\" in filename.lower() or "应用" in filename:
            return "application"
        elif "/reasoning/" in filename.lower() or "\\reasoning\\" in filename.lower() or "推理" in filename or "logic" in filename.lower():
            return "reasoning"
        elif "/creativity/" in filename.lower() or "\\creativity\\" in filename.lower() or "创造" in filename or "writing" in filename.lower():
            return "creativity"
        elif "/ethics/" in filename.lower() or "\\ethics\\" in filename.lower() or "伦理" in filename or "moral" in filename.lower():
            return "ethics"
        else:
            # 默认分类
            return "general"
    
    def save_results(self, questions, outputs, answers, subjects=None):
        """
        保存评估结果到文件
        
        Args:
            questions: 问题列表
            outputs: 模型输出列表
            answers: 标准答案列表
            subjects: 学科列表，可选
            
        Returns:
            str: 保存的文件路径
        """
        # 确定输出文件路径
        if self.output_file:
            output_file = self.output_file
        else:
            # 从任务路径生成输出路径
            basename = os.path.basename(self.task_path)
            model_name_safe = self.model_name.replace('/', '_').replace(' ', '_')
            
            # 创建输出目录
            output_dir = os.path.join(os.path.dirname(self.task_path), "..", "outputs", model_name_safe)
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成输出文件名
            output_file = os.path.join(output_dir, f"{model_name_safe}_{basename}")
            
        print(f"Saving results to {output_file}")
        
        # 推断评估分类
        category = self.infer_category_from_filename(self.task_path)
        
        # 保存结果
        with jsonlines.open(output_file, mode='w') as writer:
            for i in range(len(questions)):
                # 准备结果数据
                result = {
                    "id": i,
                    "question": questions[i],
                    "model_output": outputs[i],
                    "reference_answer": answers[i] if i < len(answers) else "",
                    "model_name": self.model_name,
                    "task_type": self.task_type,
                    "category": category
                }
                
                # 添加学科信息（如果有）
                if subjects and i < len(subjects) and subjects[i]:
                    result["subject"] = subjects[i]
                    
                # 写入文件
                writer.write(result)
                
        print(f"Saved {len(questions)} results to {output_file}")
        return output_file 