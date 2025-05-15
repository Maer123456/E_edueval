"""
Evaluator module for EngEdueval framework

This module provides evaluation capabilities for different educational task types.
It calculates metrics such as accuracy, BLEU score, and other relevant metrics
depending on the task type.
"""

import os
import sys
import json
import jsonlines
import re
import numpy as np
import math
from pathlib import Path
from collections import defaultdict
import logging
import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    logger.warning("NLTK is not available. BLEU score calculation will be skipped.")
    NLTK_AVAILABLE = False

try:
    from rouge import Rouge
    ROUGE_AVAILABLE = True
except ImportError:
    logger.warning("Rouge is not available. ROUGE score calculation will be skipped.")
    ROUGE_AVAILABLE = False

# 新增：openai包用于API调用
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI is not available. LLM scoring will be skipped.")
    OPENAI_AVAILABLE = False

class Evaluator:
    """Base evaluator class for educational evaluation tasks"""
    
    def __init__(self, result_file, output_file=None, openai_api_key=None, openai_base_url=None, openai_model=None):
        """
        Initialize the evaluator
        
        Args:
            result_file: Path to the result file (JSONL format)
            output_file: Path to the output evaluation file (optional)
            openai_api_key: API key for OpenAI API (optional)
            openai_base_url: Base URL for OpenAI API (optional)
            openai_model: Model name for OpenAI API (optional)
        """
        self.result_file = result_file
        self.output_file = output_file
        
 
        self.openai_api_key = openai_api_key 
        self.openai_base_url = openai_base_url 
        self.openai_model = openai_model 
        self.openai_client = None  
        
        # Load results
        self.results = self.load_results()
        
        # Determine task type
        self.task_type = self.determine_task_type()
        logger.info(f"Task type determined: {self.task_type}")
        
    def load_results(self):
        """
        Load results from the result file
        
        Returns:
            list: List of result dictionaries
        """
        if not os.path.exists(self.result_file):
            raise FileNotFoundError(f"Result file not found: {self.result_file}")
            
        try:
            with jsonlines.open(self.result_file, 'r') as reader:
                results = list(reader)
            logger.info(f"Loaded {len(results)} results from {self.result_file}")
            return results
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            raise
            
    def determine_task_type(self):
        """
        Determine the task type from the result file
        
        Returns:
            str: Task type
        """
        # If results contain task_type field, use it
        if self.results and "task_type" in self.results[0]:
            return self.results[0]["task_type"]
            
        # Otherwise infer from file path
        filename = os.path.basename(self.result_file).lower()
        
        # 新增：判断LLM评分相关任务
        if "qg_100" in filename:
            return "qg_llm_scoring"
        elif "teachingdesign_50" in filename:
            return "teachingdesign_llm_scoring"
        elif "writing_50" in filename:
            return "writing_llm_scoring"
            
        if "multiple_choice" in filename or "primary" in filename or "junior" in filename or "senior" in filename:
            return "multiple_choice"
        elif "essay" in filename or "writing" in filename:
            return "essay"
        elif "short_answer" in filename or "reading" in filename or "poetry" in filename:
            return "short_answer"
        elif "essay_grading" in filename:
            return "essay_grading"
        elif "question_generation" in filename:
            return "question_generation"
        elif "teaching_design" in filename:
            return "teaching_design"
        elif "conversation_classification" in filename or "dialogue_classification" in filename:
            return "conversation_classification"
        else:
            # Default to multiple_choice as the most common type
            logger.warning(f"Could not determine task type from filename {filename}. Using default.")
            return "multiple_choice"
            
    def evaluate(self):
        """
        Evaluate the results based on task type
        
        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"Evaluating {len(self.results)} results for task type: {self.task_type}")
        
        if self.task_type == "multiple_choice":
            metrics = self.evaluate_multiple_choice()
        elif self.task_type == "short_answer":
            metrics = self.evaluate_short_answer()
        elif self.task_type == "essay":
            metrics = self.evaluate_essay()
        elif self.task_type == "essay_grading":
            metrics = self.evaluate_essay_grading()
        elif self.task_type == "question_generation":
            metrics = self.evaluate_question_generation()
        elif self.task_type == "teaching_design":
            metrics = self.evaluate_teaching_design()
        elif self.task_type == "conversation_classification":
            metrics = self.evaluate_conversation_classification()
        elif self.task_type in ["qg_llm_scoring", "teachingdesign_llm_scoring", "writing_llm_scoring"]:
            metrics = self.evaluate_llm_scoring()
        else:
            logger.warning(f"Unknown task type: {self.task_type}. Using default evaluation.")
            metrics = self.evaluate_default()
            
        # Save evaluation results if output file is specified
        if self.output_file:
            self.save_evaluation(metrics)
            
        return metrics
        
    def evaluate_multiple_choice(self):
        """
        Evaluate multiple choice questions
        
        Returns:
            dict: Evaluation metrics
        """
        correct = 0
        total = 0
        per_subject_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for result in self.results:
            # Skip results without reference answer
            if not result.get("reference_answer"):
                continue
                
            reference = result["reference_answer"].strip().upper()
            # 如果参考答案为空，跳过
            if not reference:
                continue
                
            # 处理模型输出
            model_output = result["model_output"].strip().upper()
            
            # Extract first letter if needed
            if len(model_output) > 1:
                # Try to extract option letter
                choices = re.findall(r'[A-D]', model_output)
                if choices:
                    model_output = choices[0]
                else:
                    # 如果无法提取到选项字母，尝试其他提取方法
                    if re.search(r'选择?\s*([A-D])', model_output):
                        model_output = re.search(r'选择?\s*([A-D])', model_output).group(1)
                    elif re.search(r'答案[是为]?\s*([A-D])', model_output):
                        model_output = re.search(r'答案[是为]?\s*([A-D])', model_output).group(1)
                    elif re.search(r'[Tt]he answer is\s*([A-D])', model_output):
                        model_output = re.search(r'[Tt]he answer is\s*([A-D])', model_output).group(1)
                    else:
                        # 如果还是无法提取，使用第一个字符
                        model_output = model_output[0] if model_output else ""
            
            # Compare model output with reference answer
            if model_output == reference:
                correct += 1
                
            total += 1
            
            # 记录每个学科的指标
            subject = result.get("subject", "unknown")
            per_subject_metrics[subject]["total"] += 1
            if model_output == reference:
                per_subject_metrics[subject]["correct"] += 1
                
        # 计算总正确率
        accuracy = correct / total if total > 0 else 0
        
        # 计算每个学科的正确率
        subject_accuracy = {}
        for subject, metrics in per_subject_metrics.items():
            subject_accuracy[subject] = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
            
        return {
            "task_type": self.task_type,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "per_subject": subject_accuracy
        }
        
    def evaluate_short_answer(self):
        """
        Evaluate short answer questions
        
        Returns:
            dict: Evaluation metrics
        """
        if not NLTK_AVAILABLE:
            logger.warning("NLTK is not available. Using simple matching for short answers.")
            return self.evaluate_default()
            
        exact_match = 0
        total = 0
        rouge_scores = []
        bleu_scores = []
        
        # For BLEU score calculation
        smoothing = SmoothingFunction().method1
        
        # For ROUGE score calculation
        if ROUGE_AVAILABLE:
            rouge = Rouge()
        
        for result in self.results:
            # Skip results without reference answer
            if not result.get("reference_answer"):
                continue
                
            reference = result["reference_answer"].strip()
            model_output = result["model_output"].strip()
            
            # Skip empty outputs
            if not model_output or not reference:
                continue
                
            # Check exact match
            if model_output == reference:
                exact_match += 1
                
            # Calculate BLEU score
            try:
                reference_tokens = reference.split()
                model_tokens = model_output.split()
                bleu = sentence_bleu([reference_tokens], model_tokens, smoothing_function=smoothing)
                bleu_scores.append(bleu)
            except Exception as e:
                logger.warning(f"Failed to calculate BLEU score: {e}")
                
            # Calculate ROUGE score
            if ROUGE_AVAILABLE:
                try:
                    # 确保文本不为空
                    if len(model_output) > 0 and len(reference) > 0:
                        rouge_score = rouge.get_scores(model_output, reference)[0]
                        rouge_scores.append(rouge_score)
                except Exception as e:
                    logger.warning(f"Failed to calculate ROUGE score: {e}")
                
            total += 1
            
        # Calculate metrics
        exact_match_ratio = exact_match / total if total > 0 else 0
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
        
        # Calculate average ROUGE scores
        avg_rouge = {
            "rouge-1": {"f": 0, "p": 0, "r": 0},
            "rouge-2": {"f": 0, "p": 0, "r": 0},
            "rouge-l": {"f": 0, "p": 0, "r": 0}
        }
        
        if ROUGE_AVAILABLE and rouge_scores:
            for metric in ["rouge-1", "rouge-2", "rouge-l"]:
                for key in ["f", "p", "r"]:
                    scores = [score[metric][key] for score in rouge_scores]
                    avg_rouge[metric][key] = np.mean(scores)
        
        return {
            "task_type": self.task_type,
            "exact_match_ratio": exact_match_ratio,
            "bleu_score": avg_bleu,
            "rouge_scores": avg_rouge if ROUGE_AVAILABLE else "Not available",
            "exact_matches": exact_match,
            "total": total
        }
        
    def evaluate_essay(self):
        """
        Evaluate essays - this is more subjective and requires human evaluation
        
        Returns:
            dict: Evaluation metrics
        """
        total_essays = 0
        avg_length = 0
        
        for result in self.results:
            model_output = result["model_output"].strip()
            
            # Skip empty outputs
            if not model_output:
                continue
                
            # Calculate essay length
            avg_length += len(model_output.split())
            total_essays += 1
            
        # Calculate average essay length
        avg_length = avg_length / total_essays if total_essays > 0 else 0
        
        return {
            "task_type": self.task_type,
            "total_essays": total_essays,
            "average_length": avg_length,
            "note": "Essay evaluation requires human judgment for quality assessment"
        }
        
    def evaluate_essay_grading(self):
        """
        Evaluate essay grading accuracy
        
        Returns:
            dict: Evaluation metrics
        """
        valid_scores = 0
        total = 0
        score_diffs = []
        
        for result in self.results:
            # Skip results without reference score
            reference = result.get("reference_answer", "")
            if not reference:
                continue
                
            model_output = result["model_output"].strip()
            
            # Extract scores
            try:
                # Extract reference score
                ref_score_match = re.search(r'(\d+)', reference)
                ref_score = int(ref_score_match.group(1)) if ref_score_match else None
                
                # Extract model score
                model_score_match = re.search(r'(\d+)', model_output)
                model_score = int(model_score_match.group(1)) if model_score_match else None
                
                # Calculate score difference
                if ref_score is not None and model_score is not None:
                    score_diff = abs(ref_score - model_score)
                    score_diffs.append(score_diff)
                    valid_scores += 1
            except Exception as e:
                logger.warning(f"Failed to extract scores: {e}")
                
            total += 1
            
        # Calculate metrics
        avg_score_diff = np.mean(score_diffs) if score_diffs else float('inf')
        score_extraction_rate = valid_scores / total if total > 0 else 0
        
        return {
            "task_type": self.task_type,
            "average_score_difference": avg_score_diff,
            "score_extraction_rate": score_extraction_rate,
            "valid_scores": valid_scores,
            "total": total
        }
        
    def evaluate_question_generation(self):
        """
        Evaluate question generation - mostly qualitative
        
        Returns:
            dict: Evaluation metrics
        """
        total_questions = 0
        avg_length = 0
        
        for result in self.results:
            model_output = result["model_output"].strip()
            
            # Skip empty outputs
            if not model_output:
                continue
                
            # Calculate question length
            avg_length += len(model_output.split())
            total_questions += 1
            
        # Calculate average question length
        avg_length = avg_length / total_questions if total_questions > 0 else 0
        
        return {
            "task_type": self.task_type,
            "total_questions": total_questions,
            "average_length": avg_length,
            "note": "Question generation evaluation requires human judgment for quality assessment"
        }
        
    def evaluate_teaching_design(self):
        """
        Evaluate teaching design - mostly qualitative
        
        Returns:
            dict: Evaluation metrics
        """
        total_designs = 0
        avg_length = 0
        
        for result in self.results:
            model_output = result["model_output"].strip()
            
            # Skip empty outputs
            if not model_output:
                continue
                
            # Calculate design length
            avg_length += len(model_output.split())
            total_designs += 1
            
        # Calculate average design length
        avg_length = avg_length / total_designs if total_designs > 0 else 0
        
        return {
            "task_type": self.task_type,
            "total_designs": total_designs,
            "average_length": avg_length,
            "note": "Teaching design evaluation requires human judgment for quality assessment"
        }
        
    def evaluate_conversation_classification(self):
        """
        Evaluate conversation classification accuracy
        
        Returns:
            dict: Evaluation metrics
        """
        correct = 0
        total = 0
        per_category_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for result in self.results:
            # Skip results without reference answer
            if not result.get("reference_answer"):
                continue
                
            reference = result["reference_answer"].strip()
            model_output = result["model_output"].strip()
            
            # Extract category number
            try:
                # 提取参考答案中的类别编号
                ref_category_match = re.search(r'(\d+)', reference)
                ref_category = ref_category_match.group(1) if ref_category_match else None
                
                # 提取模型输出中的类别编号
                model_category_match = re.search(r'(\d+)', model_output)
                model_category = model_category_match.group(1) if model_category_match else None
                
                # 跳过无效的类别
                if not ref_category or not model_category:
                    continue
                    
                # 比较类别编号
                if model_category == ref_category:
                    correct += 1
                    per_category_metrics[ref_category]["correct"] += 1
                    
                total += 1
                per_category_metrics[ref_category]["total"] += 1
                
            except Exception as e:
                logger.warning(f"Failed to extract category: {e}")
                
        # 计算总正确率
        accuracy = correct / total if total > 0 else 0
        
        # 计算每个类别的正确率
        category_accuracy = {}
        for category, metrics in per_category_metrics.items():
            category_accuracy[category] = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
            
        return {
            "task_type": self.task_type,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "per_category": category_accuracy
        }
        
    def evaluate_default(self):
        """
        Default evaluation method for unspecified task types
        
        Returns:
            dict: Evaluation metrics
        """
        exact_match = 0
        total = 0
        
        for result in self.results:
            # Skip results without reference answer
            if not result.get("reference_answer"):
                continue
                
            reference = result["reference_answer"].strip()
            model_output = result["model_output"].strip()
            
            # Skip empty outputs
            if not model_output or not reference:
                continue
                
            # Check exact match
            if model_output == reference:
                exact_match += 1
                
            total += 1
            
        # Calculate exact match ratio
        exact_match_ratio = exact_match / total if total > 0 else 0
        
        return {
            "task_type": self.task_type,
            "exact_match_ratio": exact_match_ratio,
            "exact_matches": exact_match,
            "total": total,
            "note": "Default evaluation using exact matching"
        }
        
    def evaluate_llm_scoring(self):
        """使用LLM评分对创造类任务进行评估"""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI is not available. LLM scoring will be skipped.")
            return {"llm_score": 0.5}  # 默认分数
            
        # 如果从外部传入了客户端，优先使用
        if hasattr(self, 'openai_client') and self.openai_client:
            openai_client = self.openai_client
            logger.info(f"使用外部传入的OpenAI客户端，模型: {self.openai_model}")
        elif hasattr(self, 'openai_api_key') and hasattr(self, 'openai_base_url') and self.openai_api_key and self.openai_base_url:
            # 使用配置的API密钥和URL创建客户端
            openai_client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url
            )
            logger.info(f"使用配置的API参数创建OpenAI客户端，Base URL: {self.openai_base_url}")
        else:
            logger.warning(f"警告：未配置OpenAI Client，无法进行LLM评分")
            return {"llm_score": 0.5}  # 默认分数
        
        scores = []
        
        # 根据任务类型确定评分提示和评分维度
        if self.task_type == "teachingdesign_llm_scoring":
            prompt_template = """请按照以下评分标准，对这份教学设计进行评分（总分100分）：

教学设计内容：
{content}

评分标准：
教学目标（20分）：
明确性（10分）：教学目标是否清晰、具体，能够明确阐述学生在课程结束后应掌握的知识、技能和态度。
适切性（10分）：教学目标是否符合课程标准和学生的年龄特点及学习水平。
教学内容（20分）：
准确性（10分）：教学内容是否准确无误，是否符合学科知识体系。
相关性（10分）：教学内容是否与教学目标紧密相关，是否有助于学生达成学习目标。
教学方法（20分）：
多样性（10分）：是否采用了多种教学方法（如讲授、讨论、实验、案例分析等），以满足不同学生的学习需求。
适切性（10分）：所选教学方法是否适合教学内容和学生的认知水平，能否有效促进学生的学习。
教学过程（20分）：
逻辑性（10分）：教学过程是否逻辑清晰，环节之间过渡自然，是否有助于学生逐步理解和掌握知识。
完整性（10分）：教学过程是否包括导入、新课讲授、实践练习、总结评价等完整环节，是否合理分配了各环节的时间。
教学资源（10分）：
丰富性（5分）：是否充分考虑了多种教学资源（如教材、教具、多媒体、实验器材等）的使用，以增强教学效果。
适切性（5分）：所选教学资源是否适合教学内容和学生的实际情况，能否有效支持教学活动。
教学评价（10分）：
多样性（5分）：是否采用了多种评价方式（如课堂提问、作业、测验、项目作业等），全面评估学生的学习效果。
有效性（5分）：评价方式是否能够有效测量学生对教学目标的达成度，是否注重过程性评价与终结性评价相结合。
创新性（10分）：
独特性（5分）：教学设计是否有独特之处，是否采用了新颖的教学策略或方法，能够激发学生的学习兴趣。
实用性（5分）：创新的教学策略是否具有实际可操作性，是否能够在实际教学中有效实施。
可行性（10分）：
时间管理（5分）：教学设计是否考虑了实际教学时间的限制，各环节时间分配是否合理。
资源可行性（5分）：所设计的教学活动是否能够在现有的教学资源和条件下顺利实施。

请根据以上标准进行评分，并给出总分（满分100分）。请只返回分数，不要有任何解释。
"""
        elif self.task_type == "qg_llm_scoring":
            prompt_template = """请按照以下评分标准，对这个题目生成内容进行评分（总分100分）：

题目内容：
{content}

评分标准：
一、题目准确性（20分）
知识准确性（10分）：题目所涉及的知识点是否准确无误，是否符合学科的基本概念、原理和规律。
表述清晰性（10分）：题目表述是否清晰、准确，是否存在歧义或模糊之处，是否能够使学生明确题意。
二、题目难易程度（20分）
难度适切性（10分）：题目难度是否符合相应年级和学科的要求，是否能够区分不同水平的学生。
层次分明性（10分）：题目是否具有层次性，是否能够涵盖不同难度层次的知识点，是否有利于学生逐步提高。
三、题目实用性（15分）
教学相关性（8分）：题目是否与教学内容紧密相关，是否能够有效支持教学目标的实现。
可操作性（7分）：题目是否具有可操作性，是否能够在实际教学中有效实施，是否考虑到了教学时间和资源的限制。
四、题目创新性（15分）
题目新颖性（8分）：题目是否具有新颖性，是否能够激发学生的学习兴趣和创造力，是否与传统题目有所不同。
形式多样性（7分）：题目是否采用了多样化的形式，是否能够综合考查学生的不同能力和素质。
五、题目灵活性（10分）
思维灵活性（6分）：题目是否能够考查学生的灵活思维能力，是否鼓励学生从不同角度思考问题。
解法多样性（4分）：题目是否具有多种解法，是否能够培养学生的发散思维和创新能力。
六、题目综合性（10分）
知识综合度（6分）：题目是否能够综合考查多个知识点，是否能够促进学生对知识的系统理解和综合运用。
能力覆盖面（4分）：题目是否能够考查学生的多种能力，如记忆、理解、应用、分析、综合和评价等。
七、题目思政融入情况（10分）
思政元素贴合度（6分）：题目是否自然地融入了思政元素，是否与学科知识有机结合，是否生硬或牵强。
育人功能（4分）：题目是否具有积极的育人功能，是否能够引导学生树立正确的价值观和人生观。

请根据以上标准进行评分，并给出总分（满分100分）。请只返回分数，不要有任何解释。
"""
        elif self.task_type == "writing_llm_scoring":
            prompt_template = """请按照以下评分标准，对这篇写作内容进行评分（总分100分）：

写作内容：
{content}

评分标准：
一、内容完整性（20分）
主题明确性（10分）：文本是否紧扣主题，中心思想是否鲜明且贯穿始终，是否存在偏题或离题的现象。
内容丰富度（10分）：文本是否围绕主题展开了充分的论述或描述，是否提供了足够的细节、例证或论据来支撑观点或丰富情节。
二、结构逻辑性（20分）
整体连贯性（10分）：文本段落之间是否过渡自然，是否形成了紧密的逻辑联系，使读者能够顺畅地理解作者的思路。
段落合理性（10分）：每个段落是否都有明确的主题句或中心思想，句子之间的组织是否逻辑清晰，是否符合常见的写作结构（如总分总、分总等）。
三、语言表达能力（20分）
语法正确性（10分）：文本是否存在语法错误，如主谓一致、时态、语态、句子结构等方面的问题。
用词准确性（10分）：用词是否恰当、准确，是否能够精确地表达作者的意图，是否存在滥用、误用或不当的词汇选择。
四、创意与想象力（15分）
新颖性（8分）：文本在主题、观点、情节或表现手法上是否具有新颖性，是否能够给人耳目一新的感觉。
想象力（7分）：在描述性或叙事性文本中，是否展现了丰富的想象力，是否创造出了独特的场景、人物或情节。
五、写作风格与适应性（10分）
风格一致性（6分）：文本是否保持了一致的写作风格，包括语气、语调和表达方式等，是否符合目标读者的期望和需求。
读者适应性（4分）：文本是否能够根据目标读者的年龄、背景知识和兴趣等特点，选择合适的表达方式和内容深度，以提高文本的可读性和吸引力。
六、深度与洞察力（15分）
思考深度（6分）：文本是否对主题进行了深入的思考和分析，是否展现出了作者对问题的深刻理解和独到见解。
批判性思维（9分）：在论述性或评论性文本中，是否体现了批判性思维，是否能够对不同的观点或现象进行客观的评价和分析。

请根据以上标准进行评分，并给出总分（满分100分）。请只返回分数，不要有任何解释。
"""
        else:
            # 对于其他类型的任务，使用通用提示
            prompt_template = """请评估以下内容的质量，给出1-100的分数：

内容：
{content}

请根据内容的准确性、完整性、逻辑性、创新性等方面进行综合评分，满分100分。
请只返回分数，不要有任何解释。
"""
        
        # 评估所有样本
        logger.info(f"正在使用 {self.openai_model} 模型评估所有 {len(self.results)} 个样本...")
        
        for i, result in enumerate(self.results):
            prompt = result.get("input") or result.get("question", "")
            output = result.get("output") or result.get("model_output") or result.get("model_answer") or ""
            
            # 跳过无输出的情况
            if not output:
                logger.warning(f"样本 #{i+1} 无输出，跳过")
                continue
            
            try:
                # 构造评分内容
                if self.task_type in ["teachingdesign_llm_scoring", "qg_llm_scoring", "writing_llm_scoring"]:
                    # 使用详细的评分标准模板
                    prompt_content = prompt_template.format(content=output)
                    messages = [
                        {"role": "system", "content": "你是一位专业的教育评估专家，善于客观公正地评价内容质量。"},
                        {"role": "user", "content": prompt_content}
                    ]
                else:
                    # 使用简化的评分请求
                    messages = [
                        {"role": "system", "content": "你是一位专业的教育评估专家。请根据提供的题目和回答，给回答质量评分(1-10分)。评分应基于回答的准确性、完整性、逻辑性和创新性。"},
                        {"role": "user", "content": f"题目：{prompt}\n\n回答：{output}\n\n请给这个回答打分（1-10分），只需返回分数。"}
                    ]
                
                # 调用API获取评分
                try:
                    # 使用传入的客户端或创建的客户端
                    response = openai_client.chat.completions.create(
                        model=self.openai_model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=10
                    )
                    
                    # 提取分数
                    score_text = response.choices[0].message.content.strip()
                    score_nums = re.findall(r'\d+\.?\d*', score_text)
                    
                    if score_nums:
                        score = float(score_nums[0])
                        # 归一化到0-1区间
                        if self.task_type in ["teachingdesign_llm_scoring", "qg_llm_scoring", "writing_llm_scoring"]:
                            normalized_score = score / 100.0  # 这些任务使用100分制
                        else:
                            normalized_score = score / 10.0   # 其他任务使用10分制
                        scores.append(normalized_score)
                        logger.info(f"LLM评分: 样本 #{i+1}/{len(self.results)}: {score}分 -> {normalized_score:.4f}")
                    else:
                        logger.warning(f"无法从API响应中提取分数: {score_text}")
                except Exception as api_error:
                    logger.error(f"调用API时出错: {api_error}")
                
                # 避免API限流
                import time
                time.sleep(1.0)  # 增加等待时间，避免API限流
                
            except Exception as e:
                logger.error(f"LLM评分时出错 (样本 #{i+1}): {e}")
        
        # 计算平均分
        avg_score = sum(scores) / len(scores) if scores else 0
        logger.info(f"评分完成。共评估了 {len(scores)} 个有效样本，平均分: {avg_score:.4f}")
        
        return {"llm_score": avg_score}
        
    def save_evaluation(self, metrics):
        """
        Save evaluation metrics to file
        
        Args:
            metrics: Evaluation metrics
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # Add timestamp and result file info
        metrics["result_file"] = self.result_file
        metrics["timestamp"] = datetime.datetime.now().isoformat()
        
        # Save metrics
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            logger.info(f"Evaluation metrics saved to {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to save evaluation metrics: {e}")
            
def main():
    """Main entry point for evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model outputs for educational tasks")
    parser.add_argument("--result_file", type=str, required=True, help="Path to result file")
    parser.add_argument("--output_file", type=str, default=None, help="Path to output evaluation file")
    parser.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--openai_base_url", type=str, default=None, help="OpenAI API base URL")
    parser.add_argument("--openai_model", type=str, default="x1", help="OpenAI model name")
    
    args = parser.parse_args()
    
    # Generate default output file if not specified
    if not args.output_file:
        result_basename = os.path.basename(args.result_file)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(args.result_file), "evaluations")
        os.makedirs(output_dir, exist_ok=True)
        args.output_file = os.path.join(output_dir, f"eval_{result_basename}_{timestamp}.json")
    
    # Create evaluator and run evaluation
    evaluator = Evaluator(
        args.result_file, 
        args.output_file,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        openai_model=args.openai_model
    )
    metrics = evaluator.evaluate()
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(json.dumps(metrics, indent=2))
    
if __name__ == "__main__":
    main() 
