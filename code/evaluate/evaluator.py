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
from pathlib import Path
from collections import defaultdict
import logging

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

class Evaluator:
    """Base evaluator class for educational evaluation tasks"""
    
    def __init__(self, result_file, output_file=None):
        """
        Initialize the evaluator
        
        Args:
            result_file: Path to the result file (JSONL format)
            output_file: Path to the output evaluation file (optional)
        """
        self.result_file = result_file
        self.output_file = output_file
        
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
    import datetime
    
    parser = argparse.ArgumentParser(description="Evaluate model outputs for educational tasks")
    parser.add_argument("--result_file", type=str, required=True, help="Path to result file")
    parser.add_argument("--output_file", type=str, default=None, help="Path to output evaluation file")
    
    args = parser.parse_args()
    
    # Generate default output file if not specified
    if not args.output_file:
        result_basename = os.path.basename(args.result_file)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(args.result_file), "evaluations")
        os.makedirs(output_dir, exist_ok=True)
        args.output_file = os.path.join(output_dir, f"eval_{result_basename}_{timestamp}.json")
    
    # Create evaluator and run evaluation
    evaluator = Evaluator(args.result_file, args.output_file)
    metrics = evaluator.evaluate()
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(json.dumps(metrics, indent=2))
    
if __name__ == "__main__":
    main() 