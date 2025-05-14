"""
Main entry point for EngEdueval framework

This script provides a command-line interface for running different tasks
in the EngEdueval framework.
"""

import os
import sys
import argparse
import logging
import json
import jsonlines
import importlib
from EngEdueval.code.generate.model_gen import ModelGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('edubenchmark.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='EngEdueval - English Educational Evaluation Framework')
    
    # Required arguments
    parser.add_argument('--task_path', type=str, required=True,
                        help='Path to the task file (JSONL format)')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path or name of the model to use')
    parser.add_argument('--model_name', type=str, default="default",
                        help='Display name for the model (used in logging and output files)')
    parser.add_argument('--model_class', type=str, default=None,
                        help='Custom model class to use for generation (e.g., "models.llama.LlamaGenerator")')
    parser.add_argument('--device', type=str, default="0",
                        help='Device ID for GPU (e.g., "0" or "0,1,2,3")')
                        
    # Few-shot parameters
    parser.add_argument('--few_shot', action='store_true',
                        help='Use few-shot prompting')
    parser.add_argument('--few_shot_path', type=str, default=None,
                        help='Path to few-shot examples file')
                        
    # vLLM parameters
    parser.add_argument('--vllm', action='store_true',
                        help='Use vLLM for faster inference')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='Tensor parallel size for vLLM')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='GPU memory utilization for vLLM')
                        
    # Output parameters
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to output file')
                        
    # Processing parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing')
    parser.add_argument('--max_items', type=int, default=None,
                        help='Maximum number of items to process')
    parser.add_argument('--offset', type=int, default=0,
                        help='Starting offset in the task file')
    
    return parser.parse_args()

def load_custom_model_class(model_class_path):
    """
    Dynamically load a custom model class from string path
    
    Args:
        model_class_path: String in format "module.submodule.ClassName"
        
    Returns:
        Class reference that can be instantiated
    """
    try:
        module_path, class_name = model_class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load custom model class {model_class_path}: {e}")
        raise

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Log execution parameters
    logger.info(f"Starting EngEdueval with task: {args.task_path}")
    logger.info(f"Model: {args.model_path} (display name: {args.model_name})")
    logger.info(f"Few-shot: {args.few_shot}, Few-shot path: {args.few_shot_path}")
    logger.info(f"vLLM: {args.vllm}, Tensor parallel size: {args.tensor_parallel_size}")
    
    try:
        # Initialize the appropriate model generator class
        if args.model_class:
            # Load custom model class
            generator_class = load_custom_model_class(args.model_class)
            generator = generator_class(
                task_path=args.task_path,
                model_path=args.model_path,
                model_name=args.model_name,
                device=args.device,
                is_few_shot=args.few_shot,
                few_shot_path=args.few_shot_path,
                is_vllm=args.vllm,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                output_file=args.output_file
            )
        else:
            # Use base ModelGenerator - note that you'll need to subclass this
            # for specific model implementations
            generator = ModelGenerator(
                task_path=args.task_path,
                model_path=args.model_path,
                model_name=args.model_name,
                device=args.device,
                is_few_shot=args.few_shot,
                few_shot_path=args.few_shot_path,
                is_vllm=args.vllm,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                output_file=args.output_file
            )
        
        # Initialize model
        if args.vllm:
            model, tokenizer = generator.model_init_vllm()
        else:
            # For non-vLLM models, should be implemented in the subclass
            if args.model_class:
                model, tokenizer = generator.model_init()
            else:
                logger.error("Base ModelGenerator cannot initialize non-vLLM models. Please provide a custom model class.")
                sys.exit(1)
        
        # Generate outputs
        output_file, completed = generator.generate_output(
            tokenizer=tokenizer,
            model=model,
            batch_size=args.batch_size,
            max_items=args.max_items,
            offset=args.offset
        )
        
        if output_file:
            logger.info(f"Generation completed successfully. Processed {completed} items. Results saved to: {output_file}")
        else:
            logger.warning("Generation completed but no output file was produced.")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 