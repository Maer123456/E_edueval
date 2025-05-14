import json
import os
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import jsonlines
from pathlib import Path
import sys

# Import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate.model_gen import ModelGenerator

class QwenGenerator(ModelGenerator):
    """
    Generator class for Qwen models (Qwen-7B, Qwen-14B, etc.)
    """
    def __init__(self, task_path, model_path, model_name, device="0", is_few_shot=False,
                 few_shot_path=None, is_vllm=False, tensor_parallel_size=1,
                 gpu_memory_utilization=0.9, output_file=None):
        """
        Initialize Qwen generator
        
        Args:
            task_path: Path to the task file
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
        super().__init__(
            task_path=task_path,
            model_path=model_path,
            model_name=model_name,
            device=device,
            is_few_shot=is_few_shot,
            few_shot_path=few_shot_path,
            is_vllm=is_vllm,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            output_file=output_file
        )

    def model_init(self):
        """
        Initialize Qwen model and tokenizer
        
        Returns:
            tuple: (model, tokenizer)
        """
        if self.is_vllm:
            # Use vLLM for faster inference
            return self.model_init_vllm()

        try:
            # Check if the path is a local directory or HuggingFace model ID
            is_local_dir = os.path.isdir(self.model_path)

            print(f"Loading tokenizer from {self.model_path} (local_dir={is_local_dir})")
            tokenizer_kwargs = {
                "trust_remote_code": True,
            }

            # Add local_files_only=True for local paths
            if is_local_dir:
                tokenizer_kwargs["local_files_only"] = True

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                **tokenizer_kwargs
            )

            # Explicitly set the chat template if it's missing
            if tokenizer.chat_template is None:
                print("Tokenizer missing chat template. Setting Qwen default template.")
                tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "{% if loop.first %}"
                    "{% if message['role'] == 'system' %}"
                    "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
                    "{% else %}"
                    "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
                    "{% endif %}"
                    "{% elif message['role'] == 'user' %}"
                    "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
                    "{% elif message['role'] == 'assistant' %}"
                    "{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
                    "{% endif %}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}"
                    "{{ '<|im_start|>assistant\n' }}"
                    "{% endif %}"
                )

            print(f"Loading model from {self.model_path}")
            # Keep settings adjusted for Qwen models
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": f"cuda:{self.device}",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "max_memory": {0: "40GiB"},
            }

            # Add local_files_only=True for local paths
            if is_local_dir:
                model_kwargs["local_files_only"] = True

            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )

            model.eval()

        except Exception as e:
            print(f"Error loading model/tokenizer from {self.model_path}: {e}")
            print("Trying to load with low_cpu_mem_usage=True (fallback)")
            # Try again with low_cpu_mem_usage (fallback, keep params consistent)
            is_local_dir = os.path.isdir(self.model_path)
            tokenizer_kwargs = {
                "trust_remote_code": True,
            }

            if is_local_dir:
                tokenizer_kwargs["local_files_only"] = True

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                **tokenizer_kwargs
            )
            
            # Explicitly set the chat template if it's missing (in except block too)
            if tokenizer.chat_template is None:
                print("Tokenizer missing chat template (fallback). Setting Qwen default template.")
                tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "{% if loop.first %}"
                    "{% if message['role'] == 'system' %}"
                    "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
                    "{% else %}"
                    "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
                    "{% endif %}"
                    "{% elif message['role'] == 'user' %}"
                    "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
                    "{% elif message['role'] == 'assistant' %}"
                    "{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
                    "{% endif %}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}"
                    "{{ '<|im_start|>assistant\n' }}"
                    "{% endif %}"
                )

            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": f"cuda:{self.device}",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "max_memory": {0: "40GiB"},
            }

            if is_local_dir:
                model_kwargs["local_files_only"] = True

            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )

            model.eval()

        return model, tokenizer

    def format_prompt(self, prompt):
        """Format prompt for Qwen models using chat template"""
        # For Qwen models, it's better to format as a chat
        if hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return formatted_prompt
        return prompt

    def generate_output(self, tokenizer, model, batch_size=1, max_items=None, offset=0):
        """
        Override generate_output to handle Qwen-specific prompt formatting
        
        Args:
            tokenizer: Tokenizer object
            model: Model object
            batch_size: Batch size for processing
            max_items: Maximum number of items to process
            offset: Starting offset in the dataset
            
        Returns:
            tuple: (output_file_path, completed_items)
        """
        self.tokenizer = tokenizer  # Store tokenizer for use in format_prompt
        
        print(f"Generating outputs for {self.task_path} with model {self.model_name}")
        
        # Prepare prompts 
        prompts, answers, questions, subjects = self.prepare_prompts(max_items, offset)
        
        if not prompts:
            print("No prompts generated, exiting.")
            return None, 0
            
        print(f"Generating responses for {len(prompts)} prompts")
        outputs = []
        
        # Use tqdm to show progress
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch_prompts = prompts[i:i+batch_size]
            
            # Format prompts for Qwen
            formatted_prompts = [self.format_prompt(prompt) for prompt in batch_prompts]
            
            try:
                if self.is_vllm:
                    # Use vLLM for generation
                    from vllm import SamplingParams
                    
                    sampling_params = SamplingParams(
                        temperature=0.7,
                        top_p=0.95,
                        max_tokens=512
                    )
                    
                    result_outputs = model.generate(formatted_prompts, sampling_params)
                    
                    for j, output in enumerate(result_outputs):
                        output_text = output.outputs[0].text
                        processed_output = self.post_process_output(output_text, self.task_type)
                        outputs.append(processed_output)
                else:
                    # Use standard HuggingFace generation
                    for j, prompt in enumerate(formatted_prompts):
                        truncated_prompt = self.truncate_prompt(prompt, tokenizer)
                        
                        inputs = tokenizer(truncated_prompt, return_tensors="pt").to(model.device)
                        
                        with torch.no_grad():
                            output_ids = model.generate(
                                **inputs, 
                                max_new_tokens=512,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.95
                            )
                        
                        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                        
                        # Remove the prompt part from the output
                        if truncated_prompt in output_text:
                            output_text = output_text.replace(truncated_prompt, "").strip()
                        
                        processed_output = self.post_process_output(output_text, self.task_type)
                        outputs.append(processed_output)
                        
            except Exception as e:
                print(f"Error during generation for batch starting at index {i}: {e}")
                outputs.extend([""] * len(batch_prompts))
        
        print(f"Generated {len(outputs)} outputs")
        
        # Save results
        output_file = self.save_results(questions, outputs, answers, subjects)
        
        return output_file, len(outputs)

    def extract_choice_answer(self, text):
        """Improved choice answer extraction for Qwen models"""
        # First try parent class method
        answer = super().extract_choice_answer(text)
        if answer and answer in "ABCD":
            return answer
            
        # Try Qwen-specific extraction if parent method failed
        cleaned_text = text.strip()
        
        # Check if the answer is at the end (Qwen often puts the answer as the last letter)
        last_paragraph = cleaned_text.split('\n')[-1].strip()
        choices = re.findall(r'[A-Da-d]', last_paragraph)
        if choices:
            return choices[0].upper()
            
        # Try to find the answer after specific markers
        markers = ["答案是", "答案为", "选择", "选项是", "应选", "正确的是", "正确答案是"]
        for marker in markers:
            if marker in cleaned_text:
                after_marker = cleaned_text.split(marker, 1)[1].strip()
                choices = re.findall(r'[A-Da-d]', after_marker)
                if choices:
                    return choices[0].upper()
        
        # As a last resort, return the original text (parent method)
        return answer 