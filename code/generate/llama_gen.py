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

class LlamaGenerator(ModelGenerator):
    """
    Generator class for Llama models (LLama-2, LLama-3, Vicuna, etc.)
    """
    def __init__(self, task_path, model_path, model_name, device="0", is_few_shot=False,
                 few_shot_path=None, is_vllm=False, tensor_parallel_size=1,
                 gpu_memory_utilization=0.9, output_file=None):
        """Initialize Llama generator"""
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
        """Initialize Llama model and tokenizer"""
        if self.is_vllm:
            return self.model_init_vllm()

        try:
            is_local_dir = os.path.isdir(self.model_path)
            tokenizer_kwargs = {
                "trust_remote_code": True,
                "use_fast": False,
            }
            if is_local_dir:
                tokenizer_kwargs["local_files_only"] = True

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                **tokenizer_kwargs
            )

            # Set padding token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": f"cuda:{self.device}",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            if is_local_dir:
                model_kwargs["local_files_only"] = True

            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            model.eval()

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        return model, tokenizer

    def format_prompt(self, prompt):
        """Format prompt for Llama models"""
        if hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(messages, tokenize=False)
        
        # Model-specific formats
        if "llama-2" in self.model_name.lower():
            system = "You are a helpful assistant. Always answer as helpfully as possible."
            return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{prompt} [/INST]"
        elif "llama-3" in self.model_name.lower():
            return f"<|begin_of_text|><|user|>\n{prompt}<|end_of_turn|>\n<|assistant|>\n"
        elif "vicuna" in self.model_name.lower():
            return f"USER: {prompt}\nASSISTANT:"
        
        # Generic format
        return f"<s>User: {prompt}\n\nAssistant:"

    def generate_output(self, tokenizer, model, batch_size=1, max_items=None, offset=0):
        """Generate outputs for Llama models"""
        self.tokenizer = tokenizer
        
        prompts, answers, questions, subjects = self.prepare_prompts(max_items, offset)
        
        if not prompts:
            return None, 0
            
        outputs = []
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch_prompts = prompts[i:i+batch_size]
            formatted_prompts = [self.format_prompt(prompt) for prompt in batch_prompts]
            
            try:
                if self.is_vllm:
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
                        
                        # Extract response based on model type
                        if "llama-2" in self.model_name.lower() and "[/INST]" in output_text:
                            output_text = output_text.split("[/INST]", 1)[1].strip()
                        elif "llama-3" in self.model_name.lower() and "<|assistant|>" in output_text:
                            output_text = output_text.split("<|assistant|>", 1)[1].strip()
                            if "<|end_of_turn|>" in output_text:
                                output_text = output_text.split("<|end_of_turn|>", 1)[0].strip()
                        elif "vicuna" in self.model_name.lower() and "ASSISTANT:" in output_text:
                            output_text = output_text.split("ASSISTANT:", 1)[1].strip()
                        elif "Assistant:" in output_text:
                            output_text = output_text.split("Assistant:", 1)[1].strip()
                        elif truncated_prompt in output_text:
                            output_text = output_text.replace(truncated_prompt, "").strip()
                        
                        processed_output = self.post_process_output(output_text, self.task_type)
                        outputs.append(processed_output)
                        
            except Exception as e:
                print(f"Error in generation: {e}")
                outputs.extend([""] * len(batch_prompts))
        
        # Save results
        output_file = self.save_results(questions, outputs, answers, subjects)
        
        return output_file, len(outputs)

    def extract_choice_answer(self, text):
        """Extract multiple choice answers from Llama outputs"""
        # First try parent method
        answer = super().extract_choice_answer(text)
        if answer and answer in "ABCD":
            return answer
            
        cleaned_text = text.strip()
        
        # Check single letters on lines
        lines = cleaned_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) <= 3:
                choices = re.findall(r'[A-Da-d]', line)
                if choices and len(choices) == 1:
                    return choices[0].upper()
            
        # Try English markers
        markers = ["answer is", "answer:", "option", "choice", "select", "I choose"]
        for marker in markers:
            if marker.lower() in cleaned_text.lower():
                after_marker = cleaned_text.lower().split(marker.lower(), 1)[1].strip()
                choices = re.findall(r'[a-d]', after_marker)
                if choices:
                    return choices[0].upper()
        
        # Look for patterns like "(A)" or "A." at beginning of lines
        patterns = [r'\(([A-Da-d])\)', r'^\s*([A-Da-d])[\.\)]']
        for line in lines:
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    return match.group(1).upper()
                
        return answer 