# EngEdueval Framework

**English Educational Evaluation Framework**

A framework for evaluating large language models (LLMs) on English educational tasks.

## Overview

EngEdueval is a comprehensive framework designed to assess the performance of language models across various educational dimensions, from basic knowledge recall to creative problem-solving. It mirrors the structure and methodology of the Chinese edueval5_5 framework while adapting it specifically for English educational content.

## Directory Structure

```
EngEdueval/
├── code/
│   ├── generate/            # Model generation modules
│   │   ├── model_gen.py     # Base generator class
│   │   ├── qwen_gen.py      # Qwen model generator
│   │   ├── glm_gen.py       # GLM model generator
│   │   ├── baichuan_gen.py  # Baichuan model generator
│   │   └── llama_gen.py     # Llama model generator
│   ├── evaluate/            # Evaluation modules
│   │   └── evaluator.py     # Evaluation metrics and methods
│   ├── prompt_builder.py    # Utilities for building prompts
│   └── main.py              # Main entry point
├── Edata/                   # English educational data
│   ├── Memory/              # Knowledge recall tasks
│   ├── Understanding/       # Reading comprehension tasks
│   ├── Application/         # Application tasks
│   ├── Reasoning/           # Logical reasoning tasks
│   ├── Creativity/          # Creative tasks
│   └── Ethics/              # Ethical reasoning tasks
└── README.md                # This documentation
```

## Framework Components

### Generation Module

The generation module is responsible for running language models on educational tasks. It includes:

- **ModelGenerator (model_gen.py)**: Base class that handles the core functionality for all model types
- **QwenGenerator (qwen_gen.py)**: Specialized for Qwen models (Qwen-7B, Qwen-14B)
- **GLMGenerator (glm_gen.py)**: Specialized for GLM models (ChatGLM, GLM-3)
- **BaichuanGenerator (baichuan_gen.py)**: Specialized for Baichuan models
- **LlamaGenerator (llama_gen.py)**: Specialized for Llama models (Llama-2, Llama-3, Vicuna)

Each generator handles model-specific prompt formatting, output processing, and answer extraction.

### Evaluation Module

The evaluation module assesses model outputs against reference answers, including:

- Accuracy metrics for multiple-choice questions
- BLEU/ROUGE scores for open-ended questions
- Custom scoring for various task types

### Prompt Building

The framework includes utilities for constructing appropriate prompts based on task types:

- Multiple-choice prompts
- Short answer prompts
- Essay prompts
- Reading comprehension prompts

## Supported Models

EngEdueval supports various LLM architectures:

- **Qwen models**: Qwen-7B, Qwen-14B
- **GLM models**: ChatGLM, GLM-3
- **Baichuan models**: Baichuan-7B, Baichuan2-13B
- **Llama models**: Llama-2, Llama-3, Vicuna

Models can be loaded from local paths or Hugging Face.

## Supported Task Types

The framework supports six dimensions of educational tasks:

1. **Memory**: Basic knowledge recall and memorization
2. **Understanding**: Reading comprehension and interpretation
3. **Application**: Applying knowledge to solve problems
4. **Reasoning**: Logical reasoning and inference
5. **Creativity**: Creative problem-solving and generation
6. **Ethics**: Ethical reasoning and decision-making

## Usage

### Running Evaluation Tasks

```bash
python -m EngEdueval.code.main \
  --task_path Edata/Memory/Primary_Formula_Recall.jsonl \
  --model_path /path/to/model \
  --model_name "Model Name" \
  --device 0 \
  --model_type qwen  # Options: qwen, glm, baichuan, llama
```

### Command-Line Options

- `--task_path`: Path to the task JSONL file
- `--model_path`: Path to the model weights or HuggingFace model ID
- `--model_name`: Name for logging and output files
- `--device`: GPU device ID (e.g., "0" or "0,1,2,3")
- `--model_type`: Type of model (qwen, glm, baichuan, or llama)
- `--is_few_shot`: Use few-shot examples (optional)
- `--few_shot_path`: Path to few-shot examples file (optional)
- `--is_vllm`: Use vLLM for faster inference (optional)
- `--tensor_parallel_size`: Number of GPUs for tensor parallelism (vLLM only)
- `--output_file`: Custom output file path (optional)

### Using vLLM for Acceleration

For faster inference on large models, you can enable vLLM:

```bash
python -m EngEdueval.code.main \
  --task_path Edata/Memory/Primary_Formula_Recall.jsonl \
  --model_path /path/to/model \
  --model_name "Model Name" \
  --device 0 \
  --model_type llama \
  --is_vllm \
  --tensor_parallel_size 4
```

## File Formats

### Task Files

Task files are in JSONL format with the following structure:

```json
{
  "subject": "mathematics",
  "ques_content": "What is 2+2?",
  "options": ["3", "4", "5", "6"],
  "ques_answer": "B",
  "ques_analyze": "The sum of 2 and 2 is 4, which corresponds to option B."
}
```

### Output Files

Model outputs are saved in JSONL format with:

```json
{
  "id": 42,
  "task_type": "multiple_choice",
  "question": "What is 2+2?",
  "reference_answer": "B",
  "model_output": "The answer is 4, so the correct option is B.",
  "processed_output": "B",
  "model_name": "Model Name",
  "timestamp": "2023-09-15 14:30:00"
}
```

## Contributing

Contributions are welcome! You can extend the framework by:

1. Adding support for new model architectures
2. Creating new task types
3. Implementing additional evaluation metrics
4. Improving prompt templates

## License

This project is available under the MIT License.

## Acknowledgments

EngEdueval is inspired by the edueval5_5 framework and adapted for English educational content evaluation. 