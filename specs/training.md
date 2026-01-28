# Training Pipeline Specification

## Overview

The training pipeline uses a large LLM to generate high-quality training data, then fine-tunes a small model using supervised learning followed by reinforcement learning (DPO).

## Named Trainings

Training is organized by CLI name. Each CLI has its own schema, data, and models:

```
schemas/{name}.yaml           # CLI schema definition
training/data/{name}/         # Training data
  ├── train.jsonl             # Training examples
  ├── train_val.jsonl         # Validation examples
  ├── train_progress.jsonl    # Resume checkpoint (temp)
  └── preferences.jsonl       # DPO preference pairs
training/models/{name}/       # Trained models
  ├── sft/                    # SFT checkpoint
  ├── dpo/                    # DPO checkpoint
  └── clitron-{name}-{quant}.gguf  # Final quantized model
```

### Usage

```bash
# Full pipeline for a CLI
just train name=gh                    # Train for GitHub CLI
just train name=docker                # Train for Docker CLI
just train name=gh num=5000           # Custom example count
just train name=gh quant=q8_0         # Custom quantization

# Individual stages
just train-generate name=gh           # Generate data (skips if exists)
just train-generate-force name=gh     # Force regenerate data
just train-sft name=gh                # Run SFT training
just train-generate-preferences name=gh
just train-dpo name=gh
just train-quantize name=gh
just train-eval name=gh

# List available schemas
just train-list-schemas
```

### Resume Support

Data generation automatically resumes from interruptions:
- Progress saved to `train_progress.jsonl` after each batch
- On restart, generation continues from last checkpoint
- Use `train-generate-force` to start fresh

## Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 1: Data Generation                      │
│                                                                  │
│  Schema → Opus 4.5 → Synthetic Examples → Validation → Dataset  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 2: SFT Training                         │
│                                                                  │
│  Base Model + Dataset → Supervised Fine-Tuning → SFT Model      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 3: DPO Training                         │
│                                                                  │
│  SFT Model + Preference Data → DPO → Final Model                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 4: Quantization                         │
│                                                                  │
│  Final Model → GGUF Conversion → Q4_K_M Quantization            │
└─────────────────────────────────────────────────────────────────┘
```

## Stage 1: Data Generation

### Input Schema Format

```yaml
# schema.yaml
cli_name: "gh"
description: "GitHub CLI"
commands:
  - name: "pr"
    description: "Work with pull requests"
    subcommands:
      - name: "list"
        description: "List pull requests"
        args:
          - name: "state"
            type: "enum"
            values: ["open", "closed", "merged", "all"]
            default: "open"
          - name: "author"
            type: "string"
            description: "Filter by author"
          - name: "limit"
            type: "integer"
            default: 30
        flags:
          - name: "web"
            short: "w"
            description: "Open in browser"
```

### Generation Prompt Template

```
You are generating training data for a CLI command interpreter.

Given this CLI schema:
{schema_yaml}

Generate {n} diverse examples of natural language inputs that map to CLI commands.
For each example, provide:
1. Natural language input (how a human might say it)
2. Structured JSON output

Vary the examples by:
- Formality level (casual to precise)
- Completeness (full command vs implied defaults)
- Synonyms (show/display/list, make/create, etc.)
- Word order variations
- With/without optional arguments

Output as JSON array:
[
  {
    "input": "show me open prs",
    "output": {"command": "pr", "subcommand": "list", "args": {"state": "open"}, "flags": []}
  },
  ...
]
```

### Data Generation Script

```python
# training/src/generate_dataset.py

import anthropic
import json
from pathlib import Path

def generate_training_data(
    schema_path: Path,
    output_path: Path,
    num_examples: int = 10000,
    batch_size: int = 50
) -> None:
    """Generate training data using Opus 4.5."""
    client = anthropic.Anthropic()

    schema = load_schema(schema_path)
    examples = []

    for batch in range(num_examples // batch_size):
        response = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": format_generation_prompt(schema, batch_size)
            }]
        )

        batch_examples = parse_examples(response.content)
        validated = validate_examples(batch_examples, schema)
        examples.extend(validated)

    save_dataset(examples, output_path)
```

### Validation Rules

1. **JSON Validity**: Output must be valid JSON
2. **Schema Compliance**: Command/subcommand must exist in schema
3. **Arg Types**: Arguments must match schema types
4. **Required Args**: All required args must be present
5. **Diversity Check**: Ensure input variation across dataset

## Stage 2: Supervised Fine-Tuning (SFT)

### Training Configuration

```yaml
# training/configs/sft_config.yaml
# Note: data and output paths are overridden by CLI args for named trainings
model:
  name: "meta-llama/Llama-3.2-1B-Instruct"

training:
  output_dir: "./models/sft"  # Overridden to ./models/{name}/sft
  num_train_epochs: 3
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 2e-5
  warmup_ratio: 0.1
  lr_scheduler_type: "cosine"

  # LoRA configuration for efficient training
  use_lora: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

data:
  train_file: "./data/train.jsonl"        # Overridden to ./data/{name}/train.jsonl
  validation_file: "./data/train_val.jsonl"  # Overridden to ./data/{name}/train_val.jsonl
  max_seq_length: 2048

logging:
  report_to: "none"
  logging_steps: 10
```

### Training Script

```python
# training/src/train_sft.py

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

def train_sft(config_path: str) -> None:
    config = load_config(config_path)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    # Apply LoRA
    lora_config = LoraConfig(
        r=config.training.lora_r,
        lora_alpha=config.training.lora_alpha,
        lora_dropout=config.training.lora_dropout,
        target_modules=config.training.lora_target_modules
    )
    model = get_peft_model(model, lora_config)

    # Load and format dataset
    dataset = load_dataset("json", data_files={
        "train": config.data.train_file,
        "validation": config.data.validation_file
    })

    # Train
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**config.training),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model()
```

## Stage 3: DPO Training

### Preference Data Generation

Generate preference pairs using Opus 4.5 to evaluate outputs:

```python
def generate_preference_data(
    sft_model_path: str,
    schema: dict,
    num_pairs: int = 5000
) -> list[dict]:
    """Generate preference pairs for DPO training."""

    preferences = []
    sft_model = load_model(sft_model_path)
    opus_client = anthropic.Anthropic()

    for prompt in generate_test_prompts(schema, num_pairs):
        # Get SFT model output
        model_output = sft_model.generate(prompt)

        # Get "ideal" output from Opus
        ideal_output = opus_client.messages.create(
            model="claude-opus-4-5-20251101",
            messages=[{"role": "user", "content": prompt}]
        )

        # Create preference pair
        preferences.append({
            "prompt": prompt,
            "chosen": ideal_output,  # Opus output preferred
            "rejected": model_output  # SFT output if different
        })

    return preferences
```

### DPO Configuration

```yaml
# training/configs/dpo_config.yaml
# Note: model and data paths are overridden by CLI args for named trainings
model:
  name: "./models/sft"  # Overridden to ./models/{name}/sft

training:
  output_dir: "./models/dpo"  # Overridden to ./models/{name}/dpo
  num_train_epochs: 1
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 5e-6

  # DPO specific
  beta: 0.1  # KL penalty coefficient
  loss_type: "sigmoid"  # or "hinge"

data:
  preference_file: "./data/preferences.jsonl"  # Overridden to ./data/{name}/preferences.jsonl
```

### DPO Training Script

```python
# training/src/train_dpo.py

from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def train_dpo(config_path: str) -> None:
    config = load_config(config_path)

    # Load SFT model as starting point
    model = AutoModelForCausalLM.from_pretrained(config.model.name)
    ref_model = AutoModelForCausalLM.from_pretrained(config.model.name)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    # Load preference data
    dataset = load_dataset("json", data_files=config.data.preference_file)

    # DPO training
    dpo_config = DPOConfig(
        beta=config.training.beta,
        loss_type=config.training.loss_type,
        **config.training
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset["train"],
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model()
```

## Stage 4: Quantization

### GGUF Conversion

```bash
# Convert to GGUF format
python -m llama_cpp.convert \
    --input ./models/dpo \
    --output ./models/clitron.gguf \
    --outtype f16

# Quantize to Q4_K_M
./llama-quantize \
    ./models/clitron.gguf \
    ./models/clitron-q4_k_m.gguf \
    q4_k_m
```

### Quantization Script

```python
# training/src/quantize.py

import subprocess
from pathlib import Path

def quantize_model(
    input_path: Path,
    output_path: Path,
    quantization: str = "q4_k_m"
) -> None:
    """Quantize model to GGUF format."""

    # First convert to GGUF if needed
    if input_path.suffix != ".gguf":
        gguf_path = input_path.with_suffix(".gguf")
        subprocess.run([
            "python", "-m", "llama_cpp.convert",
            "--input", str(input_path),
            "--output", str(gguf_path),
            "--outtype", "f16"
        ], check=True)
        input_path = gguf_path

    # Quantize
    subprocess.run([
        "llama-quantize",
        str(input_path),
        str(output_path),
        quantization
    ], check=True)
```

## Automated Pipeline

### Full Training Pipeline

The training pipeline is managed via `just` (see `justfile`):

```bash
# Full pipeline - runs all stages sequentially
just train name=gh num=10000 quant=q4_k_m

# This executes:
# 1. train-setup      - Set up Python environment with uv
# 2. train-generate   - Generate training data (skips if exists)
# 3. train-sft        - Supervised fine-tuning
# 4. train-generate-preferences - Generate DPO preference pairs
# 5. train-dpo        - Direct Preference Optimization
# 6. train-quantize   - Convert to GGUF and quantize
# 7. train-eval       - Evaluate model accuracy
```

### Pipeline Implementation

Each stage is a separate `just` recipe that can be run independently:

```just
# justfile (simplified)

train name="gh" num="10000" quant="q4_k_m": \
    train-setup \
    (train-generate name num) \
    (train-sft name) \
    (train-generate-preferences name) \
    (train-dpo name) \
    (train-quantize name quant) \
    (train-eval name quant)

train-generate name="gh" num="10000":
    cd training && uv run python -m clitron_training.generate_dataset \
        --schema ../schemas/{{name}}.yaml \
        --output ./data/{{name}}/train.jsonl \
        --num-examples {{num}}

train-sft name="gh":
    cd training && uv run python -m clitron_training.train_sft \
        --config ./configs/sft_config.yaml \
        --data-dir ./data/{{name}} \
        --output-dir ./models/{{name}}/sft

train-dpo name="gh":
    cd training && uv run python -m clitron_training.train_dpo \
        --config ./configs/dpo_config.yaml \
        --model-dir ./models/{{name}}/sft \
        --output-dir ./models/{{name}}/dpo \
        --preference-file ./data/{{name}}/preferences.jsonl
```

## Hardware Requirements

### Minimum (Training)
- GPU: 1x RTX 3090 or A100 (24GB VRAM)
- RAM: 32GB
- Storage: 100GB SSD

### Recommended (Training)
- GPU: 2x A100 80GB
- RAM: 128GB
- Storage: 500GB NVMe

### For Data Generation Only
- CPU: Any modern CPU
- RAM: 16GB
- API access to Opus 4.5

## Cost Estimates

| Stage | Resource | Estimated Cost |
|-------|----------|----------------|
| Data Generation | Opus 4.5 API | ~$50 for 10k examples |
| SFT Training | A100 x 2hrs | ~$10 (cloud) |
| DPO Training | A100 x 1hr | ~$5 (cloud) |
| Evaluation | CPU only | Free |
| **Total** | | **~$65** |

## Monitoring and Logging

### Weights & Biases Integration

```python
import wandb

wandb.init(
    project="clitron",
    config={
        "model": "llama-3.2-1b",
        "training_type": "sft+dpo",
        "dataset_size": 10000
    }
)
```

### Metrics to Track

- Training loss
- Validation loss
- Exact match accuracy
- JSON validity rate
- Inference latency
- Model perplexity
