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
just train name=mycli                 # Train for a CLI
just train name=mycli num=5000        # Custom example count
just train name=mycli quant=q8_0      # Custom quantization

# Individual stages
just train-generate name=mycli        # Generate data (skips if exists)
just train-generate-force name=mycli  # Force regenerate data
just train-sft name=mycli             # Run SFT training
just train-generate-preferences name=mycli
just train-dpo name=mycli
just train-quantize name=mycli
just train-eval name=mycli

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

## Training Data Size

### Analysis

This is a **constrained mapping task**, not open-ended generation. The model needs to:
- Map phrases → command/subcommand (classification)
- Extract arguments from natural language (slot filling)
- Recognize flags from context
- Handle ambiguous/incomplete inputs

**Coverage calculation:**

```
N subcommands
× ~10 phrasing variations each
× ~5 arg/flag combinations
= N × 50 core examples

+ negative cases (ambiguous, incomplete, off-topic)
+ edge cases (typos, synonyms, formality variations)
= estimated total
```

### Recommended Dataset Size

For a typical CLI with ~30 subcommands: **~3,000 examples**

**Rationale:**
- Intent classification research: 100-500 examples/class works well
- Diminishing returns hit around 3,000-5,000 for constrained tasks
- Quality over quantity - diverse, well-crafted examples beat volume

**Distribution:**
- ~60% positive cases with context
- ~20% positive cases without context
- ~20% negative cases (ambiguous, incomplete, clarification needed)

**Context variations:**
- With full context (all relevant environment info)
- With partial context (some info missing)
- Without any context (context: null)

## Response Types

### Training Data Format

Each training example includes optional context:

```json
{
  "instruction": "<natural language input>",
  "context": {
    "git": {
      "current_branch": "feature/example",
      "repo_owner": "owner",
      "repo_name": "repo",
      "current_pr": 123,
      "has_uncommitted_changes": false
    }
  },
  "output": "{\"type\": \"command\", \"command\": \"...\", \"subcommand\": \"...\", \"args\": {...}, \"flags\": [...], \"confidence\": 0.95}"
}
```

Examples without context (for when context is unavailable):

```json
{
  "instruction": "<natural language input>",
  "context": null,
  "output": "{\"type\": \"command\", \"command\": \"...\", \"subcommand\": \"...\", \"args\": {...}, \"flags\": [], \"confidence\": 0.95}"
}
```

### Positive Response (Clear Command)

When input clearly maps to a command:

```json
{
  "type": "command",
  "command": "<command>",
  "subcommand": "<subcommand>",
  "args": {"<arg>": "<value>"},
  "flags": ["<flag>"],
  "confidence": 0.95
}
```

### Clarification Response (Ambiguous Input)

When input is ambiguous or incomplete, respond with suggestions:

```json
{
  "type": "clarification",
  "message": "Did you mean X or Y?",
  "suggestions": [
    {"label": "Option A", "command": "...", "subcommand": "..."},
    {"label": "Option B", "command": "...", "subcommand": "..."}
  ],
  "confidence": 0.4
}
```

### Negative Cases to Handle

| Case | Description | Expected Response |
|------|-------------|-------------------|
| Ambiguous command | Command name without subcommand | Clarification with options |
| Missing required arg | Required argument not provided | Ask for the missing value |
| Unclear intent | Vague or unclear request | Ask what user wants to do |
| Off-topic | Request outside CLI scope | Explain what CLI can do |
| Multiple interpretations | Input matches multiple commands | Clarification with options |
| Typos with ambiguity | Typo makes intent unclear | Suggest correction |

### Context-Dependent Interpretations

| Input Pattern | Without Context | With Context |
|---------------|-----------------|--------------|
| "this", "current" | ❌ Clarification needed | ✅ Use context to resolve |
| Pronouns ("it", "them") | ❌ Ambiguous | ✅ Resolve from context |
| Implicit targets | ❌ Ask for target | ✅ Infer from environment |

### Confidence Thresholds

| Confidence | Action |
|------------|--------|
| >= 0.9 | Execute command |
| 0.7 - 0.9 | Execute with confirmation |
| 0.5 - 0.7 | Show clarification with suggestions |
| < 0.5 | Ask for clarification |

## Stage 1: Data Generation

### Input Schema Format

```yaml
# schema.yaml
cli_name: "<cli-name>"
description: "<CLI description>"
commands:
  - name: "<command>"
    description: "<description>"
    subcommands:
      - name: "<subcommand>"
        description: "<description>"
        args:
          - name: "<arg>"
            type: "string|integer|enum"
            required: true|false
            default: "<default>"
        flags:
          - name: "<flag>"
            short: "<char>"
            description: "<description>"
```

### Generation Prompt Template

```
You are generating training data for a CLI command interpreter.

Given this CLI schema:
{schema_yaml}

Generate {n} diverse examples of natural language inputs that map to CLI commands.
For each example, provide:
1. Natural language input (how a human might say it)
2. Optional context (environment state)
3. Structured JSON output

Vary the examples by:
- Formality level (casual to precise)
- Completeness (full command vs implied defaults)
- Synonyms (show/display/list, make/create, etc.)
- Word order variations
- With/without optional arguments
- With/without context

Include negative cases:
- Ambiguous inputs requiring clarification
- Incomplete inputs missing required info
- Off-topic requests

Output as JSON array:
[
  {
    "input": "<natural language>",
    "context": {...} or null,
    "output": {"type": "command|clarification", ...}
  },
  ...
]
```

### Validation Rules

1. **JSON Validity**: Output must be valid JSON
2. **Schema Compliance**: Command/subcommand must exist in schema
3. **Arg Types**: Arguments must match schema types
4. **Required Args**: All required args must be present (or clarification requested)
5. **Diversity Check**: Ensure input variation across dataset

## Stage 2: Supervised Fine-Tuning (SFT)

### Response-Only Loss Masking

**Critical**: The SFT training uses **response-only loss masking** to ensure the model learns to generate correct JSON outputs rather than memorizing prompts.

**Problem Discovered**: When training with `labels = input_ids` (standard causal LM), the model learns to predict the entire sequence including system prompts and user input. This results in:
- Low training loss (appears to converge)
- Poor actual performance (generates wrong format outputs)
- Model "passes" by predicting prompt tokens well, but fails at the actual task

**Solution**: Mask non-response tokens by setting their labels to `-100` (PyTorch's ignore index), and **append EOS token** to teach the model when to stop:

```python
# Tokenize prompt and response separately
prompt_tokens = tokenizer(prompt, add_special_tokens=True)
response_tokens = tokenizer(response, add_special_tokens=False)

# Combine for input, adding EOS token at the end
eos_token_id = tokenizer.eos_token_id
input_ids = prompt_tokens["input_ids"] + response_tokens["input_ids"] + [eos_token_id]

# Labels: -100 for prompt (ignored), actual IDs for response + EOS
labels = [-100] * len(prompt_tokens["input_ids"]) + response_tokens["input_ids"] + [eos_token_id]
```

This ensures:
1. The loss is computed **only on the JSON response**, forcing the model to learn the output format
2. The model learns to **stop generating** after the JSON output (EOS token)

### Training Configuration

```yaml
# training/configs/sft_config.yaml
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
  train_file: "./data/train.jsonl"
  validation_file: "./data/train_val.jsonl"
  max_seq_length: 2048

logging:
  report_to: "none"
  logging_steps: 10
```

## Stage 3: DPO Training

### Preference Data Generation

Generate preference pairs using a large LLM to evaluate outputs:

```python
def generate_preference_data(
    sft_model_path: str,
    schema: dict,
    num_pairs: int = 1000
) -> list[dict]:
    """Generate preference pairs for DPO training."""
    # Get SFT model output
    # Get "ideal" output from large LLM
    # Create preference pair where ideal is preferred
```

### DPO Configuration

```yaml
# training/configs/dpo_config.yaml
model:
  name: "./models/sft"  # Overridden to ./models/{name}/sft

training:
  output_dir: "./models/dpo"
  num_train_epochs: 1
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 5e-6
  beta: 0.1  # KL penalty coefficient
  loss_type: "sigmoid"

data:
  preference_file: "./data/preferences.jsonl"
```

## Stage 4: Quantization

### LoRA Adapter Merging

Since training uses LoRA adapters, the quantization process must first merge adapters with the base model:

1. **Detect LoRA adapters** - Check for `adapter_config.json` and `adapter_model.safetensors`
2. **Load base model** - From the path specified in adapter config
3. **Merge adapters** - Use PEFT's `merge_and_unload()` to create full model
4. **Convert to GGUF** - Use llama.cpp's `convert_hf_to_gguf.py`
5. **Quantize** - Apply Q4_K_M or other quantization

### GGUF Conversion

The quantization script handles this automatically:

```bash
# Full pipeline (detects LoRA, merges, converts, quantizes)
just train-quantize name=gh quant=q4_k_m
```

Manual process:

```bash
# 1. Merge LoRA adapters (handled by quantize.py)
# 2. Convert to GGUF format
python convert_hf_to_gguf.py ./merged_model --outfile model-f16.gguf --outtype f16

# 3. Quantize to Q4_K_M
llama-quantize model-f16.gguf model-q4_k_m.gguf q4_k_m
```

### Dependencies

- `llama.cpp` tools (install via `brew install llama.cpp` on macOS)
- `gguf` Python package
- `peft` for LoRA merging

## Hardware Requirements

### Apple Silicon (MPS) - Recommended for Local Training

Apple Silicon Macs can train using PyTorch's MPS backend:

- **Minimum**: M1/M2/M3 with 16GB unified memory
- **Recommended**: M1 Pro/Max/Ultra or M2/M3/M4 Pro with 32GB+ memory
- **Storage**: 50GB free space

The training code uses `device_map="auto"` which automatically detects MPS.

```bash
# Verify MPS is available
python -c "import torch; print(torch.backends.mps.is_available())"

# Run training (MPS is auto-detected)
just train-sft name=gh
```

**Note**: MPS training is slower than NVIDIA GPUs but much faster than CPU-only training.

### NVIDIA GPU (Cloud or Local)

**Minimum**:
- GPU: 1x RTX 3090 or A100 (24GB VRAM)
- RAM: 32GB
- Storage: 100GB SSD

**Recommended**:
- GPU: 2x A100 80GB
- RAM: 128GB
- Storage: 500GB NVMe

### For Data Generation Only
- CPU: Any modern CPU
- RAM: 16GB
- API access to large LLM

## Cost Estimates

| Stage | Resource | Estimated Cost |
|-------|----------|----------------|
| Data Generation | LLM API | ~$35 for 3k examples |
| SFT Training | A100 x 2hrs | ~$10 (cloud) |
| DPO Training | A100 x 1hr | ~$5 (cloud) |
| Evaluation | CPU only | Free |
| **Total** | | **~$50** |

Note: Data generation can be done manually or via conversation with Claude to avoid API costs.

## Monitoring and Logging

### Metrics to Track

- Training loss
- Validation loss
- Exact match accuracy
- JSON validity rate
- Inference latency
- Model perplexity
