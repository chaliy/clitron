# Model Selection and Requirements

## Overview

Clitron requires a small, fast language model capable of understanding natural language CLI commands and outputting structured JSON. The model must be small enough to potentially bundle with CLI applications.

## Model Candidates

### Primary Candidates

| Model | Parameters | Quantized Size | Notes |
|-------|------------|----------------|-------|
| **Llama 3.2 1B Instruct** | 1B | ~500MB (Q4) | Best balance of size/capability |
| **Llama 3.2 3B Instruct** | 3B | ~1.5GB (Q4) | Better accuracy, larger |
| **Gemma 2 2B** | 2B | ~1GB (Q4) | Good instruction following |
| **SmolLM2 1.7B** | 1.7B | ~800MB (Q4) | Designed for edge deployment |

### Recommended: Llama 3.2 1B Instruct

**Rationale:**
- Excellent instruction-following capabilities
- Well-documented and widely supported
- Good tokenizer for code/CLI content
- Available in GGUF format
- Strong community and tooling support

## Model Requirements

### Functional Requirements

1. **Instruction Following**: Must reliably follow structured prompts
2. **JSON Generation**: Must output valid, parseable JSON
3. **Context Understanding**: Must understand CLI domain vocabulary
4. **Few-shot Learning**: Must effectively use examples in context

### Non-Functional Requirements

1. **Size**: < 1GB quantized (ideal: < 500MB)
2. **Latency**: < 100ms inference on modern CPU
3. **Memory**: < 512MB runtime memory
4. **Format**: GGUF for easy deployment

## Quantization Strategy

### Recommended: Q4_K_M

- Best balance of size and quality
- ~4 bits per weight average
- Minimal quality degradation for this task
- Wide tooling support

### Quantization Comparison

| Method | Size (1B model) | Quality Loss | Speed |
|--------|-----------------|--------------|-------|
| Q8_0 | ~1GB | Minimal | Fast |
| Q6_K | ~750MB | Very Low | Fast |
| Q4_K_M | ~500MB | Low | Fast |
| Q4_0 | ~400MB | Moderate | Fastest |
| Q2_K | ~250MB | High | Fastest |

## Input/Output Format

### System Prompt Template

```
You are a CLI command interpreter. Given a natural language command, output a JSON object representing the structured command.

Available commands:
{schema_description}

Examples:
{few_shot_examples}

Output only valid JSON. No explanations.
```

### Input Format

```
User input: "show my open prs"
```

### Output Format

```json
{
  "command": "pr",
  "subcommand": "list",
  "args": {
    "state": "open",
    "author": "@me"
  },
  "flags": [],
  "confidence": 0.95
}
```

## Training Approach

### Base Model Fine-tuning

1. Start with instruction-tuned base model
2. Fine-tune on CLI command interpretation task
3. Use RLHF/DPO to improve output quality

### Training Data Requirements

- **Synthetic Generation**: Use large LLM (Opus 4.5) to generate training pairs
- **Diversity**: Cover all commands, variations, edge cases
- **Quality**: Validate all outputs against schema

### Training Configuration

```yaml
base_model: "meta-llama/Llama-3.2-1B-Instruct"
training:
  method: "sft"  # Supervised Fine-Tuning first
  epochs: 3
  learning_rate: 2e-5
  batch_size: 8

rl_training:
  method: "dpo"  # Direct Preference Optimization
  epochs: 1
  beta: 0.1
```

## Inference Configuration

### Runtime Settings

```rust
pub struct ModelConfig {
    pub model_path: PathBuf,
    pub context_size: usize,      // 2048 tokens
    pub max_tokens: usize,        // 256 tokens (JSON output)
    pub temperature: f32,         // 0.1 (deterministic)
    pub top_p: f32,               // 0.9
    pub repeat_penalty: f32,      // 1.1
}
```

### Optimization Flags

- Use `mmap` for model loading
- Enable Flash Attention if available
- Use batch size 1 for single queries
- Pre-warm model on first load

## Model Bundling Options

### Option 1: Separate Download

```rust
// Model downloaded on first run
let model = Interpreter::download_model().await?;
```

### Option 2: Embedded in Binary

```rust
// Model embedded at compile time (large binary)
const MODEL_BYTES: &[u8] = include_bytes!("model.gguf");
```

### Option 3: Lazy Download with Cache

```rust
// Download once, cache in ~/.clitron/models/
let model = Interpreter::load_or_download().await?;
```

**Recommended**: Option 3 for flexibility and smaller initial binary.

## Llama License Requirements

When distributing fine-tuned Llama models, you must comply with the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/):

### Required for Distribution

1. **Model Naming**: Model filename must start with "Llama" (e.g., `Llama-clitron-gh-q4_k_m.gguf`)
2. **Attribution**: Display "Built with Llama" on website, UI, or documentation
3. **License Copy**: Bundle a copy of the Llama 3.2 Community License with the distribution

### Usage Restrictions

- Must follow Meta's Acceptable Use Policy
- Free for commercial use if < 700M monthly active users
- Companies with >= 700M MAU need a separate commercial agreement with Meta

### Example Distribution Structure

```
my-cli/
├── bin/
│   └── my-cli
├── models/
│   └── Llama-clitron-gh-q4_k_m.gguf
├── LICENSE                    # Your app license
├── LLAMA_LICENSE             # Llama 3.2 Community License
└── README.md                  # Include "Built with Llama"
```

## Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Exact Match | > 90% | JSON output exactly matches expected |
| Structural Match | > 95% | Correct command/subcommand identified |
| Arg Accuracy | > 93% | Correct arguments parsed |
| Invalid JSON Rate | < 1% | Outputs that fail JSON parsing |
| Latency P50 | < 50ms | Median inference time |
| Latency P99 | < 150ms | 99th percentile inference time |

## Fallback Strategy

When model confidence is low or output is invalid:

1. **Suggest Corrections**: "Did you mean: `gh pr list --state open`?"
2. **Show Similar**: "Similar commands: pr list, pr view, pr create"
3. **Fallback Mode**: "Use --raw to enter traditional CLI mode"
