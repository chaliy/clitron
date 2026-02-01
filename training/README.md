# Clitron Training

Training pipeline for clitron CLI interpreter models.

## Setup

```bash
# Using uv (recommended)
uv sync --dev

# Or using pip
pip install -e ".[dev]"
```

## Usage

### Generate Training Data

```bash
uv run python -m clitron_training.generate_dataset \
    --schema ../schemas/gh.yaml \
    --output ./data/train.jsonl \
    --num-examples 10000
```

### Train Model (SFT)

```bash
uv run python -m clitron_training.train_sft \
    --config ./configs/sft_config.yaml
```

### Train Model (DPO)

```bash
uv run python -m clitron_training.train_dpo \
    --config ./configs/dpo_config.yaml
```

### Quantize Model

```bash
uv run python -m clitron_training.quantize \
    --input ./models/dpo \
    --output ./models/clitron-q4_k_m.gguf \
    --quantization q4_k_m
```

### Evaluate Model

```bash
uv run python -m clitron_training.evaluate \
    --model ./models/clitron-q4_k_m.gguf \
    --test-data ./data/test.jsonl
```

## Requirements

- Python 3.10+
- ANTHROPIC_API_KEY for data generation
- GPU recommended for training (but not required)
