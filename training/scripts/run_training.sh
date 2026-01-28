#!/bin/bash
# Full training pipeline for clitron
#
# This script runs the complete training pipeline:
# 1. Generate training data using Opus 4.5
# 2. Run SFT training
# 3. Generate preference data
# 4. Run DPO training
# 5. Quantize model
# 6. Evaluate
#
# Prerequisites:
# - ANTHROPIC_API_KEY environment variable set
# - Python venv with clitron-training installed
# - llama.cpp installed for quantization
#
# Usage:
#   ./scripts/run_training.sh [schema.yaml] [num_examples]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SCHEMA_PATH="${1:-$PROJECT_DIR/../schemas/gh.yaml}"
NUM_EXAMPLES="${2:-10000}"
DATA_DIR="$PROJECT_DIR/data"
MODEL_DIR="$PROJECT_DIR/models"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check prerequisites
check_prereqs() {
    log "Checking prerequisites..."

    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        error "ANTHROPIC_API_KEY environment variable not set"
    fi

    if ! command -v python &> /dev/null; then
        error "Python not found"
    fi

    if ! python -c "import clitron_training" &> /dev/null; then
        warn "clitron_training not installed. Installing..."
        pip install -e "$PROJECT_DIR"
    fi

    log "Prerequisites OK"
}

# Stage 1: Generate training data
generate_data() {
    log "=== Stage 1: Generating Training Data ==="

    mkdir -p "$DATA_DIR"

    python -m clitron_training.generate_dataset \
        --schema "$SCHEMA_PATH" \
        --output "$DATA_DIR/train.jsonl" \
        --num-examples "$NUM_EXAMPLES" \
        --batch-size 50 \
        --validation-split 0.1

    log "Training data generated: $DATA_DIR/train.jsonl"
}

# Stage 2: SFT Training
train_sft() {
    log "=== Stage 2: SFT Training ==="

    python -m clitron_training.train_sft \
        --config "$PROJECT_DIR/configs/sft_config.yaml"

    log "SFT model saved: $MODEL_DIR/sft"
}

# Stage 3: Generate preference data
generate_preferences() {
    log "=== Stage 3: Generating Preference Data ==="

    python -m clitron_training.generate_preferences \
        --model "$MODEL_DIR/sft" \
        --schema "$SCHEMA_PATH" \
        --output "$DATA_DIR/preferences.jsonl" \
        --num-pairs 5000

    log "Preference data generated: $DATA_DIR/preferences.jsonl"
}

# Stage 4: DPO Training
train_dpo() {
    log "=== Stage 4: DPO Training ==="

    python -m clitron_training.train_dpo \
        --config "$PROJECT_DIR/configs/dpo_config.yaml"

    log "DPO model saved: $MODEL_DIR/dpo"
}

# Stage 5: Quantization
quantize_model() {
    log "=== Stage 5: Quantization ==="

    python -m clitron_training.quantize \
        --input "$MODEL_DIR/dpo" \
        --output "$MODEL_DIR/clitron-q4_k_m.gguf" \
        --quantization q4_k_m

    log "Quantized model saved: $MODEL_DIR/clitron-q4_k_m.gguf"
}

# Stage 6: Evaluation
evaluate_model() {
    log "=== Stage 6: Evaluation ==="

    # Create test set from validation data
    TEST_FILE="$DATA_DIR/train_val.jsonl"

    if [[ -f "$TEST_FILE" ]]; then
        python -m clitron_training.evaluate \
            --model "$MODEL_DIR/clitron-q4_k_m.gguf" \
            --test-data "$TEST_FILE" \
            --output "$MODEL_DIR/eval_results.json"
    else
        warn "Validation file not found, skipping evaluation"
    fi
}

# Main
main() {
    log "Starting clitron training pipeline"
    log "Schema: $SCHEMA_PATH"
    log "Examples: $NUM_EXAMPLES"

    check_prereqs
    generate_data
    train_sft
    generate_preferences
    train_dpo
    quantize_model
    evaluate_model

    log "=== Training Complete ==="
    log "Final model: $MODEL_DIR/clitron-q4_k_m.gguf"
}

main "$@"
