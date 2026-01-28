# Clitron - Human CLI Library
# Run `just` to see all available commands

# Default recipe: show help
default:
    @just --list

# ============================================================================
# Development
# ============================================================================

# Build all Rust crates
build:
    cargo build --workspace

# Build in release mode
build-release:
    cargo build --workspace --release

# Run all tests
test:
    cargo test --workspace

# Run tests with output
test-verbose:
    cargo test --workspace -- --nocapture

# Check code without building
check:
    cargo check --workspace

# Format code
fmt:
    cargo fmt --all

# Check formatting
fmt-check:
    cargo fmt --all -- --check

# Run clippy lints
lint:
    cargo clippy --workspace -- -D warnings

# Run all checks (format, lint, test)
ci: fmt-check lint test

# Clean build artifacts
clean:
    cargo clean
    rm -rf training/.venv
    rm -rf models/

# ============================================================================
# Clitron Library
# ============================================================================

# Build clitron library only
lib-build:
    cargo build -p clitron

# Test clitron library
lib-test:
    cargo test -p clitron

# Generate library documentation
lib-docs:
    cargo doc -p clitron --no-deps --open

# ============================================================================
# HGH Demo CLI
# ============================================================================

# Build hgh CLI
hgh-build:
    cargo build -p hgh

# Build hgh in release mode
hgh-release:
    cargo build -p hgh --release

# Run hgh with arguments
hgh *ARGS:
    cargo run -p hgh -- {{ARGS}}

# Install hgh locally
hgh-install:
    cargo install --path hgh

# Test hgh
hgh-test:
    cargo test -p hgh

# ============================================================================
# Training Pipeline (uses uv for fast dependency management)
# ============================================================================
#
# Training is organized by CLI name. Each CLI has its own:
#   - Schema: schemas/{name}.yaml
#   - Data: training/data/{name}/
#   - Models: training/models/{name}/
#
# Examples:
#   just train name=gh              # Train model for GitHub CLI
#   just train name=docker          # Train model for Docker CLI
#   just train-generate name=gh     # Only generate data for gh

# Set up Python training environment
train-setup:
    cd training && uv sync --dev

# Full training pipeline for a CLI
train name="gh" num="3000" quant="q4_k_m": train-setup (train-generate name num) (train-sft name) (train-generate-preferences name) (train-dpo name) (train-quantize name quant) (train-eval name quant)
    @echo "Training complete for {{name}}!"
    @echo "Model: training/models/{{name}}/clitron-{{name}}-{{quant}}.gguf"

# Generate training data using Opus 4.5 (skips if data exists, resumes if interrupted)
train-generate name="gh" num="3000":
    #!/usr/bin/env bash
    set -e
    cd training
    mkdir -p "./data/{{name}}"
    if [[ -f "./data/{{name}}/train.jsonl" ]]; then
        echo "Training data already exists (./data/{{name}}/train.jsonl). Skipping generation."
        echo "Use 'just train-generate-force name={{name}}' to regenerate."
    else
        uv run python -m clitron_training.generate_dataset \
            --schema ../schemas/{{name}}.yaml \
            --output ./data/{{name}}/train.jsonl \
            --num-examples {{num}}
    fi

# Force regenerate training data (ignores existing data)
train-generate-force name="gh" num="3000":
    #!/usr/bin/env bash
    set -e
    cd training
    mkdir -p "./data/{{name}}"
    rm -f "./data/{{name}}/train.jsonl" "./data/{{name}}/train_progress.jsonl"
    uv run python -m clitron_training.generate_dataset \
        --schema ../schemas/{{name}}.yaml \
        --output ./data/{{name}}/train.jsonl \
        --num-examples {{num}} \
        --no-resume

# Run SFT training
train-sft name="gh":
    #!/usr/bin/env bash
    set -e
    cd training
    mkdir -p "./models/{{name}}"
    uv run python -m clitron_training.train_sft \
        --config ./configs/sft_config.yaml \
        --data-dir ./data/{{name}} \
        --output-dir ./models/{{name}}/sft

# Generate preference data for DPO
train-generate-preferences name="gh":
    cd training && uv run python -m clitron_training.generate_preferences \
        --model ./models/{{name}}/sft \
        --schema ../schemas/{{name}}.yaml \
        --output ./data/{{name}}/preferences.jsonl

# Run DPO training
train-dpo name="gh":
    #!/usr/bin/env bash
    set -e
    cd training
    uv run python -m clitron_training.train_dpo \
        --config ./configs/dpo_config.yaml \
        --model-dir ./models/{{name}}/sft \
        --output-dir ./models/{{name}}/dpo \
        --preference-file ./data/{{name}}/preferences.jsonl

# Quantize model to GGUF
train-quantize name="gh" quant="q4_k_m":
    cd training && uv run python -m clitron_training.quantize \
        --input ./models/{{name}}/dpo \
        --output ./models/{{name}}/clitron-{{name}}-{{quant}}.gguf \
        --quantization {{quant}}

# Evaluate model
train-eval name="gh" quant="q4_k_m":
    cd training && uv run python -m clitron_training.evaluate \
        --model ./models/{{name}}/clitron-{{name}}-{{quant}}.gguf \
        --test-data ./data/{{name}}/train_val.jsonl

# List available schemas
train-list-schemas:
    @echo "Available schemas:"
    @ls -1 schemas/*.yaml 2>/dev/null | xargs -I{} basename {} .yaml | sed 's/^/  /'

# Run full training pipeline
train-all: train-setup train-generate train-sft train-generate-preferences train-dpo train-quantize train-eval
    @echo "Training complete!"

# ============================================================================
# Model Management
# ============================================================================

# Download pre-trained model
model-download:
    mkdir -p models
    @echo "Downloading pre-trained clitron model..."
    # TODO: Add actual download URL when model is published
    @echo "Model download not yet available. Run 'just train-all' to train locally."

# List available models
model-list:
    @echo "Available models:"
    @ls -lh models/*.gguf 2>/dev/null || echo "  No models found. Run 'just train-all' or 'just model-download'."

# Copy model to clitron cache directory
model-install:
    mkdir -p ~/.clitron/models
    cp models/clitron-q4_k_m.gguf ~/.clitron/models/

# ============================================================================
# Schema Management
# ============================================================================

# Validate a schema file
schema-validate file:
    cd training && .venv/bin/python -m clitron_training.validate_schema \
        --schema {{file}}

# Generate examples from schema (for testing)
schema-examples file num="10":
    cd training && .venv/bin/python -m clitron_training.generate_examples \
        --schema {{file}} \
        --num {{num}}

# ============================================================================
# Documentation
# ============================================================================

# Build all documentation
docs:
    cargo doc --workspace --no-deps

# Open documentation in browser
docs-open:
    cargo doc --workspace --no-deps --open

# ============================================================================
# Release
# ============================================================================

# Create release builds for all platforms (requires cross)
release-all:
    cross build --release --target x86_64-unknown-linux-gnu
    cross build --release --target x86_64-apple-darwin
    cross build --release --target aarch64-apple-darwin
    cross build --release --target x86_64-pc-windows-msvc

# Package release for current platform
release-package:
    #!/usr/bin/env bash
    set -euo pipefail
    VERSION=$(cargo metadata --format-version 1 | jq -r '.packages[] | select(.name == "hgh") | .version')
    TARGET=$(rustc -vV | grep host | cut -d' ' -f2)
    mkdir -p dist
    cargo build --release -p hgh
    cp target/release/hgh dist/hgh-${VERSION}-${TARGET}
    echo "Created dist/hgh-${VERSION}-${TARGET}"

# ============================================================================
# Development Helpers
# ============================================================================

# Watch for changes and rebuild
watch:
    cargo watch -x check

# Run benchmarks
bench:
    cargo bench --workspace

# Open coverage report (requires cargo-llvm-cov)
coverage:
    cargo llvm-cov --workspace --html --open

# Update dependencies
update:
    cargo update

# Check for security vulnerabilities
audit:
    cargo audit

# ============================================================================
# Quick Start
# ============================================================================

# Initialize project for first-time setup
init: train-setup
    @echo "Clitron development environment ready!"
    @echo ""
    @echo "Next steps:"
    @echo "  1. Run 'just train-all' to train a model (requires API key)"
    @echo "  2. Run 'just hgh-build' to build the demo CLI"
    @echo "  3. Run 'just hgh \"show my prs\"' to test"

# Quick demo (assumes model is available)
demo:
    @echo "Demo: Interpreting 'show my open pull requests'"
    just hgh "show my open pull requests"
