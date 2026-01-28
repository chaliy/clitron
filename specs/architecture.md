# Clitron Architecture

## Overview

Clitron is a library that transforms natural language CLI input into structured commands. It uses a small, fine-tuned language model to interpret human-friendly input and map it to traditional CLI argument structures (compatible with clap).

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Input                                │
│              "show my open pull requests"                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Clitron Library (Rust)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Tokenizer  │→ │ Small LLM   │→ │  Command Parser/Mapper  │  │
│  │  (GGUF)     │  │ (GGUF)      │  │  (to clap Args)         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Structured Command                            │
│              Command { action: "pr", subaction: "list",          │
│                        filters: { state: "open", author: "me" }} │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Input Phase**: User provides natural language input
2. **Tokenization**: Input is tokenized using the model's tokenizer
3. **Inference**: Small LLM generates structured output (JSON)
4. **Parsing**: JSON is parsed into Rust structs
5. **Mapping**: Structs are mapped to clap-compatible arguments
6. **Execution**: Host CLI executes the interpreted command

## Key Design Decisions

### 1. Model Format: GGUF

- Use GGUF format for model distribution
- Enables CPU inference without GPU dependencies
- Can be embedded directly in CLI binary
- Use `llama.cpp` bindings via `llama-cpp-rs` or `candle`

### 2. Model Output Format: JSON

The model outputs structured JSON:

```json
{
  "command": "pr",
  "subcommand": "list",
  "args": {
    "state": "open",
    "author": "@me"
  },
  "flags": ["--web"]
}
```

### 3. Schema-Driven Interpretation

Each CLI defines a schema that:
- Describes available commands and their arguments
- Provides examples for few-shot learning
- Is used both for training and runtime context

### 4. Confidence Scoring

Model provides confidence scores. Low confidence triggers:
- Clarification prompts
- Suggestion of closest matches
- Fallback to traditional CLI mode

## Component Details

### Clitron Library (Rust Crate)

```rust
// Core trait that CLI applications implement
pub trait HumanCli {
    /// Define the command schema
    fn schema() -> CommandSchema;

    /// Execute the interpreted command
    fn execute(cmd: InterpretedCommand) -> Result<()>;
}

// Main interpreter
pub struct Interpreter {
    model: Model,
    schema: CommandSchema,
}

impl Interpreter {
    pub fn interpret(&self, input: &str) -> Result<InterpretedCommand>;
}
```

### Training Pipeline (Python)

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Schema     │ →  │  Large LLM   │ →  │   Training   │
│ Definition   │    │  (Opus 4.5)  │    │   Dataset    │
└──────────────┘    └──────────────┘    └──────────────┘
                                              │
                                              ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Fine-tuned  │ ←  │     RL       │ ←  │    Base      │
│    Model     │    │   Training   │    │    Model     │
└──────────────┘    └──────────────┘    └──────────────┘
```

## File Structure

```
clitron/
├── specs/                    # Specifications
├── clitron/                  # Main Rust library
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs           # Library entry
│       ├── interpreter.rs   # Core interpreter
│       ├── schema.rs        # Command schema definitions
│       ├── model.rs         # Model loading/inference
│       └── mapper.rs        # clap integration
├── hgh/                      # Demo: Human GitHub CLI
│   ├── Cargo.toml
│   └── src/
│       └── main.rs
├── training/                 # Training infrastructure
│   ├── pyproject.toml
│   ├── src/
│   │   ├── generate_dataset.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── configs/
│   │   └── training_config.yaml
│   └── scripts/
│       └── run_training.sh
├── models/                   # Trained models (gitignored)
├── AGENTS.md
├── justfile
└── README.md
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Model Size | < 500MB (quantized) |
| Inference Latency | < 100ms |
| Accuracy | > 95% on common commands |
| Memory Usage | < 512MB |

## Security Considerations

1. **Input Sanitization**: All interpreted commands are validated against schema
2. **No Code Execution**: Model output is never executed directly
3. **Sandboxed Inference**: Model runs in isolated context
4. **Audit Logging**: All interpretations can be logged for review
