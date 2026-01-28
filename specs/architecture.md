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

1. **Context Gathering**: CLI collects environment context (git state, etc.)
2. **Input Phase**: User provides natural language input
3. **Prompt Construction**: Input + context formatted for model
4. **Tokenization**: Prompt is tokenized using the model's tokenizer
5. **Inference**: Small LLM generates structured output (JSON)
6. **Parsing**: JSON is parsed into Rust structs
7. **Mapping**: Structs are mapped to clap-compatible arguments
8. **Execution**: Host CLI executes the interpreted command

## Context-Aware Interpretation

The interpreter accepts optional context that improves interpretation accuracy:

```rust
pub struct Context {
    /// Git repository context (if in a git repo)
    pub git: Option<GitContext>,
    /// Custom key-value context from the CLI
    pub custom: HashMap<String, String>,
}

pub struct GitContext {
    pub current_branch: Option<String>,
    pub has_uncommitted_changes: bool,
    pub has_staged_changes: bool,
    pub is_worktree: bool,
    pub repo_owner: Option<String>,
    pub repo_name: Option<String>,
    pub current_pr: Option<u64>,      // PR associated with current branch
    pub upstream_branch: Option<String>,
}
```

### Context Examples

| User Input | Without Context | With Context |
|------------|-----------------|--------------|
| "merge this" | ❌ Clarification needed | ✅ `pr merge 123` (knows current PR) |
| "push" | ❌ Ambiguous | ✅ `git push origin feature-x` (knows branch) |
| "show the pr" | ❌ Which PR? | ✅ `pr view 123` (knows current PR) |
| "diff" | ❌ Diff what? | ✅ `pr diff` (has uncommitted changes) |

### Prompt Format with Context

```
<|system|>
You are a CLI interpreter. Output only valid JSON.
<|context|>
repo: owner/repo-name
branch: feature/add-login
pr: #123
uncommitted: true
staged: false
<|user|>
merge this and delete the branch
<|assistant|>
{"type": "command", "command": "pr", "subcommand": "merge", "args": {"number": 123}, "flags": ["delete-branch"], "confidence": 0.95}
```

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
