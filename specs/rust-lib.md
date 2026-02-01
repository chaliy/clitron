# Rust Library Specification

## Overview

The `clitron` crate provides the core functionality for interpreting natural language CLI commands using a small embedded language model.

## Crate Structure

```
clitron/
├── Cargo.toml
└── src/
    ├── lib.rs           # Public API exports
    ├── interpreter.rs   # Core interpretation logic
    ├── schema.rs        # Command schema definitions
    ├── model.rs         # Model loading and inference
    ├── mapper.rs        # Clap integration
    ├── tokenizer.rs     # Tokenization utilities
    ├── cache.rs         # Model caching
    └── error.rs         # Error types
```

## Public API

### Core Types

```rust
/// Represents a CLI command schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandSchema {
    pub name: String,
    pub description: String,
    pub commands: Vec<Command>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Command {
    pub name: String,
    pub description: String,
    pub subcommands: Vec<Subcommand>,
    pub args: Vec<Argument>,
    pub flags: Vec<Flag>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subcommand {
    pub name: String,
    pub description: String,
    pub args: Vec<Argument>,
    pub flags: Vec<Flag>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Argument {
    pub name: String,
    pub arg_type: ArgType,
    pub required: bool,
    pub default: Option<serde_json::Value>,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArgType {
    String,
    Integer,
    Float,
    Boolean,
    Enum(Vec<String>),
    Path,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flag {
    pub name: String,
    pub short: Option<char>,
    pub description: String,
}
```

### Interpreted Command

```rust
/// The result of interpreting a natural language command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretedCommand {
    pub command: String,
    pub subcommand: Option<String>,
    pub args: HashMap<String, serde_json::Value>,
    pub flags: Vec<String>,
    pub confidence: f32,
    pub raw_output: String,
}

impl InterpretedCommand {
    /// Check if interpretation confidence is above threshold
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }

    /// Convert to clap-style argument vector
    pub fn to_args(&self) -> Vec<String>;

    /// Get argument value as specific type
    pub fn get_arg<T: DeserializeOwned>(&self, name: &str) -> Option<T>;

    /// Check if flag is set
    pub fn has_flag(&self, name: &str) -> bool;
}
```

### Context

Context provides environmental information that improves interpretation accuracy:

```rust
/// Environmental context for interpretation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Context {
    /// Git repository context
    pub git: Option<GitContext>,
    /// Custom key-value pairs for CLI-specific context
    pub custom: HashMap<String, String>,
}

/// Git repository state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitContext {
    /// Current branch name
    pub current_branch: Option<String>,
    /// Whether there are uncommitted changes
    pub has_uncommitted_changes: bool,
    /// Whether there are staged changes
    pub has_staged_changes: bool,
    /// Whether we're in a worktree
    pub is_worktree: bool,
    /// Repository owner (from remote URL)
    pub repo_owner: Option<String>,
    /// Repository name (from remote URL)
    pub repo_name: Option<String>,
    /// PR number associated with current branch (if any)
    pub current_pr: Option<u64>,
    /// Upstream tracking branch
    pub upstream_branch: Option<String>,
}

impl Context {
    /// Create empty context
    pub fn new() -> Self;

    /// Create context with git information (auto-detected)
    pub fn with_git() -> Result<Self, ClitronError>;

    /// Add custom context value
    pub fn set(&mut self, key: &str, value: &str) -> &mut Self;

    /// Format context for model prompt
    pub fn to_prompt_string(&self) -> String;
}

impl GitContext {
    /// Detect git context from current directory
    pub fn detect() -> Result<Option<Self>, ClitronError>;

    /// Detect with PR lookup via GitHub API
    pub fn detect_with_pr(github_token: Option<&str>) -> Result<Option<Self>, ClitronError>;
}
```

### Interpreter

```rust
/// Configuration for the interpreter
#[derive(Debug, Clone)]
pub struct InterpreterConfig {
    /// Path to GGUF model file
    pub model_path: PathBuf,

    /// Context window size (default: 2048)
    pub context_size: usize,

    /// Maximum tokens to generate (default: 256)
    pub max_tokens: usize,

    /// Temperature for generation (default: 0.1)
    pub temperature: f32,

    /// Minimum confidence threshold (default: 0.7)
    pub confidence_threshold: f32,

    /// Number of threads for inference (default: num_cpus)
    pub num_threads: usize,
}

impl Default for InterpreterConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("~/.clitron/models/clitron.gguf"),
            context_size: 2048,
            max_tokens: 256,
            temperature: 0.1,
            confidence_threshold: 0.7,
            num_threads: num_cpus::get(),
        }
    }
}

/// Main interpreter struct
pub struct Interpreter {
    model: Model,
    schema: CommandSchema,
    config: InterpreterConfig,
}

impl Interpreter {
    /// Create new interpreter with schema and config
    pub fn new(
        schema: CommandSchema,
        config: InterpreterConfig
    ) -> Result<Self, ClitronError>;

    /// Create with default config
    pub fn with_schema(schema: CommandSchema) -> Result<Self, ClitronError>;

    /// Interpret natural language input (no context)
    pub fn interpret(&self, input: &str) -> Result<InterpretedCommand, ClitronError>;

    /// Interpret with environmental context
    pub fn interpret_with_context(
        &self,
        input: &str,
        context: &Context
    ) -> Result<InterpretedCommand, ClitronError>;

    /// Get suggestions for ambiguous input
    pub fn suggest(&self, input: &str, context: Option<&Context>) -> Result<Vec<Suggestion>, ClitronError>;

    /// Validate command against schema
    pub fn validate(&self, cmd: &InterpretedCommand) -> Result<(), ValidationError>;
}
```

### Model Management

```rust
/// Model download and caching utilities
pub struct ModelManager {
    cache_dir: PathBuf,
}

impl ModelManager {
    /// Create with default cache directory (~/.clitron/models)
    pub fn new() -> Self;

    /// Create with custom cache directory
    pub fn with_cache_dir(path: PathBuf) -> Self;

    /// Download model if not cached
    pub async fn ensure_model(&self, model_id: &str) -> Result<PathBuf, ClitronError>;

    /// Get path to cached model
    pub fn model_path(&self, model_id: &str) -> PathBuf;

    /// Check if model is cached
    pub fn is_cached(&self, model_id: &str) -> bool;

    /// Clear model cache
    pub fn clear_cache(&self) -> Result<(), ClitronError>;
}
```

### Clap Integration

```rust
/// Trait for types that can be derived from interpreted commands
pub trait FromInterpretedCommand: Sized {
    fn from_interpreted(cmd: &InterpretedCommand) -> Result<Self, ClitronError>;
}

/// Extension trait for clap Command
pub trait ClapHumanExt {
    /// Add human-friendly interpretation to clap command
    fn with_human_interface(self, schema: CommandSchema) -> HumanCommand;
}

/// Wrapper that adds human interpretation to clap
pub struct HumanCommand {
    inner: clap::Command,
    interpreter: Interpreter,
}

impl HumanCommand {
    /// Parse arguments, using interpretation if needed
    pub fn get_matches_human(self) -> ArgMatches;

    /// Try to parse traditional args first, fall back to interpretation
    pub fn get_matches_smart(self) -> ArgMatches;
}

// Derive macro for automatic implementation
#[derive(clap::Parser, clitron::HumanCli)]
#[clitron(schema = "schema.yaml")]
pub struct MyCli {
    #[clap(subcommand)]
    command: Commands,
}
```

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum ClitronError {
    #[error("Model loading failed: {0}")]
    ModelLoad(String),

    #[error("Inference failed: {0}")]
    Inference(String),

    #[error("Invalid JSON output: {0}")]
    InvalidJson(#[from] serde_json::Error),

    #[error("Schema validation failed: {0}")]
    Validation(String),

    #[error("Low confidence ({confidence:.2}): {suggestion}")]
    LowConfidence {
        confidence: f32,
        suggestion: String,
    },

    #[error("Unknown command: {0}")]
    UnknownCommand(String),

    #[error("Missing required argument: {0}")]
    MissingArgument(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Model not found. Run `clitron model download` first.")]
    ModelNotFound,
}

#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Unknown command: {0}")]
    UnknownCommand(String),

    #[error("Unknown subcommand: {0}")]
    UnknownSubcommand(String),

    #[error("Invalid argument type for {name}: expected {expected}")]
    InvalidArgType { name: String, expected: String },

    #[error("Unknown argument: {0}")]
    UnknownArgument(String),

    #[error("Unknown flag: {0}")]
    UnknownFlag(String),
}
```

## Usage Examples

### Basic Usage

```rust
use clitron::{Interpreter, CommandSchema, InterpreterConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load schema from YAML
    let schema = CommandSchema::from_yaml_file("schema.yaml")?;

    // Create interpreter
    let interpreter = Interpreter::with_schema(schema)?;

    // Interpret user input
    let input = "show me open pull requests";
    let cmd = interpreter.interpret(input)?;

    println!("Command: {}", cmd.command);
    println!("Subcommand: {:?}", cmd.subcommand);
    println!("Args: {:?}", cmd.args);
    println!("Confidence: {:.2}", cmd.confidence);

    Ok(())
}
```

### With Clap Integration

```rust
use clap::Parser;
use clitron::ClapHumanExt;

#[derive(Parser)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Parser)]
enum Commands {
    Pr(PrCommand),
    Issue(IssueCommand),
}

fn main() {
    // Get raw args
    let args: Vec<String> = std::env::args().collect();

    // If it looks like natural language, interpret it
    if args.len() == 2 && !args[1].starts_with('-') {
        let interpreter = get_interpreter();
        match interpreter.interpret(&args[1]) {
            Ok(cmd) if cmd.is_confident(0.8) => {
                // Convert to traditional args and re-parse
                let new_args = cmd.to_args();
                let cli = Cli::parse_from(new_args);
                run(cli);
            }
            Ok(cmd) => {
                eprintln!("Did you mean: gh {}", cmd.to_args().join(" "));
            }
            Err(e) => {
                eprintln!("Could not interpret: {}", e);
            }
        }
    } else {
        // Traditional parsing
        let cli = Cli::parse();
        run(cli);
    }
}
```

### Schema Definition

```rust
use clitron::{CommandSchema, Command, Subcommand, Argument, ArgType, Flag};

let schema = CommandSchema {
    name: "gh".into(),
    description: "GitHub CLI".into(),
    commands: vec![
        Command {
            name: "pr".into(),
            description: "Work with pull requests".into(),
            subcommands: vec![
                Subcommand {
                    name: "list".into(),
                    description: "List pull requests".into(),
                    args: vec![
                        Argument {
                            name: "state".into(),
                            arg_type: ArgType::Enum(vec![
                                "open".into(),
                                "closed".into(),
                                "merged".into(),
                                "all".into(),
                            ]),
                            required: false,
                            default: Some(json!("open")),
                            description: "PR state filter".into(),
                        },
                    ],
                    flags: vec![
                        Flag {
                            name: "web".into(),
                            short: Some('w'),
                            description: "Open in browser".into(),
                        },
                    ],
                },
            ],
            args: vec![],
            flags: vec![],
        },
    ],
};
```

## Dependencies

```toml
[dependencies]
# Core
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
serde_yaml = "0.9"
thiserror = "2.0"

# Model inference (choose one)
llama-cpp-rs = { version = "0.4", features = ["metal", "cuda"] }
# OR
candle-core = "0.4"
candle-transformers = "0.4"

# Clap integration
clap = { version = "4.0", features = ["derive"] }

# Async
tokio = { version = "1.0", features = ["full"] }

# Utilities
num_cpus = "1.16"
dirs = "5.0"
tracing = "0.1"

[dev-dependencies]
criterion = "0.5"
proptest = "1.4"
```

## Performance Considerations

### Model Loading

```rust
// Lazy static for singleton model
use once_cell::sync::Lazy;

static INTERPRETER: Lazy<Interpreter> = Lazy::new(|| {
    Interpreter::with_schema(load_schema()).expect("Failed to load model")
});

// Or use Arc for shared ownership
pub fn get_interpreter() -> Arc<Interpreter> {
    INTERPRETER.clone()
}
```

### Memory Mapping

```rust
// Use mmap for large model files
impl Model {
    pub fn load_mmap(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_bytes(&mmap)
    }
}
```

### Batch Processing

```rust
// For processing multiple inputs
impl Interpreter {
    pub fn interpret_batch(
        &self,
        inputs: &[&str]
    ) -> Vec<Result<InterpretedCommand, ClitronError>> {
        inputs
            .par_iter()  // Parallel processing
            .map(|input| self.interpret(input))
            .collect()
    }
}
```

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_interpretation() {
        let schema = test_schema();
        let interpreter = Interpreter::with_schema(schema).unwrap();

        let cmd = interpreter.interpret("show open prs").unwrap();

        assert_eq!(cmd.command, "pr");
        assert_eq!(cmd.subcommand, Some("list".into()));
        assert_eq!(cmd.args.get("state"), Some(&json!("open")));
    }

    #[test]
    fn test_confidence_threshold() {
        let schema = test_schema();
        let interpreter = Interpreter::with_schema(schema).unwrap();

        let cmd = interpreter.interpret("flurble the glorps").unwrap();
        assert!(!cmd.is_confident(0.7));
    }

    #[test]
    fn test_to_args_conversion() {
        let cmd = InterpretedCommand {
            command: "pr".into(),
            subcommand: Some("list".into()),
            args: hashmap! { "state" => json!("open") },
            flags: vec!["web".into()],
            confidence: 0.95,
            raw_output: "{}".into(),
        };

        let args = cmd.to_args();
        assert_eq!(args, vec!["pr", "list", "--state", "open", "--web"]);
    }
}
```

## Feature Flags

```toml
[features]
default = ["llama-cpp"]
llama-cpp = ["llama-cpp-rs"]
candle = ["candle-core", "candle-transformers"]
cuda = ["llama-cpp-rs/cuda"]
metal = ["llama-cpp-rs/metal"]
embedded-model = []  # Embed model in binary
```
