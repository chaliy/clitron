//! # Clitron
//!
//! A library for building human-friendly CLI interfaces using small language models.
//!
//! Clitron allows users to interact with CLI tools using natural language instead of
//! memorizing exact command syntax. It uses a small, fine-tuned language model to
//! interpret natural language input and map it to structured CLI commands.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use clitron::{Interpreter, CommandSchema};
//!
//! // Load your CLI schema
//! let schema = CommandSchema::from_yaml_file("schema.yaml")?;
//!
//! // Create an interpreter
//! let interpreter = Interpreter::with_schema(schema)?;
//!
//! // Interpret natural language input
//! let cmd = interpreter.interpret("show my open pull requests")?;
//!
//! println!("Command: {}", cmd.command);
//! println!("Args: {:?}", cmd.args);
//! ```
//!
//! ## Architecture
//!
//! Clitron consists of several components:
//!
//! - **Schema**: Defines the structure of your CLI commands
//! - **Interpreter**: Translates natural language to structured commands
//! - **Model**: Handles inference using a small language model (GGUF format)
//!
//! ## Training
//!
//! Clitron models are trained using:
//!
//! 1. A large LLM (e.g., Claude Opus) to generate training data
//! 2. Supervised fine-tuning on a small base model (e.g., Llama 3.2 1B)
//! 3. DPO (Direct Preference Optimization) for improved output quality
//! 4. Quantization to GGUF format for efficient deployment
//!
//! See the `training/` directory for training scripts and documentation.

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

pub mod command;
pub mod context;
pub mod error;
pub mod interpreter;
pub mod model;
pub mod progress;
pub mod schema;

// Re-export main types at crate root
pub use command::InterpretedCommand;
pub use context::{Context, GitContext};
pub use error::{ClitronError, Result, ValidationError};
pub use interpreter::{Interpreter, InterpreterConfig, Suggestion};
pub use model::{Model, ModelConfig, ModelManager, ModelStatus};
pub use progress::{
    clear_terminal_progress, set_terminal_progress, DownloadProgress, DownloadTracker,
    TerminalOutput, TerminalProgressState,
};
pub use schema::{ArgType, Argument, Command, CommandSchema, Flag, Subcommand};

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default confidence threshold for accepting interpretations.
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.7;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::command::InterpretedCommand;
    pub use crate::context::{Context, GitContext};
    pub use crate::error::{ClitronError, Result};
    pub use crate::interpreter::Interpreter;
    pub use crate::schema::CommandSchema;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_prelude_imports() {
        // Just verify that prelude types are accessible
        use crate::prelude::*;

        let _ = InterpretedCommand::new("test");
    }
}
