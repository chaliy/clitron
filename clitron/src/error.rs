//! Error types for clitron.

use thiserror::Error;

/// Main error type for clitron operations.
#[derive(Debug, Error)]
pub enum ClitronError {
    /// Model loading failed.
    #[error("Model loading failed: {0}")]
    ModelLoad(String),

    /// Inference failed.
    #[error("Inference failed: {0}")]
    Inference(String),

    /// Invalid JSON output from model.
    #[error("Invalid JSON output: {0}")]
    InvalidJson(#[from] serde_json::Error),

    /// Schema validation failed.
    #[error("Schema validation failed: {0}")]
    SchemaValidation(String),

    /// Command validation failed.
    #[error("Command validation failed: {0}")]
    CommandValidation(#[from] ValidationError),

    /// Low confidence interpretation.
    #[error("Low confidence ({confidence:.2}): {suggestion}")]
    LowConfidence {
        /// The confidence score.
        confidence: f32,
        /// Suggested command.
        suggestion: String,
    },

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Model not found.
    #[error("Model not found at {path}. Run model download first.")]
    ModelNotFound {
        /// Expected model path.
        path: String,
    },

    /// YAML parsing error.
    #[error("YAML parsing error: {0}")]
    YamlParse(#[from] serde_yaml::Error),
}

/// Validation errors for interpreted commands.
#[derive(Debug, Error)]
pub enum ValidationError {
    /// Unknown command.
    #[error("Unknown command: {0}")]
    UnknownCommand(String),

    /// Unknown subcommand.
    #[error("Unknown subcommand '{subcommand}' for command '{command}'")]
    UnknownSubcommand {
        /// The parent command.
        command: String,
        /// The unknown subcommand.
        subcommand: String,
    },

    /// Invalid argument type.
    #[error("Invalid argument type for '{name}': expected {expected}, got {actual}")]
    InvalidArgType {
        /// Argument name.
        name: String,
        /// Expected type.
        expected: String,
        /// Actual type.
        actual: String,
    },

    /// Unknown argument.
    #[error("Unknown argument: {0}")]
    UnknownArgument(String),

    /// Unknown flag.
    #[error("Unknown flag: {0}")]
    UnknownFlag(String),

    /// Missing required argument.
    #[error("Missing required argument: {0}")]
    MissingRequired(String),
}

/// Result type alias for clitron operations.
pub type Result<T> = std::result::Result<T, ClitronError>;
