//! Core interpreter for natural language CLI commands.

use std::path::Path;

use crate::command::InterpretedCommand;
use crate::context::Context;
use crate::error::{ClitronError, Result, ValidationError};
use crate::model::{Model, ModelConfig};
use crate::schema::{ArgType, CommandSchema};

/// Configuration for the interpreter.
#[derive(Debug, Clone)]
pub struct InterpreterConfig {
    /// Model configuration.
    pub model: ModelConfig,

    /// Minimum confidence threshold for accepting interpretations.
    pub confidence_threshold: f32,

    /// Whether to validate commands against schema.
    pub validate: bool,
}

impl Default for InterpreterConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            confidence_threshold: 0.7,
            validate: true,
        }
    }
}

/// Main interpreter for natural language CLI commands.
pub struct Interpreter {
    model: Model,
    schema: CommandSchema,
    config: InterpreterConfig,
}

impl Interpreter {
    /// Create a new interpreter with the given schema and configuration.
    pub fn new(schema: CommandSchema, config: InterpreterConfig) -> Result<Self> {
        let model = Model::load_with_config(config.model.clone(), schema.clone())?;

        Ok(Self {
            model,
            schema,
            config,
        })
    }

    /// Create an interpreter with default configuration.
    pub fn with_schema(schema: CommandSchema) -> Result<Self> {
        Self::new(schema, InterpreterConfig::default())
    }

    /// Create an interpreter with a custom model path.
    pub fn with_model_path(schema: CommandSchema, model_path: impl AsRef<Path>) -> Result<Self> {
        let config = InterpreterConfig {
            model: ModelConfig {
                model_path: model_path.as_ref().to_path_buf(),
                ..Default::default()
            },
            ..Default::default()
        };

        Self::new(schema, config)
    }

    /// Get the schema.
    pub fn schema(&self) -> &CommandSchema {
        &self.schema
    }

    /// Get the configuration.
    pub fn config(&self) -> &InterpreterConfig {
        &self.config
    }

    /// Interpret a natural language input.
    ///
    /// Returns an `InterpretedCommand` if successful. If confidence is below
    /// the threshold, returns a `LowConfidence` error with a suggestion.
    pub fn interpret(&self, input: &str) -> Result<InterpretedCommand> {
        let input = input.trim();

        if input.is_empty() {
            return Err(ClitronError::Inference("Empty input".into()));
        }

        tracing::debug!("Interpreting: {}", input);

        // Run inference
        let mut cmd = self.model.infer(input)?;
        cmd.raw_output = input.to_string();

        // Validate against schema
        if self.config.validate {
            self.validate(&cmd)?;
        }

        // Check confidence
        if !cmd.is_confident(self.config.confidence_threshold) {
            return Err(ClitronError::LowConfidence {
                confidence: cmd.confidence,
                suggestion: cmd.to_shell_command(&self.schema.cli_name),
            });
        }

        Ok(cmd)
    }

    /// Interpret with environmental context.
    ///
    /// Context provides information about the current environment (git state, etc.)
    /// that helps the model make better interpretations.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let context = Context::with_git()?;
    /// let cmd = interpreter.interpret_with_context("merge this", &context)?;
    /// ```
    pub fn interpret_with_context(
        &self,
        input: &str,
        context: &Context,
    ) -> Result<InterpretedCommand> {
        let input = input.trim();

        if input.is_empty() {
            return Err(ClitronError::Inference("Empty input".into()));
        }

        tracing::debug!("Interpreting with context: {}", input);
        tracing::debug!("Context: {}", context.to_prompt_string());

        // Run inference with context
        let mut cmd = self.model.infer_with_context(input, context)?;
        cmd.raw_output = input.to_string();

        // Validate against schema
        if self.config.validate {
            self.validate(&cmd)?;
        }

        // Check confidence
        if !cmd.is_confident(self.config.confidence_threshold) {
            return Err(ClitronError::LowConfidence {
                confidence: cmd.confidence,
                suggestion: cmd.to_shell_command(&self.schema.cli_name),
            });
        }

        Ok(cmd)
    }

    /// Interpret with a lower confidence threshold.
    ///
    /// Useful when you want to get a result even with low confidence.
    pub fn interpret_lenient(&self, input: &str) -> Result<InterpretedCommand> {
        let input = input.trim();

        if input.is_empty() {
            return Err(ClitronError::Inference("Empty input".into()));
        }

        let mut cmd = self.model.infer(input)?;
        cmd.raw_output = input.to_string();

        if self.config.validate {
            self.validate(&cmd)?;
        }

        Ok(cmd)
    }

    /// Validate a command against the schema.
    pub fn validate(&self, cmd: &InterpretedCommand) -> Result<()> {
        // Find the command
        let command = self
            .schema
            .find_command(&cmd.command)
            .ok_or_else(|| ValidationError::UnknownCommand(cmd.command.clone()))?;

        // If there's a subcommand, validate it
        if let Some(ref sub_name) = cmd.subcommand {
            let subcommand = command.find_subcommand(sub_name).ok_or_else(|| {
                ValidationError::UnknownSubcommand {
                    command: cmd.command.clone(),
                    subcommand: sub_name.clone(),
                }
            })?;

            // Validate arguments
            for (arg_name, arg_value) in &cmd.args {
                let arg_def = subcommand
                    .args
                    .iter()
                    .find(|a| &a.name == arg_name)
                    .or_else(|| command.args.iter().find(|a| &a.name == arg_name))
                    .ok_or_else(|| ValidationError::UnknownArgument(arg_name.clone()))?;

                validate_arg_type(arg_name, arg_value, &arg_def.arg_type)?;
            }

            // Validate flags
            for flag_name in &cmd.flags {
                let flag_exists = subcommand.flags.iter().any(|f| &f.name == flag_name)
                    || command.flags.iter().any(|f| &f.name == flag_name);

                if !flag_exists {
                    return Err(ValidationError::UnknownFlag(flag_name.clone()).into());
                }
            }

            // Check required arguments
            for arg in &subcommand.args {
                if arg.required && !cmd.args.contains_key(&arg.name) {
                    return Err(ValidationError::MissingRequired(arg.name.clone()).into());
                }
            }
        }

        Ok(())
    }

    /// Get suggestions for ambiguous or unknown input.
    pub fn suggest(&self, input: &str) -> Vec<Suggestion> {
        let input_lower = input.to_lowercase();
        let mut suggestions = Vec::new();

        // Find similar commands
        for command in &self.schema.commands {
            let similarity = string_similarity(&input_lower, &command.name);

            if similarity > 0.3 {
                for subcommand in &command.subcommands {
                    suggestions.push(Suggestion {
                        command: format!("{} {}", command.name, subcommand.name),
                        description: subcommand.description.clone(),
                        score: similarity,
                    });
                }
            }
        }

        // Sort by score
        suggestions.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        suggestions.truncate(5);

        suggestions
    }
}

/// A command suggestion.
#[derive(Debug, Clone)]
pub struct Suggestion {
    /// The suggested command.
    pub command: String,

    /// Description of the command.
    pub description: String,

    /// Similarity score (0.0 to 1.0).
    pub score: f32,
}

/// Validate an argument value against its expected type.
fn validate_arg_type(name: &str, value: &serde_json::Value, expected: &ArgType) -> Result<()> {
    use crate::schema::SimpleArgType;

    let valid = match expected {
        ArgType::Simple(simple) => match simple {
            SimpleArgType::String => value.is_string(),
            SimpleArgType::Integer => value.is_i64() || value.is_u64(),
            SimpleArgType::Float => value.is_f64() || value.is_i64(),
            SimpleArgType::Boolean => value.is_boolean(),
            SimpleArgType::Path => value.is_string(),
        },
        ArgType::Enum(enum_type) => value
            .as_str()
            .map(|s| enum_type.values.contains(&s.to_string()))
            .unwrap_or(false),
    };

    if !valid {
        return Err(ValidationError::InvalidArgType {
            name: name.to_string(),
            expected: expected.to_string(),
            actual: value_type_name(value),
        }
        .into());
    }

    Ok(())
}

/// Get a human-readable name for a JSON value type.
fn value_type_name(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".into(),
        serde_json::Value::Bool(_) => "boolean".into(),
        serde_json::Value::Number(_) => "number".into(),
        serde_json::Value::String(_) => "string".into(),
        serde_json::Value::Array(_) => "array".into(),
        serde_json::Value::Object(_) => "object".into(),
    }
}

/// Simple string similarity (Jaccard index on words).
fn string_similarity(a: &str, b: &str) -> f32 {
    let a_words: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let b_words: std::collections::HashSet<&str> = b.split_whitespace().collect();

    if a_words.is_empty() && b_words.is_empty() {
        return 1.0;
    }

    let intersection = a_words.intersection(&b_words).count();
    let union = a_words.union(&b_words).count();

    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_schema() -> CommandSchema {
        CommandSchema::from_yaml(
            r#"
cli_name: "test"
description: "Test CLI"
commands:
  - name: "pr"
    description: "Pull requests"
    aliases: ["pull-request"]
    subcommands:
      - name: "list"
        description: "List PRs"
        aliases: ["show", "display"]
        args:
          - name: "state"
            type: string
            required: false
            default: "open"
          - name: "author"
            type: string
        flags:
          - name: "web"
            short: "w"
"#,
        )
        .unwrap()
    }

    // Note: This test is ignored because it requires a downloaded model.
    // Run with: cargo test --ignored test_interpret_basic
    #[test]
    #[ignore]
    fn test_interpret_basic() {
        let schema = test_schema();
        let interpreter = Interpreter::with_schema(schema).unwrap();

        let result = interpreter.interpret_lenient("show my open prs");

        // Should at least not error
        assert!(result.is_ok() || matches!(result, Err(ClitronError::LowConfidence { .. })));
    }

    #[test]
    fn test_validate_command() {
        let schema = test_schema();

        // Test validation using schema directly (doesn't require model)
        // Valid command
        let cmd = InterpretedCommand::new("pr")
            .with_subcommand("list")
            .with_arg("state", "open");

        // Check command exists in schema
        let command = schema.find_command(&cmd.command);
        assert!(command.is_some());

        let command = command.unwrap();
        let subcommand = command.find_subcommand(cmd.subcommand.as_ref().unwrap());
        assert!(subcommand.is_some());

        // Invalid command
        let invalid_cmd = InterpretedCommand::new("unknown");
        assert!(schema.find_command(&invalid_cmd.command).is_none());
    }

    #[test]
    fn test_string_similarity() {
        // "show prs" vs "show prs" should be 1.0 (exact match)
        assert_eq!(string_similarity("show prs", "show prs"), 1.0);
        // "show prs" and "show" share "show", so similarity = 1/2 = 0.5
        assert!(string_similarity("show prs", "show") >= 0.3);
        // Completely different words should have low similarity
        assert!(string_similarity("completely different", "nothing similar") < 0.3);
    }
}
