//! Model loading and inference.
//!
//! This module handles loading the GGUF model and running inference.
//! Currently provides a mock implementation; real inference will be added
//! when integrating llama.cpp or candle backends.

use std::path::{Path, PathBuf};

use crate::command::InterpretedCommand;
use crate::error::{ClitronError, Result};
use crate::schema::CommandSchema;

/// Configuration for model inference.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Path to the GGUF model file.
    pub model_path: PathBuf,

    /// Context window size in tokens.
    pub context_size: usize,

    /// Maximum tokens to generate.
    pub max_tokens: usize,

    /// Temperature for generation (lower = more deterministic).
    pub temperature: f32,

    /// Top-p sampling parameter.
    pub top_p: f32,

    /// Repetition penalty.
    pub repeat_penalty: f32,

    /// Number of threads for inference.
    pub num_threads: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: default_model_path(),
            context_size: 2048,
            max_tokens: 256,
            temperature: 0.1,
            top_p: 0.9,
            repeat_penalty: 1.1,
            num_threads: num_cpus(),
        }
    }
}

fn default_model_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".clitron")
        .join("models")
        .join("clitron.gguf")
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Model wrapper for inference.
pub struct Model {
    config: ModelConfig,
    schema: CommandSchema,
    // TODO: Add actual model state when integrating inference backend
    // model: llama_cpp::LlamaModel,
}

impl Model {
    /// Load a model from the given path.
    pub fn load(path: impl AsRef<Path>, schema: CommandSchema) -> Result<Self> {
        let path = path.as_ref();

        // For now, we don't actually load a model
        // This will be implemented when adding the inference backend
        if !path.exists() {
            tracing::warn!("Model file not found at {:?}, using mock inference", path);
        }

        Ok(Self {
            config: ModelConfig {
                model_path: path.to_path_buf(),
                ..Default::default()
            },
            schema,
        })
    }

    /// Load with custom configuration.
    pub fn load_with_config(config: ModelConfig, schema: CommandSchema) -> Result<Self> {
        if !config.model_path.exists() {
            tracing::warn!(
                "Model file not found at {:?}, using mock inference",
                config.model_path
            );
        }

        Ok(Self { config, schema })
    }

    /// Run inference on the given input.
    pub fn infer(&self, input: &str) -> Result<InterpretedCommand> {
        let prompt = self.build_prompt(input);

        // TODO: Implement actual inference
        // For now, use simple pattern matching as a fallback
        tracing::debug!("Prompt: {}", prompt);

        self.mock_inference(input)
    }

    /// Build the prompt for the model.
    fn build_prompt(&self, input: &str) -> String {
        format!(
            r#"<|system|>
You are a CLI command interpreter. Given a natural language command, output a JSON object representing the structured command.

Available commands:
{}

Output only valid JSON in this format:
{{"command": "...", "subcommand": "...", "args": {{}}, "flags": [], "confidence": 0.95}}

<|user|>
{}

<|assistant|>
"#,
            self.schema.to_summary(),
            input
        )
    }

    /// Mock inference using simple pattern matching.
    /// This is a fallback when the model is not available.
    fn mock_inference(&self, input: &str) -> Result<InterpretedCommand> {
        let input_lower = input.to_lowercase();
        let words: Vec<&str> = input_lower.split_whitespace().collect();

        // Try to identify command and subcommand
        let mut cmd = InterpretedCommand::default();
        cmd.confidence = 0.5; // Lower confidence for mock

        for command in &self.schema.commands {
            let cmd_matches = words.iter().any(|w| {
                *w == command.name
                    || command
                        .aliases
                        .as_ref()
                        .map_or(false, |a| a.iter().any(|alias| alias.contains(w)))
            });

            if cmd_matches {
                cmd.command = command.name.clone();
                cmd.confidence = 0.7;

                // Look for subcommand
                for subcommand in &command.subcommands {
                    let sub_matches = words.iter().any(|w| {
                        *w == subcommand.name
                            || subcommand
                                .aliases
                                .as_ref()
                                .map_or(false, |a| a.iter().any(|alias| alias.contains(w)))
                    });

                    if sub_matches {
                        cmd.subcommand = Some(subcommand.name.clone());
                        cmd.confidence = 0.8;

                        // Extract common patterns
                        self.extract_args(&input_lower, subcommand, &mut cmd);
                        break;
                    }
                }

                // If no subcommand but command has default (list), use it
                if cmd.subcommand.is_none() && !command.subcommands.is_empty() {
                    if let Some(list_sub) = command.subcommands.iter().find(|s| s.name == "list") {
                        cmd.subcommand = Some("list".into());
                        self.extract_args(&input_lower, list_sub, &mut cmd);
                    }
                }

                break;
            }
        }

        // If no command found, return with very low confidence
        if cmd.command.is_empty() {
            cmd.confidence = 0.1;
            return Err(ClitronError::LowConfidence {
                confidence: cmd.confidence,
                suggestion: "Could not identify command".into(),
            });
        }

        Ok(cmd)
    }

    /// Extract arguments from input using simple patterns.
    fn extract_args(
        &self,
        input: &str,
        subcommand: &crate::schema::Subcommand,
        cmd: &mut InterpretedCommand,
    ) {
        // Common patterns for state
        if input.contains("open") {
            if subcommand.args.iter().any(|a| a.name == "state") {
                cmd.args.insert("state".into(), "open".into());
                cmd.confidence += 0.05;
            }
        } else if input.contains("closed") {
            if subcommand.args.iter().any(|a| a.name == "state") {
                cmd.args.insert("state".into(), "closed".into());
                cmd.confidence += 0.05;
            }
        } else if input.contains("merged") {
            if subcommand.args.iter().any(|a| a.name == "state") {
                cmd.args.insert("state".into(), "merged".into());
                cmd.confidence += 0.05;
            }
        }

        // Common patterns for author
        if input.contains("my") || input.contains("mine") || input.contains("i ") {
            if subcommand.args.iter().any(|a| a.name == "author") {
                cmd.args.insert("author".into(), "@me".into());
                cmd.confidence += 0.05;
            }
            if subcommand.args.iter().any(|a| a.name == "assignee") {
                cmd.args.insert("assignee".into(), "@me".into());
                cmd.confidence += 0.05;
            }
        }

        // Check for flags
        for flag in &subcommand.flags {
            let flag_triggers = flag
                .natural_language
                .as_ref()
                .map(|nl| nl.iter().any(|t| input.contains(t)))
                .unwrap_or(false);

            if flag_triggers || input.contains(&flag.name) {
                cmd.flags.push(flag.name.clone());
                cmd.confidence += 0.02;
            }
        }

        // Extract numbers (for PR/issue numbers)
        if let Some(num) = extract_number(input) {
            if subcommand.args.iter().any(|a| a.name == "number") {
                cmd.args.insert("number".into(), num.into());
                cmd.confidence += 0.05;
            }
        }

        // Clamp confidence
        cmd.confidence = cmd.confidence.min(0.95);
    }
}

/// Extract a number from the input string.
fn extract_number(input: &str) -> Option<i64> {
    // Look for patterns like "#123", "number 123", or just "123"
    let re_patterns = [r"#(\d+)", r"number\s+(\d+)", r"\b(\d+)\b"];

    for pattern in &re_patterns {
        // Simple extraction without regex for now
        if pattern.contains('#') {
            if let Some(idx) = input.find('#') {
                let rest = &input[idx + 1..];
                let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                if let Ok(n) = num_str.parse() {
                    return Some(n);
                }
            }
        }
    }

    // Fallback: find any standalone number
    for word in input.split_whitespace() {
        if let Ok(n) = word.trim_matches(|c: char| !c.is_ascii_digit()).parse() {
            return Some(n);
        }
    }

    None
}

/// Model manager for downloading and caching models.
pub struct ModelManager {
    cache_dir: PathBuf,
}

impl ModelManager {
    /// Create with default cache directory (~/.clitron/models).
    pub fn new() -> Self {
        Self {
            cache_dir: dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".clitron")
                .join("models"),
        }
    }

    /// Create with custom cache directory.
    pub fn with_cache_dir(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    /// Get the path where a model would be cached.
    pub fn model_path(&self, model_id: &str) -> PathBuf {
        self.cache_dir.join(format!("{}.gguf", model_id))
    }

    /// Check if a model is already cached.
    pub fn is_cached(&self, model_id: &str) -> bool {
        self.model_path(model_id).exists()
    }

    /// Ensure the cache directory exists.
    pub fn ensure_cache_dir(&self) -> Result<()> {
        std::fs::create_dir_all(&self.cache_dir)?;
        Ok(())
    }

    /// Download a model if not cached.
    ///
    /// Returns the path to the model file.
    pub async fn ensure_model(&self, model_id: &str) -> Result<PathBuf> {
        let path = self.model_path(model_id);

        if path.exists() {
            return Ok(path);
        }

        self.ensure_cache_dir()?;

        // TODO: Implement actual download
        tracing::warn!(
            "Model download not yet implemented. Please manually place model at {:?}",
            path
        );

        Err(ClitronError::ModelNotFound {
            path: path.to_string_lossy().into(),
        })
    }

    /// Clear the model cache.
    pub fn clear_cache(&self) -> Result<()> {
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir)?;
        }
        Ok(())
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_number() {
        assert_eq!(extract_number("view pr #123"), Some(123));
        assert_eq!(extract_number("view pr 456"), Some(456));
        assert_eq!(extract_number("no number here"), None);
    }

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.context_size, 2048);
        assert_eq!(config.max_tokens, 256);
        assert!(config.temperature < 0.5); // Should be low for deterministic output
    }
}
