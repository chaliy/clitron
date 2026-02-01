//! Model loading and inference.
//!
//! This module handles loading the GGUF model and running inference using Candle.

use std::path::{Path, PathBuf};

use crate::command::InterpretedCommand;
use crate::context::Context;
use crate::error::{ClitronError, Result};
use crate::schema::CommandSchema;

#[cfg(feature = "candle")]
mod candle_backend;

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
        .join(ModelManager::DEFAULT_MODEL_FILE)
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Model wrapper for inference.
pub struct Model {
    #[allow(dead_code)]
    config: ModelConfig,
    schema: CommandSchema,
    #[cfg(feature = "candle")]
    backend: Option<candle_backend::CandleBackend>,
}

impl Model {
    /// Load a model from the given path.
    pub fn load(path: impl AsRef<Path>, schema: CommandSchema) -> Result<Self> {
        let path = path.as_ref();
        let config = ModelConfig {
            model_path: path.to_path_buf(),
            ..Default::default()
        };

        Self::load_with_config(config, schema)
    }

    /// Load with custom configuration.
    pub fn load_with_config(config: ModelConfig, schema: CommandSchema) -> Result<Self> {
        #[cfg(feature = "candle")]
        let backend = if config.model_path.exists() {
            match candle_backend::CandleBackend::load(&config) {
                Ok(b) => {
                    tracing::info!("Loaded Candle backend from {:?}", config.model_path);
                    Some(b)
                }
                Err(e) => {
                    tracing::warn!("Failed to load Candle backend: {}, using mock", e);
                    None
                }
            }
        } else {
            tracing::warn!(
                "Model file not found at {:?}, using mock inference",
                config.model_path
            );
            None
        };

        Ok(Self {
            config,
            schema,
            #[cfg(feature = "candle")]
            backend,
        })
    }

    /// Run inference on the given input.
    pub fn infer(&self, input: &str) -> Result<InterpretedCommand> {
        self.infer_with_context(input, &Context::default())
    }

    /// Run inference with environmental context.
    pub fn infer_with_context(&self, input: &str, context: &Context) -> Result<InterpretedCommand> {
        let prompt = self.build_prompt(input, Some(context));
        tracing::debug!("Prompt: {}", prompt);

        #[cfg(feature = "candle")]
        if let Some(ref backend) = self.backend {
            return backend
                .generate(&prompt)
                .and_then(|output| self.parse_output(&output));
        }

        // Fallback to mock inference
        self.mock_inference(input, Some(context))
    }

    /// Build the prompt for the model.
    fn build_prompt(&self, input: &str, context: Option<&Context>) -> String {
        let context_section = context
            .map(|ctx| {
                let ctx_str = ctx.to_prompt_string();
                if ctx_str.is_empty() {
                    String::new()
                } else {
                    format!(
                        r#"<|context|>
{}

"#,
                        ctx_str
                    )
                }
            })
            .unwrap_or_default();

        // Use the same prompt format as training
        format!(
            r#"<|system|>
You are a CLI command interpreter. Output only valid JSON.

{}<|user|>
{}
<|assistant|>
"#,
            context_section, input
        )
    }

    /// Parse the model output into an InterpretedCommand.
    fn parse_output(&self, output: &str) -> Result<InterpretedCommand> {
        // Find the JSON in the output (model might produce extra text)
        let json_start = output.find('{');
        let json_end = output.rfind('}');

        match (json_start, json_end) {
            (Some(start), Some(end)) if start < end => {
                let json_str = &output[start..=end];
                serde_json::from_str(json_str).map_err(|e| ClitronError::InvalidOutput {
                    output: output.to_string(),
                    reason: format!("Invalid JSON: {}", e),
                })
            }
            _ => Err(ClitronError::InvalidOutput {
                output: output.to_string(),
                reason: "No JSON object found in output".to_string(),
            }),
        }
    }

    /// Mock inference using simple pattern matching.
    fn mock_inference(&self, input: &str, context: Option<&Context>) -> Result<InterpretedCommand> {
        let input_lower = input.to_lowercase();
        let words: Vec<&str> = input_lower.split_whitespace().collect();

        let mut cmd = InterpretedCommand {
            confidence: 0.5,
            ..Default::default()
        };

        // Check for context-dependent phrases
        if let Some(ctx) = context {
            if let Some(ref git) = ctx.git {
                if let Some(pr_num) = git.current_pr {
                    if input_lower.contains("this")
                        || input_lower.contains("the pr")
                        || input_lower.contains("current")
                    {
                        cmd.args
                            .insert("number".into(), serde_json::Value::from(pr_num));
                        cmd.confidence += 0.1;
                    }
                }
            }
        }

        for command in &self.schema.commands {
            let cmd_matches = words.iter().any(|w| {
                *w == command.name
                    || command
                        .aliases
                        .as_ref()
                        .is_some_and(|a| a.iter().any(|alias| alias.contains(w)))
            });

            if cmd_matches {
                cmd.command = command.name.clone();
                cmd.confidence = 0.7;

                for subcommand in &command.subcommands {
                    let sub_matches = words.iter().any(|w| {
                        *w == subcommand.name
                            || subcommand
                                .aliases
                                .as_ref()
                                .is_some_and(|a| a.iter().any(|alias| alias.contains(w)))
                    });

                    if sub_matches {
                        cmd.subcommand = Some(subcommand.name.clone());
                        cmd.confidence = 0.8;
                        self.extract_args(&input_lower, subcommand, &mut cmd);
                        break;
                    }
                }

                if cmd.subcommand.is_none() && !command.subcommands.is_empty() {
                    if let Some(list_sub) = command.subcommands.iter().find(|s| s.name == "list") {
                        cmd.subcommand = Some("list".into());
                        self.extract_args(&input_lower, list_sub, &mut cmd);
                    }
                }

                break;
            }
        }

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
        if input.contains("open") && subcommand.args.iter().any(|a| a.name == "state") {
            cmd.args.insert("state".into(), "open".into());
            cmd.confidence += 0.05;
        } else if input.contains("closed") && subcommand.args.iter().any(|a| a.name == "state") {
            cmd.args.insert("state".into(), "closed".into());
            cmd.confidence += 0.05;
        } else if input.contains("merged") && subcommand.args.iter().any(|a| a.name == "state") {
            cmd.args.insert("state".into(), "merged".into());
            cmd.confidence += 0.05;
        }

        if (input.contains("my") || input.contains("mine") || input.contains("i "))
            && subcommand.args.iter().any(|a| a.name == "author")
        {
            cmd.args.insert("author".into(), "@me".into());
            cmd.confidence += 0.05;
        }

        for flag in &subcommand.flags {
            let flag_triggers = flag
                .natural_language
                .as_ref()
                .is_some_and(|nl| nl.iter().any(|t| input.contains(t)));

            if flag_triggers || input.contains(&flag.name) {
                cmd.flags.push(flag.name.clone());
                cmd.confidence += 0.02;
            }
        }

        if let Some(num) = extract_number(input) {
            if subcommand.args.iter().any(|a| a.name == "number") {
                cmd.args.insert("number".into(), num.into());
                cmd.confidence += 0.05;
            }
        }

        cmd.confidence = cmd.confidence.min(0.95);
    }
}

fn extract_number(input: &str) -> Option<i64> {
    if let Some(idx) = input.find('#') {
        let rest = &input[idx + 1..];
        let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(n) = num_str.parse() {
            return Some(n);
        }
    }

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
    /// Default HuggingFace model repository.
    pub const DEFAULT_REPO: &'static str = "chalyi/clitron-gh";

    /// Default model filename.
    pub const DEFAULT_MODEL_FILE: &'static str = "clitron-gh-q4_k_m.gguf";

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

    /// Get the default model path.
    pub fn default_model_path(&self) -> PathBuf {
        self.cache_dir.join(Self::DEFAULT_MODEL_FILE)
    }

    /// Download a model if not cached.
    #[cfg(feature = "candle")]
    pub fn ensure_model(&self, repo_id: &str, filename: &str) -> Result<PathBuf> {
        use hf_hub::api::sync::Api;

        let target_path = self.cache_dir.join(filename);

        if target_path.exists() {
            tracing::info!("Model already cached at {:?}", target_path);
            return Ok(target_path);
        }

        self.ensure_cache_dir()?;

        tracing::info!("Downloading model from {}/{}...", repo_id, filename);

        let api = Api::new().map_err(|e| ClitronError::ModelDownloadFailed {
            reason: format!("Failed to create HuggingFace API: {}", e),
        })?;

        let repo = api.model(repo_id.to_string());
        let downloaded_path =
            repo.get(filename)
                .map_err(|e| ClitronError::ModelDownloadFailed {
                    reason: format!("Failed to download model: {}", e),
                })?;

        // Copy to cache dir (hf-hub downloads to its own cache)
        std::fs::copy(&downloaded_path, &target_path)?;

        tracing::info!("Model downloaded to {:?}", target_path);
        Ok(target_path)
    }

    /// Download the default model.
    #[cfg(feature = "candle")]
    pub fn ensure_default_model(&self) -> Result<PathBuf> {
        self.ensure_model(Self::DEFAULT_REPO, Self::DEFAULT_MODEL_FILE)
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
        assert!(config.temperature < 0.5);
    }
}
