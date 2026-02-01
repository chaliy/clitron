//! Model loading and inference.
//!
//! This module handles loading the GGUF model and running inference using Candle.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::command::InterpretedCommand;
use crate::context::Context;
use crate::error::{ClitronError, Result};
use crate::progress::{DownloadTracker, TerminalOutput};
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

    /// Whether to auto-download the model if not found.
    pub auto_download: bool,
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
            auto_download: true,
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
    #[allow(dead_code)]
    schema: CommandSchema,
    #[cfg(feature = "candle")]
    backend: candle_backend::CandleBackend,
}

impl Model {
    /// Load a model from the given path.
    pub fn load(path: impl AsRef<Path>, schema: CommandSchema) -> Result<Self> {
        let path = path.as_ref();
        let config = ModelConfig {
            model_path: path.to_path_buf(),
            auto_download: false, // Explicit path = no auto-download
            ..Default::default()
        };

        Self::load_with_config(config, schema)
    }

    /// Load with custom configuration.
    ///
    /// If `auto_download` is enabled and the model doesn't exist, it will be
    /// automatically downloaded from HuggingFace.
    pub fn load_with_config(config: ModelConfig, schema: CommandSchema) -> Result<Self> {
        let model_path = &config.model_path;

        // Auto-download if enabled and model doesn't exist
        if !model_path.exists() {
            if config.auto_download {
                tracing::info!("Model not found, downloading...");
                let manager = ModelManager::new();
                manager.download_default_model_with_progress()?;
            } else {
                return Err(ClitronError::ModelNotFound {
                    path: model_path.display().to_string(),
                });
            }
        }

        // Load the backend
        #[cfg(feature = "candle")]
        let backend = candle_backend::CandleBackend::load(&config).map_err(|e| {
            ClitronError::ModelLoadFailed {
                reason: format!("Failed to load Candle backend: {}", e),
            }
        })?;

        tracing::info!("Loaded model from {:?}", config.model_path);

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
        {
            self.backend
                .generate(&prompt)
                .and_then(|output| self.parse_output(&output))
        }

        #[cfg(not(feature = "candle"))]
        {
            Err(ClitronError::ModelLoadFailed {
                reason: "Candle feature not enabled".to_string(),
            })
        }
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
}

/// Model manager for downloading and caching models.
pub struct ModelManager {
    cache_dir: PathBuf,
    output: TerminalOutput,
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
            output: TerminalOutput::new(),
        }
    }

    /// Create with custom cache directory.
    pub fn with_cache_dir(cache_dir: PathBuf) -> Self {
        Self {
            cache_dir,
            output: TerminalOutput::new(),
        }
    }

    /// Get the path where a model would be cached.
    pub fn model_path(&self, model_id: &str) -> PathBuf {
        self.cache_dir.join(format!("{}.gguf", model_id))
    }

    /// Check if a model is already cached.
    pub fn is_cached(&self, model_id: &str) -> bool {
        self.model_path(model_id).exists()
    }

    /// Check if the default model is cached.
    pub fn is_default_cached(&self) -> bool {
        self.default_model_path().exists()
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

    /// Get the terminal output handler.
    pub fn output(&self) -> &TerminalOutput {
        &self.output
    }

    /// Download the default model with progress reporting.
    pub fn download_default_model_with_progress(&self) -> Result<PathBuf> {
        self.download_model_with_progress(Self::DEFAULT_REPO, Self::DEFAULT_MODEL_FILE)
    }

    /// Download a model with progress reporting.
    #[cfg(feature = "candle")]
    pub fn download_model_with_progress(&self, repo_id: &str, filename: &str) -> Result<PathBuf> {
        use hf_hub::api::sync::Api;

        let target_path = self.cache_dir.join(filename);

        if target_path.exists() {
            self.output.info(&format!(
                "Model already downloaded: {}",
                target_path.display()
            ));
            return Ok(target_path);
        }

        self.ensure_cache_dir()?;

        // Build the download URL for display
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo_id, filename
        );

        self.output.download_start(filename, &url);

        // Use hf-hub to get the file URL and download with progress
        let api = Api::new().map_err(|e| ClitronError::ModelDownloadFailed {
            reason: format!("Failed to create HuggingFace API: {}", e),
        })?;

        let repo = api.model(repo_id.to_string());

        // Get the file URL for direct download with progress
        let file_url = repo.url(filename);

        // Download with progress
        match self.download_with_progress(&file_url, &target_path) {
            Ok(_) => Ok(target_path),
            Err(e) => {
                // Fall back to hf-hub download without progress
                tracing::warn!("Direct download failed, falling back to hf-hub: {}", e);

                let downloaded_path =
                    repo.get(filename)
                        .map_err(|e| ClitronError::ModelDownloadFailed {
                            reason: format!("Failed to download model: {}", e),
                        })?;

                // Copy to cache dir (hf-hub downloads to its own cache)
                std::fs::copy(&downloaded_path, &target_path)?;

                self.output
                    .download_complete(&target_path.display().to_string());
                Ok(target_path)
            }
        }
    }

    /// Download a file with progress reporting.
    fn download_with_progress(&self, url: &str, target_path: &Path) -> Result<()> {
        // Use a simple HTTP client for progress tracking
        let response = ureq::get(url)
            .call()
            .map_err(|e| ClitronError::ModelDownloadFailed {
                reason: format!("HTTP request failed: {}", e),
            })?;

        let total_size = response
            .header("Content-Length")
            .and_then(|s| s.parse::<u64>().ok());

        let mut tracker = DownloadTracker::new(total_size);

        // Create temp file
        let temp_path = target_path.with_extension("download");
        let mut file = File::create(&temp_path)?;

        let mut reader = response.into_reader();
        let mut buffer = [0u8; 8192];
        let mut downloaded: u64 = 0;

        loop {
            let bytes_read =
                reader
                    .read(&mut buffer)
                    .map_err(|e| ClitronError::ModelDownloadFailed {
                        reason: format!("Failed to read response: {}", e),
                    })?;

            if bytes_read == 0 {
                break;
            }

            file.write_all(&buffer[..bytes_read])?;
            downloaded += bytes_read as u64;
            tracker.update(downloaded);
        }

        file.flush()?;
        drop(file);

        // Rename temp file to target
        std::fs::rename(&temp_path, target_path)?;

        tracker.complete(&target_path.display().to_string());

        Ok(())
    }

    /// Download a model if not cached (simple version for compatibility).
    #[cfg(feature = "candle")]
    pub fn ensure_model(&self, repo_id: &str, filename: &str) -> Result<PathBuf> {
        self.download_model_with_progress(repo_id, filename)
    }

    /// Download the default model.
    #[cfg(feature = "candle")]
    pub fn ensure_default_model(&self) -> Result<PathBuf> {
        self.download_default_model_with_progress()
    }

    /// Download a model without progress (fallback).
    #[cfg(not(feature = "candle"))]
    pub fn download_model_with_progress(&self, _repo_id: &str, _filename: &str) -> Result<PathBuf> {
        Err(ClitronError::ModelDownloadFailed {
            reason: "Candle feature not enabled".to_string(),
        })
    }

    /// Clear the model cache.
    pub fn clear_cache(&self) -> Result<()> {
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir)?;
        }
        Ok(())
    }

    /// Get model status information.
    pub fn model_status(&self) -> ModelStatus {
        let path = self.default_model_path();
        if path.exists() {
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            ModelStatus::Downloaded { path, size }
        } else {
            ModelStatus::NotDownloaded {
                expected_path: path,
            }
        }
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Status of the model cache.
#[derive(Debug)]
pub enum ModelStatus {
    /// Model is downloaded and ready.
    Downloaded {
        /// Path to the model file.
        path: PathBuf,
        /// Size in bytes.
        size: u64,
    },
    /// Model is not downloaded.
    NotDownloaded {
        /// Expected path where model would be stored.
        expected_path: PathBuf,
    },
}

impl ModelStatus {
    /// Check if model is available.
    pub fn is_available(&self) -> bool {
        matches!(self, ModelStatus::Downloaded { .. })
    }

    /// Get the model path if available.
    pub fn path(&self) -> &Path {
        match self {
            ModelStatus::Downloaded { path, .. } => path,
            ModelStatus::NotDownloaded { expected_path } => expected_path,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.context_size, 2048);
        assert_eq!(config.max_tokens, 256);
        assert!(config.temperature < 0.5);
        assert!(config.auto_download);
    }

    #[test]
    fn test_model_manager_paths() {
        let manager = ModelManager::new();
        let path = manager.default_model_path();
        assert!(path.to_string_lossy().contains("clitron-gh-q4_k_m.gguf"));
    }

    #[test]
    fn test_model_status() {
        let manager = ModelManager::new();
        let status = manager.model_status();
        // Just verify it returns something
        let _ = status.is_available();
        let _ = status.path();
    }
}
