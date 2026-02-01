//! Candle inference backend for GGUF models.

use std::sync::Mutex;

use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama as llama;
use tokenizers::Tokenizer;

use crate::error::{ClitronError, Result};
use crate::model::ModelConfig;

/// Candle-based inference backend.
pub struct CandleBackend {
    model: Mutex<llama::ModelWeights>,
    tokenizer: Tokenizer,
    device: Device,
    config: ModelConfig,
}

impl CandleBackend {
    /// Load the model from the given configuration.
    pub fn load(config: &ModelConfig) -> Result<Self> {
        let device = Device::Cpu;

        // Load the GGUF model
        let mut file = std::fs::File::open(&config.model_path)?;
        let model_content =
            candle_core::quantized::gguf_file::Content::read(&mut file).map_err(|e| {
                ClitronError::ModelLoadFailed {
                    reason: format!("Failed to read GGUF file: {}", e),
                }
            })?;

        let model =
            llama::ModelWeights::from_gguf(model_content, &mut file, &device).map_err(|e| {
                ClitronError::ModelLoadFailed {
                    reason: format!("Failed to load model weights: {}", e),
                }
            })?;
        tracing::info!("Loaded model weights from {:?}", config.model_path);

        // Load tokenizer from HuggingFace hub
        let tokenizer = Self::load_tokenizer()?;

        Ok(Self {
            model: Mutex::new(model),
            tokenizer,
            device,
            config: config.clone(),
        })
    }

    /// Load the tokenizer for Llama 3.2.
    fn load_tokenizer() -> Result<Tokenizer> {
        use hf_hub::api::sync::Api;

        let api = Api::new().map_err(|e| ClitronError::ModelLoadFailed {
            reason: format!("Failed to create HuggingFace API: {}", e),
        })?;

        let repo = api.model("meta-llama/Llama-3.2-1B-Instruct".to_string());
        let tokenizer_path =
            repo.get("tokenizer.json")
                .map_err(|e| ClitronError::ModelLoadFailed {
                    reason: format!("Failed to download tokenizer: {}", e),
                })?;

        Tokenizer::from_file(&tokenizer_path).map_err(|e| ClitronError::ModelLoadFailed {
            reason: format!("Failed to load tokenizer: {}", e),
        })
    }

    /// Generate text from a prompt.
    pub fn generate(&self, prompt: &str) -> Result<String> {
        // Tokenize the prompt
        let tokens =
            self.tokenizer
                .encode(prompt, true)
                .map_err(|e| ClitronError::InferenceFailed {
                    reason: format!("Tokenization failed: {}", e),
                })?;

        let prompt_tokens: Vec<u32> = tokens.get_ids().to_vec();
        let mut all_tokens = prompt_tokens.clone();

        tracing::debug!("Prompt has {} tokens", prompt_tokens.len());

        // Create logits processor for sampling (greedy with low temperature)
        let mut logits_processor = LogitsProcessor::new(
            42, // seed
            Some(self.config.temperature as f64),
            None, // disable top_p for more deterministic output
        );

        // EOS token for Llama 3.2
        let eos_token_id = 128009u32; // <|eot_id|>

        // Generate tokens
        let mut generated = String::new();
        let start_gen = all_tokens.len();

        // Lock the model for the entire generation loop
        let mut model = self
            .model
            .lock()
            .map_err(|e| ClitronError::InferenceFailed {
                reason: format!("Failed to lock model: {}", e),
            })?;

        // Process the initial prompt
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)
            .map_err(|e| ClitronError::InferenceFailed {
                reason: format!("Failed to create input tensor: {}", e),
            })?
            .unsqueeze(0)
            .map_err(|e| ClitronError::InferenceFailed {
                reason: format!("Failed to unsqueeze: {}", e),
            })?;

        let logits = model
            .forward(&input, 0)
            .map_err(|e| ClitronError::InferenceFailed {
                reason: format!("Forward pass failed: {}", e),
            })?;

        // Model returns [batch, vocab] - just squeeze the batch dimension
        let logits = logits
            .squeeze(0)
            .map_err(|e| ClitronError::InferenceFailed {
                reason: format!("Failed to squeeze: {}", e),
            })?;

        let next_token =
            logits_processor
                .sample(&logits)
                .map_err(|e| ClitronError::InferenceFailed {
                    reason: format!("Sampling failed: {}", e),
                })?;

        all_tokens.push(next_token);

        if let Ok(text) = self.tokenizer.decode(&[next_token], false) {
            generated.push_str(&text);
        }

        // Generate remaining tokens one at a time
        for _ in 1..self.config.max_tokens {
            if *all_tokens.last().unwrap() == eos_token_id {
                break;
            }

            // Check for complete JSON
            if generated.contains('}')
                && generated.matches('{').count() == generated.matches('}').count()
            {
                break;
            }

            let input = Tensor::new(&[*all_tokens.last().unwrap()], &self.device)
                .map_err(|e| ClitronError::InferenceFailed {
                    reason: format!("Failed to create input tensor: {}", e),
                })?
                .unsqueeze(0)
                .map_err(|e| ClitronError::InferenceFailed {
                    reason: format!("Failed to unsqueeze: {}", e),
                })?;

            let logits = model.forward(&input, all_tokens.len() - 1).map_err(|e| {
                ClitronError::InferenceFailed {
                    reason: format!("Forward pass failed: {}", e),
                }
            })?;

            // Model returns [batch, vocab] - just squeeze the batch dimension
            let logits = logits
                .squeeze(0)
                .map_err(|e| ClitronError::InferenceFailed {
                    reason: format!("Failed to squeeze: {}", e),
                })?;

            let next_token =
                logits_processor
                    .sample(&logits)
                    .map_err(|e| ClitronError::InferenceFailed {
                        reason: format!("Sampling failed: {}", e),
                    })?;

            all_tokens.push(next_token);

            if let Ok(text) = self.tokenizer.decode(&[next_token], false) {
                generated.push_str(&text);
            }
        }

        // Decode the generated tokens
        let generated_tokens = &all_tokens[start_gen..];
        let output = self.tokenizer.decode(generated_tokens, true).map_err(|e| {
            ClitronError::InferenceFailed {
                reason: format!("Decoding failed: {}", e),
            }
        })?;

        tracing::debug!("Generated output: {}", output);
        Ok(output)
    }
}
