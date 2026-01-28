"""Direct Preference Optimization (DPO) training script.

This module applies DPO training to improve model output quality by learning
from preference pairs.
"""

import argparse
import logging
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load training configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_dpo(
    config_path: Path,
    model_dir: Path | None = None,
    output_dir: Path | None = None,
    preference_file: Path | None = None,
) -> None:
    """Run DPO training."""
    config = load_config(config_path)

    # Override paths from command line
    if model_dir:
        config["model"]["name"] = str(model_dir)
    if output_dir:
        config["training"]["output_dir"] = str(output_dir)
    if preference_file:
        config["data"]["preference_file"] = str(preference_file)

    model_name = config["model"]["name"]
    logger.info(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (starting from SFT checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Reference model (frozen copy)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load preference dataset
    logger.info("Loading preference dataset...")
    dataset = load_dataset("json", data_files=config["data"]["preference_file"])

    # Format dataset for DPO
    def format_dpo(example):
        """Format example for DPO training."""
        prompt = f"<|system|>\nYou are a CLI command interpreter. Output only valid JSON.\n<|user|>\n{example['prompt']}\n<|assistant|>\n"
        return {
            "prompt": prompt,
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    dataset = dataset.map(format_dpo)

    # LoRA config for DPO
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["training"].get("lora_r", 8),
        lora_alpha=config["training"].get("lora_alpha", 16),
        lora_dropout=config["training"].get("lora_dropout", 0.05),
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    # DPO config
    dpo_config = DPOConfig(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"].get("num_train_epochs", 1),
        per_device_train_batch_size=config["training"].get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 8),
        learning_rate=config["training"].get("learning_rate", 5e-6),
        beta=config["training"].get("beta", 0.1),
        loss_type=config["training"].get("loss_type", "sigmoid"),
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        remove_unused_columns=False,
    )

    # DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # Train
    logger.info("Starting DPO training...")
    trainer.train()

    # Save
    logger.info(f"Saving model to {config['training']['output_dir']}")
    trainer.save_model()
    tokenizer.save_pretrained(config["training"]["output_dir"])

    logger.info("DPO training complete!")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run DPO training for clitron")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    parser.add_argument("--model-dir", type=Path, help="Override model directory (SFT checkpoint)")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--preference-file", type=Path, help="Override preference file path")

    args = parser.parse_args()
    train_dpo(args.config, args.model_dir, args.output_dir, args.preference_file)


if __name__ == "__main__":
    main()
