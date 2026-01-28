"""Supervised Fine-Tuning (SFT) training script.

This module fine-tunes a small base model on the generated training data
using LoRA for efficient training.
"""

import argparse
import logging
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load training configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def format_prompt(example: dict) -> str:
    """Format a training example as a prompt."""
    return f"""<|system|>
You are a CLI command interpreter. Output only valid JSON.
<|user|>
{example['instruction']}
<|assistant|>
{example['output']}"""


def preprocess_function(examples: dict, tokenizer) -> dict:
    """Tokenize examples for training."""
    prompts = [format_prompt({"instruction": i, "output": o})
               for i, o in zip(examples["instruction"], examples["output"])]

    model_inputs = tokenizer(
        prompts,
        max_length=2048,
        truncation=True,
        padding=False,
    )

    # For causal LM, labels are the same as input_ids
    model_inputs["labels"] = model_inputs["input_ids"].copy()

    return model_inputs


def train_sft(config_path: Path, data_dir: Path | None = None, output_dir: Path | None = None) -> None:
    """Run SFT training."""
    config = load_config(config_path)

    # Override paths from command line
    if data_dir:
        config["data"]["train_file"] = str(data_dir / "train.jsonl")
        config["data"]["validation_file"] = str(data_dir / "train_val.jsonl")
    if output_dir:
        config["training"]["output_dir"] = str(output_dir)

    model_name = config["model"]["name"]
    logger.info(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    if config["training"].get("use_lora", True):
        logger.info("Applying LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config["training"].get("lora_r", 16),
            lora_alpha=config["training"].get("lora_alpha", 32),
            lora_dropout=config["training"].get("lora_dropout", 0.05),
            target_modules=config["training"].get(
                "lora_target_modules",
                ["q_proj", "v_proj", "k_proj", "o_proj"]
            ),
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load dataset
    logger.info("Loading dataset...")
    train_file = config["data"]["train_file"]
    val_file = config["data"].get("validation_file")

    data_files = {"train": train_file}
    if val_file and Path(val_file).exists():
        data_files["validation"] = val_file

    dataset = load_dataset("json", data_files=data_files)

    # Preprocess
    logger.info("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"].get("num_train_epochs", 3),
        per_device_train_batch_size=config["training"].get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 4),
        learning_rate=config["training"].get("learning_rate", 2e-5),
        warmup_ratio=config["training"].get("warmup_ratio", 0.1),
        lr_scheduler_type=config["training"].get("lr_scheduler_type", "cosine"),
        logging_steps=config.get("logging", {}).get("logging_steps", 10),
        save_strategy="epoch",
        evaluation_strategy="epoch" if "validation" in tokenized_dataset else "no",
        bf16=True,
        report_to=config.get("logging", {}).get("report_to", "none"),
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save
    logger.info(f"Saving model to {config['training']['output_dir']}")
    trainer.save_model()
    tokenizer.save_pretrained(config["training"]["output_dir"])

    logger.info("Training complete!")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run SFT training for clitron")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    parser.add_argument("--data-dir", type=Path, help="Override data directory")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")

    args = parser.parse_args()
    train_sft(args.config, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
