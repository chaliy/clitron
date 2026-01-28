"""Model evaluation script.

This module evaluates a trained model's performance on a test set.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_data(test_path: Path) -> list[dict[str, Any]]:
    """Load test data from JSONL file."""
    examples = []
    with open(test_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def parse_json_output(text: str) -> dict[str, Any] | None:
    """Parse JSON from model output."""
    text = text.strip()

    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def compute_metrics(predictions: list[dict], references: list[dict]) -> dict[str, float]:
    """Compute evaluation metrics."""
    metrics = defaultdict(int)
    total = len(predictions)

    for pred, ref in zip(predictions, references):
        # Check if prediction is valid JSON
        if pred is None:
            metrics["invalid_json"] += 1
            continue

        # Exact match
        if pred == ref:
            metrics["exact_match"] += 1

        # Command match
        if pred.get("command") == ref.get("command"):
            metrics["command_match"] += 1

        # Subcommand match
        if pred.get("subcommand") == ref.get("subcommand"):
            metrics["subcommand_match"] += 1

        # Args match
        if pred.get("args") == ref.get("args"):
            metrics["args_match"] += 1

        # Flags match
        pred_flags = set(pred.get("flags", []))
        ref_flags = set(ref.get("flags", []))
        if pred_flags == ref_flags:
            metrics["flags_match"] += 1

    # Convert to percentages
    return {
        "exact_match": metrics["exact_match"] / total * 100,
        "command_accuracy": metrics["command_match"] / total * 100,
        "subcommand_accuracy": metrics["subcommand_match"] / total * 100,
        "args_accuracy": metrics["args_match"] / total * 100,
        "flags_accuracy": metrics["flags_match"] / total * 100,
        "invalid_json_rate": metrics["invalid_json"] / total * 100,
        "total_examples": total,
    }


def evaluate_with_llama_cpp(
    model_path: Path,
    test_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate using llama-cpp-python."""
    try:
        from llama_cpp import Llama
    except ImportError:
        raise RuntimeError("llama-cpp-python not installed. Run: pip install llama-cpp-python")

    logger.info(f"Loading model from {model_path}...")
    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=4,
        verbose=False,
    )

    predictions = []

    for example in tqdm(test_data, desc="Evaluating"):
        prompt = f"""<|system|>
You are a CLI command interpreter. Output only valid JSON.
<|user|>
{example['instruction']}
<|assistant|>
"""

        output = llm(
            prompt,
            max_tokens=256,
            temperature=0.1,
            stop=["<|user|>", "<|system|>"],
        )

        generated = output["choices"][0]["text"].strip()
        predictions.append(parse_json_output(generated))

    return predictions


def evaluate_with_transformers(
    model_path: Path,
    test_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate using transformers (for non-GGUF models)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    predictions = []

    for example in tqdm(test_data, desc="Evaluating"):
        prompt = f"""<|system|>
You are a CLI command interpreter. Output only valid JSON.
<|user|>
{example['instruction']}
<|assistant|>
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant's response
        if "<|assistant|>" in generated:
            generated = generated.split("<|assistant|>")[-1].strip()

        predictions.append(parse_json_output(generated))

    return predictions


def evaluate(
    model_path: Path,
    test_path: Path,
    output_path: Path | None = None,
) -> dict[str, float]:
    """Evaluate a model on test data."""
    # Load test data
    test_data = load_test_data(test_path)
    logger.info(f"Loaded {len(test_data)} test examples")

    # Parse references
    references = [json.loads(ex["output"]) for ex in test_data]

    # Run inference
    if model_path.suffix == ".gguf":
        predictions = evaluate_with_llama_cpp(model_path, test_data)
    else:
        predictions = evaluate_with_transformers(model_path, test_data)

    # Compute metrics
    metrics = compute_metrics(predictions, references)

    # Print results
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Total examples: {metrics['total_examples']}")
    logger.info(f"Exact match: {metrics['exact_match']:.1f}%")
    logger.info(f"Command accuracy: {metrics['command_accuracy']:.1f}%")
    logger.info(f"Subcommand accuracy: {metrics['subcommand_accuracy']:.1f}%")
    logger.info(f"Args accuracy: {metrics['args_accuracy']:.1f}%")
    logger.info(f"Flags accuracy: {metrics['flags_accuracy']:.1f}%")
    logger.info(f"Invalid JSON rate: {metrics['invalid_json_rate']:.1f}%")
    logger.info("=" * 50)

    # Save results if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "predictions": [
                        {"input": ex["instruction"], "predicted": pred, "expected": ref}
                        for ex, pred, ref in zip(test_data, predictions, references)
                    ],
                },
                f,
                indent=2,
            )
        logger.info(f"Results saved to {output_path}")

    return metrics


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate clitron model")
    parser.add_argument("--model", type=Path, required=True, help="Model path (GGUF or HF)")
    parser.add_argument("--test-data", type=Path, required=True, help="Test data JSONL")
    parser.add_argument("--output", type=Path, help="Output JSON path for detailed results")

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        test_path=args.test_data,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
