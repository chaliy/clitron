"""Generate preference data for DPO training.

This module generates preference pairs by comparing SFT model outputs
with ideal outputs from Claude Opus.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import anthropic
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_schema(schema_path: Path) -> dict[str, Any]:
    """Load CLI schema from YAML."""
    with open(schema_path) as f:
        return yaml.safe_load(f)


def generate_test_prompts(schema: dict[str, Any], num_prompts: int) -> list[str]:
    """Generate diverse test prompts from schema."""
    prompts = []

    # Generate variations for each command/subcommand
    for cmd in schema.get("commands", []):
        cmd_name = cmd["name"]

        for sub in cmd.get("subcommands", []):
            sub_name = sub["name"]

            # Generate natural language variations
            templates = [
                f"show {cmd_name}s",
                f"list all {cmd_name}s",
                f"display {sub_name} {cmd_name}",
                f"get my {cmd_name}s",
                f"{sub_name} {cmd_name}s please",
                f"can you {sub_name} the {cmd_name}s",
                f"i want to see {cmd_name}s",
            ]

            # Add argument variations
            for arg in sub.get("args", []):
                if isinstance(arg.get("type"), dict) and "enum" in arg["type"]:
                    for value in arg["type"]["enum"][:3]:  # First 3 enum values
                        templates.append(f"show {value} {cmd_name}s")
                        templates.append(f"list {cmd_name}s that are {value}")

            prompts.extend(templates)

    # Shuffle and limit
    random.shuffle(prompts)
    return prompts[:num_prompts]


def get_sft_output(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
) -> str:
    """Get output from the SFT model."""
    full_prompt = f"""<|system|>
You are a CLI command interpreter. Output only valid JSON.
<|user|>
{prompt}
<|assistant|>
"""

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the assistant's response
    if "<|assistant|>" in generated:
        generated = generated.split("<|assistant|>")[-1].strip()

    return generated


def get_ideal_output(
    client: anthropic.Anthropic,
    prompt: str,
    schema_summary: str,
) -> str:
    """Get ideal output from Claude Opus."""
    system_prompt = f"""You are a CLI command interpreter. Given a natural language command, output ONLY a valid JSON object.

Available commands:
{schema_summary}

Output format:
{{"command": "...", "subcommand": "...", "args": {{}}, "flags": [], "confidence": 0.95}}

CRITICAL: Output ONLY the JSON, nothing else."""

    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()


def is_valid_json(text: str) -> bool:
    """Check if text is valid JSON."""
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def generate_preferences(
    model_path: Path,
    schema_path: Path,
    output_path: Path,
    num_pairs: int = 5000,
) -> None:
    """Generate preference pairs for DPO training."""
    # Load schema
    schema = load_schema(schema_path)

    # Create schema summary for prompts
    from .generate_dataset import schema_to_summary
    schema_summary = schema_to_summary(schema)

    # Load SFT model
    logger.info(f"Loading SFT model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Initialize Anthropic client
    client = anthropic.Anthropic()

    # Generate test prompts
    prompts = generate_test_prompts(schema, num_pairs)

    preferences = []
    logger.info(f"Generating {len(prompts)} preference pairs...")

    for prompt in tqdm(prompts, desc="Generating preferences"):
        try:
            # Get SFT output
            sft_output = get_sft_output(model, tokenizer, prompt)

            # Get ideal output from Opus
            ideal_output = get_ideal_output(client, prompt, schema_summary)

            # Only create preference if outputs differ and ideal is valid JSON
            if sft_output != ideal_output and is_valid_json(ideal_output):
                preferences.append({
                    "prompt": prompt,
                    "chosen": ideal_output,
                    "rejected": sft_output,
                })
            elif is_valid_json(sft_output):
                # If SFT output is good, we can still use it as chosen
                # with a slightly worse version as rejected
                pass  # Skip for now

        except Exception as e:
            logger.warning(f"Error generating preference for '{prompt}': {e}")
            continue

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for pref in preferences:
            f.write(json.dumps(pref, ensure_ascii=False) + "\n")

    logger.info(f"Generated {len(preferences)} preference pairs -> {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate preference data for DPO")
    parser.add_argument("--model", type=Path, required=True, help="Path to SFT model")
    parser.add_argument("--schema", type=Path, required=True, help="Path to schema YAML")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--num-pairs", type=int, default=5000, help="Number of pairs")

    args = parser.parse_args()

    generate_preferences(
        model_path=args.model,
        schema_path=args.schema,
        output_path=args.output,
        num_pairs=args.num_pairs,
    )


if __name__ == "__main__":
    main()
