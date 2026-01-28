"""Generate training data using a large LLM (Claude Opus).

This module generates synthetic training examples from CLI schemas using
Claude Opus 4.5 to create diverse, high-quality input-output pairs.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import anthropic
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


GENERATION_PROMPT = """You are generating training data for a CLI command interpreter.

Given this CLI schema:
{schema}

Generate {batch_size} diverse examples of natural language inputs that map to CLI commands.

For each example, provide:
1. Natural language input (how a human might say it)
2. Structured JSON output

IMPORTANT: Vary the examples by:
- Formality level (casual "show me prs" vs formal "list all pull requests")
- Completeness (full command with all args vs implied defaults)
- Synonyms (show/display/list/get, make/create/new, etc.)
- Word order variations ("my open prs" vs "open prs that I created")
- With/without optional arguments
- Including typos occasionally (realistic user input)
- Different sentence structures (questions, commands, requests)

Output as a JSON array with exactly {batch_size} items:
[
  {{
    "input": "show me open prs",
    "output": {{"command": "pr", "subcommand": "list", "args": {{"state": "open"}}, "flags": [], "confidence": 0.95}}
  }},
  ...
]

CRITICAL: Output ONLY valid JSON, no explanations or markdown.
"""


def load_schema(schema_path: Path) -> dict[str, Any]:
    """Load a CLI schema from YAML file."""
    with open(schema_path) as f:
        return yaml.safe_load(f)


def schema_to_summary(schema: dict[str, Any]) -> str:
    """Convert schema to a readable summary for the prompt."""
    lines = [f"CLI: {schema['cli_name']} - {schema['description']}", "", "Commands:"]

    for cmd in schema.get("commands", []):
        lines.append(f"  {cmd['name']} - {cmd['description']}")
        if "aliases" in cmd:
            lines.append(f"    Aliases: {', '.join(cmd['aliases'])}")

        for sub in cmd.get("subcommands", []):
            lines.append(f"    {sub['name']} - {sub['description']}")
            if "aliases" in sub:
                lines.append(f"      Aliases: {', '.join(sub['aliases'])}")

            if sub.get("args"):
                lines.append("      Arguments:")
                for arg in sub["args"]:
                    arg_type = arg.get("type", "string")
                    if isinstance(arg_type, dict) and "enum" in arg_type:
                        arg_type = f"enum({', '.join(arg_type['enum'])})"
                    default = f" (default: {arg['default']})" if "default" in arg else ""
                    required = " [required]" if arg.get("required") else ""
                    lines.append(f"        --{arg['name']}: {arg_type}{default}{required}")

            if sub.get("flags"):
                lines.append("      Flags:")
                for flag in sub["flags"]:
                    short = f" (-{flag['short']})" if "short" in flag else ""
                    lines.append(f"        --{flag['name']}{short}: {flag.get('description', '')}")

    return "\n".join(lines)


def generate_batch(
    client: anthropic.Anthropic,
    schema: dict[str, Any],
    batch_size: int = 50,
) -> list[dict[str, Any]]:
    """Generate a batch of training examples using Claude."""
    schema_summary = schema_to_summary(schema)

    prompt = GENERATION_PROMPT.format(
        schema=schema_summary,
        batch_size=batch_size,
    )

    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract text content
    text = response.content[0].text

    # Parse JSON
    try:
        # Handle potential markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        examples = json.loads(text.strip())

        if not isinstance(examples, list):
            logger.warning("Response was not a list, wrapping")
            examples = [examples]

        return examples

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.debug(f"Raw response: {text[:500]}...")
        return []


def validate_example(example: dict[str, Any], schema: dict[str, Any]) -> bool:
    """Validate that an example conforms to the schema."""
    if "input" not in example or "output" not in example:
        return False

    output = example["output"]
    if "command" not in output:
        return False

    # Find the command in schema
    cmd_name = output["command"]
    cmd = next((c for c in schema.get("commands", []) if c["name"] == cmd_name), None)
    if cmd is None:
        logger.debug(f"Unknown command: {cmd_name}")
        return False

    # Validate subcommand if present
    if "subcommand" in output and output["subcommand"]:
        sub_name = output["subcommand"]
        sub = next((s for s in cmd.get("subcommands", []) if s["name"] == sub_name), None)
        if sub is None:
            logger.debug(f"Unknown subcommand: {sub_name}")
            return False

    return True


def format_for_training(example: dict[str, Any]) -> dict[str, str]:
    """Format an example for training (instruction format)."""
    return {
        "instruction": example["input"],
        "output": json.dumps(example["output"], ensure_ascii=False),
    }


def load_existing_examples(output_path: Path) -> list[dict[str, str]]:
    """Load existing examples from a partial run."""
    examples = []
    progress_path = output_path.parent / f"{output_path.stem}_progress.jsonl"

    if progress_path.exists():
        logger.info(f"Found progress file: {progress_path}")
        with open(progress_path) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        logger.info(f"Loaded {len(examples)} existing examples")

    return examples


def save_progress(examples: list[dict[str, str]], output_path: Path) -> None:
    """Save progress incrementally."""
    progress_path = output_path.parent / f"{output_path.stem}_progress.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(progress_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def generate_dataset(
    schema_path: Path,
    output_path: Path,
    num_examples: int = 10000,
    batch_size: int = 50,
    validation_split: float = 0.1,
    resume: bool = True,
) -> None:
    """Generate a complete training dataset."""
    client = anthropic.Anthropic()
    schema = load_schema(schema_path)

    # Try to resume from existing progress
    examples = []
    if resume:
        examples = load_existing_examples(output_path)

    start_batch = len(examples) // batch_size
    num_batches = (num_examples + batch_size - 1) // batch_size

    if start_batch > 0:
        logger.info(f"Resuming from batch {start_batch + 1}/{num_batches}")
    else:
        logger.info(f"Generating {num_examples} examples in {num_batches} batches...")

    try:
        for batch_idx in tqdm(range(start_batch, num_batches), desc="Generating", initial=start_batch, total=num_batches):
            batch = generate_batch(client, schema, batch_size)

            # Validate and collect
            for example in batch:
                if validate_example(example, schema):
                    examples.append(format_for_training(example))
                else:
                    logger.debug(f"Invalid example: {example}")

            # Save progress after each batch
            save_progress(examples, output_path)

            if len(examples) >= num_examples:
                break

    except KeyboardInterrupt:
        logger.info(f"\nInterrupted! Saved {len(examples)} examples. Run again to resume.")
        return
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.info(f"Saved {len(examples)} examples. Run again to resume.")
        raise

    # Shuffle
    random.shuffle(examples)
    examples = examples[:num_examples]

    # Split into train/validation
    split_idx = int(len(examples) * (1 - validation_split))
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    # Save final files
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_path = output_path
    val_path = output_path.parent / output_path.name.replace(".jsonl", "_val.jsonl")

    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(val_path, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Clean up progress file
    progress_path = output_path.parent / f"{output_path.stem}_progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    logger.info(f"Generated {len(train_examples)} training examples -> {train_path}")
    logger.info(f"Generated {len(val_examples)} validation examples -> {val_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate training data for clitron")
    parser.add_argument("--schema", type=Path, required=True, help="Path to schema YAML")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--num-examples", type=int, default=10000, help="Number of examples")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for generation")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, don't resume")

    args = parser.parse_args()

    generate_dataset(
        schema_path=args.schema,
        output_path=args.output,
        num_examples=args.num_examples,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
