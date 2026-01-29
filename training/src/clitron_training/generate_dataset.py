"""Generate training data using a large LLM (Claude Opus).

This module generates synthetic training examples from CLI schemas using
Claude Opus 4.5 to create diverse, high-quality input-output pairs.

Training data distribution:
- 60% positive cases WITH context (git state, current branch, etc.)
- 20% positive cases WITHOUT context
- 20% negative cases (ambiguous, incomplete, off-topic inputs)
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


# Prompt for generating positive examples WITH context (60%)
POSITIVE_WITH_CONTEXT_PROMPT = """You are generating training data for a CLI command interpreter.

Given this CLI schema:
{schema}

Generate {batch_size} diverse examples of natural language inputs that map to CLI commands.
These examples should include environmental context (git state) that helps interpretation.

For each example, provide:
1. Natural language input (how a human might say it)
2. Environmental context (git state like current branch, repo, PR number)
3. Structured JSON output with type "command"

Context enables interpreting phrases like:
- "this", "current", "the pr" → resolve from current_pr in context
- "merge this branch" → resolve from current_branch
- Implicit targets → infer from repo_owner/repo_name

IMPORTANT variations:
- Formality level (casual "merge this" vs formal "merge the current pull request")
- Context-dependent phrases ("this pr", "current branch", "my repo")
- Synonyms (show/display/list/get, make/create/new, merge/land/complete)
- Word order variations
- With/without optional arguments
- Include occasional typos

Output as JSON array with exactly {batch_size} items:
[
  {{
    "input": "merge this and delete the branch",
    "context": {{
      "git": {{
        "current_branch": "feature/add-login",
        "repo_owner": "acme",
        "repo_name": "webapp",
        "current_pr": 123,
        "has_uncommitted_changes": false
      }}
    }},
    "output": {{"type": "command", "command": "pr", "subcommand": "merge", "args": {{"number": 123}}, "flags": ["delete-branch"], "confidence": 0.95}}
  }},
  ...
]

CRITICAL: Output ONLY valid JSON, no explanations or markdown.
"""

# Prompt for generating positive examples WITHOUT context (20%)
POSITIVE_NO_CONTEXT_PROMPT = """You are generating training data for a CLI command interpreter.

Given this CLI schema:
{schema}

Generate {batch_size} diverse examples of natural language inputs that map to CLI commands.
These examples have NO environmental context - all information must be explicit in the input.

For each example, provide:
1. Natural language input with all details explicit
2. context: null
3. Structured JSON output with type "command"

IMPORTANT variations:
- Formality level (casual "show me prs" vs formal "list all pull requests")
- Explicit identifiers ("pr #42", "issue 123", "branch main")
- Synonyms (show/display/list/get, make/create/new)
- Word order variations
- With/without optional arguments
- Include occasional typos

Output as JSON array with exactly {batch_size} items:
[
  {{
    "input": "show me open prs by @octocat",
    "context": null,
    "output": {{"type": "command", "command": "pr", "subcommand": "list", "args": {{"state": "open", "author": "@octocat"}}, "flags": [], "confidence": 0.95}}
  }},
  ...
]

CRITICAL: Output ONLY valid JSON, no explanations or markdown.
"""

# Prompt for generating negative/clarification examples (20%)
NEGATIVE_CASES_PROMPT = """You are generating training data for a CLI command interpreter.

Given this CLI schema:
{schema}

Generate {batch_size} examples of AMBIGUOUS, INCOMPLETE, or OFF-TOPIC inputs that require clarification.
The model should learn to ask for clarification instead of guessing.

Categories to cover:
1. Ambiguous command - "show something" (show what? pr? issue? repo?)
2. Missing required args - "checkout pr" (which PR number?)
3. Unclear intent - "help me with github" (too vague)
4. Off-topic requests - "what's the weather?" (outside CLI scope)
5. Multiple interpretations - "close it" without context (close what?)
6. Typos with ambiguity - "mege" could be merge but unclear

For each example, provide:
1. Natural language input (ambiguous/incomplete)
2. context: null or partial context that doesn't resolve ambiguity
3. Clarification response with suggestions

Output as JSON array with exactly {batch_size} items:
[
  {{
    "input": "show something",
    "context": null,
    "output": {{"type": "clarification", "message": "What would you like to show?", "suggestions": [{{"label": "Pull requests", "command": "pr", "subcommand": "list"}}, {{"label": "Issues", "command": "issue", "subcommand": "list"}}, {{"label": "Repositories", "command": "repo", "subcommand": "list"}}], "confidence": 0.3}}
  }},
  {{
    "input": "checkout pr",
    "context": null,
    "output": {{"type": "clarification", "message": "Which PR number would you like to checkout?", "suggestions": [], "confidence": 0.4}}
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
    example_type: str = "positive_with_context",
) -> list[dict[str, Any]]:
    """Generate a batch of training examples using Claude.

    Args:
        client: Anthropic client
        schema: CLI schema dict
        batch_size: Number of examples to generate
        example_type: One of "positive_with_context", "positive_no_context", "negative"
    """
    schema_summary = schema_to_summary(schema)

    prompt_templates = {
        "positive_with_context": POSITIVE_WITH_CONTEXT_PROMPT,
        "positive_no_context": POSITIVE_NO_CONTEXT_PROMPT,
        "negative": NEGATIVE_CASES_PROMPT,
    }

    prompt = prompt_templates[example_type].format(
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

    # Context can be None or a dict, both are valid
    if "context" not in example:
        return False

    output = example["output"]

    # Check response type
    resp_type = output.get("type")
    if resp_type not in ("command", "clarification"):
        logger.debug(f"Invalid response type: {resp_type}")
        return False

    # Validate clarification responses
    if resp_type == "clarification":
        if "message" not in output:
            logger.debug("Clarification missing message")
            return False
        if "suggestions" not in output:
            logger.debug("Clarification missing suggestions")
            return False
        return True

    # Validate command responses
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


def format_for_training(example: dict[str, Any]) -> dict[str, Any]:
    """Format an example for training (instruction format with context)."""
    return {
        "instruction": example["input"],
        "context": example.get("context"),  # Can be None or dict
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
    num_examples: int = 3000,
    batch_size: int = 50,
    validation_split: float = 0.1,
    resume: bool = True,
) -> None:
    """Generate a complete training dataset.

    Distribution:
    - 60% positive cases WITH context
    - 20% positive cases WITHOUT context
    - 20% negative cases (clarification responses)
    """
    client = anthropic.Anthropic()
    schema = load_schema(schema_path)

    # Calculate target counts for each type
    target_with_context = int(num_examples * 0.6)
    target_no_context = int(num_examples * 0.2)
    target_negative = num_examples - target_with_context - target_no_context

    # Try to resume from existing progress
    examples: dict[str, list[dict[str, Any]]] = {
        "positive_with_context": [],
        "positive_no_context": [],
        "negative": [],
    }
    if resume:
        all_examples = load_existing_examples(output_path)
        # Categorize existing examples
        for ex in all_examples:
            if ex.get("context") is None:
                output = json.loads(ex["output"])
                if output.get("type") == "clarification":
                    examples["negative"].append(ex)
                else:
                    examples["positive_no_context"].append(ex)
            else:
                examples["positive_with_context"].append(ex)

    total_existing = sum(len(v) for v in examples.values())
    if total_existing > 0:
        logger.info(
            f"Resuming with {len(examples['positive_with_context'])} with-context, "
            f"{len(examples['positive_no_context'])} no-context, "
            f"{len(examples['negative'])} negative examples"
        )
    else:
        logger.info(f"Generating {num_examples} examples...")
        logger.info(f"  - {target_with_context} with context (60%)")
        logger.info(f"  - {target_no_context} without context (20%)")
        logger.info(f"  - {target_negative} negative/clarification (20%)")

    # Generate each type
    generation_plan = [
        ("positive_with_context", target_with_context),
        ("positive_no_context", target_no_context),
        ("negative", target_negative),
    ]

    try:
        for example_type, target_count in generation_plan:
            current_count = len(examples[example_type])
            if current_count >= target_count:
                logger.info(f"Already have enough {example_type} examples")
                continue

            remaining = target_count - current_count
            num_batches = (remaining + batch_size - 1) // batch_size

            logger.info(f"Generating {remaining} {example_type} examples...")

            for _ in tqdm(range(num_batches), desc=example_type):
                batch = generate_batch(client, schema, batch_size, example_type)

                # Validate and collect
                for example in batch:
                    if validate_example(example, schema):
                        examples[example_type].append(format_for_training(example))
                    else:
                        logger.debug(f"Invalid example: {example}")

                # Save progress after each batch
                all_examples = (
                    examples["positive_with_context"]
                    + examples["positive_no_context"]
                    + examples["negative"]
                )
                save_progress(all_examples, output_path)

                if len(examples[example_type]) >= target_count:
                    break

    except KeyboardInterrupt:
        total = sum(len(v) for v in examples.values())
        logger.info(f"\nInterrupted! Saved {total} examples. Run again to resume.")
        return
    except Exception as e:
        logger.error(f"Error: {e}")
        total = sum(len(v) for v in examples.values())
        logger.info(f"Saved {total} examples. Run again to resume.")
        raise

    # Combine and shuffle
    all_examples = (
        examples["positive_with_context"][:target_with_context]
        + examples["positive_no_context"][:target_no_context]
        + examples["negative"][:target_negative]
    )
    random.shuffle(all_examples)

    # Split into train/validation
    split_idx = int(len(all_examples) * (1 - validation_split))
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

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

    # Log distribution
    train_with_ctx = sum(1 for ex in train_examples if ex.get("context") is not None)
    train_clarification = sum(
        1 for ex in train_examples if json.loads(ex["output"]).get("type") == "clarification"
    )
    logger.info(
        f"Distribution: {train_with_ctx} with context, "
        f"{len(train_examples) - train_with_ctx - train_clarification} no context, "
        f"{train_clarification} clarification"
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate training data for clitron")
    parser.add_argument("--schema", type=Path, required=True, help="Path to schema YAML")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--num-examples", type=int, default=3000, help="Number of examples")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for generation")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--start-from-scratch", action="store_true", help="Start fresh, ignore existing progress")

    args = parser.parse_args()

    generate_dataset(
        schema_path=args.schema,
        output_path=args.output,
        num_examples=args.num_examples,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        resume=not args.start_from_scratch,
    )


if __name__ == "__main__":
    main()
