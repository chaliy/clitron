"""Generate preference data for DPO training (offline, no API calls).

This module generates preference pairs by:
1. Creating prompts with known correct outputs from the schema
2. Running the SFT model to get its outputs (batched for speed)
3. Creating preference pairs where SFT output differs from correct output

Features:
- Batched inference for 5-10x speedup
- Checkpoint/recovery support - resume from where you left off
- Proper JSON extraction from model outputs
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Checkpoint frequency (save every N pairs)
CHECKPOINT_INTERVAL = 50


def load_schema(schema_path: Path) -> dict[str, Any]:
    """Load CLI schema from YAML."""
    with open(schema_path) as f:
        return yaml.safe_load(f)


# Templates for generating natural language prompts
PROMPT_TEMPLATES = {
    "list": [
        "list {objects}",
        "show {objects}",
        "show me {objects}",
        "display {objects}",
        "get {objects}",
        "what are my {objects}",
        "show all {objects}",
        "list all {objects}",
        "give me {objects}",
        "i want to see {objects}",
        "can you show {objects}",
    ],
    "list_filtered": [
        "list {filter} {objects}",
        "show {filter} {objects}",
        "show me {filter} {objects}",
        "get {filter} {objects}",
        "display {filter} {objects}",
        "what {objects} are {filter}",
        "{filter} {objects}",
        "show all {filter} {objects}",
        "list all {filter} {objects}",
    ],
    "create": [
        "create {object}",
        "make {object}",
        "new {object}",
        "create a {object}",
        "make a {object}",
        "add {object}",
        "i want to create {object}",
        "let's create {object}",
    ],
    "view": [
        "view {object} {id}",
        "show {object} {id}",
        "display {object} {id}",
        "get {object} {id}",
        "open {object} {id}",
        "show me {object} {id}",
        "what is {object} {id}",
        "{object} {id}",
    ],
    "close": [
        "close {object} {id}",
        "close {object} #{id}",
        "shut {object} {id}",
    ],
    "merge": [
        "merge {object} {id}",
        "merge {object} #{id}",
        "land {object} {id}",
        "complete {object} {id}",
    ],
    "delete": [
        "delete {object} {id}",
        "remove {object} {id}",
        "del {object} {id}",
    ],
    "checkout": [
        "checkout {object} {id}",
        "switch to {object} {id}",
        "go to {object} {id}",
    ],
    "comment": [
        "comment on {object} {id}",
        "add comment to {object} {id}",
        "leave a comment on {object} {id}",
    ],
    "edit": [
        "edit {object} {id}",
        "update {object} {id}",
        "modify {object} {id}",
        "change {object} {id}",
    ],
    "status": [
        "status",
        "show status",
        "what's the status",
        "current status",
    ],
}

# Object name variations
OBJECT_NAMES = {
    "pr": ["pr", "prs", "pull request", "pull requests", "PR", "PRs"],
    "issue": ["issue", "issues", "bug", "bugs", "ticket", "tickets"],
    "repo": ["repo", "repos", "repository", "repositories"],
    "branch": ["branch", "branches"],
    "release": ["release", "releases"],
    "workflow": ["workflow", "workflows", "action", "actions"],
    "run": ["run", "runs", "workflow run", "workflow runs"],
    "gist": ["gist", "gists"],
}

# Filter variations for list commands
FILTER_VARIATIONS = {
    "open": ["open", "opened", "active"],
    "closed": ["closed", "completed", "done", "finished"],
    "merged": ["merged", "landed"],
    "draft": ["draft", "wip", "work in progress"],
    "all": ["all", "every"],
}


def generate_prompt_output_pairs(schema: dict[str, Any], num_pairs: int) -> list[dict]:
    """Generate prompt-output pairs from schema with known correct outputs."""
    pairs = []

    for cmd in schema.get("commands", []):
        cmd_name = cmd["name"]
        obj_variations = OBJECT_NAMES.get(cmd_name, [cmd_name, cmd_name + "s"])

        for sub in cmd.get("subcommands", []):
            sub_name = sub["name"]

            # Generate list commands
            if sub_name == "list":
                # Basic list
                for template in PROMPT_TEMPLATES["list"]:
                    for obj in obj_variations:
                        prompt = template.format(objects=obj)
                        output = {
                            "type": "command",
                            "command": cmd_name,
                            "subcommand": "list",
                            "args": {},
                            "flags": [],
                            "confidence": 0.95,
                        }
                        pairs.append({"prompt": prompt, "correct_output": output})

                # List with filters (state argument)
                for arg in sub.get("args", []):
                    if arg["name"] == "state" and isinstance(arg.get("type"), dict):
                        for enum_val in arg["type"].get("enum", []):
                            filter_vars = FILTER_VARIATIONS.get(enum_val, [enum_val])
                            for template in PROMPT_TEMPLATES["list_filtered"]:
                                for obj in obj_variations:
                                    for filt in filter_vars:
                                        prompt = template.format(objects=obj, filter=filt)
                                        output = {
                                            "type": "command",
                                            "command": cmd_name,
                                            "subcommand": "list",
                                            "args": {"state": enum_val},
                                            "flags": [],
                                            "confidence": 0.95,
                                        }
                                        pairs.append({"prompt": prompt, "correct_output": output})

            # Generate view commands
            elif sub_name == "view":
                for template in PROMPT_TEMPLATES["view"]:
                    for obj in obj_variations[:2]:  # Limit variations
                        for id_val in [42, 123, 7, 99]:
                            prompt = template.format(object=obj, id=id_val)
                            output = {
                                "type": "command",
                                "command": cmd_name,
                                "subcommand": "view",
                                "args": {"number": id_val} if cmd_name in ["pr", "issue"] else {"id": str(id_val)},
                                "flags": [],
                                "confidence": 0.95,
                            }
                            pairs.append({"prompt": prompt, "correct_output": output})

            # Generate create commands
            elif sub_name == "create":
                for template in PROMPT_TEMPLATES["create"]:
                    for obj in obj_variations[:2]:
                        prompt = template.format(object=obj)
                        output = {
                            "type": "command",
                            "command": cmd_name,
                            "subcommand": "create",
                            "args": {},
                            "flags": [],
                            "confidence": 0.9,
                        }
                        pairs.append({"prompt": prompt, "correct_output": output})

            # Generate close commands
            elif sub_name == "close":
                for template in PROMPT_TEMPLATES["close"]:
                    for obj in obj_variations[:2]:
                        for id_val in [42, 123, 7]:
                            prompt = template.format(object=obj, id=id_val)
                            output = {
                                "type": "command",
                                "command": cmd_name,
                                "subcommand": "close",
                                "args": {"number": id_val} if cmd_name in ["pr", "issue"] else {"id": str(id_val)},
                                "flags": [],
                                "confidence": 0.95,
                            }
                            pairs.append({"prompt": prompt, "correct_output": output})

            # Generate merge commands (PR specific)
            elif sub_name == "merge" and cmd_name == "pr":
                for template in PROMPT_TEMPLATES["merge"]:
                    for obj in OBJECT_NAMES["pr"][:2]:
                        for id_val in [42, 123, 7]:
                            prompt = template.format(object=obj, id=id_val)
                            output = {
                                "type": "command",
                                "command": "pr",
                                "subcommand": "merge",
                                "args": {"number": id_val},
                                "flags": [],
                                "confidence": 0.95,
                            }
                            pairs.append({"prompt": prompt, "correct_output": output})

            # Generate checkout commands (PR specific)
            elif sub_name == "checkout" and cmd_name == "pr":
                for template in PROMPT_TEMPLATES["checkout"]:
                    for obj in OBJECT_NAMES["pr"][:2]:
                        for id_val in [42, 123]:
                            prompt = template.format(object=obj, id=id_val)
                            output = {
                                "type": "command",
                                "command": "pr",
                                "subcommand": "checkout",
                                "args": {"number": id_val},
                                "flags": [],
                                "confidence": 0.95,
                            }
                            pairs.append({"prompt": prompt, "correct_output": output})

            # Generate comment commands
            elif sub_name == "comment":
                for template in PROMPT_TEMPLATES["comment"]:
                    for obj in obj_variations[:2]:
                        for id_val in [42, 123]:
                            prompt = template.format(object=obj, id=id_val)
                            output = {
                                "type": "command",
                                "command": cmd_name,
                                "subcommand": "comment",
                                "args": {"number": id_val} if cmd_name in ["pr", "issue"] else {"id": str(id_val)},
                                "flags": [],
                                "confidence": 0.9,
                            }
                            pairs.append({"prompt": prompt, "correct_output": output})

            # Generate edit commands
            elif sub_name == "edit":
                for template in PROMPT_TEMPLATES["edit"]:
                    for obj in obj_variations[:2]:
                        for id_val in [42, 123]:
                            prompt = template.format(object=obj, id=id_val)
                            output = {
                                "type": "command",
                                "command": cmd_name,
                                "subcommand": "edit",
                                "args": {"number": id_val} if cmd_name in ["pr", "issue"] else {"id": str(id_val)},
                                "flags": [],
                                "confidence": 0.9,
                            }
                            pairs.append({"prompt": prompt, "correct_output": output})

            # Generate status commands
            elif sub_name == "status":
                for template in PROMPT_TEMPLATES["status"]:
                    output = {
                        "type": "command",
                        "command": cmd_name,
                        "subcommand": "status",
                        "args": {},
                        "flags": [],
                        "confidence": 0.95,
                    }
                    pairs.append({"prompt": template, "correct_output": output})

    # Shuffle and limit
    random.shuffle(pairs)
    return pairs[:num_pairs]


def extract_json(text: str) -> str:
    """Extract JSON object from text, handling common model output issues."""
    # Try to find JSON object pattern
    # Look for { ... } pattern, handling nested braces
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if match:
        return match.group(0)

    # Fallback: return stripped text
    return text.strip()


def format_prompt(prompt: str) -> str:
    """Format a prompt for the model."""
    return f"""<|system|>
You are a CLI command interpreter. Output only valid JSON.

<|user|>
{prompt}
<|assistant|>
"""


def get_sft_outputs_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    batch_size: int = 8,
) -> list[str]:
    """Get outputs from the SFT model in batches for speed."""
    results = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        full_prompts = [format_prompt(p) for p in batch_prompts]

        # Tokenize with padding
        inputs = tokenizer(
            full_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode each output
        for j, output in enumerate(outputs):
            generated = tokenizer.decode(output, skip_special_tokens=True)

            # Extract the assistant's response
            if "<|assistant|>" in generated:
                generated = generated.split("<|assistant|>")[-1].strip()

            # Extract JSON from the response
            generated = extract_json(generated)
            results.append(generated)

    return results


def get_sft_output(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
) -> str:
    """Get output from the SFT model (single prompt, for compatibility)."""
    results = get_sft_outputs_batch(model, tokenizer, [prompt], batch_size=1)
    return results[0] if results else ""


def is_valid_json(text: str) -> bool:
    """Check if text is valid JSON."""
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def outputs_match(correct: dict, generated_str: str) -> bool:
    """Check if generated output matches correct output."""
    try:
        generated = json.loads(generated_str)

        # Check key fields
        if generated.get("command") != correct.get("command"):
            return False
        if generated.get("subcommand") != correct.get("subcommand"):
            return False

        # Check critical args (allow extra args in generated)
        correct_args = correct.get("args", {})
        generated_args = generated.get("args", {})
        for key, value in correct_args.items():
            if generated_args.get(key) != value:
                return False

        return True
    except (json.JSONDecodeError, AttributeError):
        return False


def load_checkpoint(checkpoint_path: Path) -> tuple[list[dict], set[str], dict]:
    """Load checkpoint if exists. Returns (preferences, processed_prompts, stats)."""
    if not checkpoint_path.exists():
        return [], set(), {"matches": 0, "mismatches": 0, "invalid_json": 0}

    preferences = []
    processed_prompts = set()

    with open(checkpoint_path) as f:
        for line in f:
            try:
                pref = json.loads(line)
                preferences.append(pref)
                processed_prompts.add(pref["prompt"])
            except json.JSONDecodeError:
                continue

    # Load stats if available
    stats_path = checkpoint_path.with_suffix(".stats.json")
    stats = {"matches": 0, "mismatches": 0, "invalid_json": 0}
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)

    logger.info(f"Resumed from checkpoint: {len(preferences)} pairs already generated")
    return preferences, processed_prompts, stats


def save_checkpoint(
    checkpoint_path: Path,
    preferences: list[dict],
    stats: dict,
) -> None:
    """Save checkpoint for recovery."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    with open(checkpoint_path, "w") as f:
        for pref in preferences:
            f.write(json.dumps(pref, ensure_ascii=False) + "\n")

    stats_path = checkpoint_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f)


def generate_preferences(
    model_path: Path,
    schema_path: Path,
    output_path: Path,
    num_pairs: int = 1000,
    batch_size: int = 8,
) -> None:
    """Generate preference pairs for DPO training (offline).

    Features:
    - Batched inference for speed
    - Checkpoint/recovery support
    """
    # Checkpoint path (alongside output)
    checkpoint_path = output_path.with_suffix(".checkpoint.jsonl")

    # Load checkpoint if exists
    preferences, processed_prompts, stats = load_checkpoint(checkpoint_path)
    matches = stats["matches"]
    mismatches = stats["mismatches"]
    invalid_json = stats["invalid_json"]

    if len(preferences) >= num_pairs:
        logger.info(f"Already have {len(preferences)} pairs, nothing to do")
        # Just copy checkpoint to output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for pref in preferences:
                f.write(json.dumps(pref, ensure_ascii=False) + "\n")
        return

    # Load schema
    schema = load_schema(schema_path)

    # Load SFT model
    logger.info(f"Loading SFT model from {model_path}")

    # Check if this is a LoRA adapter or full model
    adapter_config = model_path / "adapter_config.json"
    if adapter_config.exists():
        # Load base model and apply LoRA adapter
        with open(adapter_config) as f:
            config = json.load(f)
        base_model_name = config.get("base_model_name_or_path", "meta-llama/Llama-3.2-1B-Instruct")

        logger.info(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        logger.info(f"Loading LoRA adapter from {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        # Load full model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure left padding for batched generation
    tokenizer.padding_side = "left"

    # Generate test prompts with correct outputs
    remaining_needed = num_pairs - len(preferences)
    logger.info(f"Need {remaining_needed} more pairs (have {len(preferences)})")

    # Generate extra pairs to account for matches and already processed
    test_pairs = generate_prompt_output_pairs(schema, remaining_needed * 3)

    # Filter out already processed prompts
    test_pairs = [p for p in test_pairs if p["prompt"] not in processed_prompts]
    logger.info(f"Running SFT model on {len(test_pairs)} prompts (batch_size={batch_size})...")

    # Process in batches
    pbar = tqdm(total=remaining_needed, desc="Generating preferences", initial=0)
    pairs_since_checkpoint = 0

    for batch_start in range(0, len(test_pairs), batch_size):
        if len(preferences) >= num_pairs:
            break

        batch_pairs = test_pairs[batch_start:batch_start + batch_size]
        batch_prompts = [p["prompt"] for p in batch_pairs]

        try:
            # Get SFT outputs for entire batch
            sft_outputs = get_sft_outputs_batch(model, tokenizer, batch_prompts, batch_size=len(batch_prompts))

            for pair, sft_output in zip(batch_pairs, sft_outputs):
                if len(preferences) >= num_pairs:
                    break

                prompt = pair["prompt"]
                correct_output = pair["correct_output"]

                if not is_valid_json(sft_output):
                    invalid_json += 1
                    # Invalid JSON is definitely rejected
                    preferences.append({
                        "prompt": prompt,
                        "chosen": json.dumps(correct_output),
                        "rejected": sft_output,
                    })
                    pbar.update(1)
                    pairs_since_checkpoint += 1
                elif outputs_match(correct_output, sft_output):
                    matches += 1
                    # Outputs match, no preference pair needed
                else:
                    mismatches += 1
                    # Outputs differ, create preference pair
                    preferences.append({
                        "prompt": prompt,
                        "chosen": json.dumps(correct_output),
                        "rejected": sft_output,
                    })
                    pbar.update(1)
                    pairs_since_checkpoint += 1

        except Exception as e:
            logger.warning(f"Error processing batch: {e}")
            continue

        # Save checkpoint periodically
        if pairs_since_checkpoint >= CHECKPOINT_INTERVAL:
            save_checkpoint(
                checkpoint_path,
                preferences,
                {"matches": matches, "mismatches": mismatches, "invalid_json": invalid_json},
            )
            pairs_since_checkpoint = 0
            logger.info(f"Checkpoint saved: {len(preferences)} pairs")

    pbar.close()

    # Save final output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for pref in preferences:
            f.write(json.dumps(pref, ensure_ascii=False) + "\n")

    # Clean up checkpoint files
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    stats_path = checkpoint_path.with_suffix(".stats.json")
    if stats_path.exists():
        stats_path.unlink()

    logger.info(f"Results: {matches} matches, {mismatches} mismatches, {invalid_json} invalid JSON")
    logger.info(f"Generated {len(preferences)} preference pairs -> {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate preference data for DPO (offline)")
    parser.add_argument("--model", type=Path, required=True, help="Path to SFT model")
    parser.add_argument("--schema", type=Path, required=True, help="Path to schema YAML")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--num-pairs", type=int, default=1000, help="Target number of pairs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference (default: 8)")

    args = parser.parse_args()

    generate_preferences(
        model_path=args.model,
        schema_path=args.schema,
        output_path=args.output,
        num_pairs=args.num_pairs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
