"""Model quantization to GGUF format.

This module handles converting trained models to GGUF format and
applying quantization for efficient deployment.

Workflow:
1. Detect if input has LoRA adapters and merge them
2. Convert merged model to GGUF using llama.cpp scripts
3. Quantize GGUF to target precision
"""

import argparse
import logging
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URL for llama.cpp conversion script (pinned to b4867 which matches gguf 0.17.1)
CONVERT_SCRIPT_URL = "https://raw.githubusercontent.com/ggerganov/llama.cpp/b4867/convert_hf_to_gguf.py"


def check_llama_cpp() -> bool:
    """Check if llama.cpp tools are available."""
    try:
        subprocess.run(["llama-quantize", "--help"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def merge_lora_adapters(adapter_path: Path, output_path: Path) -> Path:
    """Merge LoRA adapters with base model."""
    logger.info(f"Merging LoRA adapters from {adapter_path}...")

    # Load adapter config to get base model name
    import json
    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise RuntimeError(f"No adapter_config.json found in {adapter_path}")

    with open(adapter_config_path) as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise RuntimeError("Could not determine base model from adapter config")

    logger.info(f"Loading base model: {base_model_name}")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",  # Use CPU for merging to save GPU memory
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    # Load and merge LoRA adapters
    logger.info("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    logger.info("Merging adapters into base model...")
    merged_model = model.merge_and_unload()

    # Save merged model
    logger.info(f"Saving merged model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    return output_path


def get_convert_script(script_dir: Path) -> Path:
    """Download or locate the GGUF conversion script."""
    script_path = script_dir / "convert_hf_to_gguf.py"

    if not script_path.exists():
        logger.info("Downloading convert_hf_to_gguf.py from llama.cpp...")
        urllib.request.urlretrieve(CONVERT_SCRIPT_URL, script_path)
        logger.info(f"Downloaded to {script_path}")

    return script_path


def convert_to_gguf(
    input_path: Path,
    output_path: Path,
    output_type: str = "f16",
) -> Path:
    """Convert a HuggingFace model to GGUF format."""
    logger.info(f"Converting {input_path} to GGUF (type: {output_type})...")

    # Get or download the conversion script
    script_dir = Path(__file__).parent
    convert_script = get_convert_script(script_dir)

    # Run conversion
    try:
        subprocess.run(
            [
                "python", str(convert_script),
                str(input_path),
                "--outfile", str(output_path),
                "--outtype", output_type,
            ],
            check=True,
        )
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert model to GGUF: {e}")


def quantize_gguf(
    input_path: Path,
    output_path: Path,
    quantization: str = "q4_k_m",
) -> Path:
    """Quantize a GGUF model."""
    logger.info(f"Quantizing {input_path} with {quantization}...")

    try:
        subprocess.run(
            ["llama-quantize", str(input_path), str(output_path), quantization],
            check=True,
        )
        return output_path
    except FileNotFoundError:
        raise RuntimeError(
            "llama-quantize not found. Please install llama.cpp:\n"
            "  git clone https://github.com/ggerganov/llama.cpp\n"
            "  cd llama.cpp && make"
        )


def has_lora_adapters(model_path: Path) -> bool:
    """Check if the model path contains LoRA adapters."""
    adapter_config = model_path / "adapter_config.json"
    adapter_model = model_path / "adapter_model.safetensors"
    return adapter_config.exists() and adapter_model.exists()


def quantize_model(
    input_path: Path,
    output_path: Path,
    quantization: str = "q4_k_m",
) -> None:
    """Convert and quantize a model to GGUF format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if input is already GGUF
    if input_path.suffix == ".gguf":
        logger.info("Input is already GGUF, skipping conversion")
        if quantization in ("f16", "fp16"):
            shutil.copy(input_path, output_path)
            logger.info(f"Model saved to {output_path}")
            return
        quantize_gguf(input_path, output_path, quantization)
        logger.info(f"Quantized model saved to {output_path}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Step 1: Check for and merge LoRA adapters
        if has_lora_adapters(input_path):
            logger.info("Detected LoRA adapters, merging with base model...")
            merged_path = tmpdir_path / "merged"
            merge_lora_adapters(input_path, merged_path)
            model_path = merged_path
        else:
            model_path = input_path

        # Step 2: Convert to GGUF (f16)
        gguf_path = tmpdir_path / "model-f16.gguf"
        convert_to_gguf(model_path, gguf_path, "f16")

        # Step 3: If quantization is f16, we're done
        if quantization in ("f16", "fp16"):
            shutil.copy(gguf_path, output_path)
            logger.info(f"Model saved to {output_path}")
            return

        # Step 4: Quantize to target precision
        quantize_gguf(gguf_path, output_path, quantization)

    logger.info(f"Quantized model saved to {output_path}")

    # Print output size
    if output_path.exists():
        output_size = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Output size: {output_size:.1f}MB")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Quantize model to GGUF")
    parser.add_argument("--input", type=Path, required=True, help="Input model path")
    parser.add_argument("--output", type=Path, required=True, help="Output GGUF path")
    parser.add_argument(
        "--quantization",
        type=str,
        default="q4_k_m",
        choices=["f16", "q8_0", "q6_k", "q5_k_m", "q5_0", "q4_k_m", "q4_k_s", "q4_0", "q3_k_m", "q2_k"],
        help="Quantization type (default: q4_k_m)",
    )

    args = parser.parse_args()

    quantize_model(
        input_path=args.input,
        output_path=args.output,
        quantization=args.quantization,
    )


if __name__ == "__main__":
    main()
