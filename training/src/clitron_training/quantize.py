"""Model quantization to GGUF format.

This module handles converting trained models to GGUF format and
applying quantization for efficient deployment.
"""

import argparse
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_llama_cpp() -> bool:
    """Check if llama.cpp tools are available."""
    try:
        subprocess.run(["llama-quantize", "--help"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def convert_to_gguf(
    input_path: Path,
    output_path: Path,
    output_type: str = "f16",
) -> Path:
    """Convert a HuggingFace model to GGUF format."""
    logger.info(f"Converting {input_path} to GGUF (type: {output_type})...")

    # Try using llama.cpp's convert script
    try:
        subprocess.run(
            [
                "python", "-m", "llama_cpp.convert",
                "--input", str(input_path),
                "--output", str(output_path),
                "--outtype", output_type,
            ],
            check=True,
        )
        return output_path
    except (subprocess.CalledProcessError, ModuleNotFoundError):
        pass

    # Fallback: try convert.py from llama.cpp
    try:
        subprocess.run(
            [
                "python", "convert.py",
                str(input_path),
                "--outfile", str(output_path),
                "--outtype", output_type,
            ],
            check=True,
        )
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    raise RuntimeError(
        "Could not convert model to GGUF. Please install llama.cpp or llama-cpp-python:\n"
        "  pip install llama-cpp-python\n"
        "  # or\n"
        "  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make"
    )


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
        gguf_path = input_path
    else:
        # Convert to GGUF first (fp16)
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "model-f16.gguf"
            convert_to_gguf(input_path, gguf_path, "f16")

            # If quantization is f16, we're done
            if quantization in ("f16", "fp16"):
                shutil.copy(gguf_path, output_path)
                logger.info(f"Model saved to {output_path}")
                return

            # Quantize
            quantize_gguf(gguf_path, output_path, quantization)

    # If input was already GGUF, quantize directly
    if input_path.suffix == ".gguf" and quantization not in ("f16", "fp16"):
        quantize_gguf(input_path, output_path, quantization)

    logger.info(f"Quantized model saved to {output_path}")

    # Print size comparison
    if input_path.exists():
        input_size = input_path.stat().st_size / (1024 * 1024)
        output_size = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Size: {input_size:.1f}MB -> {output_size:.1f}MB "
                    f"({output_size/input_size*100:.1f}%)")


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
