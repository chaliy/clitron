"""Offline checks for the locked CPU training dependency graph."""

import inspect

from clitron_training import generate_dataset, train_dpo, train_sft
from transformers import Trainer, TrainingArguments
from trl import DPOConfig, DPOTrainer


def test_training_entry_points_import():
    assert callable(generate_dataset.main)
    assert callable(train_sft.main)
    assert callable(train_dpo.main)


def test_trainers_accept_pipeline_arguments(tmp_path):
    assert "processing_class" in inspect.signature(Trainer).parameters
    assert "processing_class" in inspect.signature(DPOTrainer).parameters
    args = TrainingArguments(output_dir=str(tmp_path), report_to="none", use_cpu=True)
    config = DPOConfig(output_dir=str(tmp_path), report_to="none", use_cpu=True, bf16=False)
    assert args.device.type == "cpu"
    assert config.beta > 0


def test_response_only_preprocessing_preserves_eos():
    class Tokenizer:
        eos_token_id = 99

        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2] if kwargs["add_special_tokens"] else [3]}

    result = train_sft.preprocess_function(
        {"instruction": ["show my prs"], "output": ['{"command":"pr"}']}, Tokenizer()
    )
    assert result["input_ids"] == [[1, 2, 3, 99]]
    assert result["labels"] == [[-100, -100, 3, 99]]
    assert result["attention_mask"] == [[1, 1, 1, 1]]
