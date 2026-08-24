from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2

class PreferenceTrainer:
    """Interface for DPO/ORPO training implementations."""
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(self, train_path: str | Path, output_dir: str | Path, model_name: str = "distilgpt2") -> dict[str, Any]:
        """Train the policy.

        Uses the TRL-backed DPO trainer and writes the checkpoint to the configured output directory.
        Keep side effects explicit: checkpoints and metrics should go to configured output_dir.
        """
        if self.config.method != "dpo":
            raise ValueError("the runnable trainer currently supports method=\"dpo\" only")
        try:
            from datasets import load_dataset
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from trl import DPOConfig, DPOTrainer
        except ImportError as exc:
            raise RuntimeError("install the [train] dependencies to run DPO") from exc
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        dataset = load_dataset("json", data_files=str(train_path), split="train")
        args = DPOConfig(output_dir=str(output_dir), beta=self.config.beta, max_length=self.config.max_length, per_device_train_batch_size=self.config.batch_size, num_train_epochs=1, max_steps=2, learning_rate=5e-5, logging_steps=1, save_strategy="no", report_to="none", remove_unused_columns=False, use_cpu=True, bf16=False, fp16=False)
        trainer = DPOTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
        result = trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        return {"train_loss": float(result.training_loss), "model_name": model_name}
