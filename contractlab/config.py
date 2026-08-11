from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainingConfig:
    run_name: str = "clause-classifier"
    dataset_path: str = ""
    output_dir: str = ""
    model_name: str = "distilbert/distilbert-base-uncased"
    epochs: int = 3
    learning_rate: float = 2e-5
    train_batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    max_length: int = 256
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 2
    early_stopping_min_delta: float = 0.001
    overfitting_warning_gap: float = 0.15
    overfitting_loss_ratio: float = 1.5
    split_strategy: str = "grouped"
    deduplicate_examples: bool = True
    loss_function: str = "cross_entropy"
    focal_gamma: float = 2.0
    seed: int = 42
    precision: str = "auto"
    freeze_encoder: bool = False
    wandb_mode: str = "disabled"
    wandb_project: str = "ai-enabled-contracts-lab"
    log_every: int = 10
    data_license: str = "Unknown"
    data_source: str = "User supplied"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.run_name.strip():
            errors.append("Run name is required.")
        if self.epochs < 1:
            errors.append("Epochs must be at least 1.")
        if self.learning_rate <= 0:
            errors.append("Learning rate must be positive.")
        if self.train_batch_size < 1 or self.eval_batch_size < 1:
            errors.append("Batch sizes must be at least 1.")
        if self.gradient_accumulation_steps < 1:
            errors.append("Gradient accumulation must be at least 1.")
        if self.early_stopping_patience < 0:
            errors.append("Early-stopping patience cannot be negative.")
        if not 0 <= self.early_stopping_min_delta < 1:
            errors.append("Early-stopping minimum delta must be between 0 and 1.")
        if not 0 < self.overfitting_warning_gap < 1:
            errors.append("Overfitting warning gap must be between 0 and 1.")
        if self.overfitting_loss_ratio <= 1:
            errors.append("Overfitting loss ratio must be greater than 1.")
        if not 32 <= self.max_length <= 4096:
            errors.append("Maximum token length must be between 32 and 4096.")
        if self.loss_function not in {"cross_entropy", "class_weighted", "focal"}:
            errors.append("Loss function must be cross_entropy, class_weighted, or focal.")
        if self.focal_gamma < 0:
            errors.append("Focal gamma cannot be negative.")
        if self.split_strategy not in {"grouped", "predefined"}:
            errors.append("Split strategy must be grouped or predefined.")
        if self.precision not in {"auto", "fp32", "fp16", "bf16"}:
            errors.append("Precision must be auto, fp32, fp16, or bf16.")
        if self.wandb_mode not in {"disabled", "offline", "online"}:
            errors.append("W&B mode must be disabled, offline, or online.")
        return errors

    @property
    def safe_run_name(self) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", self.run_name.strip()).strip("-.")
        return safe or "clause-classifier"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> TrainingConfig:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**values)
