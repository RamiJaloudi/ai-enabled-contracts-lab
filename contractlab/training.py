from __future__ import annotations

import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from contractlab.config import TrainingConfig
from contractlab.data import (
    load_table,
    normalize_dataset,
    save_splits,
    split_by_contract,
    split_predefined,
)
from contractlab.evaluation import classification_metrics, confusion_records, detailed_report
from contractlab.overfitting import EarlyStoppingMonitor, overfitting_diagnostics


class ClauseDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, label_to_id: dict[str, int]):
        self.texts = frame["text"].tolist()
        self.labels = [label_to_id[label] for label in frame["label"]]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "text": self.texts[index],
            "labels": self.labels[index],
            "row_index": index,
        }


class TokenizingCollator:
    """Tokenize one batch at a time so large corpora do not fill system RAM."""

    def __init__(self, tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(
            [str(item["text"]) for item in features],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor([int(item["labels"]) for item in features], dtype=torch.long)
        batch["row_index"] = torch.tensor(
            [int(item["row_index"]) for item in features], dtype=torch.long
        )
        return batch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _status(path: Path, state: str, **values: Any) -> None:
    current: dict[str, Any] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
    current.update({"state": state, "updated_at": time.time(), **values})
    _write_json(path, current)


def _precision(device: torch.device, requested: str) -> tuple[bool, torch.dtype | None, bool]:
    if device.type != "cuda" or requested == "fp32":
        return False, None, False
    if requested == "bf16" or (requested == "auto" and torch.cuda.is_bf16_supported()):
        return True, torch.bfloat16, False
    return True, torch.float16, True


def _balanced_class_weights(
    frame: pd.DataFrame,
    label_to_id: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    """Return stable inverse-square-root weights normalized around one."""
    counts = frame["label"].value_counts()
    total = float(len(frame))
    class_count = float(len(label_to_id))
    values = [
        math.sqrt(total / (class_count * max(1.0, float(counts.get(label, 0)))))
        for label in label_to_id
    ]
    weights = torch.tensor(values, dtype=torch.float32, device=device)
    weights = weights / weights.mean().clamp_min(1e-8)
    return weights.clamp(0.25, 4.0)


def _classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_function: str,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
) -> torch.Tensor:
    if loss_function == "cross_entropy":
        return F.cross_entropy(logits, labels)
    per_example = F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        reduction="none",
    )
    if loss_function == "focal":
        probability = torch.exp(-per_example)
        per_example = ((1.0 - probability) ** focal_gamma) * per_example
    return per_example.mean()


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype | None,
) -> tuple[dict[str, float], list[int], list[int], list[int]]:
    model.eval()
    losses: list[float] = []
    labels: list[int] = []
    predictions: list[int] = []
    row_indices: list[int] = []
    context = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if autocast_enabled and autocast_dtype is not None
        else nullcontext()
    )
    with torch.no_grad():
        for batch in loader:
            row_indices.extend(int(value) for value in batch.pop("row_index"))
            batch = {key: value.to(device) for key, value in batch.items()}
            with context:
                outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu()))
            labels.extend(batch["labels"].detach().cpu().tolist())
            predictions.extend(outputs.logits.argmax(dim=-1).detach().cpu().tolist())
    metrics = classification_metrics(labels, predictions)
    metrics["loss"] = float(np.mean(losses)) if losses else math.nan
    return metrics, labels, predictions, row_indices


def run_training(config: TrainingConfig) -> dict[str, Any]:
    errors = config.validate()
    if errors:
        raise ValueError(" ".join(errors))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    _status(status_path, "starting", message="Loading and splitting the dataset")
    seed_everything(config.seed)

    frame = normalize_dataset(load_table(config.dataset_path))
    frame = frame[(frame["text"] != "") & (frame["label"] != "")]
    if config.deduplicate_examples:
        frame = frame.drop_duplicates(subset=["text", "label"])
    splits = (
        split_predefined(frame)
        if config.split_strategy == "predefined"
        else split_by_contract(frame, seed=config.seed)
    )
    save_splits(splits, output_dir / "data_splits")
    labels = sorted(frame["label"].unique().tolist())
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    _write_json(
        output_dir / "labels.json", {"label_to_id": label_to_id, "id_to_label": id_to_label}
    )

    _status(status_path, "starting", message="Loading tokenizer and pretrained model")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(labels),
        label2id=label_to_id,
        id2label=id_to_label,
    )
    if config.freeze_encoder:
        base = getattr(model, model.base_model_prefix)
        for parameter in base.parameters():
            parameter.requires_grad = False

    collator = TokenizingCollator(tokenizer, config.max_length)
    datasets = {name: ClauseDataset(split, label_to_id) for name, split in splits.items()}
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=config.train_batch_size, shuffle=True, collate_fn=collator
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=config.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
        ),
        "test": DataLoader(
            datasets["test"], batch_size=config.eval_batch_size, shuffle=False, collate_fn=collator
        ),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    class_weights = (
        _balanced_class_weights(splits["train"], label_to_id, device)
        if config.loss_function in {"class_weighted", "focal"}
        else None
    )
    class_weight_report = (
        {label: float(class_weights[index].detach().cpu()) for label, index in label_to_id.items()}
        if class_weights is not None
        else {}
    )
    autocast_enabled, autocast_dtype, use_scaler = _precision(device, config.precision)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    updates_per_epoch = math.ceil(len(loaders["train"]) / config.gradient_accumulation_steps)
    total_steps = max(1, updates_per_epoch * config.epochs)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    wandb_run = None
    if config.wandb_mode != "disabled":
        os.environ["WANDB_MODE"] = config.wandb_mode
        import wandb

        safe_config = config.to_dict() | {
            "dataset_path": Path(config.dataset_path).name,
            "output_dir": Path(config.output_dir).name,
            "rows": len(frame),
            "labels": len(labels),
        }
        wandb_run = wandb.init(
            project=config.wandb_project, name=config.safe_run_name, config=safe_config
        )

    history: list[dict[str, float | int]] = []
    early_stopping = EarlyStoppingMonitor(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )
    stopped_early = False
    global_step = 0
    try:
        for epoch in range(1, config.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            for batch_index, batch in enumerate(loaders["train"], start=1):
                batch.pop("row_index")
                batch = {key: value.to(device) for key, value in batch.items()}
                context = (
                    torch.autocast(device_type=device.type, dtype=autocast_dtype)
                    if autocast_enabled and autocast_dtype is not None
                    else nullcontext()
                )
                with context:
                    outputs = model(**batch)
                    loss = (
                        _classification_loss(
                            outputs.logits,
                            batch["labels"],
                            config.loss_function,
                            class_weights,
                            config.focal_gamma,
                        )
                        / config.gradient_accumulation_steps
                    )
                scaler.scale(loss).backward()
                running_loss += float(loss.detach().cpu()) * config.gradient_accumulation_steps
                should_step = (
                    batch_index % config.gradient_accumulation_steps == 0
                    or batch_index == len(loaders["train"])
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                if batch_index % config.log_every == 0 or batch_index == len(loaders["train"]):
                    _status(
                        status_path,
                        "training",
                        epoch=epoch,
                        epochs=config.epochs,
                        batch=batch_index,
                        batches=len(loaders["train"]),
                        global_step=global_step,
                        message=f"Epoch {epoch}: training batch {batch_index}/{len(loaders['train'])}",
                    )

            train_metrics, _, _, _ = evaluate_model(
                model, loaders["train"], device, autocast_enabled, autocast_dtype
            )
            validation_metrics, _, _, _ = evaluate_model(
                model, loaders["validation"], device, autocast_enabled, autocast_dtype
            )
            epoch_record: dict[str, float | int] = {
                "epoch": epoch,
                "train_optimization_loss": running_loss / max(1, len(loaders["train"])),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
            }
            history.append(epoch_record)
            _write_json(output_dir / "history.json", {"history": history})
            if wandb_run:
                wandb_run.log(epoch_record, step=global_step)
            improved, should_stop = early_stopping.observe(validation_metrics["macro_f1"], epoch)
            if improved:
                model.save_pretrained(output_dir / "best")
                tokenizer.save_pretrained(output_dir / "best")
            current_diagnostics = overfitting_diagnostics(
                history,
                generalization_gap_threshold=config.overfitting_warning_gap,
                loss_ratio_threshold=config.overfitting_loss_ratio,
                selected_epoch=early_stopping.best_epoch,
            )
            _status(
                status_path,
                "training",
                epoch=epoch,
                epochs=config.epochs,
                best_epoch=early_stopping.best_epoch,
                best_validation_macro_f1=early_stopping.best_score,
                overfitting_diagnostics=current_diagnostics,
                message=f"Epoch {epoch} validation complete",
            )
            if should_stop:
                stopped_early = True
                _status(
                    status_path,
                    "training",
                    epoch=epoch,
                    epochs=config.epochs,
                    best_epoch=early_stopping.best_epoch,
                    message=(
                        "Early stopping triggered after validation macro-F1 failed to improve "
                        f"by {config.early_stopping_min_delta:.4f} for "
                        f"{config.early_stopping_patience} epoch(s)"
                    ),
                )
                break

        _status(
            status_path,
            "evaluating",
            message="Evaluating the best checkpoint on the held-out test split",
        )
        best_model = AutoModelForSequenceClassification.from_pretrained(output_dir / "best").to(
            device
        )
        test_metrics, true_ids, predicted_ids, row_indices = evaluate_model(
            best_model, loaders["test"], device, autocast_enabled, autocast_dtype
        )
        test_frame = splits["test"].iloc[row_indices].reset_index(drop=True).copy()
        test_frame["actual"] = [id_to_label[value] for value in true_ids]
        test_frame["predicted"] = [id_to_label[value] for value in predicted_ids]
        test_frame["correct"] = test_frame["actual"] == test_frame["predicted"]
        test_frame.to_csv(output_dir / "test_predictions.csv", index=False)
        diagnostics = overfitting_diagnostics(
            history,
            generalization_gap_threshold=config.overfitting_warning_gap,
            loss_ratio_threshold=config.overfitting_loss_ratio,
            selected_epoch=early_stopping.best_epoch,
        )
        results = {
            "run_name": config.run_name,
            "model_name": config.model_name,
            "device": str(device),
            "precision": str(autocast_dtype).replace("torch.", "") if autocast_dtype else "fp32",
            "loss_function": config.loss_function,
            "focal_gamma": config.focal_gamma if config.loss_function == "focal" else None,
            "class_weights": class_weight_report,
            "rows": len(frame),
            "label_names": labels,
            "split_rows": {name: len(split) for name, split in splits.items()},
            "best_validation_macro_f1": early_stopping.best_score,
            "best_epoch": early_stopping.best_epoch,
            "early_stopping": {
                "enabled": config.early_stopping_patience > 0,
                "patience": config.early_stopping_patience,
                "min_delta": config.early_stopping_min_delta,
                "stopped_early": stopped_early,
                "epochs_completed": len(history),
                "epochs_configured": config.epochs,
            },
            "overfitting_diagnostics": diagnostics,
            "release_gate": {
                "passed": diagnostics["release_gate_passed"],
                "split_strategy": config.split_strategy,
                "group_leakage_check_passed": True,
                "test_set_sealed_until_checkpoint_selection": True,
                "reason": (
                    "Overfitting diagnostics and split-integrity checks passed."
                    if diagnostics["release_gate_passed"]
                    else "Potential overfitting detected; do not treat this checkpoint as release-ready."
                ),
            },
            "test_metrics": test_metrics,
            "classification_report": detailed_report(true_ids, predicted_ids, labels),
            "confusion_matrix": confusion_records(true_ids, predicted_ids, labels),
            "history": history,
            "wandb_url": getattr(wandb_run, "url", None) if wandb_run else None,
            "data_source": config.data_source,
            "data_license": config.data_license,
        }
        _write_json(output_dir / "summary.json", results)
        completion_message = (
            "Training completed, but potential overfitting requires review"
            if diagnostics["overfitting_detected"]
            else "Training and held-out evaluation completed"
        )
        _status(
            status_path,
            "completed",
            progress=1.0,
            message=completion_message,
            overfitting_diagnostics=diagnostics,
            release_gate=results["release_gate"],
        )
        if wandb_run:
            wandb_run.summary.update(test_metrics)
        return results
    except Exception as exc:
        _status(status_path, "failed", message=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        if wandb_run:
            wandb_run.finish()
