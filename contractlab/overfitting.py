from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class EarlyStoppingMonitor:
    patience: int
    min_delta: float
    best_score: float = -math.inf
    best_epoch: int | None = None
    epochs_without_improvement: int = 0

    def observe(self, score: float, epoch: int) -> tuple[bool, bool]:
        improved = score > self.best_score + self.min_delta
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        should_stop = self.patience > 0 and self.epochs_without_improvement >= self.patience
        return improved, should_stop


def overfitting_diagnostics(
    history: list[dict[str, Any]],
    *,
    generalization_gap_threshold: float,
    loss_ratio_threshold: float,
    selected_epoch: int | None = None,
) -> dict[str, Any]:
    comparable = [
        row
        for row in history
        if all(
            key in row
            for key in (
                "epoch",
                "train_macro_f1",
                "validation_macro_f1",
                "train_loss",
                "validation_loss",
            )
        )
    ]
    if not comparable:
        return {
            "status": "not_available",
            "overfitting_detected": None,
            "release_gate_passed": False,
            "reasons": ["Comparable train and validation metrics are unavailable."],
        }

    selected = next(
        (row for row in comparable if int(row["epoch"]) == selected_epoch),
        max(comparable, key=lambda row: float(row["validation_macro_f1"])),
    )
    final = comparable[-1]
    train_f1 = float(selected["train_macro_f1"])
    validation_f1 = float(selected["validation_macro_f1"])
    train_loss = float(selected["train_loss"])
    validation_loss = float(selected["validation_loss"])
    generalization_gap = train_f1 - validation_f1
    loss_ratio = validation_loss / max(train_loss, 1e-12)
    validation_deterioration = validation_f1 - float(final["validation_macro_f1"])
    reasons: list[str] = []
    if generalization_gap > generalization_gap_threshold:
        reasons.append(
            f"Train macro-F1 exceeds validation macro-F1 by {generalization_gap:.3f}, "
            f"above the configured {generalization_gap_threshold:.3f} threshold."
        )
    if loss_ratio > loss_ratio_threshold:
        reasons.append(
            f"Validation loss is {loss_ratio:.2f}x training loss, above the configured "
            f"{loss_ratio_threshold:.2f}x threshold."
        )
    detected = bool(reasons)
    return {
        "status": "warning" if detected else "passed",
        "overfitting_detected": detected,
        "release_gate_passed": not detected,
        "selected_epoch": int(selected["epoch"]),
        "epochs_observed": len(comparable),
        "train_macro_f1": train_f1,
        "validation_macro_f1": validation_f1,
        "generalization_gap": generalization_gap,
        "generalization_gap_threshold": generalization_gap_threshold,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "validation_to_train_loss_ratio": loss_ratio,
        "loss_ratio_threshold": loss_ratio_threshold,
        "validation_deterioration_after_best_epoch": max(0.0, validation_deterioration),
        "reasons": reasons or ["Configured train/validation gap checks passed."],
    }
