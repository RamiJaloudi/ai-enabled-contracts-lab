from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def detailed_report(
    labels: Sequence[int], predictions: Sequence[int], label_names: Sequence[str]
) -> dict:
    all_ids = list(range(len(label_names)))
    return classification_report(
        labels,
        predictions,
        labels=all_ids,
        target_names=list(label_names),
        output_dict=True,
        zero_division=0,
    )


def confusion_records(
    labels: Sequence[int], predictions: Sequence[int], label_names: Sequence[str]
) -> list[dict[str, str | int]]:
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(label_names))))
    records: list[dict[str, str | int]] = []
    for actual_index, predicted_index in np.ndindex(matrix.shape):
        if matrix[actual_index, predicted_index] == 0:
            continue
        records.append(
            {
                "actual": label_names[actual_index],
                "predicted": label_names[predicted_index],
                "count": int(matrix[actual_index, predicted_index]),
            }
        )
    return records
