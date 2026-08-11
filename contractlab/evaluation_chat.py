from __future__ import annotations

from typing import Any


def _label_rows(summary: dict[str, Any]) -> list[dict[str, float | str]]:
    report = summary.get("classification_report", {})
    rows: list[dict[str, float | str]] = []
    for label in summary.get("label_names", []):
        values = report.get(label, {})
        rows.append(
            {
                "label": label,
                "precision": float(values.get("precision", 0.0)),
                "recall": float(values.get("recall", 0.0)),
                "f1": float(values.get("f1-score", 0.0)),
                "support": float(values.get("support", 0.0)),
            }
        )
    return rows


def _top_confusions(summary: dict[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    rows = [
        row
        for row in summary.get("confusion_matrix", [])
        if row.get("actual") != row.get("predicted") and int(row.get("count", 0)) > 0
    ]
    return sorted(rows, key=lambda row: int(row.get("count", 0)), reverse=True)[:limit]


def evaluation_diagnosis(summary: dict[str, Any]) -> str:
    """Build an evidence-based diagnosis from saved held-out artifacts."""
    metrics = summary.get("test_metrics", {})
    accuracy = float(metrics.get("accuracy", 0.0))
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    rows = _label_rows(summary)
    zero_rows = [row for row in rows if float(row["f1"]) == 0.0]
    weakest = sorted(rows, key=lambda row: (float(row["f1"]), -float(row["support"])))[:4]
    strongest = sorted(rows, key=lambda row: float(row["f1"]), reverse=True)[:4]
    confusions = _top_confusions(summary)

    lines = [
        "### Evaluation diagnosis",
        f"Accuracy is **{accuracy:.3f}**, while macro F1 is **{macro_f1:.3f}**. "
        f"The **{accuracy - macro_f1:.3f} gap** indicates that frequent labels are "
        "carrying the aggregate accuracy while performance is uneven across labels.",
    ]
    if rows:
        lines.append(
            f"**{len(zero_rows)} of {len(rows)} labels have zero held-out F1.** "
            "That is the clearest sign that the current checkpoint does not provide "
            "reliable coverage of the full label set."
        )
        lines.append(
            "**Weakest labels:** "
            + ", ".join(
                f"{row['label']} (F1 {float(row['f1']):.3f}, n={int(float(row['support']))})"
                for row in weakest
            )
            + "."
        )
        lines.append(
            "**Strongest labels:** "
            + ", ".join(
                f"{row['label']} (F1 {float(row['f1']):.3f}, n={int(float(row['support']))})"
                for row in strongest
            )
            + "."
        )
    if confusions:
        lines.append(
            "**Largest off-diagonal confusions:** "
            + "; ".join(
                f"{row['actual']} -> {row['predicted']} ({int(row['count'])})" for row in confusions
            )
            + "."
        )
    lines.extend(
        [
            "### What to try next",
            "1. Compare the same contract-grouped split with **Contracts-BERT** rather than changing the test set.",
            "2. Use **class-weighted cross-entropy** first; compare focal loss as a separate experiment.",
            "3. Review zero-F1 labels for tiny support, overlapping definitions, and passages that need surrounding context.",
            "4. Select the next checkpoint on validation macro F1 and inspect per-label recall before viewing test results.",
            "5. Treat confidence as uncalibrated until a separate calibration set is available.",
        ]
    )
    return "\n\n".join(lines)


def answer_metrics_question(prompt: str, summary: dict[str, Any]) -> str | None:
    """Answer analysis questions from saved evaluation artifacts without inventing evidence."""
    normalized = prompt.casefold()
    if any(term in normalized for term in ("overfit", "generalization gap", "release-ready")):
        diagnostics = summary.get("overfitting_diagnostics", {})
        if diagnostics.get("status") == "warning":
            return (
                "The run **failed the configured overfitting gate** and is not release-ready. "
                + " ".join(diagnostics.get("reasons", []))
            )
        if diagnostics.get("status") == "passed":
            return (
                "The run passed its configured overfitting checks. At the selected epoch, train "
                f"macro F1 was **{diagnostics.get('train_macro_f1', 0):.3f}**, validation macro "
                f"F1 was **{diagnostics.get('validation_macro_f1', 0):.3f}**, and the measured "
                f"gap was **{diagnostics.get('generalization_gap', 0):.3f}**. Passing this gate "
                "reduces evidence of overfitting but does not prove universal generalization."
            )
        return (
            "This run does not contain comparable train and validation metrics, so overfitting "
            "has **not been verified**. Retrain it with the current safeguards before release."
        )
    if any(term in normalized for term in ("analy", "why", "poor", "improv", "diagnos", "perform")):
        return evaluation_diagnosis(summary)

    metrics = summary.get("test_metrics", {})
    if "macro f1" in normalized or "f1 score" in normalized:
        return f"The held-out macro F1 is **{metrics.get('macro_f1', 0.0):.3f}**."
    if "accuracy" in normalized:
        return f"The held-out accuracy is **{metrics.get('accuracy', 0.0):.3f}**."
    if "precision" in normalized:
        return f"The held-out macro precision is **{metrics.get('macro_precision', 0.0):.3f}**."
    if "recall" in normalized:
        return f"The held-out macro recall is **{metrics.get('macro_recall', 0.0):.3f}**."
    if "how many" in normalized and ("test" in normalized or "held-out" in normalized):
        count = summary.get("split_rows", {}).get("test", 0)
        return f"The held-out test partition contains **{count:,} examples**."

    label_rows = _label_rows(summary)
    if label_rows and any(word in normalized for word in ("weakest", "worst", "lowest")):
        row = min(label_rows, key=lambda item: float(item["f1"]))
        return (
            f"The weakest label by held-out F1 is **{row['label']}** at "
            f"**{float(row['f1']):.3f}** with support **{int(float(row['support']))}**."
        )
    if label_rows and any(word in normalized for word in ("strongest", "best", "highest")):
        row = max(label_rows, key=lambda item: float(item["f1"]))
        return (
            f"The strongest label by held-out F1 is **{row['label']}** at "
            f"**{float(row['f1']):.3f}** with support **{int(float(row['support']))}**."
        )
    for row in label_rows:
        label = str(row["label"])
        if label.casefold() in normalized:
            return (
                f"**{label}** has precision **{float(row['precision']):.3f}**, recall "
                f"**{float(row['recall']):.3f}**, F1 **{float(row['f1']):.3f}**, "
                f"and held-out support **{int(float(row['support']))}**."
            )
    return None


def prediction_markdown(predictions: list[dict[str, Any]]) -> str:
    if not predictions:
        return "The checkpoint did not return a prediction."
    best = predictions[0]
    alternatives = ", ".join(
        f"{item['label']} ({item['confidence']:.1%})" for item in predictions[1:]
    )
    response = (
        f"The checkpoint predicts **{best['label']}** with **{best['confidence']:.1%} confidence**."
    )
    if alternatives:
        response += f" The next candidates are {alternatives}."
    if len(predictions) > 1:
        margin = float(best["confidence"]) - float(predictions[1]["confidence"])
        if float(best["confidence"]) < 0.5 or margin < 0.15:
            response += (
                " **This is an uncertain prediction** because the leading score or margin is low."
            )
    return (
        response + "\n\nThis is a clause-classification result, not a legal interpretation. "
        "Confidence is uncalibrated; compare the passage with the label definition and similar held-out errors."
    )
