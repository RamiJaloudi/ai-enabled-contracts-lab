from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from contractlab.legal_ai.calibration import TemperatureCalibration, load_temperature_calibration


class ClausePredictor:
    def __init__(self, checkpoint: str | Path, *, device: str | None = None):
        self.checkpoint = str(checkpoint)
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(selected_device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint).to(
            self.device
        )
        self.model.eval()
        self.model_version = Path(self.checkpoint).name or self.checkpoint
        label_map = json.dumps(self.model.config.id2label, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(label_map).hexdigest()[:12]
        self.taxonomy_version = f"id2label-sha256:{digest}"
        self.calibration = load_temperature_calibration(self.checkpoint)

    def predict_top_k(
        self,
        texts: list[str],
        max_length: int = 256,
        top_k: int = 3,
        prediction_type: str = "doctrine",
    ) -> list[list[dict[str, Any]]]:
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
            calibration = getattr(self, "calibration", TemperatureCalibration())
            probabilities = (logits / calibration.temperature).softmax(dim=-1)
        mapping = self.model.config.id2label
        count = min(max(1, top_k), probabilities.shape[-1])
        confidence, label_ids = probabilities.topk(count, dim=-1)
        results: list[list[dict[str, Any]]] = []
        for row_ids, row_scores in zip(label_ids.cpu(), confidence.cpu(), strict=True):
            results.append(
                [
                    {
                        "label": mapping.get(
                            int(label_id),
                            mapping.get(str(int(label_id)), str(int(label_id))),
                        ),
                        "confidence": float(score),
                        "model_version": self.model_version,
                        "taxonomy_version": self.taxonomy_version,
                        "prediction_type": prediction_type,
                        "rank": rank,
                        "calibrated": calibration.calibrated,
                        "calibration_version": calibration.version,
                    }
                    for rank, (label_id, score) in enumerate(
                        zip(row_ids, row_scores, strict=True), start=1
                    )
                ]
            )
        return results

    def __call__(self, texts: list[str], max_length: int = 256) -> list[dict[str, Any]]:
        return [predictions[0] for predictions in self.predict_top_k(texts, max_length, 1)]
