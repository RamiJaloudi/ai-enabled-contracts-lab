from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class TemperatureCalibration:
    temperature: float = 1.0
    version: str = 'uncalibrated'
    expected_calibration_error: float | None = None

    @property
    def calibrated(self) -> bool:
        return self.version != 'uncalibrated' and self.temperature > 0


def expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    bins: int = 15,
) -> float:
    if probabilities.ndim != 2 or labels.ndim != 1:
        raise ValueError('Probabilities must be 2D and labels must be 1D.')
    confidence, predictions = probabilities.max(dim=1)
    correctness = predictions.eq(labels).float()
    boundaries = torch.linspace(0, 1, bins + 1, device=probabilities.device)
    error = torch.zeros((), device=probabilities.device)
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        in_bin = confidence.gt(lower) & confidence.le(upper)
        if in_bin.any():
            weight = in_bin.float().mean()
            error += weight * (correctness[in_bin].mean() - confidence[in_bin].mean()).abs()
    return float(error.cpu())


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    minimum: float = 0.25,
    maximum: float = 5.0,
    steps: int = 191,
) -> TemperatureCalibration:
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError('Logits and labels have incompatible shapes.')
    if not 0 < minimum < maximum or steps < 2:
        raise ValueError('Temperature search bounds are invalid.')
    candidates = torch.linspace(minimum, maximum, steps, device=logits.device)
    losses = torch.stack([F.cross_entropy(logits / value, labels) for value in candidates])
    temperature = float(candidates[int(losses.argmin())].cpu())
    probabilities = (logits / temperature).softmax(dim=-1)
    ece = expected_calibration_error(probabilities, labels)
    return TemperatureCalibration(
        temperature=temperature,
        version='temperature-scaling-v1',
        expected_calibration_error=ece,
    )


def load_temperature_calibration(checkpoint: str | Path) -> TemperatureCalibration:
    path = Path(checkpoint) / 'calibration.json'
    if not path.exists():
        return TemperatureCalibration()
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        temperature = float(payload['temperature'])
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        ece = payload.get('expected_calibration_error')
        return TemperatureCalibration(
            temperature=temperature,
            version=str(payload.get('version') or 'temperature-scaling-v1'),
            expected_calibration_error=float(ece) if ece is not None else None,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return TemperatureCalibration()


def save_temperature_calibration(
    checkpoint: str | Path,
    calibration: TemperatureCalibration,
) -> Path:
    target = Path(checkpoint) / 'calibration.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'temperature': calibration.temperature,
        'version': calibration.version,
        'expected_calibration_error': calibration.expected_calibration_error,
    }
    target.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return target
