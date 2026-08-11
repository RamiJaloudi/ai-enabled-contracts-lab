from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ModelPreset:
    name: str
    model_id: str
    license: str
    level: str
    notes: str
    approximate_parameters_millions: int


MODEL_PRESETS = [
    ModelPreset(
        "DistilBERT — recommended",
        "distilbert/distilbert-base-uncased",
        "Apache-2.0",
        "Beginner",
        "Fast, widely documented, and suitable for full fine-tuning on 12 GB VRAM.",
        66,
    ),
    ModelPreset(
        "Contracts-BERT - recommended legal-domain comparison",
        "nlpaueb/bert-base-uncased-contracts",
        "CC-BY-SA-4.0",
        "Intermediate",
        "BERT-base pre-trained on U.S. contracts; use batch size 4 with accumulation on 12 GB VRAM.",
        110,
    ),
    ModelPreset(
        "LEGAL-BERT base - broader legal-domain comparison",
        "nlpaueb/legal-bert-base-uncased",
        "CC-BY-SA-4.0",
        "Intermediate",
        "BERT-base pre-trained on mixed English legal corpora; useful as a domain-ablation baseline.",
        110,
    ),
    ModelPreset(
        "DeBERTa v3 small — advanced",
        "microsoft/deberta-v3-small",
        "MIT",
        "Intermediate",
        "Stronger encoder architecture with a somewhat heavier tokenizer and training footprint.",
        44,
    ),
]


def device_summary() -> dict[str, str | bool | int]:
    cuda = torch.cuda.is_available()
    summary: dict[str, str | bool | int] = {
        "cuda_available": cuda,
        "device": "cuda" if cuda else "cpu",
        "bf16_supported": bool(cuda and torch.cuda.is_bf16_supported()),
    }
    if cuda:
        properties = torch.cuda.get_device_properties(0)
        summary.update(
            {
                "gpu": properties.name,
                "vram_gb": round(properties.total_memory / (1024**3), 1),
                "cuda_version": torch.version.cuda or "unknown",
            }
        )
    return summary


def estimate_training_memory_gb(
    parameters_millions: int,
    batch_size: int,
    max_length: int,
    precision: str,
) -> float:
    """Conservative educational estimate, not a CUDA allocator guarantee."""
    bytes_per_value = 4 if precision == "fp32" else 2
    weights_and_states = parameters_millions * 1_000_000 * (bytes_per_value + 12)
    activations = (
        parameters_millions * 1_000_000 * 0.18 * batch_size * (max_length / 256) * bytes_per_value
    )
    overhead = 1.2 * 1024**3
    return round((weights_and_states + activations + overhead) / (1024**3), 1)
