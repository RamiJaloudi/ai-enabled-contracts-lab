from __future__ import annotations

from collections import defaultdict
from typing import Any

from contractlab.assistant import split_passages
from contractlab.legal_ai.schemas import ClassifierPrediction, DocumentPassage


def segment_document(
    text: str,
    *,
    maximum_passage_characters: int,
    overlap_characters: int,
    maximum_passages: int,
) -> tuple[list[DocumentPassage], bool]:
    if maximum_passage_characters < 200:
        raise ValueError('maximum_passage_characters must be at least 200.')
    if not 0 <= overlap_characters < maximum_passage_characters:
        raise ValueError('overlap_characters must be smaller than the passage limit.')
    if maximum_passages < 1:
        raise ValueError('maximum_passages must be positive.')

    base = split_passages(text, minimum_length=1)
    passages: list[DocumentPassage] = []
    step = maximum_passage_characters - overlap_characters
    for parent in base:
        parent_text = str(parent['text'])
        parent_start = int(parent['start'])
        if len(parent_text) <= maximum_passage_characters:
            passages.append(
                DocumentPassage(
                    passage_id=f'P{len(passages) + 1}',
                    text=parent_text,
                    start=parent_start,
                    end=int(parent['end']),
                    parent_passage_id=str(parent['passage_id']),
                )
            )
            continue
        for offset in range(0, len(parent_text), step):
            chunk = parent_text[offset : offset + maximum_passage_characters].strip()
            if not chunk:
                continue
            leading = len(parent_text[offset : offset + maximum_passage_characters]) - len(
                parent_text[offset : offset + maximum_passage_characters].lstrip()
            )
            start = parent_start + offset + leading
            passages.append(
                DocumentPassage(
                    passage_id=f'P{len(passages) + 1}',
                    text=chunk,
                    start=start,
                    end=start + len(chunk),
                    parent_passage_id=str(parent['passage_id']),
                )
            )
            if offset + maximum_passage_characters >= len(parent_text):
                break

    truncated = len(passages) > maximum_passages
    return passages[:maximum_passages], truncated


def classify_passages(
    passages: list[DocumentPassage],
    predictor: Any,
    *,
    top_k: int,
) -> list[list[ClassifierPrediction]]:
    if not passages:
        return []
    texts = [passage.text for passage in passages]
    if hasattr(predictor, 'predict_top_k'):
        raw = predictor.predict_top_k(texts, top_k=top_k, prediction_type='doctrine')
    else:
        raw = [[predictor([text])[0]] for text in texts]
    results: list[list[ClassifierPrediction]] = []
    for passage, rows in zip(passages, raw, strict=True):
        validated: list[ClassifierPrediction] = []
        for rank, row in enumerate(rows, start=1):
            prediction = ClassifierPrediction.model_validate(row)
            prediction.prediction_type = 'doctrine'
            prediction.rank = rank
            prediction.passage_ids = [passage.passage_id]
            validated.append(prediction)
        results.append(validated)
    return results


def aggregate_passage_predictions(
    passage_predictions: list[list[ClassifierPrediction]],
    *,
    top_k: int,
) -> list[ClassifierPrediction]:
    scores: dict[str, list[float]] = defaultdict(list)
    evidence: dict[str, list[str]] = defaultdict(list)
    prototypes: dict[str, ClassifierPrediction] = {}
    for rows in passage_predictions:
        for prediction in rows:
            scores[prediction.label].append(prediction.confidence)
            for passage_id in prediction.passage_ids:
                if passage_id not in evidence[prediction.label]:
                    evidence[prediction.label].append(passage_id)
            prototypes.setdefault(prediction.label, prediction)

    aggregated: list[ClassifierPrediction] = []
    for label, values in scores.items():
        prototype = prototypes[label]
        strongest = max(values)
        support_bonus = min(0.08, 0.02 * (len(values) - 1))
        aggregated.append(
            prototype.model_copy(
                update={
                    'confidence': min(1.0, strongest + support_bonus),
                    'passage_ids': evidence[label],
                }
            )
        )
    aggregated.sort(key=lambda item: item.confidence, reverse=True)
    for rank, prediction in enumerate(aggregated[:top_k], start=1):
        prediction.rank = rank
    return aggregated[:top_k]


def evidence_payload(
    passages: list[DocumentPassage],
    predictions: list[ClassifierPrediction],
) -> list[dict[str, Any]]:
    relevant = {item for prediction in predictions for item in prediction.passage_ids}
    selected = [passage for passage in passages if passage.passage_id in relevant]
    if not selected:
        selected = passages
    return [passage.model_dump() for passage in selected]
