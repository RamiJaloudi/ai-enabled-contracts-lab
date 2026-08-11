from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from contractlab.legal_ai.schemas import (
    ClassificationRoute,
    ClassifierPrediction,
)


class TopKPredictor(Protocol):
    def predict_top_k(
        self,
        texts: list[str],
        *,
        top_k: int,
        prediction_type: str = 'doctrine',
    ) -> list[list[dict[str, object]]]: ...


def _validated(
    rows: list[dict[str, object]] | list[ClassifierPrediction],
    prediction_type: str,
) -> list[ClassifierPrediction]:
    predictions: list[ClassifierPrediction] = []
    for rank, row in enumerate(rows, start=1):
        prediction = (
            row
            if isinstance(row, ClassifierPrediction)
            else ClassifierPrediction.model_validate(row)
        )
        prediction.prediction_type = prediction_type  # type: ignore[assignment]
        prediction.rank = rank
        predictions.append(prediction)
    return sorted(predictions, key=lambda item: item.confidence, reverse=True)


def _margin(predictions: list[ClassifierPrediction]) -> float:
    if not predictions:
        return 0.0
    runner_up = predictions[1].confidence if len(predictions) > 1 else 0.0
    return predictions[0].confidence - runner_up


def derive_domain_predictions(
    doctrine_predictions: list[ClassifierPrediction],
) -> list[ClassifierPrediction]:
    by_domain: dict[str, ClassifierPrediction] = {}
    for prediction in doctrine_predictions:
        if '::' not in prediction.label:
            continue
        domain, _ = prediction.label.split('::', 1)
        previous = by_domain.get(domain)
        if previous is None or prediction.confidence > previous.confidence:
            by_domain[domain] = ClassifierPrediction(
                label=domain,
                confidence=prediction.confidence,
                model_version=prediction.model_version,
                taxonomy_version=prediction.taxonomy_version,
                prediction_type='domain',
                calibrated=prediction.calibrated,
                calibration_version=prediction.calibration_version,
                passage_ids=prediction.passage_ids,
            )
    return _validated(list(by_domain.values()), 'domain')


def build_classification_route(
    *,
    domain_predictions: list[dict[str, object]] | list[ClassifierPrediction],
    doctrine_predictions: list[dict[str, object]] | list[ClassifierPrediction],
    domain_threshold: float,
    doctrine_review_threshold: float,
    auto_route_threshold: float,
    minimum_margin: float,
) -> ClassificationRoute:
    domains = _validated(domain_predictions, 'domain')
    doctrines = _validated(doctrine_predictions, 'doctrine')
    if not domains:
        domains = derive_domain_predictions(doctrines)
    reasons: list[str] = []
    if not domains or not doctrines:
        return ClassificationRoute(
            status='ABSTAIN',
            domain_predictions=domains,
            doctrine_predictions=doctrines,
            reasons=['Both domain and doctrine evidence are required for automatic routing.'],
        )

    domain = domains[0]
    doctrine = doctrines[0]
    if domain.confidence < domain_threshold:
        reasons.append(
            f'Domain confidence {domain.confidence:.3f} is below {domain_threshold:.3f}.'
        )
    if doctrine.confidence < doctrine_review_threshold:
        reasons.append(
            f'Doctrine confidence {doctrine.confidence:.3f} is below '
            f'{doctrine_review_threshold:.3f}.'
        )
        return ClassificationRoute(
            status='ABSTAIN',
            selected_domain=domain.label,
            domain_predictions=domains,
            doctrine_predictions=doctrines,
            reasons=reasons,
        )
    domain_margin = _margin(domains)
    doctrine_margin = _margin(doctrines)
    if domain_margin < minimum_margin:
        reasons.append(f'Domain margin {domain_margin:.3f} is below {minimum_margin:.3f}.')
    if doctrine_margin < minimum_margin:
        reasons.append(f'Doctrine margin {doctrine_margin:.3f} is below {minimum_margin:.3f}.')

    automatic = (
        domain.confidence >= auto_route_threshold
        and doctrine.confidence >= auto_route_threshold
        and domain_margin >= minimum_margin
        and doctrine_margin >= minimum_margin
    )
    if automatic:
        reasons.append('Domain and doctrine scores satisfy automatic-routing gates.')
    elif not reasons:
        reasons.append('Scores remain hypotheses and require human review.')
    return ClassificationRoute(
        status='AUTO_ROUTE' if automatic else 'REVIEW_REQUIRED',
        selected_domain=domain.label,
        domain_predictions=domains,
        doctrine_predictions=doctrines,
        reasons=reasons,
    )


class HierarchicalClassifier:
    def __init__(
        self,
        domain_predictor: TopKPredictor,
        doctrine_predictors: Mapping[str, TopKPredictor],
        *,
        top_k: int = 5,
    ) -> None:
        self.domain_predictor = domain_predictor
        self.doctrine_predictors = dict(doctrine_predictors)
        self.top_k = top_k

    def predict(self, text: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        domains = self.domain_predictor.predict_top_k(
            [text], top_k=self.top_k, prediction_type='domain'
        )[0]
        selected = str(domains[0]['label']) if domains else ''
        predictor = self.doctrine_predictors.get(selected) or self.doctrine_predictors.get('*')
        doctrines = (
            predictor.predict_top_k(
                [text], top_k=self.top_k, prediction_type='doctrine'
            )[0]
            if predictor is not None
            else []
        )
        return domains, doctrines
