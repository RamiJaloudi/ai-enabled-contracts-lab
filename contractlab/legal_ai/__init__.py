"""Hybrid classifier and generative legal-analysis services."""

from contractlab.legal_ai.config import LegalAISettings, load_legal_ai_settings
from contractlab.legal_ai.contract_artifacts import build_contract_docx, contract_text_bytes
from contractlab.legal_ai.llm import OllamaClient
from contractlab.legal_ai.orchestrator import LegalAnalysisOrchestrator
from contractlab.legal_ai.routing import HierarchicalClassifier, build_classification_route

__all__ = [
    "LegalAISettings",
    "LegalAnalysisOrchestrator",
    "OllamaClient",
    "HierarchicalClassifier",
    "build_classification_route",
    "build_contract_docx",
    "contract_text_bytes",
    "load_legal_ai_settings",
]
