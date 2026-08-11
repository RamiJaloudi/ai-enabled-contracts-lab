import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from contractlab.assistant import analyze_contract, contract_chat_response, heuristic_prediction
from contractlab.inference import ClausePredictor
from contractlab.legal_ai.chat_service import (
    answer_contract_question,
    answer_evaluation_question,
    normalize_chat_history,
)
from contractlab.legal_ai.config import load_legal_ai_settings
from contractlab.legal_ai.jurisdiction import route_jurisdiction
from contractlab.legal_ai.llm import LLMError
from contractlab.legal_ai.loaders import load_markdown, load_prompt, load_skill
from contractlab.legal_ai.orchestrator import (
    LegalAnalysisOrchestrator,
    contains_unverified_authority,
)
from contractlab.legal_ai.schemas import ChatResult
from contractlab.paths import PROJECT_ROOT


class RecordingLLM:
    model = "recording-model"
    provider = "test"

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.messages: list[dict[str, str]] = []
        self.schema: dict[str, Any] | None = None

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> str:
        self.messages = messages
        self.schema = schema
        return json.dumps(self.payload)


class FakeTokenizer:
    def __call__(self, texts: list[str], **_: Any) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[1, 2, 3] for _ in texts])}


class FakeModel:
    config = SimpleNamespace(id2label={0: "Alpha", 1: "Beta", 2: "Gamma"})

    def __call__(self, **_: Any) -> SimpleNamespace:
        return SimpleNamespace(logits=torch.tensor([[0.1, 1.4, 0.8]]))


def _valid_analysis_payload() -> dict[str, Any]:
    return {
        "domain": "Contract Law",
        "jurisdiction": None,
        "jurisdiction_confidence": 0.0,
        "classifier_predictions": [],
        "issues": [
            {
                "taxonomy_label": "Governing Law",
                "issue": "Which law may govern the agreement?",
                "analysis_confidence": 0.61,
                "rule": "The governing law depends on the agreement and forum rules.",
                "supporting_facts": ["The text refers to New Jersey law."],
                "adverse_facts": [],
                "missing_facts": ["The forum and complete governing-law clause."],
                "application": "The reference supports New Jersey as a candidate jurisdiction.",
                "counterarguments": ["A conflicts analysis may point elsewhere."],
                "jurisdiction_status": "JURISDICTION_SPECIFIC_VERIFICATION_REQUIRED",
                "conclusion": "Jurisdiction-dependent and subject to verification.",
            }
        ],
    }


def test_instruction_pack_prompts_skills_and_schema_load():
    settings = load_legal_ai_settings()
    for name in (
        "system",
        "issue_spotting",
        "analysis",
        "jurisdiction",
        "bridge",
        "document_analysis",
        "contract_review",
        "evaluation_chat",
        "contract_drafting",
    ):
        assert load_prompt(name, settings.prompt_paths)

    for name in (
        "legal-model-routing",
        "legal-issue-spotting",
        "legal-jurisdiction",
        "legal-fact-extraction",
        "legal-counterarguments",
        "legal-analysis",
    ):
        assert "description:" in load_skill(name)

    starter = load_markdown(PROJECT_ROOT / "prompts" / "CODEX_STARTER_TASK.md")
    assert "Do not implement RAG" in starter

    schema = json.loads(settings.schema_path.read_text(encoding="utf-8"))
    prediction_properties = schema["properties"]["classifier_predictions"]["items"]["properties"]
    assert {"label", "confidence", "model_version", "taxonomy_version"} <= set(
        prediction_properties
    )
    required_issue_fields = set(schema["properties"]["issues"]["items"]["required"])
    assert {
        "taxonomy_label",
        "issue",
        "analysis_confidence",
        "rule",
        "supporting_facts",
        "adverse_facts",
        "missing_facts",
        "application",
        "counterarguments",
        "jurisdiction_status",
        "conclusion",
    } == required_issue_fields
    assert settings.rag_enabled is False


def test_classifier_top_k_preserves_confidence_and_provenance():
    predictor = object.__new__(ClausePredictor)
    predictor.device = torch.device("cpu")
    predictor.tokenizer = FakeTokenizer()
    predictor.model = FakeModel()
    predictor.model_version = "fake-checkpoint-v1"
    predictor.taxonomy_version = "id2label-sha256:test"

    predictions = predictor.predict_top_k(["sample"], top_k=2)[0]

    assert [row["label"] for row in predictions] == ["Beta", "Gamma"]
    assert predictions[0]["confidence"] > predictions[1]["confidence"]
    assert predictions[0]["model_version"] == "fake-checkpoint-v1"
    assert predictions[0]["taxonomy_version"] == "id2label-sha256:test"


def test_heuristic_prediction_includes_taxonomy_provenance():
    prediction = heuristic_prediction("This agreement is governed by New Jersey law.")

    assert prediction["model_version"] == "heuristic-keyword-v1"
    assert prediction["taxonomy_version"] == "0.1.0"
    assert 0 <= prediction["confidence"] <= 1


def test_jurisdiction_router_preserves_ambiguity():
    route = route_jurisdiction(
        "New Jersey law applies, and a United States District Court action is pending."
    )

    assert route.likely_jurisdiction == "New Jersey"
    assert route.alternative_jurisdictions == ["U.S. federal"]
    assert route.confidence < 0.5
    assert route.verification_required is True


def test_orchestrator_preserves_low_confidence_and_missing_facts():
    llm = RecordingLLM(_valid_analysis_payload())
    orchestrator = LegalAnalysisOrchestrator(
        settings=load_legal_ai_settings(),
        llm=llm,
    )
    predictions = [
        {
            "label": "Governing Law",
            "confidence": 0.21,
            "model_version": "classifier-v2",
            "taxonomy_version": "taxonomy-v1",
        },
        {
            "label": "Venue",
            "confidence": 0.12,
            "model_version": "classifier-v2",
            "taxonomy_version": "taxonomy-v1",
        },
    ]

    result = orchestrator.analyze(
        text="The contract states that New Jersey law applies.",
        predictions=predictions,
    )

    assert [row.confidence for row in result.classifier_predictions] == [0.21, 0.12]
    assert result.classifier_predictions[0].model_version == "classifier-v2"
    assert result.issues[0].analysis_confidence == 0.61
    assert result.issues[0].missing_facts
    assert result.jurisdiction == "New Jersey"
    assert llm.schema is not None
    system_prompt = llm.messages[0]["content"]
    assert "Treat those predictions as suggestions" in system_prompt
    assert "JURISDICTION_SPECIFIC_VERIFICATION_REQUIRED" in system_prompt


def test_orchestrator_rejects_invalid_structured_output():
    llm = RecordingLLM({"domain": "Contract Law", "issues": [{}]})
    orchestrator = LegalAnalysisOrchestrator(
        settings=load_legal_ai_settings(),
        llm=llm,
    )

    with pytest.raises(LLMError, match="schema validation"):
        orchestrator.analyze(text="No jurisdiction facts are supplied.")


def test_authority_signal_flags_unverified_case_style_citation():
    assert contains_unverified_authority("Smith v. Jones is cited for this proposition.")
    assert not contains_unverified_authority(
        "This general principle requires jurisdiction-specific verification."
    )


class SuccessfulChatOrchestrator:
    def contract_chat(self, **_: Any) -> ChatResult:
        return ChatResult(
            content="Generated contract answer",
            model="primary-model",
            provider="ollama",
        )

    def evaluation_chat(self, **_: Any) -> ChatResult:
        return ChatResult(
            content="Generated evaluation answer",
            model="fallback-model",
            provider="ollama",
            used_fallback=True,
        )


class FailingChatOrchestrator:
    def contract_chat(self, **_: Any) -> ChatResult:
        raise LLMError("Ollama unavailable")

    def evaluation_chat(self, **_: Any) -> ChatResult:
        raise LLMError("Ollama unavailable")


def test_chat_service_uses_generative_results_and_tracks_model_fallback():
    orchestrator = SuccessfulChatOrchestrator()

    contract_reply = answer_contract_question(
        orchestrator,  # type: ignore[arg-type]
        contract_text="Agreement text",
        analysis_result={},
        question="What should be reviewed?",
        history=[],
        fallback=lambda _question, _result: "Local answer",
    )
    evaluation_reply = answer_evaluation_question(
        orchestrator,  # type: ignore[arg-type]
        summary={},
        question="Why is recall low?",
        history=[],
        fallback_content="Local metrics answer",
    )

    assert contract_reply.content == "Generated contract answer"
    assert contract_reply.source_label == "ollama: primary-model"
    assert contract_reply.used_local_fallback is False
    assert evaluation_reply.content == "Generated evaluation answer"
    assert evaluation_reply.used_model_fallback is True
    assert evaluation_reply.source_label == "ollama: fallback-model (fallback model)"


def test_chat_service_uses_explicit_local_fallback_when_ollama_fails():
    orchestrator = FailingChatOrchestrator()

    contract_reply = answer_contract_question(
        orchestrator,  # type: ignore[arg-type]
        contract_text="Agreement text",
        analysis_result={"engine": "test"},
        question="Summarize",
        history=[],
        fallback=lambda _question, _result: "Grounded local answer",
    )
    evaluation_reply = answer_evaluation_question(
        orchestrator,  # type: ignore[arg-type]
        summary={},
        question="What was macro F1?",
        history=[],
        fallback_content="Saved metrics answer",
    )

    assert contract_reply.content == "Grounded local answer"
    assert contract_reply.used_local_fallback is True
    assert contract_reply.error == "Ollama unavailable"
    assert evaluation_reply.content == "Saved metrics answer"
    assert evaluation_reply.source_label == "Local deterministic fallback"


def test_chat_history_normalization_strips_metadata_and_limits_context():
    messages = [
        {"role": "system", "content": "ignored"},
        {"role": "assistant", "content": "first", "source_label": "local"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "third"},
    ]

    assert normalize_chat_history(messages, limit=2) == [
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "third"},
    ]


def test_evaluation_chat_includes_ad_hoc_classifier_hypotheses():
    llm = RecordingLLM({"answer": "ok"})
    orchestrator = LegalAnalysisOrchestrator(
        settings=load_legal_ai_settings(),
        llm=llm,
    )
    summary = {
        "run_name": "test-run",
        "classification_report": {},
        "label_names": [],
        "ad_hoc_query": "Either party may terminate for breach.",
        "ad_hoc_predictions": [
            {
                "label": "Termination",
                "confidence": 0.72,
                "model_version": "checkpoint-v1",
                "taxonomy_version": "taxonomy-v1",
            }
        ],
    }

    orchestrator.evaluation_chat(
        summary=summary,
        question="Explain the prediction.",
    )

    evidence_message = llm.messages[1]["content"]
    assert '"ad_hoc_query": "Either party may terminate for breach."' in evidence_message
    assert '"label": "Termination"' in evidence_message


def test_streamlit_pages_call_generative_chat_services():
    assistant_page = (PROJECT_ROOT / "app_pages" / "assistant.py").read_text(encoding="utf-8")
    evaluate_page = (PROJECT_ROOT / "app_pages" / "evaluate.py").read_text(encoding="utf-8")

    assert "answer_contract_question(" in assistant_page
    assert '"Draft complete contract"' in assistant_page
    assert "answer_evaluation_question(" in evaluate_page
    assert 'submit_mode="disable"' in assistant_page
    assert 'submit_mode="disable"' in evaluate_page
    assert "typed_prompt = st.chat_input(" in assistant_page
    assert "prompt = pending_prompt or typed_prompt" in assistant_page
    assert "typed_prompt = st.chat_input(" in evaluate_page
    assert "prompt = pending_prompt or typed_prompt" in evaluate_page


def test_contract_chat_routes_drafting_requests_to_concrete_proposal_mode():
    llm = RecordingLLM({"answer": "ok"})
    orchestrator = LegalAnalysisOrchestrator(
        settings=load_legal_ai_settings(),
        llm=llm,
    )

    orchestrator.contract_chat(
        contract_text="A short services agreement.",
        analysis_result={"contract_analysis": {"clauses": []}, "findings": []},
        question="What additions do you propose to the contract?",
    )

    request = llm.messages[-1]["content"]
    assert "DRAFTING RECOMMENDATION MODE" in request
    assert "Proposed language (not in contract)" in request
    assert "three highest-impact" in request
    assert "limited to 90 words" in request
    assert "Do not use a table" in request
    assert "Never cite proposed language" in request


def test_contract_chat_uses_question_answer_mode_for_existing_language_questions():
    llm = RecordingLLM({"answer": "ok"})
    orchestrator = LegalAnalysisOrchestrator(
        settings=load_legal_ai_settings(),
        llm=llm,
    )

    orchestrator.contract_chat(
        contract_text="A short services agreement.",
        analysis_result={"contract_analysis": {"clauses": []}, "findings": []},
        question="What does the termination section say?",
    )

    assert "QUESTION-ANSWER MODE" in llm.messages[-1]["content"]


def test_deterministic_contract_chat_handles_addition_requests_without_passage_retrieval():
    text = """SOFTWARE SERVICES AGREEMENT
Supplier shall provide hosted software services. Customer shall pay invoices within 30 days.
This Agreement is governed by New York law.
"""
    result = analyze_contract(text)

    response = contract_chat_response("What additions do you propose?", result)

    assert "Candidate additions for human review" in response
    assert "Closest supporting passages" not in response
    assert "not legal requirements" in response
