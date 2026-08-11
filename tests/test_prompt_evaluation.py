from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from contractlab.legal_ai.prompt_evaluation import (
    PromptEvalAssertion,
    PromptEvalCaseResult,
    PromptEvalRun,
    compare_prompt_runs,
    evaluate_prompt_case,
    load_prompt_eval_cases,
    rescore_prompt_run,
)
from contractlab.legal_ai.schemas import (
    ContractDraft,
    ContractDraftSection,
    DraftGenerationResult,
)
from contractlab.paths import PROJECT_ROOT


def _draft() -> ContractDraft:
    return ContractDraft(
        title="Software Services Agreement",
        contract_type="Software services agreement",
        jurisdiction=None,
        parties=["Harbor Systems Inc.", "Northstar Analytics LLC"],
        effective_date="[EFFECTIVE DATE]",
        purpose="Northstar Analytics LLC will provide the services in [STATEMENT OF WORK].",
        assumptions=[],
        unresolved_terms=["[Liability Cap]", "[Notice Address]"],
        sections=[
            ContractDraftSection(
                heading="Services",
                paragraphs=[
                    "Northstar Analytics LLC will provide the services in [STATEMENT OF WORK]."
                ],
            ),
            ContractDraftSection(
                heading="Payment",
                paragraphs=[
                    "Harbor Systems Inc. will pay undisputed invoices within thirty (30) days."
                ],
            ),
            ContractDraftSection(
                heading="Confidentiality",
                paragraphs=["Each party must protect the other party's confidential information."],
            ),
            ContractDraftSection(
                heading="Intellectual Property",
                paragraphs=["Each party retains its background intellectual property."],
            ),
            ContractDraftSection(
                heading="Termination",
                paragraphs=["Termination rights require [CURE PERIOD] written notice."],
            ),
            ContractDraftSection(
                heading="Limitation of Liability",
                paragraphs=["The liability cap is [LIABILITY CAP]."],
            ),
            ContractDraftSection(
                heading="Governing Law and Notices",
                paragraphs=[
                    "The governing law is [GOVERNING LAW]; notices go to [NOTICE ADDRESS]."
                ],
            ),
            ContractDraftSection(
                heading="Execution",
                paragraphs=["Harbor: [NAME], [TITLE], [DATE]. Northstar: [NAME], [TITLE], [DATE]."],
            ),
        ],
        drafting_note=(
            "Draft for human and counsel review. JURISDICTION_SPECIFIC_VERIFICATION_REQUIRED."
        ),
    )


class DraftingOrchestrator:
    llm = SimpleNamespace(provider="test", model="prompt-model")

    def draft_contract(self, **_):
        return DraftGenerationResult(
            draft=_draft(),
            model="prompt-model",
            provider="test",
        )


def test_prompt_eval_dataset_covers_drafting_review_and_analysis():
    cases = load_prompt_eval_cases(PROJECT_ROOT / "prompt_evals" / "cases.json")

    assert {case.task for case in cases} == {
        "full_contract_draft",
        "contract_recommendation",
        "contract_workspace_chat",
        "legal_analysis",
    }
    exact_case = next(case for case in cases if case.id == "draft-improve-existing-services")
    assert exact_case.request == "Draft a better contract, please add the missing clauses."
    assert exact_case.minimum_placeholders > 0
    assert {case.split for case in cases} == {"development", "holdout"}
    assert all(
        case.id != "workspace-compare-and-redraft" for case in cases if case.split == "holdout"
    )


def test_full_draft_prompt_case_scores_user_visible_artifacts_and_content():
    case = next(
        case
        for case in load_prompt_eval_cases(PROJECT_ROOT / "prompt_evals" / "cases.json")
        if case.id == "draft-improve-existing-services"
    )

    result = evaluate_prompt_case(case, DraftingOrchestrator())  # type: ignore[arg-type]

    assert result.passed is True
    assert result.score == 1.0
    assert {row["filename"].rsplit(".", 1)[-1] for row in result.output["artifacts"]} == {
        "docx",
        "txt",
    }


def _run(label: str, score: float) -> PromptEvalRun:
    assertion = PromptEvalAssertion(name="quality", passed=score == 1.0, detail="fixture")
    result = PromptEvalCaseResult(
        case_id="case-1",
        task="full_contract_draft",
        score=score,
        passed=score == 1.0,
        provider="test",
        model="model",
        latency_seconds=0,
        assertions=[assertion],
        output={},
    )
    return PromptEvalRun(
        label=label,
        created_at=datetime.now(UTC),
        dataset_path="cases.json",
        dataset_sha256="dataset",
        prompt_bundle_sha256="prompts",
        prompt_sha256={},
        provider="test",
        configured_model="model",
        fallback_model="fallback",
        temperature=0,
        case_count=1,
        pass_rate=float(result.passed),
        mean_score=score,
        results=[result],
    )


def test_prompt_comparison_adopts_only_improvement_without_regression():
    comparison = compare_prompt_runs(_run("baseline", 0.5), _run("candidate", 1.0))

    assert comparison["mean_score_delta"] == 0.5
    assert comparison["regressions"] == []
    assert comparison["adopt_candidate"] is True

    regression = compare_prompt_runs(_run("baseline", 1.0), _run("candidate", 0.5))
    assert regression["adopt_candidate"] is False
    assert regression["regressions"] == ["case-1"]


def test_rescore_uses_semantic_number_and_parenthetical_equivalence():
    case = next(
        case
        for case in load_prompt_eval_cases(PROJECT_ROOT / "prompt_evals" / "cases.json")
        if case.id == "draft-ignore-source-injection"
    )
    run = _run("retained", 0.5)
    retained = run.results[0].model_copy(
        update={
            "case_id": case.id,
            "output": {"content": "Confidentiality continues for two (2) years."},
            "assertions": [
                PromptEvalAssertion(name="preserves:two years", passed=False, detail="old rubric")
            ],
        }
    )
    run = run.model_copy(update={"results": [retained]})

    rescored = rescore_prompt_run(run, [case])

    preservation = next(
        assertion
        for assertion in rescored.results[0].assertions
        if assertion.name == "preserves:two years"
    )
    assert preservation.passed is True
