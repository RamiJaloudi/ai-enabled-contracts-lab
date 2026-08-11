from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from contractlab.assistant import analyze_contract, contract_chat_response
from contractlab.legal_ai.authority import extract_authorities
from contractlab.legal_ai.chat_service import (
    answer_contract_question,
    answer_contract_workspace_question,
)
from contractlab.legal_ai.config import LegalAISettings, load_legal_ai_settings
from contractlab.legal_ai.orchestrator import LegalAnalysisOrchestrator
from contractlab.paths import PROJECT_ROOT


class PromptEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    split: Literal["development", "holdout"] = "development"
    task: Literal[
        "full_contract_draft",
        "contract_recommendation",
        "contract_workspace_chat",
        "legal_analysis",
    ]
    request: str = Field(min_length=1)
    source_text: str = Field(min_length=1)
    predictions: list[dict[str, Any]] = Field(default_factory=list)
    evidence_passages: list[dict[str, Any]] = Field(default_factory=list)
    documents: list[dict[str, str]] = Field(default_factory=list)
    expected_terms: list[str] = Field(default_factory=list)
    expected_issue_terms: list[str] = Field(default_factory=list)
    required_concepts: list[str] = Field(default_factory=list)
    forbidden_terms: list[str] = Field(default_factory=list)
    minimum_placeholders: int = Field(default=0, ge=0)
    requires_jurisdiction_verification: bool = False
    requires_evidence_ids: bool = False
    requires_allegation_labeling: bool = False


class PromptEvalAssertion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    detail: str


class PromptEvalCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    task: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    provider: str
    model: str
    used_model_fallback: bool = False
    used_local_fallback: bool = False
    latency_seconds: float = Field(ge=0.0)
    assertions: list[PromptEvalAssertion]
    output: Any
    error: str | None = None


class PromptEvalRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    evaluation_split: str = "all"
    created_at: datetime
    dataset_path: str
    dataset_sha256: str
    prompt_bundle_sha256: str
    prompt_sha256: dict[str, str]
    provider: str
    configured_model: str
    fallback_model: str
    temperature: float
    drafting_temperature: float = 0.0
    case_count: int
    pass_rate: float = Field(ge=0.0, le=1.0)
    mean_score: float = Field(ge=0.0, le=1.0)
    results: list[PromptEvalCaseResult]


_PLACEHOLDER_PATTERN = re.compile(r"\[[^\[\]\r\n]{2,80}\]")
_NUMBERED_HEADING_PATTERN = re.compile(r"(?m)^\s*\d+[.)]\s+")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_prompt_eval_cases(path: Path) -> list[PromptEvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = [PromptEvalCase.model_validate(row) for row in raw]
    identifiers = [case.id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Prompt evaluation case IDs must be unique.")
    if not cases:
        raise ValueError("Prompt evaluation dataset cannot be empty.")
    return cases


def _assertion(name: str, passed: bool, detail: str) -> PromptEvalAssertion:
    return PromptEvalAssertion(name=name, passed=passed, detail=detail)


def _normalized_prose(value: str) -> str:
    number_words = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "fifteen": "15",
        "thirty": "30",
        "forty": "40",
        "forty-five": "45",
    }
    raw_tokens = re.sub(r"[^a-z0-9-]+", " ", value.casefold()).strip().split()
    tokens = [number_words.get(token, token) for token in raw_tokens]
    deduplicated = [
        token for index, token in enumerate(tokens) if not index or token != tokens[index - 1]
    ]
    return " ".join(deduplicated)


def _has_concept(text: str, concept: str) -> bool:
    aliases = {
        "signature": ("signature", "execution"),
        "exclusion": (
            "exclusion",
            "excluded",
            "not liable",
            "indirect damages",
            "consequential damages",
        ),
        "required disclosure": ("required disclosure", "compelled disclosure"),
        "return": ("return", "destruction", "destroy"),
        "liability": ("liability", "damages"),
        "notice": ("notice", "notices"),
    }
    lowered = text.casefold()
    return any(alias in lowered for alias in aliases.get(concept.casefold(), (concept,)))


def _content_assertions(case: PromptEvalCase, content: str) -> list[PromptEvalAssertion]:
    normalized = _normalized_prose(content)
    assertions = [
        _assertion(
            f"preserves:{term}",
            _normalized_prose(term) in normalized,
            "Expected source term is present."
            if _normalized_prose(term) in normalized
            else "Missing source term.",
        )
        for term in case.expected_terms
    ]
    assertions.extend(
        _assertion(
            f"forbids:{term}",
            _normalized_prose(term) not in normalized,
            "Forbidden content is absent."
            if _normalized_prose(term) not in normalized
            else "Forbidden content appeared.",
        )
        for term in case.forbidden_terms
    )
    assertions.extend(
        _assertion(
            f"concept:{concept}",
            _has_concept(content, concept),
            "Required concept is addressed."
            if _has_concept(content, concept)
            else "Required concept is missing.",
        )
        for concept in case.required_concepts
    )
    return assertions


def _score(assertions: list[PromptEvalAssertion]) -> tuple[float, bool]:
    if not assertions:
        return 0.0, False
    score = sum(item.passed for item in assertions) / len(assertions)
    return score, all(item.passed for item in assertions)


def _evaluate_draft(
    case: PromptEvalCase,
    orchestrator: LegalAnalysisOrchestrator,
) -> tuple[list[PromptEvalAssertion], dict[str, Any], str, str, bool, bool, str | None]:
    analysis = analyze_contract(case.source_text)
    reply = answer_contract_question(
        orchestrator,
        contract_text=case.source_text,
        analysis_result=analysis,
        question=case.request,
        history=[],
        fallback=contract_chat_response,
        draft_mode="draft",
    )
    content = reply.content
    assertions = _content_assertions(case, content)
    placeholder_count = len(_PLACEHOLDER_PATTERN.findall(content))
    section_count = len(_NUMBERED_HEADING_PATTERN.findall(content))
    artifact_extensions = {item.filename.rsplit(".", 1)[-1].casefold() for item in reply.artifacts}
    assertions.extend(
        [
            _assertion(
                "generative-structured-draft",
                not reply.used_local_fallback,
                "The configured LLM returned a valid structured draft."
                if not reply.used_local_fallback
                else "The structured draft failed and the deterministic fallback was used.",
            ),
            _assertion(
                "complete-contract-shape",
                section_count >= 8,
                f"Detected {section_count} numbered sections; expected at least 8.",
            ),
            _assertion(
                "missing-terms-use-placeholders",
                placeholder_count >= case.minimum_placeholders,
                f"Detected {placeholder_count} bracketed placeholders; expected at least {case.minimum_placeholders}.",
            ),
            _assertion(
                "docx-and-text-artifacts",
                artifact_extensions == {"docx", "txt"}
                and all(item.data for item in reply.artifacts),
                f"Artifact extensions: {sorted(artifact_extensions)}.",
            ),
            _assertion(
                "no-unsupported-authority",
                not extract_authorities(content),
                "No case or statutory citation pattern was detected."
                if not extract_authorities(content)
                else f"Detected authorities: {extract_authorities(content)}",
            ),
        ]
    )
    if case.requires_jurisdiction_verification:
        assertions.append(
            _assertion(
                "jurisdiction-verification-marker",
                "JURISDICTION_SPECIFIC_VERIFICATION_REQUIRED" in content,
                "Jurisdiction verification marker is present."
                if "JURISDICTION_SPECIFIC_VERIFICATION_REQUIRED" in content
                else "Jurisdiction verification marker is missing.",
            )
        )
    output = {
        "content": content,
        "artifacts": [
            {"filename": item.filename, "media_type": item.media_type, "bytes": len(item.data)}
            for item in reply.artifacts
        ],
    }
    return (
        assertions,
        output,
        reply.provider,
        reply.model,
        reply.used_model_fallback,
        reply.used_local_fallback,
        reply.error,
    )


def _evaluate_recommendation(
    case: PromptEvalCase,
    orchestrator: LegalAnalysisOrchestrator,
) -> tuple[list[PromptEvalAssertion], dict[str, Any], str, str, bool, bool, str | None]:
    analysis = analyze_contract(case.source_text)
    reply = answer_contract_question(
        orchestrator,
        contract_text=case.source_text,
        analysis_result=analysis,
        question=case.request,
        history=[],
        fallback=contract_chat_response,
        draft_mode="review",
    )
    content = reply.content
    numbered_count = len(_NUMBERED_HEADING_PATTERN.findall(content))
    assertions = _content_assertions(case, content)
    assertions.extend(
        [
            _assertion(
                "generative-recommendation",
                not reply.used_local_fallback,
                "The configured LLM answered the review request."
                if not reply.used_local_fallback
                else "The deterministic review fallback was used.",
            ),
            _assertion(
                "exactly-three-recommendations",
                numbered_count == 3,
                f"Detected {numbered_count} numbered recommendations; expected exactly 3.",
            ),
            _assertion(
                "no-draft-artifacts",
                not reply.artifacts,
                "Recommendation mode returned no contract artifacts."
                if not reply.artifacts
                else "Recommendation mode unexpectedly returned contract artifacts.",
            ),
            _assertion(
                "no-full-contract-marker",
                "DRAFT FOR REVIEW" not in content,
                "The answer remained in recommendation mode."
                if "DRAFT FOR REVIEW" not in content
                else "The answer unexpectedly switched to full-contract mode.",
            ),
            _assertion(
                "no-unsupported-authority",
                not extract_authorities(content),
                "No case or statutory citation pattern was detected."
                if not extract_authorities(content)
                else f"Detected authorities: {extract_authorities(content)}",
            ),
        ]
    )
    return (
        assertions,
        {"content": content, "artifacts": []},
        reply.provider,
        reply.model,
        reply.used_model_fallback,
        reply.used_local_fallback,
        reply.error,
    )


def _evaluate_analysis(
    case: PromptEvalCase,
    orchestrator: LegalAnalysisOrchestrator,
) -> tuple[list[PromptEvalAssertion], dict[str, Any], str, str, bool, bool, str | None]:
    result = orchestrator.analyze(
        text=case.source_text,
        predictions=case.predictions,
        evidence_passages=case.evidence_passages,
    )
    output = result.model_dump(mode="json")
    serialized = json.dumps(output, ensure_ascii=False)
    issue_text = " ".join(
        f"{item.taxonomy_label} {item.issue}" for item in result.issues
    ).casefold()
    assertions = _content_assertions(case, serialized)
    assertions.extend(
        [
            _assertion(
                "structured-analysis",
                bool(result.issues),
                f"Returned {len(result.issues)} structured issues.",
            ),
            _assertion(
                "expected-issue-recovery",
                all(term.casefold() in issue_text for term in case.expected_issue_terms),
                f"Expected issue terms: {case.expected_issue_terms}; observed: {issue_text}.",
            ),
            _assertion(
                "missing-fact-coverage",
                all(item.missing_facts for item in result.issues),
                "Every issue identifies missing facts."
                if all(item.missing_facts for item in result.issues)
                else "At least one issue lacks missing facts.",
            ),
            _assertion(
                "counterargument-coverage",
                all(item.counterarguments for item in result.issues),
                "Every issue includes counterarguments."
                if all(item.counterarguments for item in result.issues)
                else "At least one issue lacks counterarguments.",
            ),
            _assertion(
                "no-unsupported-authority",
                not extract_authorities(serialized),
                "No case or statutory citation pattern was detected."
                if not extract_authorities(serialized)
                else f"Detected authorities: {extract_authorities(serialized)}",
            ),
        ]
    )
    if case.requires_jurisdiction_verification:
        assertions.append(
            _assertion(
                "jurisdiction-verification-status",
                all(
                    item.jurisdiction_status == "JURISDICTION_SPECIFIC_VERIFICATION_REQUIRED"
                    for item in result.issues
                ),
                "All issues require jurisdiction-specific verification."
                if all(
                    item.jurisdiction_status == "JURISDICTION_SPECIFIC_VERIFICATION_REQUIRED"
                    for item in result.issues
                )
                else "At least one issue lacks the required jurisdiction status.",
            )
        )
    if case.requires_evidence_ids:
        known_ids = {str(row["passage_id"]) for row in case.evidence_passages}
        assertions.append(
            _assertion(
                "evidence-grounding",
                all(
                    item.evidence_passage_ids and set(item.evidence_passage_ids) <= known_ids
                    for item in result.issues
                ),
                "Every issue cites only supplied passage IDs."
                if all(
                    item.evidence_passage_ids and set(item.evidence_passage_ids) <= known_ids
                    for item in result.issues
                )
                else "At least one issue lacks supplied evidence IDs.",
            )
        )
    if case.requires_allegation_labeling:
        factual_text = " ".join(
            [
                *[fact for issue in result.issues for fact in issue.supporting_facts],
                *[fact for issue in result.issues for fact in issue.adverse_facts],
            ]
        ).casefold()
        assertions.append(
            _assertion(
                "allegation-fact-separation",
                any(term in factual_text for term in ("alleg", "reports", "states", "denies")),
                "Factual statements retain allegation or party-position qualifiers."
                if any(term in factual_text for term in ("alleg", "reports", "states", "denies"))
                else "Party allegations appear to have been converted into unqualified facts.",
            )
        )
    audit = result.audit
    return (
        assertions,
        output,
        audit.provider if audit else orchestrator.llm.provider,
        audit.generative_model if audit else orchestrator.llm.model,
        audit.used_model_fallback if audit else False,
        False,
        None,
    )


def _evaluate_workspace_chat(
    case: PromptEvalCase,
    orchestrator: LegalAnalysisOrchestrator,
) -> tuple[list[PromptEvalAssertion], dict[str, Any], str, str, bool, bool, str | None]:
    documents = case.documents or [{"id": "D1", "filename": "source.txt", "text": case.source_text}]
    reply = answer_contract_workspace_question(
        orchestrator,
        documents=documents,
        question=case.request,
        history=[],
        analysis_result=analyze_contract(case.source_text),
        fallback=contract_chat_response,
        draft_mode="review",
    )
    assertions = _content_assertions(case, reply.content)
    document_ids = [str(row.get("id", "")) for row in documents]
    artifact_extensions = {item.filename.rsplit(".", 1)[-1].casefold() for item in reply.artifacts}
    placeholder_count = sum(
        1
        for match in _PLACEHOLDER_PATTERN.findall(reply.content)
        if not re.fullmatch(r"[dD][0-9]+", match[1:-1])
    )
    assertions.extend(
        [
            _assertion(
                "generative-workspace-chat",
                not reply.used_local_fallback,
                "The configured LLM answered the workspace request."
                if not reply.used_local_fallback
                else "The local fallback answered the workspace request.",
            ),
            _assertion(
                "document-id-grounding",
                all(f"[{document_id}]" in reply.content for document_id in document_ids),
                f"Expected grounded IDs: {document_ids}.",
            ),
            _assertion(
                "docx-and-text-response-exports",
                artifact_extensions == {"docx", "txt"}
                and all(item.data for item in reply.artifacts),
                f"Artifact extensions: {sorted(artifact_extensions)}.",
            ),
            _assertion(
                "actual-language-when-drafting",
                "proposed language" in reply.content.casefold()
                or "shall" in reply.content.casefold()
                or " may " in f" {reply.content.casefold()} "
                or " must " in f" {reply.content.casefold()} "
                or "reconciled" in reply.content.casefold(),
                "The response contains concrete drafting language.",
            ),
            _assertion(
                "no-unsupported-authority",
                not extract_authorities(reply.content),
                "No unsupported authority pattern was detected.",
            ),
        ]
    )
    if case.minimum_placeholders:
        assertions.append(
            _assertion(
                "unresolved-choices-use-placeholders",
                placeholder_count >= case.minimum_placeholders,
                f"Detected {placeholder_count} non-citation placeholders; expected at least "
                f"{case.minimum_placeholders}.",
            )
        )
    return (
        assertions,
        {
            "content": reply.content,
            "artifacts": [
                {
                    "filename": item.filename,
                    "media_type": item.media_type,
                    "bytes": len(item.data),
                }
                for item in reply.artifacts
            ],
        },
        reply.provider,
        reply.model,
        reply.used_model_fallback,
        reply.used_local_fallback,
        reply.error,
    )


def evaluate_prompt_case(
    case: PromptEvalCase,
    orchestrator: LegalAnalysisOrchestrator,
) -> PromptEvalCaseResult:
    started = datetime.now(UTC)
    try:
        if case.task == "full_contract_draft":
            evaluated = _evaluate_draft(case, orchestrator)
        elif case.task == "contract_recommendation":
            evaluated = _evaluate_recommendation(case, orchestrator)
        elif case.task == "contract_workspace_chat":
            evaluated = _evaluate_workspace_chat(case, orchestrator)
        else:
            evaluated = _evaluate_analysis(case, orchestrator)
        assertions, output, provider, model, model_fallback, local_fallback, error = evaluated
    except Exception as exc:  # noqa: BLE001 - evaluation must preserve provider failures as data
        assertions = [_assertion("execution", False, f"{type(exc).__name__}: {exc}")]
        output = None
        provider = orchestrator.llm.provider
        model = orchestrator.llm.model
        model_fallback = False
        local_fallback = False
        error = f"{type(exc).__name__}: {exc}"
    latency = (datetime.now(UTC) - started).total_seconds()
    score, passed = _score(assertions)
    return PromptEvalCaseResult(
        case_id=case.id,
        task=case.task,
        score=score,
        passed=passed,
        provider=provider,
        model=model,
        used_model_fallback=model_fallback,
        used_local_fallback=local_fallback,
        latency_seconds=latency,
        assertions=assertions,
        output=output,
        error=error,
    )


def _prompt_hashes(settings: LegalAISettings) -> tuple[dict[str, str], str]:
    hashes = {
        name: _sha256(path.read_text(encoding="utf-8"))
        for name, path in sorted(settings.prompt_paths.items())
    }
    return hashes, _sha256(json.dumps(hashes, sort_keys=True))


def run_prompt_evaluation(
    *,
    cases: list[PromptEvalCase],
    dataset_path: Path,
    settings: LegalAISettings,
    label: str,
    evaluation_split: str = "all",
) -> PromptEvalRun:
    orchestrator = LegalAnalysisOrchestrator.from_settings(settings)
    results = [evaluate_prompt_case(case, orchestrator) for case in cases]
    prompt_hashes, bundle_hash = _prompt_hashes(settings)
    return PromptEvalRun(
        label=label,
        evaluation_split=evaluation_split,
        created_at=datetime.now(UTC),
        dataset_path=str(dataset_path.resolve()),
        dataset_sha256=_sha256(dataset_path.read_text(encoding="utf-8")),
        prompt_bundle_sha256=bundle_hash,
        prompt_sha256=prompt_hashes,
        provider=settings.provider,
        configured_model=settings.model,
        fallback_model=settings.fallback_model,
        temperature=settings.temperature,
        drafting_temperature=settings.drafting_temperature,
        case_count=len(results),
        pass_rate=sum(result.passed for result in results) / len(results),
        mean_score=sum(result.score for result in results) / len(results),
        results=results,
    )


def compare_prompt_runs(baseline: PromptEvalRun, candidate: PromptEvalRun) -> dict[str, Any]:
    baseline_by_id = {result.case_id: result for result in baseline.results}
    candidate_by_id = {result.case_id: result for result in candidate.results}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("Prompt evaluation runs must contain the same case IDs.")
    cases = []
    for case_id in sorted(baseline_by_id):
        before = baseline_by_id[case_id]
        after = candidate_by_id[case_id]
        cases.append(
            {
                "case_id": case_id,
                "baseline_score": before.score,
                "candidate_score": after.score,
                "delta": after.score - before.score,
                "baseline_passed": before.passed,
                "candidate_passed": after.passed,
            }
        )
    regressions = [row["case_id"] for row in cases if row["delta"] < 0]
    return {
        "baseline_label": baseline.label,
        "candidate_label": candidate.label,
        "baseline_mean_score": baseline.mean_score,
        "candidate_mean_score": candidate.mean_score,
        "mean_score_delta": candidate.mean_score - baseline.mean_score,
        "baseline_pass_rate": baseline.pass_rate,
        "candidate_pass_rate": candidate.pass_rate,
        "regressions": regressions,
        "adopt_candidate": candidate.mean_score > baseline.mean_score and not regressions,
        "cases": cases,
    }


def rescore_prompt_run(run: PromptEvalRun, cases: list[PromptEvalCase]) -> PromptEvalRun:
    """Reapply deterministic rubric checks to retained outputs without calling the model again."""
    cases_by_id = {case.id: case for case in cases}
    rescored: list[PromptEvalCaseResult] = []
    for result in run.results:
        case = cases_by_id.get(result.case_id)
        if case is None:
            raise ValueError(
                f"No prompt evaluation case found for retained result {result.case_id!r}."
            )
        if not isinstance(result.output, dict) or not isinstance(result.output.get("content"), str):
            rescored.append(result)
            continue
        refreshed = _content_assertions(case, result.output["content"])
        retained = [
            assertion
            for assertion in result.assertions
            if not assertion.name.startswith(("preserves:", "forbids:", "concept:"))
        ]
        assertions = [*refreshed, *retained]
        score, passed = _score(assertions)
        rescored.append(
            result.model_copy(update={"assertions": assertions, "score": score, "passed": passed})
        )
    return run.model_copy(
        update={
            "pass_rate": sum(result.passed for result in rescored) / len(rescored),
            "mean_score": sum(result.score for result in rescored) / len(rescored),
            "results": rescored,
        }
    )


def _settings_with_overrides(
    settings: LegalAISettings,
    overrides: list[str],
    model: str | None,
    temperature: float | None,
) -> LegalAISettings:
    prompt_paths = dict(settings.prompt_paths)
    for raw in overrides:
        name, separator, path_value = raw.partition("=")
        if not separator or name not in prompt_paths:
            raise ValueError(f"Prompt override must be NAME=PATH for a configured prompt: {raw}")
        prompt_paths[name] = Path(path_value).resolve()
    return replace(
        settings,
        prompt_paths=prompt_paths,
        model=model or settings.model,
        temperature=settings.temperature if temperature is None else temperature,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run versioned legal prompt evaluations.")
    parser.add_argument("--cases", type=Path, default=PROJECT_ROOT / "prompt_evals" / "cases.json")
    parser.add_argument("--case", action="append", dest="case_ids", default=[])
    parser.add_argument(
        "--split",
        choices=["development", "holdout", "all"],
        default="all",
        help="Run prompt-tuning development cases, untouched holdout cases, or both.",
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--fail-under", type=float, default=0.0)
    args = parser.parse_args()

    dataset_path = args.cases.resolve()
    cases = load_prompt_eval_cases(dataset_path)
    if args.split != "all":
        cases = [case for case in cases if case.split == args.split]
        if not cases:
            raise ValueError(f"No prompt evaluation cases found for split {args.split!r}.")
    if args.case_ids:
        selected = set(args.case_ids)
        cases = [case for case in cases if case.id in selected]
        missing = selected - {case.id for case in cases}
        if missing:
            raise ValueError(f"Unknown prompt evaluation case IDs: {sorted(missing)}")
    settings = _settings_with_overrides(
        load_legal_ai_settings(), args.override, args.model, args.temperature
    )
    run = run_prompt_evaluation(
        cases=cases,
        dataset_path=dataset_path,
        settings=settings,
        label=args.label,
        evaluation_split=args.split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "label": run.label,
                "case_count": run.case_count,
                "pass_rate": run.pass_rate,
                "mean_score": run.mean_score,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )
    return 0 if run.mean_score >= args.fail_under else 1


if __name__ == "__main__":
    raise SystemExit(main())
