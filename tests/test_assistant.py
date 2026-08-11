from contractlab.assistant import (
    analyze_contract,
    answer_with_passages,
    extract_facts,
    split_passages,
)

CONTRACT = """Confidentiality. Each party shall keep non-public information confidential for three years.

Payment. Customer shall pay $12,000 within 30 days after August 10, 2026.

Governing Law. This Agreement is governed by the laws of the State of New York."""


def test_passages_keep_source_offsets():
    passages = split_passages(CONTRACT)
    assert len(passages) == 3
    for passage in passages:
        assert CONTRACT[passage["start"] : passage["end"]] == passage["text"]


def test_analysis_exposes_evidence_and_missing_categories():
    result = analyze_contract(CONTRACT)
    labels = {finding["label"] for finding in result["findings"]}
    assert "Confidentiality" in labels
    assert "Payment Terms" in labels
    assert "Governing Law" in labels
    assert "Termination" in result["missing_categories"]
    answers = answer_with_passages("When must payment be made?", result["findings"])
    assert answers
    assert answers[0]["passage_id"].startswith("P")


def test_extract_facts_keeps_obligation_source_sentence():
    facts = extract_facts(CONTRACT)
    assert "August 10, 2026" in facts["dates"]
    assert "$12,000" in facts["money"]
    assert any("shall pay" in value for value in facts["obligations"])


def test_hybrid_assistant_overrides_low_confidence_out_of_scope_label():
    def weak_predictor(texts):
        return [{"label": "Audit Rights", "confidence": 0.21} for _ in texts]

    result = analyze_contract(
        "Each party shall keep all non-public information confidential.", weak_predictor
    )
    assert result["findings"][0]["label"] == "Confidentiality"
    assert result["engine"] == "hybrid checkpoint and transparent playbook"
