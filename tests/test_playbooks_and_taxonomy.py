import json
from pathlib import Path

from contractlab.assistant import analyze_contract, contract_chat_response
from contractlab.playbooks import identify_contract_profile, load_contract_taxonomy

NDA_TEXT = """MUTUAL NON-DISCLOSURE AGREEMENT

Confidential Information means all non-public technical and business information disclosed by either party.
Each party shall keep Confidential Information confidential and use it solely for evaluating a business relationship.
Confidential Information shall not include information that is publicly available or independently developed.
Disclosure may be made to employees who need to know and are bound by confidentiality duties.
If required by law or subpoena, the receiving party will provide notice when legally permitted.
On request, each party shall return or destroy all Confidential Information.
These obligations survive termination for three years.
This Agreement is governed by the laws of the State of New York.
"""


def test_nda_profile_and_chat_use_explicit_definitions():
    assert identify_contract_profile(NDA_TEXT) == "Non-Disclosure Agreement (NDA)"
    result = analyze_contract(NDA_TEXT)
    assert result["contract_profile"] == "Non-disclosure agreement (NDA)"
    assert result["contract_analysis"]["contract_type_id"] == "nda"
    assert result["nda_analysis"]["detected_count"] >= 6
    definition = contract_chat_response("What is an NDA?", result)
    assert "non-disclosure agreement" in definition
    analysis = contract_chat_response("Analyze this NDA", result)
    assert "NDA-specific analysis" in analysis


def test_general_contract_profile_drives_default_clause_review():
    text = """SOFTWARE SERVICES AGREEMENT
This agreement is between Northstar Analytics LLC and Harbor Systems Inc.
Supplier shall provide software services. Customer shall pay invoices within 30 days.
Each party shall keep confidential information secret.
Either party may terminate for material breach. This Agreement is governed by New York law.
"""
    result = analyze_contract(text)
    review = result["contract_analysis"]
    assert review["profile"] == "Master services agreement"
    assert review["domain"] == "commercial"
    detected = {row["id"] for row in review["clauses"] if row["status"] == "Detected"}
    assert {"parties", "scope", "fees_payment", "termination", "confidentiality"} <= detected
    profile = contract_chat_response("What type of contract is this?", result)
    assert "Master services agreement" in profile
    clauses = contract_chat_response("Which clauses were detected?", result)
    assert "Clauses detected" in clauses
    assert "Fees and payment" in clauses


def test_taxonomy_references_are_complete():
    path = Path("datasets/contract_taxonomy.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    clause_ids = {row["id"] for row in payload["clause_definitions"]}
    assert len(payload["domains"]) == 10
    assert len(payload["contract_types"]) >= 45
    assert len(clause_ids) >= 45
    assert load_contract_taxonomy()["metadata"]["license"] == "CC BY 4.0"
    for contract_type in payload["contract_types"]:
        assert set(contract_type["typical_clause_ids"]) <= clause_ids
