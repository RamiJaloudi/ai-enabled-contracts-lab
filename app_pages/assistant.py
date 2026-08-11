import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from contractlab.assistant import analyze_contract, contract_chat_response
from contractlab.document_parser import extract_text
from contractlab.inference import ClausePredictor
from contractlab.jobs import list_runs
from contractlab.legal_ai.chat_service import answer_contract_question, normalize_chat_history
from contractlab.ui import cached_legal_ai_orchestrator, page_intro

page_intro(
    "Contract analysis assistant",
    "Explore six analysis capabilities while keeping every observation tied to an exact source passage.",
)

st.warning(
    "This page demonstrates an ML pipeline. It does not determine enforceability, legal risk, or what you should agree to.",
    icon=":material/gavel:",
)

sample_contract = """SOFTWARE SERVICES AGREEMENT

This Software Services Agreement is entered into on August 10, 2026 between Northstar Analytics LLC and Harbor Systems Inc.

Confidentiality. Each party shall keep the other party's non-public business and technical information confidential for three years and may disclose it only to personnel who need to know it.

Payment. Customer shall pay undisputed invoices within 30 days after receipt. Late undisputed amounts accrue interest at one percent per month.

Intellectual Property. Each party retains its pre-existing intellectual property. Customer owns Customer Data, and Supplier may process that data only to provide the Services.

Limitation of Liability. Neither party is liable for indirect or consequential damages. Except for confidentiality breaches and infringement claims, aggregate liability will not exceed fees paid in the previous twelve months.

Termination. Either party may terminate for material breach that remains uncured for 30 days after written notice.

Renewal. The initial one-year term renews automatically for additional one-year terms unless either party gives 60 days' notice of non-renewal.

Governing Law. This Agreement is governed by the laws of the State of New York, and the parties consent to courts located in New York County.
"""

source_mode = st.segmented_control(
    "Contract source", ["Paste text", "Upload file"], default="Paste text"
)
contract_text = ""
if source_mode == "Paste text":
    contract_text = st.text_area("Contract text", value=sample_contract, height=320)
else:
    uploaded = st.file_uploader("Contract file", type=["txt", "md", "pdf", "docx"])
    if uploaded:
        try:
            contract_text = extract_text(uploaded.getvalue(), uploaded.name)
            st.text_area("Extracted text", value=contract_text, height=320)
            if uploaded.name.lower().endswith(".pdf") and not contract_text.strip():
                st.warning(
                    "No embedded text was found. Scanned PDFs require OCR, which is not included yet."
                )
        except Exception as exc:
            st.error(f"Could not parse the file: {exc}")

completed = [
    run
    for run in list_runs()
    if run.get("state") == "completed" and (Path(run["run_dir"]) / "best").exists()
]
engine_options = ["Transparent keyword baseline"] + [
    f"Checkpoint: {run['run_id']}" for run in completed
]
engine = st.selectbox("Clause classification engine", engine_options)
if engine.startswith("Checkpoint:"):
    selected_run_id = engine.split(":", 1)[1].strip()
    selected_run = next(run for run in completed if run["run_id"] == selected_run_id)
    release_gate = selected_run.get("summary", {}).get("release_gate", {})
    if release_gate.get("passed") is not True:
        st.warning(
            "This checkpoint has not passed the current overfitting release gate. Use it for "
            "diagnosis or comparison, not as a release-ready model.",
            icon=":material/warning:",
        )

if st.button(
    "Analyze contract",
    type="primary",
    icon=":material/find_in_page:",
    disabled=not contract_text.strip(),
):
    predictor = None
    if engine.startswith("Checkpoint:"):
        run_id = engine.split(":", 1)[1].strip()
        checkpoint = (
            Path(next(run["run_dir"] for run in completed if run["run_id"] == run_id)) / "best"
        )
        with st.spinner("Loading the checkpoint…"):
            predictor = ClausePredictor(checkpoint)
    with st.spinner("Segmenting and analyzing locally…"):
        st.session_state.analysis_result = analyze_contract(contract_text, predictor)
        st.session_state.analysis_contract_text = contract_text

result = st.session_state.get("analysis_result")
if result:
    findings = result["findings"]
    st.success(f"Analyzed {len(findings)} passages with the {result['engine']}.")

    contract_review = result["contract_analysis"]
    st.subheader("Contract profile and common-clause review")
    with st.container(horizontal=True):
        st.metric("Detected contract type", contract_review["profile"])
        st.metric(
            "Common clauses found",
            f"{contract_review['detected_count']}/{contract_review['expected_count']}",
        )
        st.metric("Domain", str(contract_review["domain"]).replace("_", " ").title())
    st.caption(contract_review["definition"])
    with st.expander("Inspect clause definitions and supporting evidence", expanded=True):
        review_rows = pd.DataFrame(contract_review["clauses"])
        review_rows["evidence"] = review_rows["evidence_ids"].apply(
            lambda values: ", ".join(values)
        )
        st.dataframe(
            review_rows[["status", "name", "definition", "evidence"]],
            hide_index=True,
        )
        st.caption(contract_review["disclaimer"])

    st.subheader("1. Identified and categorized clauses")
    finding_frame = pd.DataFrame(findings)
    st.dataframe(
        finding_frame[["passage_id", "label", "confidence", "text"]],
        hide_index=True,
        column_config={
            "confidence": st.column_config.ProgressColumn(
                "confidence", min_value=0.0, max_value=1.0, format="%.2f"
            )
        },
    )

    st.subheader("2. Extracted facts and obligations")
    facts = result["facts"]
    with st.container(horizontal=True):
        with st.container(border=True):
            st.markdown("**Dates**")
            st.write(facts["dates"] or "None detected")
        with st.container(border=True):
            st.markdown("**Amounts**")
            st.write(facts["money"] or "None detected")
        with st.container(border=True):
            st.markdown("**Party phrases**")
            st.write(facts["party_phrases"] or "None detected")
    st.markdown("**Obligation sentences**")
    for obligation in facts["obligations"]:
        st.write(f"- {obligation}")

    st.subheader("3. Plain-language explanations")
    for finding in findings:
        if finding["label"] != "Other":
            with st.expander(f"{finding['passage_id']} · {finding['label']}"):
                st.write(finding["explanation"])
                st.markdown(f"> {finding['text']}")

    st.subheader("4. Common clauses not clearly detected")
    st.caption(
        "The comparison set comes from the detected contract type in the project taxonomy. "
        "Not detected does not mean legally required, actually absent, or deficient."
    )
    undetected = [clause for clause in contract_review["clauses"] if clause["status"] != "Detected"]
    if undetected:
        st.write([clause["name"] for clause in undetected])
    else:
        st.write("Every configured common category had some matching evidence.")

    st.subheader("5. Human-review questions")
    for clause in contract_review["clauses"]:
        observed = clause["status"] == "Detected"
        with st.container(border=True):
            st.markdown(
                f"**{clause['name']} - {'detected' if observed else 'not clearly detected'}**"
            )
            st.write(clause["definition"])
            for question in clause.get("review_questions", []):
                st.write(f"- {question}")

    st.subheader("6. Chat with the contract analysis")
    orchestrator = cached_legal_ai_orchestrator()
    st.caption(
        "Ask about the contract type, detected or potentially missing clauses, obligations, "
        "review questions, a summary, a cited passage, or ask for a complete contract draft. "
        "Complete drafts can be downloaded as Word and text files directly from the chat. The "
        "configured local generative model uses the current analysis as grounded context."
    )
    st.badge(
        f"Generative model: {orchestrator.settings.model}",
        icon=":material/smart_toy:",
        color="blue",
    )
    st.caption("The first response after model startup may take longer while Ollama loads it.")
    chat_mode_label = st.segmented_control(
        "Chat response mode",
        ["Auto", "Review and explain", "Draft complete contract"],
        default="Auto",
        key="contract_chat_mode",
        help=(
            "Auto detects full-document drafting requests. Choose Draft complete contract to "
            "always return the complete revised contract with Word and text downloads."
        ),
    )
    chat_mode = {
        "Auto": "auto",
        "Review and explain": "review",
        "Draft complete contract": "draft",
    }[chat_mode_label or "Auto"]
    context_text = st.session_state.get("analysis_contract_text", "") + result["engine"]
    context_id = hashlib.sha256(context_text.encode("utf-8")).hexdigest()[:16]
    if st.session_state.get("contract_chat_context_id") != context_id:
        st.session_state.contract_chat_context_id = context_id
        st.session_state.contract_chat_messages = [
            {
                "role": "assistant",
                "content": (
                    f"I analyzed this as **{result['contract_profile']}**. Ask what type of "
                    "contract this appears to be, which clauses were detected or not clearly "
                    "detected, what obligations appear, what merits human review, or ask a "
                    "question that I can answer with passage citations."
                ),
            }
        ]

    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Clear contract chat", icon=":material/delete_sweep:"):
            st.session_state.contract_chat_messages = st.session_state.contract_chat_messages[:1]
            st.rerun()

    for message in st.session_state.contract_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("source_label"):
                st.caption(f"Response source: {message['source_label']}")
            if message.get("artifacts"):
                with st.container(horizontal=True):
                    for artifact_index, artifact in enumerate(message["artifacts"]):
                        st.download_button(
                            f"Download {Path(artifact['filename']).suffix.upper()[1:]}",
                            data=artifact["data"],
                            file_name=artifact["filename"],
                            mime=artifact["media_type"],
                            icon=":material/download:",
                            key=(
                                f"contract_artifact_{message.get('artifact_id', 'saved')}_"
                                f"{artifact_index}"
                            ),
                        )

    suggestions = {
        ":blue[:material/category:] Identify contract type": (
            "What type of contract does this appear to be?"
        ),
        ":orange[:material/checklist:] Show detected clauses": (
            "Which clauses were detected, and what do they mean?"
        ),
        ":red[:material/search_check:] What may need review?": (
            "Which common clauses are not clearly detected, and what should a human review?"
        ),
        ":green[:material/assignment:] Summarize obligations": (
            "Summarize the obligations stated in this contract with passage citations."
        ),
        ":violet[:material/contract_edit:] Draft a complete agreement": (
            "Draft a complete revised agreement based on this contract. Preserve supplied terms, "
            "use placeholders for missing business terms, and flag counsel decisions."
        ),
    }
    pending_prompt = st.session_state.pop("contract_pending_prompt", None)
    if not pending_prompt and not any(
        message["role"] == "user" for message in st.session_state.contract_chat_messages
    ):
        suggestion = st.pills(
            "Try asking",
            list(suggestions),
            label_visibility="collapsed",
            key="contract_chat_suggestion",
        )
        if suggestion:
            st.session_state.contract_pending_prompt = suggestions[suggestion]
            st.rerun()

    typed_prompt = st.chat_input(
        "Ask a question or request a complete contract draft",
        key="contract_chat_input",
        submit_mode="disable",
    )
    prompt = pending_prompt or typed_prompt
    if prompt:
        history = normalize_chat_history(st.session_state.contract_chat_messages)
        st.session_state.contract_chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.status(
                f"Generating with {orchestrator.settings.model}...", expanded=False
            ) as status:
                reply = answer_contract_question(
                    orchestrator,
                    contract_text=st.session_state.get("analysis_contract_text", ""),
                    analysis_result=result,
                    question=prompt,
                    history=history,
                    fallback=contract_chat_response,
                    draft_mode=chat_mode,
                )
                if reply.used_local_fallback:
                    fallback_label = (
                        "Structured draft failed; complete local draft and downloads created"
                        if reply.artifacts
                        else "Generative model unavailable; local fallback used"
                    )
                    status.update(label=fallback_label, state="error")
                else:
                    status.update(label="Response ready", state="complete")
            st.markdown(reply.content)
            st.caption(f"Response source: {reply.source_label}")
            if reply.artifacts:
                st.success(
                    "Draft artifacts are ready. Review all placeholders and jurisdiction-specific "
                    "terms before signature."
                )
                with st.container(horizontal=True):
                    for artifact_index, artifact in enumerate(reply.artifacts):
                        st.download_button(
                            f"Download {Path(artifact.filename).suffix.upper()[1:]}",
                            data=artifact.data,
                            file_name=artifact.filename,
                            mime=artifact.media_type,
                            icon=":material/download:",
                            key=f"new_contract_artifact_{len(st.session_state.contract_chat_messages)}_{artifact_index}",
                        )
            if reply.used_local_fallback:
                fallback_message = (
                    "Ollama did not return a valid structured contract, so the source-preserving "
                    "local drafter created a complete editable draft and both downloads. Review "
                    "all placeholders and added clauses."
                    if reply.artifacts
                    else "Ollama could not complete this response, so the grounded deterministic "
                    "chat answered instead."
                )
                st.warning(fallback_message, icon=":material/offline_bolt:")
        st.session_state.contract_chat_messages.append(
            {
                "role": "assistant",
                "content": reply.content,
                "source_label": reply.source_label,
                "used_local_fallback": reply.used_local_fallback,
                "artifact_id": hashlib.sha256(
                    (reply.content + reply.source_label).encode("utf-8")
                ).hexdigest()[:12],
                "artifacts": [
                    {
                        "filename": artifact.filename,
                        "media_type": artifact.media_type,
                        "data": artifact.data,
                    }
                    for artifact in reply.artifacts
                ],
            }
        )
