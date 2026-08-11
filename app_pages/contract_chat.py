from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from contractlab.assistant import analyze_contract, contract_chat_response
from contractlab.contract_chat_store import ContractChatStore
from contractlab.document_parser import ExtractionLimits, extract_documents
from contractlab.legal_ai.chat_service import (
    answer_contract_workspace_question,
    normalize_chat_history,
)
from contractlab.ui import cached_legal_ai_orchestrator, page_intro

SUPPORTED_TYPES = [
    "txt",
    "md",
    "json",
    "jsonl",
    "xml",
    "html",
    "htm",
    "csv",
    "tsv",
    "pdf",
    "docx",
    "rtf",
    "odt",
    "xlsx",
    "xls",
    "eml",
    "msg",
    "zip",
    "png",
    "jpg",
    "jpeg",
    "tif",
    "tiff",
    "bmp",
]


orchestrator = cached_legal_ai_orchestrator()
settings = orchestrator.settings
store = ContractChatStore(settings.contract_chat_storage_path)
limits = ExtractionLimits(
    maximum_file_bytes=settings.maximum_file_bytes,
    maximum_archive_entries=settings.maximum_archive_entries,
    maximum_archive_uncompressed_bytes=settings.maximum_archive_uncompressed_bytes,
    maximum_spreadsheet_cells=settings.maximum_spreadsheet_cells,
    maximum_ocr_pages=settings.maximum_ocr_pages,
    maximum_extracted_characters=settings.maximum_extracted_characters_per_document,
)

page_intro(
    "Contract chat",
    "Ask general contract questions, analyze or compare files, and draft complete editable agreements.",
)
st.warning(
    "Contract chat provides drafting and educational assistance, not legal advice. Verify "
    "jurisdiction-specific law and have qualified counsel review documents before signature.",
    icon=":material/gavel:",
)

conversations = store.list()
conversation = store.try_load(st.session_state.get("contract_workspace_id"))
if conversation is None:
    conversation = conversations[0] if conversations else store.create()
    st.session_state.contract_workspace_id = conversation.id
    st.session_state.pop("contract_workspace_selector", None)
    conversations = store.list()
else:
    st.session_state.contract_workspace_id = conversation.id

with st.sidebar:
    st.divider()
    st.subheader("Saved contract chats")
    if st.button("New contract chat", icon=":material/add_comment:", width="stretch"):
        created = store.create()
        st.session_state.contract_workspace_id = created.id
        st.session_state.pop("contract_workspace_selector", None)
        st.rerun()
    option_ids = [conversation.id for conversation in conversations]
    current_id = st.session_state.contract_workspace_id
    labels = {conversation.id: conversation.title for conversation in conversations}
    selected_id = st.selectbox(
        "Conversation",
        option_ids,
        index=option_ids.index(current_id),
        format_func=lambda value: labels.get(value, "New contract chat"),
        key="contract_workspace_selector",
    )
    if selected_id != current_id:
        st.session_state.contract_workspace_id = selected_id
        st.rerun()
    st.caption(f"Saved locally in {settings.contract_chat_storage_path}")

    st.divider()
    mode_label = st.selectbox(
        "Response mode",
        ["Auto", "Review and analyze", "Draft complete contract"],
        key="contract_workspace_mode",
        help=(
            "Auto detects full-document drafting requests. Draft mode always returns the complete "
            "contract with Word and text downloads."
        ),
    )

conversation = store.try_load(st.session_state.get("contract_workspace_id"))
if conversation is None:
    conversation = store.create()
    st.session_state.contract_workspace_id = conversation.id
    st.session_state.pop("contract_workspace_selector", None)
    st.rerun()
draft_mode = {
    "Auto": "auto",
    "Review and analyze": "review",
    "Draft complete contract": "draft",
}[mode_label or "Auto"]

st.caption(
    "Type a message below. Use the **+** in the chat field only when you want to attach files; "
    "attachments are never required."
)
st.badge(
    f"Generative model: {settings.model}",
    icon=":material/smart_toy:",
    color="blue",
)

with st.container(border=True):
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button(
            "Clear chat",
            icon=":material/delete_sweep:",
            key=f"clear_contract_chat_{conversation.id}",
            help="Clear this conversation's messages and keep its Local workspace documents.",
        ):
            store.clear_messages(conversation)
            st.rerun()
    chat_log = st.container(height=480, autoscroll=True)
    with chat_log:
        if not conversation.messages:
            with st.chat_message("assistant"):
                st.markdown(
                    "Ask me anything about contracts. Chat normally, or use the **+** in the "
                    "message field when you want to attach one or more files."
                )

        for message in conversation.messages:
            with st.chat_message(message.role):
                st.markdown(message.content)
                if message.source_label:
                    st.caption(f"Response source: {message.source_label}")
                if message.artifacts:
                    with st.container(horizontal=True):
                        for index, artifact in enumerate(message.artifacts):
                            st.download_button(
                                f"Download {Path(artifact.filename).suffix.upper()[1:]}",
                                data=store.artifact_bytes(conversation.id, artifact),
                                file_name=artifact.filename,
                                mime=artifact.media_type,
                                icon=":material/download:",
                                key=f"saved_workspace_artifact_{message.id}_{index}",
                            )

    chat_value = st.chat_input(
        "Message Contract chat or add files",
        key="contract_workspace_input",
        accept_file="multiple",
        file_type=SUPPORTED_TYPES,
        max_upload_size=max(1, settings.maximum_file_bytes // (1024 * 1024)),
        submit_mode="disable",
        height=96,
    )
    st.caption("Attachments are optional. Continue asking follow-up questions in the same chat.")

with st.expander("Local workspace", icon=":material/folder_open:", expanded=False):
    st.caption(
        "Uploaded originals, extracted text, chat messages, and generated Word and text files "
        "are saved locally in this conversation."
    )
    with st.form("contract_workspace_paste", clear_on_submit=True):
        paste_name = st.text_input("Document name", value="Pasted contract.txt")
        pasted_text = st.text_area("Contract text", height=160)
        add_paste = st.form_submit_button(
            "Add pasted text", icon=":material/note_add:", disabled=not pasted_text.strip()
        )
    if add_paste:
        data = pasted_text.encode("utf-8")
        extracted = extract_documents(data, paste_name or "Pasted contract.txt", limits=limits)
        if len(conversation.documents) + len(extracted) > settings.maximum_documents_per_chat:
            st.error("This chat has reached its configured document limit.")
        else:
            added = store.add_upload(
                conversation,
                upload_filename=paste_name or "Pasted contract.txt",
                upload_data=data,
                extracted=extracted,
            )
            st.success(f"Added {len(added)} document{'s' if len(added) != 1 else ''}.")
            conversation = store.load(conversation.id)
    if conversation.documents:
        st.markdown(f"**Attached documents ({len(conversation.documents)})**")
        for document in conversation.documents:
            detail = (
                f"{document.id} Â· {document.filename} Â· {len(document.text):,} characters"
                + (" Â· OCR" if document.ocr_used else "")
            )
            st.write(detail)
            for warning in document.warnings:
                st.caption(f"Warning: {warning}")
    else:
        st.caption("No documents attached. Text-only chat is ready.")
if chat_value:
    prompt = str(
        getattr(chat_value, "text", chat_value if isinstance(chat_value, str) else "")
    ).strip()
    uploaded_files = list(getattr(chat_value, "files", []))
    if len(uploaded_files) > settings.maximum_files_per_message:
        st.error("Too many files were attached to one message.")
        st.stop()

    failures: list[str] = []
    added_count = 0
    for uploaded in uploaded_files:
        try:
            data = uploaded.getvalue()
            extracted = extract_documents(data, uploaded.name, limits=limits)
            if len(conversation.documents) + len(extracted) > settings.maximum_documents_per_chat:
                raise ValueError("This chat has reached its configured document limit.")
            added = store.add_upload(
                conversation,
                upload_filename=uploaded.name,
                upload_data=data,
                extracted=extracted,
            )
            added_count += len(added)
        except Exception as exc:
            failures.append(f"{uploaded.name}: {exc}")
    conversation = store.load(conversation.id)
    if failures:
        st.error("Some attachments could not be processed:\n\n- " + "\n- ".join(failures))
    if uploaded_files and not prompt:
        prompt = (
            "Analyze the attached contract documents. Identify the contract types, key terms, "
            "obligations, risks, missing or unclear provisions, conflicts, negotiation choices, "
            "jurisdiction caveats, and questions for counsel. Compare documents when more than one "
            "is present."
        )
    if not prompt:
        st.stop()
    attachment_names = [uploaded.name for uploaded in uploaded_files]
    display_prompt = prompt
    if attachment_names:
        display_prompt += "\n\nAttached files: " + ", ".join(attachment_names)
        prompt += f"\n\n[Attached files processed in this message: {added_count}]"

    history = normalize_chat_history(
        [message.model_dump(mode="json") for message in conversation.messages],
        limit=settings.maximum_history_messages,
    )
    store.add_message(conversation, role="user", content=display_prompt)
    with chat_log:
        with st.chat_message("user"):
            st.markdown(display_prompt)

    document_payload = [
        {"id": document.id, "filename": document.filename, "text": document.text}
        for document in conversation.documents
        if document.text.strip()
    ]
    combined = "\n\n".join(
        f"[{row['id']} | {row['filename']}]\n{row['text']}" for row in document_payload
    )
    analysis_result = None
    if combined:
        analysis_key = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        cache_key = f"contract_workspace_analysis_{conversation.id}"
        cached = st.session_state.get(cache_key)
        if not cached or cached.get("key") != analysis_key:
            cached = {"key": analysis_key, "result": analyze_contract(combined)}
            st.session_state[cache_key] = cached
        analysis_result = cached["result"]

    with chat_log:
        with st.chat_message("assistant"):
            with st.status(f"Generating with {settings.model}...", expanded=False) as status:
                reply = answer_contract_workspace_question(
                    orchestrator,
                    documents=document_payload,
                    question=prompt,
                    history=history,
                    analysis_result=analysis_result,
                    fallback=contract_chat_response,
                    draft_mode=draft_mode,
                )
                status.update(
                    label="Response ready" if not reply.used_local_fallback else "Local fallback used",
                    state="complete" if not reply.used_local_fallback else "error",
                )
            st.markdown(reply.content)
            st.caption(f"Response source: {reply.source_label}")
            stored_artifacts = [
                store.save_artifact(
                    conversation,
                    filename=artifact.filename,
                    media_type=artifact.media_type,
                    data=artifact.data,
                )
                for artifact in reply.artifacts
            ]
            with st.container(horizontal=True):
                for index, artifact in enumerate(reply.artifacts):
                    st.download_button(
                        f"Download {Path(artifact.filename).suffix.upper()[1:]}",
                        data=artifact.data,
                        file_name=artifact.filename,
                        mime=artifact.media_type,
                        icon=":material/download:",
                        key=f"new_workspace_artifact_{len(conversation.messages)}_{index}",
                    )
            if draft_mode == "draft" or reply.artifacts:
                st.caption("Downloads are also saved locally with this conversation.")
    store.add_message(
        conversation,
        role="assistant",
        content=reply.content,
        source_label=reply.source_label,
        artifacts=stored_artifacts,
    )
