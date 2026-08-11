from __future__ import annotations

import streamlit as st

from contractlab.paths import PROJECT_ROOT
from contractlab.ui import initialize_state, page_intro

initialize_state()
page_intro(
    "AI-Enabled Contracts Lab",
    "A local-first legal AI workbench combining fine-tuned BERT routing with a generative LLM for structured analysis, drafting, and evaluation.",
)

st.warning(
    "Research and drafting support only. Outputs require qualified human review and jurisdiction-specific verification before legal use.",
    icon=":material/gavel:",
)

with st.container(horizontal=True):
    for label in ["Local-first", "No RAG", "Structured outputs", "Human review required"]:
        st.badge(label)

st.subheader("A modern hybrid legal-AI architecture")
st.markdown(
    "The system assigns different jobs to different models. The encoder produces auditable "
    "routing hypotheses; the generative model explains, analyzes, and drafts within explicit "
    "legal-quality constraints."
)
st.graphviz_chart(
    """
    digraph LegalAI {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fillcolor="#EAF2F8", color="#2E74B5"];
        input [label="Facts or contract"];
        prep [label="Preprocess and segment"];
        domain [label="BERT domain classifier"];
        doctrine [label="BERT doctrine top-k"];
        issues [label="Independent issue spotting"];
        jurisdiction [label="Jurisdiction router"];
        skills [label="Prompts and legal skills"];
        llm [label="Local generative LLM"];
        output [label="Structured analysis or draft"];
        review [label="Evaluation and human review"];
        input -> prep -> domain -> doctrine -> issues -> jurisdiction -> skills -> llm -> output -> review;
    }
    """,
    width="stretch",
)

encoder_col, generator_col = st.columns(2)
with encoder_col:
    st.markdown("#### Encoder layer")
    st.markdown(
        "- Fine-tuned Contracts-BERT / DistilBERT classifiers\n"
        "- Top-k labels, confidence, model version, and taxonomy version\n"
        "- Classification is a hypothesis, not a legal conclusion"
    )
with generator_col:
    st.markdown("#### Generative layer")
    st.markdown(
        "- Issue explanation, fact-to-element analysis, and counterarguments\n"
        "- Contract review, drafting, and DOCX/TXT export from chat\n"
        "- No invented authority; jurisdiction-specific law is flagged for verification"
    )

st.subheader("Production-oriented safeguards")
first_row = st.columns(3)
first_row[0].markdown("**Evidence boundaries**\n\nFacts, assumptions, missing information, and conclusions remain separate.")
first_row[1].markdown("**Jurisdiction routing**\n\nFederal, state, New Jersey, administrative, and international questions are distinguished.")
first_row[2].markdown("**Typed outputs**\n\nPydantic schemas keep analysis predictable and testable.")
second_row = st.columns(3)
second_row[0].markdown("**Held-out evaluation**\n\nMacro-F1 and per-label results expose weak classes instead of hiding them in accuracy.")
second_row[1].markdown("**Private contract workspace**\n\nContract Chat supports optional attachments and locally saved conversations.")
second_row[2].markdown("**Replaceable components**\n\nClassifier, prompts, LLM provider, and future retrieval can evolve independently.")

st.subheader("Start here")
start_cols = st.columns(3)
if start_cols[0].button("Open Contract Chat", icon=":material/chat:", width="stretch"):
    st.switch_page("app_pages/contract_chat.py")
if start_cols[1].button("Open Evaluate", icon=":material/analytics:", width="stretch"):
    st.switch_page("app_pages/evaluate.py")
if start_cols[2].button("Open Code Lab", icon=":material/code:", width="stretch"):
    st.switch_page("app_pages/code_lab.py")

st.subheader("Recommended learning and release path")
st.markdown(
    "1. Inspect the dataset and taxonomy in **Dataset Lab**.\n"
    "2. Train or load the encoder in **Train** and inspect predictions in **Model Lab**.\n"
    "3. Compare development and untouched holdout results in **Evaluate**.\n"
    "4. Test multi-turn analysis, attachments, and document export in **Contract Chat**.\n"
    "5. Generate reproducible experiment artifacts in **Experiments**.\n"
    "6. Review the private GitHub and Hugging Face release workflows in **Code Lab**."
)

st.subheader("Documentation and release destinations")
st.markdown(
    "- [Project README](https://github.com/RamiJaloudi/ai-enabled-contracts-lab#readme)\n"
    "- [Private-first GitHub repository](https://github.com/RamiJaloudi/ai-enabled-contracts-lab)\n"
    "- [Contra v2 model repository](https://huggingface.co/rjaloudi/contra-contracts-bert-cuad-clause-classifier)\n"
    "- [Architecture guide](https://github.com/RamiJaloudi/ai-enabled-contracts-lab/blob/main/docs/ARCHITECTURE.md)"
)

guide_path = PROJECT_ROOT / "docs" / "AI_Enabled_Contracts_Lab_User_Guide.docx"
if guide_path.exists():
    st.download_button(
        "Download the Word user guide",
        guide_path.read_bytes(),
        guide_path.name,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        icon=":material/download:",
    )