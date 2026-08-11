import streamlit as st

from contractlab.ui import initialize_state, sidebar_context

st.set_page_config(
    page_title="AI-Enabled Contracts Lab",
    page_icon=":material/model_training:",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_state()
sidebar_context()

pages = {
    "Start": [
        st.Page("app_pages/home.py", title="Home", icon=":material/home:"),
        st.Page("app_pages/learn.py", title="Guided course", icon=":material/school:"),
    ],
    "Build an experiment": [
        st.Page("app_pages/dataset_lab.py", title="Dataset lab", icon=":material/table_view:"),
        st.Page("app_pages/model_lab.py", title="Model lab", icon=":material/memory:"),
        st.Page("app_pages/train.py", title="Train", icon=":material/model_training:"),
        st.Page("app_pages/evaluate.py", title="Evaluate", icon=":material/analytics:"),
    ],
    "Use and understand": [
        st.Page("app_pages/contract_chat.py", title="Contract chat", icon=":material/chat:"),
        st.Page(
            "app_pages/assistant.py", title="Contract assistant", icon=":material/find_in_page:"
        ),
        st.Page("app_pages/experiments.py", title="Experiments", icon=":material/experiment:"),
        st.Page("app_pages/code_lab.py", title="Code lab", icon=":material/code:"),
        st.Page("app_pages/settings.py", title="Settings", icon=":material/settings:"),
    ],
}

navigation = st.navigation(pages, position="sidebar")
navigation.run()
