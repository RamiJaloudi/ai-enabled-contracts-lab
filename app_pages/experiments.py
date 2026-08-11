import pandas as pd
import streamlit as st

from contractlab.jobs import list_runs
from contractlab.ui import page_intro

page_intro(
    "Experiments",
    "Compare local configurations and results. One experiment should test one written hypothesis.",
)

runs = list_runs()
if not runs:
    st.info("No experiment artifacts exist yet. Launch a run from the Train page.")
    st.stop()

records = []
for run in runs:
    config = run.get("launch", {})
    summary = run.get("summary", {})
    metrics = summary.get("test_metrics", {})
    records.append(
        {
            "run": run["run_id"],
            "state": run.get("state"),
            "model": summary.get("model_name", "—"),
            "test macro F1": metrics.get("macro_f1"),
            "test accuracy": metrics.get("accuracy"),
            "W&B": summary.get("wandb_url"),
        }
    )
st.dataframe(pd.DataFrame(records), hide_index=True)

selected_id = st.selectbox("Inspect run", [run["run_id"] for run in runs])
run = next(item for item in runs if item["run_id"] == selected_id)
with st.container(horizontal=True):
    with st.container(border=True):
        st.markdown("**Status**")
        st.write(run.get("state", "unknown"))
        st.write(run.get("status", {}).get("message", "No status message"))
    with st.container(border=True):
        st.markdown("**Local artifact directory**")
        st.code(run["run_dir"], language="text")
if run.get("summary", {}).get("wandb_url"):
    st.link_button("Open W&B run", run["summary"]["wandb_url"], icon=":material/open_in_new:")
with st.expander("Run metadata"):
    st.json(run)
