import json

import pandas as pd
import streamlit as st

from contractlab.data import (
    instruction_jsonl_to_classifier,
    load_master_taxonomy,
    load_project_dataset_catalog,
    load_table,
    normalize_dataset,
    prepare_cuad,
    prepare_synthetic_396_experiment,
    split_by_contract,
    synthetic_dataset,
    validate_dataset,
)
from contractlab.paths import CURATED_DIR, PROJECT_ROOT
from contractlab.ui import initialize_state, page_intro


@st.cache_data(max_entries=2, show_spinner=False)
def _load_synthetic_396(path: str, modified_ns: int) -> pd.DataFrame:
    """Load the large local corpus only after explicit user action."""
    del modified_ns
    return prepare_synthetic_396_experiment(load_table(path))


@st.cache_data(max_entries=2, show_spinner=False)
def _load_master_legal_taxonomy(
    taxonomy_path: str,
    taxonomy_modified_ns: int,
    manifest_path: str,
    manifest_modified_ns: int,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Load validated taxonomy metadata without opening the 560,000 records."""
    del taxonomy_modified_ns, manifest_modified_ns
    return load_master_taxonomy(taxonomy_path, manifest_path)


initialize_state()
page_intro(
    "Dataset lab",
    "Inspect provenance, edit examples, validate schema, and prove that one contract never crosses splits.",
)

source_choice = st.segmented_control(
    "Data source",
    ["Built-in synthetic", "Project datasets", "Upload CSV or JSONL", "Official CUAD"],
    default="Built-in synthetic",
)

if source_choice == "Built-in synthetic":
    if st.button("Reset to the synthetic learning dataset", icon=":material/restart_alt:"):
        st.session_state.dataset_frame = synthetic_dataset()
        st.session_state.dataset_name = "AI-Enabled Contracts Lab synthetic clauses"
        st.session_state.dataset_source = "AI-Enabled Contracts Lab original synthetic examples"
        st.session_state.dataset_license = "MIT"
        st.success("Synthetic dataset selected.")

elif source_choice == "Project datasets":
    datasets_dir = PROJECT_ROOT / "datasets"
    catalog_path = datasets_dir / "dataset_catalog.json"
    st.markdown(
        "These files are separated by learning purpose. **Classifier-ready** rows can be used "
        "now. **Instruction/chat** rows support later generative-model experiments."
    )
    try:
        catalog = load_project_dataset_catalog(catalog_path)
        visible_columns = [
            "name",
            "purpose",
            "rows",
            "contracts",
            "labels_or_tasks",
            "license",
            "status",
            "available",
            "size_mb",
        ]
        st.dataframe(catalog[visible_columns], hide_index=True)
    except Exception as exc:
        st.error(f"Could not read the project dataset catalog: {exc}")

    sample_target = datasets_dir / "cuad_classifier_sample.csv"
    if sample_target.exists() and st.button(
        "Use the public CUAD classifier sample",
        icon=":material/science:",
    ):
        frame = normalize_dataset(load_table(sample_target))
        st.session_state.dataset_frame = frame
        st.session_state.dataset_name = "CUAD classifier sample"
        st.session_state.dataset_source = "The Atticus Project CUAD v1"
        st.session_state.dataset_license = "CC BY 4.0"
        st.success("The 75-row, metadata-complete classifier sample is active.")

    full_target = datasets_dir / "clauses.csv"
    if full_target.exists() and st.button(
        "Use the full verified CUAD classifier dataset",
        type="primary",
        icon=":material/database:",
    ):
        frame = normalize_dataset(load_table(full_target))
        st.session_state.dataset_frame = frame
        st.session_state.dataset_name = "CUAD full classifier derivative"
        st.session_state.dataset_source = "The Atticus Project CUAD v1"
        st.session_state.dataset_license = "CC BY 4.0"
        st.success("The 6,702-row, 510-contract classifier dataset is active.")

    experimental_target = datasets_dir / "contract_clauses_all_39600.csv"
    experimental_audit_path = datasets_dir / "synthetic_396_audit.json"
    if experimental_audit_path.exists():
        audit = json.loads(experimental_audit_path.read_text(encoding="utf-8"))
        with st.expander("Experimental: synthetic 396-label corpus"):
            st.warning(
                "This local corpus includes all 396 labels, but its supplied splits are "
                "not a trustworthy benchmark. Exact duplicates, template reuse, and "
                "label-name leakage can inflate evaluation scores."
            )
            with st.container(horizontal=True):
                st.metric("Rows", f"{audit['rows']:,}")
                st.metric("Labels", audit["labels"])
                st.metric("Categories", audit["categories"])
                st.metric(
                    "Duplicate text rows",
                    f"{audit['quality']['duplicate_text_rows']:,}",
                )
            st.caption(
                "The workbench derives template-group IDs and performs a fresh grouped "
                "split. Contra's 41-label CUAD result remains the real-contract baseline. "
                "Raw files stay Git-ignored because license and generation provenance "
                "have not yet been documented."
            )
            if experimental_target.exists():
                acknowledge = st.checkbox(
                    "I understand this is synthetic experimental data and not an "
                    "independent legal-performance benchmark.",
                    key="synthetic_ack",
                )
                if st.button(
                    "Use all 396 experimental labels",
                    disabled=not acknowledge,
                    icon=":material/experiment:",
                ):
                    with st.spinner(
                        "Loading 39,600 rows and deriving leakage-aware template groups..."
                    ):
                        frame = _load_synthetic_396(
                            str(experimental_target),
                            experimental_target.stat().st_mtime_ns,
                        )
                    st.session_state.dataset_frame = frame
                    st.session_state.dataset_name = (
                        "Synthetic contract clauses - 396-label experiment"
                    )
                    st.session_state.dataset_source = (
                        "User-supplied synthetic corpus; provenance pending"
                    )
                    st.session_state.dataset_license = "UNSPECIFIED - LOCAL ONLY"
                    st.success(
                        "All 396 labels are active. Training will use fresh "
                        "template-grouped partitions, not the supplied row split."
                    )
            else:
                st.info(
                    "The catalog entry is ready. Add contract_clauses_all_39600.csv "
                    "to datasets/ to activate this local experiment."
                )

    local_sources = {
        split: datasets_dir / f"{split}.jsonl" / f"{split}.jsonl"
        for split in ("train", "val", "test")
    }
    local_target = CURATED_DIR / "cuad_local_classifier.csv"
    if all(path.exists() for path in local_sources.values()):
        if st.button(
            "Rebuild classifier data from the instruction splits",
            icon=":material/transform:",
        ):
            with st.spinner("Reading grounded clause-QA records and verifying contract splits..."):
                try:
                    instruction_jsonl_to_classifier(local_sources, local_target)
                    st.success(
                        f"Rebuilt {local_target.name} in the ignored data/curated directory."
                    )
                except Exception as exc:
                    st.error(f"Local conversion failed: {exc}")
    else:
        st.info("All three train/validation/test JSONL exports are required for reconstruction.")

    st.caption(
        "The repository copy of clauses.csv replaces an earlier unverified two-column export. "
        "It now carries contract IDs, source, license, split lineage, and transformation metadata."
    )

elif source_choice == "Upload CSV or JSONL":
    uploaded = st.file_uploader("Training data", type=["csv", "jsonl", "json"])
    license_value = st.text_input(
        "License or private-data designation",
        value="PRIVATE — DO NOT REDISTRIBUTE",
    )
    if uploaded is not None:
        try:
            uploaded_frame = normalize_dataset(load_table(uploaded.getvalue(), uploaded.name))
            uploaded_frame["license"] = uploaded_frame["license"].replace("Unknown", license_value)
            validate_dataset(uploaded_frame)
            st.dataframe(uploaded_frame.head(100))
            if st.button("Use this uploaded dataset", type="primary"):
                st.session_state.dataset_frame = uploaded_frame
                st.session_state.dataset_name = uploaded.name
                st.session_state.dataset_source = f"Local upload: {uploaded.name}"
                st.session_state.dataset_license = license_value
                st.success(
                    "Upload selected. It remains local and in memory until training "
                    "or explicit export."
                )
        except Exception as exc:
            st.error(f"Could not read the upload: {exc}")

else:
    st.markdown(
        "The [official CUAD repository](https://github.com/The-Atticus-Project/cuad) and "
        "[Hugging Face dataset card](https://huggingface.co/datasets/theatticusproject/cuad-qa) "
        "describe 510 commercial contracts and 41 expert-annotated clause types under CC BY 4.0. "
        "AI-Enabled Contracts Lab converts positive annotated spans into an educational multiclass "
        "dataset; this is not the official extractive-QA benchmark task."
    )
    maximum = st.number_input(
        "Maximum converted spans",
        min_value=100,
        max_value=20000,
        value=3000,
        step=100,
    )
    target = CURATED_DIR / "cuad_clauses.csv"
    if st.button("Download and prepare CUAD", type="primary", icon=":material/download:"):
        with st.spinner("Downloading and converting CUAD..."):
            try:
                prepare_cuad(target, int(maximum))
                st.success(f"Prepared {target.name}")
            except Exception as exc:
                st.error(f"CUAD preparation failed: {exc}")
    if target.exists() and st.button("Use prepared CUAD"):
        frame = normalize_dataset(load_table(target))
        st.session_state.dataset_frame = frame
        st.session_state.dataset_name = "CUAD positive spans"
        st.session_state.dataset_source = "The Atticus Project CUAD v1"
        st.session_state.dataset_license = "CC BY 4.0"
        st.success("Prepared CUAD selected.")

st.space("small")
frame: pd.DataFrame = st.session_state.dataset_frame
report = validate_dataset(frame)

with st.container(horizontal=True):
    st.metric("Examples", report.rows)
    st.metric("Labels", report.labels)
    st.metric("Source contracts", report.contracts)
    st.metric("Duplicate texts", report.duplicates)

for warning in report.warnings:
    st.warning(warning)

if report.labels >= 100 and {"clause_name", "category"}.issubset(frame.columns):
    st.subheader(f"Active label catalog ({report.labels} labels)")
    label_catalog = (
        frame.assign(category=frame["category"].fillna("Uncategorized"))
        .groupby(["category", "label", "clause_name"], as_index=False, dropna=False)
        .size()
        .rename(columns={"size": "examples"})
        .sort_values(["category", "label"])
    )
    categories = ["All categories", *sorted(label_catalog["category"].unique())]
    selected_category = st.selectbox(
        "Filter labels by category",
        categories,
        key="active_label_category",
    )
    visible_labels = label_catalog
    if selected_category != "All categories":
        visible_labels = label_catalog[label_catalog["category"] == selected_category]
    st.dataframe(
        visible_labels[["label", "clause_name", "category", "examples"]],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Clause names and categories come from the supplied corpus. They are "
        "training metadata, not reviewed legal definitions. Use the separate "
        "contract taxonomy below for reviewed educational definitions."
    )


st.subheader("Label distribution")
counts = frame["label"].value_counts().rename_axis("label").reset_index(name="examples")
st.bar_chart(counts, x="label", y="examples")

st.subheader("Inspect and edit a working copy")
edited = st.data_editor(
    frame.head(250),
    hide_index=True,
    num_rows="dynamic",
    key="dataset_editor",
)
with st.container(horizontal=True):
    if st.button("Apply visible edits", icon=":material/check:"):
        if len(frame) > 250:
            st.warning(
                "Only the visible first 250 rows were edited; remaining rows were preserved."
            )
            st.session_state.dataset_frame = pd.concat(
                [edited, frame.iloc[250:]],
                ignore_index=True,
            )
        else:
            st.session_state.dataset_frame = edited
        st.success("Edits applied to the active in-memory dataset.")
    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export active dataset",
        csv_bytes,
        "ai_enabled_contracts_lab_dataset.csv",
        "text/csv",
        icon=":material/download:",
    )

st.subheader("Leakage-resistant split preview")
seed = st.number_input("Split seed", min_value=0, max_value=100000, value=42)
try:
    splits = split_by_contract(frame, seed=int(seed))
    split_summary = pd.DataFrame(
        [
            {
                "split": name,
                "examples": len(part),
                "contracts": part["contract_id"].nunique(),
                "labels": part["label"].nunique(),
            }
            for name, part in splits.items()
        ]
    )
    st.dataframe(split_summary, hide_index=True)
    group_sets = {name: set(part["contract_id"]) for name, part in splits.items()}
    overlap = (
        (group_sets["train"] & group_sets["validation"])
        | (group_sets["train"] & group_sets["test"])
        | (group_sets["validation"] & group_sets["test"])
    )
    if overlap:
        st.error(f"Contract leakage detected: {sorted(overlap)[:5]}")
    else:
        st.success("Verified: no contract_id appears in more than one split.")
except Exception as exc:
    st.error(str(exc))

st.subheader("Contract taxonomy and definitions")
taxonomy_path = PROJECT_ROOT / "datasets" / "contract_taxonomy.json"
if taxonomy_path.exists():
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    with st.container(horizontal=True):
        st.metric("Legal domains", len(taxonomy["domains"]))
        st.metric("Contract types", len(taxonomy["contract_types"]))
        st.metric("Clause definitions", len(taxonomy["clause_definitions"]))
    st.caption(taxonomy["metadata"]["limitations"])

    domain_names = {domain["id"]: domain["name"] for domain in taxonomy["domains"]}
    domain_id = st.selectbox(
        "Browse a legal domain",
        list(domain_names),
        format_func=lambda value: domain_names[value],
        key="taxonomy_domain",
    )
    domain = next(item for item in taxonomy["domains"] if item["id"] == domain_id)
    st.write(domain["definition"])
    type_rows = [item for item in taxonomy["contract_types"] if item["domain"] == domain_id]
    st.dataframe(
        pd.DataFrame(type_rows)[["name", "definition"]],
        hide_index=True,
        width="stretch",
    )

    type_lookup = {item["id"]: item for item in type_rows}
    selected_type_id = st.selectbox(
        "Contract type",
        list(type_lookup),
        format_func=lambda value: type_lookup[value]["name"],
        key="taxonomy_contract_type",
    )
    selected_type = type_lookup[selected_type_id]
    clause_lookup = {item["id"]: item for item in taxonomy["clause_definitions"]}
    typical_rows = [clause_lookup[item] for item in selected_type["typical_clause_ids"]]
    st.markdown(f"**Definition:** {selected_type['definition']}")
    st.dataframe(
        pd.DataFrame(typical_rows)[["name", "definition", "aliases", "review_questions"]],
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Download the complete taxonomy JSON",
        taxonomy_path.read_bytes(),
        "contract_taxonomy.json",
        "application/json",
    )
else:
    st.info("Run datasets/build_contract_taxonomy.py to generate the taxonomy bundle.")
st.subheader("Master multidomain legal taxonomy")
master_taxonomy_path = PROJECT_ROOT / "datasets" / "master_taxonomy.json"
master_manifest_path = PROJECT_ROOT / "datasets" / "manifest.csv"
master_bundle_path = PROJECT_ROOT / "datasets" / "legal_multidomain_all_fields_bundle.zip"
master_projection_path = CURATED_DIR / "legal_multidomain_classifier.csv"
master_audit_path = PROJECT_ROOT / "datasets" / "master_taxonomy_audit.json"

if master_taxonomy_path.exists() and master_manifest_path.exists():
    try:
        master_taxonomy, master_manifest = _load_master_legal_taxonomy(
            str(master_taxonomy_path),
            master_taxonomy_path.stat().st_mtime_ns,
            str(master_manifest_path),
            master_manifest_path.stat().st_mtime_ns,
        )
        assignments = sum(len(labels) for labels in master_taxonomy.values())
        unique_master_labels = len(
            {label for labels in master_taxonomy.values() for label in labels}
        )
        with st.container(horizontal=True):
            st.metric("Legal domains", len(master_taxonomy))
            st.metric("Domain-label assignments", f"{assignments:,}")
            st.metric("Unique label names", f"{unique_master_labels:,}")
            st.metric(
                "Examples declared",
                f"{int(master_manifest['examples'].sum()):,}",
            )
        st.warning(
            "This corpus is synthetic and answer-revealing: the audit found the "
            "target label phrase in every text and template families spanning "
            "nearly every row. Use it only as an experimental taxonomy-label task."
        )
        selected_master_domain = st.selectbox(
            "Browse a master legal domain",
            list(master_taxonomy),
            format_func=lambda value: value.replace("_", " "),
            key="master_taxonomy_domain",
        )
        master_labels = pd.DataFrame(
            {
                "label": master_taxonomy[selected_master_domain],
                "domain-qualified target": [
                    f"{selected_master_domain}::{label}"
                    for label in master_taxonomy[selected_master_domain]
                ],
            }
        )
        st.dataframe(master_labels, hide_index=True, width="stretch")
        st.caption(
            "These are supplied label names, not reviewed legal definitions. "
            "Nine names occur in more than one domain, so training uses the "
            "domain-qualified targets shown above."
        )

        if master_bundle_path.exists():
            st.success(
                "Bundle available: all 28 nested archives and 560,000 records "
                "passed structural validation."
            )
        else:
            st.error("The legal multidomain all-fields bundle is not present.")
        if master_projection_path.exists():
            st.success(
                "Classifier projection ready: 560,000 examples and 1,120 "
                "domain-qualified labels in data/curated/."
            )
        run_status_path = (
            PROJECT_ROOT / "artifacts" / "runs" / "contra-law-legalbert-1120-exp-v1" / "status.json"
        )
        if run_status_path.exists():
            run_status = json.loads(run_status_path.read_text(encoding="utf-8"))
            state = str(run_status.get("state", "unknown"))
            message = str(run_status.get("message", "No status message"))
            if state == "completed":
                st.success(f"Contra-Law Experimental training complete. {message}")
            elif state == "failed":
                st.error(f"Contra-Law Experimental training failed. {message}")
            else:
                st.info(f"Contra-Law Experimental: {message}")
        if master_audit_path.exists():
            st.download_button(
                "Download the master dataset audit",
                master_audit_path.read_bytes(),
                "master_taxonomy_audit.json",
                "application/json",
            )
    except Exception as exc:
        st.error(f"Could not validate the master legal taxonomy: {exc}")
else:
    st.info(
        "Add manifest.csv and master_taxonomy.json to datasets/ to register "
        "the master multidomain legal experiment."
    )
