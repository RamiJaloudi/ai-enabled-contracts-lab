import pandas as pd
import streamlit as st

from contractlab.modeling import MODEL_PRESETS, device_summary, estimate_training_memory_gb
from contractlab.ui import page_intro

page_intro(
    "Model lab",
    "Choose an openly licensed encoder and understand what full fine-tuning stores in GPU memory.",
)

device = device_summary()
with st.container(border=True):
    if device.get("cuda_available"):
        st.success(f"CUDA ready: {device['gpu']} with {device['vram_gb']} GB VRAM")
        st.write(
            f"PyTorch CUDA: {device.get('cuda_version')} · BF16: {device.get('bf16_supported')}"
        )
    else:
        st.warning(
            "This environment sees CPU only. Training will work but will be substantially slower."
        )

st.subheader("Open model presets")
st.dataframe(
    pd.DataFrame(
        [
            {
                "preset": preset.name,
                "model ID": preset.model_id,
                "declared license": preset.license,
                "parameters (approx. M)": preset.approximate_parameters_millions,
                "level": preset.level,
                "notes": preset.notes,
            }
            for preset in MODEL_PRESETS
        ]
    ),
    hide_index=True,
)

selected_name = st.selectbox("Model preset", [preset.name for preset in MODEL_PRESETS])
preset = next(item for item in MODEL_PRESETS if item.name == selected_name)
batch_size = st.slider("Device batch size", 1, 32, 8)
max_length = st.select_slider("Maximum token length", options=[64, 128, 256, 384, 512], value=256)
precision = st.segmented_control(
    "Precision",
    ["bf16", "fp16", "fp32"],
    default="bf16" if device.get("bf16_supported") else "fp16",
)
estimate = estimate_training_memory_gb(
    preset.approximate_parameters_millions, batch_size, max_length, precision
)

with st.container(horizontal=True):
    st.metric("Estimated training memory", f"~{estimate} GB")
    st.metric("Available VRAM", f"{device.get('vram_gb', 0)} GB")
    st.metric("Parameters", f"~{preset.approximate_parameters_millions}M")

st.caption(
    "This is a conservative teaching estimate. Actual allocation depends on sequence lengths, kernels, optimizer state, and library versions."
)
if device.get("vram_gb", 0) and estimate > float(device["vram_gb"]) * 0.9:
    st.warning("The estimate is close to available VRAM. Reduce batch size or sequence length.")
else:
    st.success(
        "The selected configuration should have reasonable memory headroom on the detected GPU."
    )

st.subheader("What gets fine-tuned")
st.markdown(
    "The pretrained encoder begins with language representations learned from a general corpus. "
    "A new classification head begins randomly initialized. In full fine-tuning, gradients update "
    "both the head and encoder. Freezing the encoder is cheaper, but it is feature extraction rather "
    "than full adaptation."
)
st.latex(r"z = W h_{[CLS]} + b \qquad p(y=k\mid x)=softmax(z)_k")

st.subheader("CUAD's published span-extraction checkpoints")
st.info(
    "The original CUAD project published RoBERTa-base, RoBERTa-large, and "
    "DeBERTa-v2-xlarge checkpoints. They solve extractive question answering "
    "(selecting answer spans from contract context), while Contra solves "
    "passage classification. They are complementary model heads, not drop-in replacements.",
    icon=":material/compare_arrows:",
)
st.markdown(
    """
**Decision for this 12 GB workstation**

- Keep the current modern Hugging Face Transformers dependency; it is already installed and
  used by the training and inference code.
- Do not downgrade to CUAD's 2021 Python 3.8 / PyTorch 1.7 / Transformers 4.3-4.4 environment.
  Those versions describe the authors' reproduction environment, not a current requirement.
- Treat the roughly 100M-parameter RoBERTa-base checkpoint as a future span-extraction
  experiment. The 300M and 900M variants are poor first fine-tuning choices for 12 GB VRAM.
- Keep span extraction separate from Contra's classifier metrics and label mapping.
"""
)
st.caption(
    "Official source: github.com/The-Atticus-Project/cuad and Zenodo record 4599830. "
    "Review checkpoint terms and security before downloading; the project does not "
    "silently import third-party model archives."
)
