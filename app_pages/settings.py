import streamlit as st

from contractlab.palettes import PALETTES, palette_swatch_html, save_palette_preference
from contractlab.ui import initialize_state, page_intro

initialize_state()

page_intro(
    "Settings",
    "Choose a visual palette for this session and remember it for future local launches.",
)

st.subheader("Color palette")
st.caption(
    "Palettes apply across the workbench. Midnight is a full dark mode; Tropical uses "
    "bright teal, coral, mango, and violet accents."
)


def update_palette() -> None:
    selected = st.session_state.settings_palette
    st.session_state.palette = selected
    save_palette_preference(selected)


selected_palette = st.selectbox(
    "Application palette",
    list(PALETTES),
    index=list(PALETTES).index(st.session_state.palette),
    format_func=lambda key: PALETTES[key].name,
    key="settings_palette",
    on_change=update_palette,
)

palette = PALETTES[selected_palette]
with st.container(border=True):
    st.markdown(f"### {palette.name}")
    st.write(palette.description)
    st.html(palette_swatch_html(selected_palette))
    st.badge(
        "Dark mode" if palette.mode == "dark" else "Light mode",
        icon=":material/dark_mode:" if palette.mode == "dark" else ":material/light_mode:",
        color="blue" if palette.mode == "dark" else "green",
    )

st.subheader("Preview")
with st.container(border=True):
    with st.container(horizontal=True):
        st.button("Primary action", type="primary", icon=":material/play_arrow:")
        st.button("Secondary action", icon=":material/tune:")
        st.badge("Ready", icon=":material/check:", color="green")
        st.badge("Review", icon=":material/warning:", color="orange")
    st.info(
        "This callout previews text, borders, links, and semantic colors.",
        icon=":material/palette:",
    )
    st.progress(0.72, text="Example training progress")

st.caption(
    "The preference is stored locally under the ignored artifacts directory. "
    "It contains only the selected palette name."
)
