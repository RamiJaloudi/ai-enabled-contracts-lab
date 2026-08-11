from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from contractlab.paths import ARTIFACTS_DIR


@dataclass(frozen=True, slots=True)
class Palette:
    name: str
    description: str
    mode: str
    primary: str
    background: str
    secondary: str
    text: str
    border: str
    sidebar: str
    sidebar_text: str
    colors: tuple[str, ...]


PALETTES: dict[str, Palette] = {
    "legal_blue": Palette(
        "Legal blue",
        "A calm, professional blue palette for focused study.",
        "light",
        "#2457A6",
        "#F8FAFD",
        "#EAF0F8",
        "#17243B",
        "#B8C7DA",
        "#E2EBF6",
        "#17243B",
        ("#2457A6", "#2F7D8C", "#D79B2E", "#7E57C2", "#C44747"),
    ),
    "parchment": Palette(
        "Parchment",
        "Warm neutrals inspired by annotated legal documents.",
        "light",
        "#6B4F2A",
        "#FFFDF7",
        "#F3EBDD",
        "#30271F",
        "#D7C7AD",
        "#EDE1CF",
        "#30271F",
        ("#6B4F2A", "#9B6A32", "#356859", "#A04A45", "#7562A8"),
    ),
    "tropical": Palette(
        "Tropical",
        "Bright teal, coral, mango, and violet accents.",
        "light",
        "#007F73",
        "#FFFDF5",
        "#DFF7F1",
        "#173B3A",
        "#7CCFC2",
        "#C7F0E7",
        "#123D39",
        ("#007F73", "#FF6B5E", "#E6A400", "#7C4DFF", "#168AAD"),
    ),
    "sunset": Palette(
        "Sunset",
        "Warm coral and plum colors with soft cream surfaces.",
        "light",
        "#B43B5A",
        "#FFF9F4",
        "#FBE5DC",
        "#432431",
        "#E8B9AF",
        "#F5D5CA",
        "#432431",
        ("#B43B5A", "#E36F47", "#D49A16", "#7652A8", "#2A7F85"),
    ),
    "forest": Palette(
        "Forest",
        "Deep green accents and low-glare natural surfaces.",
        "light",
        "#276749",
        "#F7FBF7",
        "#E3F0E6",
        "#1D3328",
        "#A8C7B1",
        "#D8E9DC",
        "#1D3328",
        ("#276749", "#4E8D5B", "#B07C27", "#4C6FA8", "#9A4D63"),
    ),
    "midnight": Palette(
        "Midnight",
        "A true dark mode with cyan and violet accents.",
        "dark",
        "#5BC0EB",
        "#111827",
        "#1F2937",
        "#F3F4F6",
        "#46556C",
        "#0B1220",
        "#F3F4F6",
        ("#5BC0EB", "#7DD3A7", "#F6C85F", "#B794F4", "#FF7B8B"),
    ),
}

DEFAULT_PALETTE = "legal_blue"
PREFERENCE_PATH = ARTIFACTS_DIR / "preferences.json"


def load_palette_preference(path: Path = PREFERENCE_PATH) -> str:
    if not path.exists():
        return DEFAULT_PALETTE
    try:
        key = json.loads(path.read_text(encoding="utf-8")).get("palette")
    except (json.JSONDecodeError, OSError):
        return DEFAULT_PALETTE
    return key if key in PALETTES else DEFAULT_PALETTE


def save_palette_preference(key: str, path: Path = PREFERENCE_PATH) -> None:
    if key not in PALETTES:
        raise ValueError(f"Unknown palette: {key}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"palette": key}, indent=2), encoding="utf-8")


def _hex_rgb(value: str) -> str:
    clean = value.lstrip("#")
    return ", ".join(str(int(clean[index : index + 2], 16)) for index in (0, 2, 4))


def palette_css(key: str) -> str:
    palette = PALETTES.get(key, PALETTES[DEFAULT_PALETTE])
    return f"""
<style>
:root, .stApp {{
  color-scheme: {palette.mode};
  --primary-color: {palette.primary};
  --primary-color-rgb: {_hex_rgb(palette.primary)};
  --background-color: {palette.background};
  --secondary-background-color: {palette.secondary};
  --text-color: {palette.text};
  --border-color: {palette.border};
}}
.stApp, [data-testid="stAppViewContainer"] {{
  background-color: {palette.background};
  color: {palette.text};
}}
[data-testid="stSidebar"] {{
  --primary-color: {palette.primary};
  --background-color: {palette.sidebar};
  --secondary-background-color: {palette.secondary};
  --text-color: {palette.sidebar_text};
  --border-color: {palette.border};
  background-color: {palette.sidebar};
  color: {palette.sidebar_text};
  border-right: 1px solid {palette.border};
}}
[data-testid="stHeader"] {{
  background-color: color-mix(in srgb, {palette.background} 92%, transparent);
}}
a {{ color: {palette.primary}; }}
code, pre {{ border-color: {palette.border}; }}
</style>
"""


def palette_swatch_html(key: str) -> str:
    palette = PALETTES[key]
    swatches = "".join(
        f'<span style="display:inline-block;width:2.25rem;height:2.25rem;'
        f"border-radius:.55rem;background:{color};border:1px solid {palette.border};"
        f'margin-right:.45rem" title="{color}"></span>'
        for color in (palette.primary, palette.background, palette.secondary, *palette.colors[1:4])
    )
    return f'<div aria-label="{palette.name} color swatches">{swatches}</div>'
