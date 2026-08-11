import pytest

from contractlab.palettes import DEFAULT_PALETTE, PALETTES, load_palette_preference


def test_required_palettes_exist():
    assert {"legal_blue", "midnight", "tropical"}.issubset(PALETTES)
    assert PALETTES["midnight"].mode == "dark"
    assert PALETTES["tropical"].mode == "light"


def test_invalid_saved_palette_falls_back(tmp_path):
    preference = tmp_path / "preferences.json"
    preference.write_text('{"palette": "not-real"}', encoding="utf-8")
    assert load_palette_preference(preference) == DEFAULT_PALETTE


def test_save_rejects_invalid_palette(tmp_path):
    from contractlab.palettes import save_palette_preference

    with pytest.raises(ValueError, match="Unknown palette"):
        save_palette_preference("not-real", tmp_path / "preferences.json")
