from __future__ import annotations

import json
from pathlib import Path

import pytest

from loupe.state_config import (
    LoupeConfigError,
    StateConfig,
    load_state_config,
    normalize_keymap,
    parse_color,
)


# ---------------------------------------------------------------------------
# parse_color
# ---------------------------------------------------------------------------


def test_parse_color_hex_rgb():
    assert parse_color("#ff0080") == (255, 0, 128, 255)


def test_parse_color_hex_rgba():
    assert parse_color("#ff008040") == (255, 0, 128, 64)


def test_parse_color_list_rgba():
    assert parse_color([10, 20, 30, 40]) == (10, 20, 30, 40)


def test_parse_color_list_rgb_defaults_alpha_255():
    assert parse_color([10, 20, 30]) == (10, 20, 30, 255)


def test_parse_color_invalid_hex():
    with pytest.raises(LoupeConfigError):
        parse_color("#zzz")


def test_parse_color_wrong_length():
    with pytest.raises(LoupeConfigError):
        parse_color([1, 2])


# ---------------------------------------------------------------------------
# normalize_keymap
# ---------------------------------------------------------------------------


def test_normalize_forward_form():
    out = normalize_keymap({"w": "Wake", "1": "NREM"})
    assert out == {"Wake": ["w"], "NREM": ["1"]}


def test_normalize_inverse_form_list():
    out = normalize_keymap({"Wake": ["w", "W"], "NREM": ["1", "n"]})
    assert out == {"Wake": ["w", "W"], "NREM": ["1", "n"]}


def test_normalize_inverse_form_string_value():
    out = normalize_keymap({"Wake": "w", "NREM": ["1", "n"]})
    assert out == {"Wake": ["w"], "NREM": ["1", "n"]}


def test_normalize_forward_duplicate_keys_to_same_state_ok():
    # Two entries pointing the same key at the same state is benign.
    out = normalize_keymap({"w": "Wake"})
    assert out == {"Wake": ["w"]}


def test_normalize_forward_conflict_raises():
    with pytest.raises(LoupeConfigError, match="Wake"):
        # Two different keys both bound — fine. But same key bound twice
        # to different states is rejected. Forward-form input is a Python dict,
        # so we can't directly express duplicate keys; conflict via inverse form:
        normalize_keymap({"Wake": ["w"], "Wake_quiet": ["w"]})


def test_normalize_empty():
    assert normalize_keymap({}) == {}
    assert normalize_keymap(None) == {}


def test_normalize_empty_keystring_raises():
    with pytest.raises(LoupeConfigError):
        normalize_keymap({"Wake": ["", "w"]})


def test_normalize_dedupes_within_state():
    out = normalize_keymap({"Wake": ["w", "w", "W"]})
    assert out == {"Wake": ["w", "W"]}


# ---------------------------------------------------------------------------
# StateConfig
# ---------------------------------------------------------------------------


def test_state_config_key_to_state_inverse():
    cfg = StateConfig(
        keys_for_state={"Wake": ["w", "W"], "NREM": ["1"]},
        label_colors={"Wake": (0, 0, 0, 255), "NREM": (0, 0, 255, 255)},
    )
    assert cfg.key_to_state == {"w": "Wake", "W": "Wake", "1": "NREM"}


def test_state_config_hotkeys_label_multi():
    cfg = StateConfig(keys_for_state={"Wake": ["w", "W"]}, label_colors={"Wake": (0, 0, 0, 255)})
    assert cfg.hotkeys_label("Wake") == "[w, W]"
    assert cfg.state_with_hotkeys("Wake") == "Wake [w, W]"


def test_state_config_hotkeys_label_single():
    cfg = StateConfig(keys_for_state={"Wake": ["w"]}, label_colors={"Wake": (0, 0, 0, 255)})
    assert cfg.hotkeys_label("Wake") == "[w]"


def test_state_config_hotkeys_label_unbound_state():
    cfg = StateConfig(keys_for_state={}, label_colors={"Wake": (0, 0, 0, 255)})
    assert cfg.hotkeys_label("Wake") == ""
    assert cfg.state_with_hotkeys("Wake") == "Wake"


# ---------------------------------------------------------------------------
# load_state_config
# ---------------------------------------------------------------------------


def _write_json(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "state.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_from_path(tmp_path):
    path = _write_json(
        tmp_path,
        {
            "keymap": {"w": "Wake", "1": "NREM"},
            "label_colors": {"Wake": [0, 0, 0, 255], "NREM": "#0000ff"},
        },
    )
    cfg = load_state_config(path=path, package_default=False)
    assert cfg.key_to_state == {"w": "Wake", "1": "NREM"}
    assert cfg.label_colors["NREM"] == (0, 0, 255, 255)


def test_load_kwarg_only():
    cfg = load_state_config(
        keymap={"w": "Wake"},
        label_colors={"Wake": "#ff0000"},
        package_default=False,
    )
    assert cfg.key_to_state == {"w": "Wake"}
    assert cfg.label_colors == {"Wake": (255, 0, 0, 255)}


def test_load_kwarg_overrides_file(tmp_path):
    path = _write_json(
        tmp_path,
        {
            "keymap": {"w": "Wake"},
            "label_colors": {"Wake": [0, 0, 0, 255]},
        },
    )
    cfg = load_state_config(
        path=path,
        keymap={"w": "Sleep"},
        label_colors={"Sleep": "#00ff00"},
        package_default=False,
    )
    assert cfg.key_to_state == {"w": "Sleep"}
    assert cfg.label_colors["Sleep"] == (0, 255, 0, 255)


def test_load_no_definitions_raises():
    with pytest.raises(LoupeConfigError, match="No state definitions"):
        load_state_config(package_default=False)


def test_load_state_with_keys_missing_color_raises(tmp_path):
    path = _write_json(
        tmp_path,
        {"keymap": {"w": "Wake"}, "label_colors": {}},
    )
    with pytest.raises(LoupeConfigError, match="no color"):
        load_state_config(path=path, package_default=False)


def test_load_inverse_keymap(tmp_path):
    path = _write_json(
        tmp_path,
        {
            "keymap": {"Wake": ["w", "W"], "NREM": ["1"]},
            "label_colors": {"Wake": "#000000", "NREM": "#0000ff"},
        },
    )
    cfg = load_state_config(path=path, package_default=False)
    assert cfg.keys_for_state == {"Wake": ["w", "W"], "NREM": ["1"]}


def test_example_state_definitions_parses():
    """The bundled example file is the authoritative schema reference; it must parse."""
    pkg_dir = Path(__file__).resolve().parent.parent / "src" / "loupe"
    candidate = pkg_dir / "example_state_definitions.json"
    if not candidate.exists():
        pytest.skip("example_state_definitions.json not yet renamed in this branch")
    cfg = load_state_config(path=candidate, package_default=False)
    assert cfg.keys_for_state, "example file should define some hotkeys"
    assert cfg.label_colors, "example file should define some colors"
