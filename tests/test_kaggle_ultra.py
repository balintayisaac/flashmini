"""Tests for train/kaggle_ultra.py — the cloud SFT pipeline helpers.

The module imports cleanly without torch/trl (heavy imports live inside
function bodies), so its pure decision logic is testable locally.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "train" / "kaggle_ultra.py"
_spec = importlib.util.spec_from_file_location("kaggle_ultra", _MODULE_PATH)
kaggle_ultra = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = kaggle_ultra
_spec.loader.exec_module(kaggle_ultra)  # type: ignore[union-attr]


class TestAssistantOnlySupported:
    def test_template_with_generation_markers_is_supported(self) -> None:
        tmpl = (
            "{% for message in messages %}"
            "{% if message.role == 'assistant' %}"
            "{% generation %}{{ message.content }}{% endgeneration %}"
            "{% endif %}{% endfor %}"
        )
        assert kaggle_ultra.assistant_only_supported(tmpl) is True

    def test_template_without_markers_is_not_supported(self) -> None:
        tmpl = "{% for message in messages %}{{ message.content }}{% endfor %}"
        assert kaggle_ultra.assistant_only_supported(tmpl) is False

    def test_missing_template_is_not_supported(self) -> None:
        assert kaggle_ultra.assistant_only_supported(None) is False
        assert kaggle_ultra.assistant_only_supported("") is False
