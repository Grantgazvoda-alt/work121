"""Streamlit AppTest: attach a loopback peer and run a real evaluation."""

from __future__ import annotations

import secrets
from pathlib import Path

import pytest

from oracle_keys import ENV_KEY
from regression_tracker import JsonlRunStore
from template_library import BenignTemplateLibrary


def _set_text(app, label: str, value: str) -> None:
    matches = [widget for widget in app.text_input if widget.label == label]
    assert matches, f"missing text input {label}"
    matches[0].set_value(value)


def test_dashboard_attaches_loopback_peer_and_writes_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv(ENV_KEY, secrets.token_hex(32))
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "dashboard.py"),
        default_timeout=90,
    )
    app.run()
    assert len(app.exception) == 0
    assert app.session_state["registry"].list_keys() == []

    try:
        _set_text(app, "Model id", "boundary-oracle")
        _set_text(app, "Model version", "strict-1")
        _set_text(app, "Architecture family", "policy-oracle")
        _set_text(app, "Artifact Output Directory", str(tmp_path))
        app.run()
        next(button for button in app.button if button.label == "Attach agent").click()
        app.run()
        assert app.session_state["registry"].list_keys() == ["boundary-oracle@strict-1"]
        assert not app.error

        next(button for button in app.button if button.label == "Load Full Default Suite").click()
        app.run()
        template_count = len(BenignTemplateLibrary().list_templates())
        assert len(app.session_state["queued_cases"]) == template_count
        selected = next(widget for widget in app.multiselect if widget.label == "Attached target agents")
        assert list(selected.value) == ["boundary-oracle@strict-1"]

        next(button for button in app.button if button.label == "Run Real-Time Evaluation").click()
        app.run()
        assert len(app.exception) == 0
        assert not app.error
        assert len(app.session_state["run_results"]) == 1
        result = next(iter(app.session_state["run_results"].values()))
        assert len(result.records) == template_count * 2
        assert any(record.is_baseline for record in result.records)
        assert any(not record.is_baseline for record in result.records)

        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        reloaded = JsonlRunStore().load_records(jsonl_files[0])
        assert len(reloaded) == template_count * 2
        assert {row.outcome for row in reloaded} <= {row.outcome for row in result.records}
        blob = " ".join(message.value for message in app.success)
        assert "policy-assistant" not in blob
        assert f"Completed {template_count * 2} evaluations" in blob
    finally:
        app.session_state["registry"].close_all()
