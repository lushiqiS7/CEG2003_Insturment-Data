"""Exercise the single review queue, detail, inference and provenance checks."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.common import ROOT

pytestmark = pytest.mark.skipif(
    not (ROOT / "outputs/metrics.json").exists(), reason="Training outputs not built"
)


def test_queue_filters_detail_and_inference():
    app = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=45).run()
    assert not app.exception
    assert app.title[0].value == "Captain risk review"
    assert app.metric[0].value == "1,509"
    assert any("severity is not recorded" in element.value for element in app.caption)
    flagged = len(app.dataframe[0].value)
    app.selectbox[0].select("All captains").run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 1509
    assert flagged < 1509
    app.button[0].click().run()
    assert not app.exception
    assert "matches" in app.success[0].value


def test_empty_search():
    app = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=45).run()
    app.text_input[0].set_value("CAPTAIN-DOES-NOT-EXIST").run()
    assert not app.exception
    assert "No captains match" in app.info[0].value
    assert len(app.button) == 0


def test_stale_preparation_is_rejected_without_touching_real_files(monkeypatch):
    original_read = Path.read_bytes
    path = ROOT / "outputs/preparation.json"

    def changed_bytes(candidate):
        content = original_read(candidate)
        return content + b" " if candidate == path else content

    st.cache_data.clear()
    monkeypatch.setattr(Path, "read_bytes", changed_bytes)
    app = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=45).run()
    assert not app.exception
    assert "Data preparation changed" in app.info[0].value
    st.cache_data.clear()


def test_changed_configuration_is_rejected(monkeypatch):
    import json

    original_read = Path.read_text

    def changed_text(path, *args, **kwargs):
        text = original_read(path, *args, **kwargs)
        if path == ROOT / "config.json":
            changed = json.loads(text)
            changed["review_fraction"] = 0.2
            return json.dumps(changed)
        return text

    st.cache_data.clear()
    monkeypatch.setattr(Path, "read_text", changed_text)
    app = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=45).run()
    assert not app.exception
    assert "Configuration changed" in app.info[0].value
    st.cache_data.clear()


@pytest.mark.parametrize(
    "relative_path",
    [
        "models/incident_model.joblib",
        "outputs/driver_scores.parquet",
        "data/processed/driver_daily.parquet",
    ],
)
def test_changed_outputs_are_rejected_before_loading(monkeypatch, relative_path):
    import hashlib

    original_digest = hashlib.file_digest

    def changed_digest(stream, *args, **kwargs):
        digest = original_digest(stream, *args, **kwargs)
        if Path(stream.name) == ROOT / relative_path:
            return hashlib.sha256(b"different model")
        return digest

    st.cache_data.clear()
    monkeypatch.setattr(hashlib, "file_digest", changed_digest)
    app = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=45).run()
    assert not app.exception
    assert "do not match" in app.info[0].value
    st.cache_data.clear()
