"""Phase 3 invariants (6) — MAD detector."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from phase3_alert import mad_zscores, detect  # noqa: E402

CHECKPOINTS = ROOT / "checkpoints"


def test_mad_zero_for_constant_series():
    z = mad_zscores(pd.Series([5] * 100))
    assert (z == 0).all()


def test_mad_flags_outlier_in_middle_of_calm_series():
    x = pd.Series([5] * 50 + [500] + [5] * 50)
    z = mad_zscores(x)
    # The 500 should be a huge outlier
    assert z.iloc[50] > 50, f"outlier z too small: {z.iloc[50]}"


def test_mad_resists_contamination():
    """A single spike should not shift the baseline for other values."""
    x = pd.Series([5] * 50 + [1000] + [5] * 50)
    z = mad_zscores(x)
    # All non-spike values should still have z=0 (because the single spike doesn't move the median)
    assert z.drop(index=50).abs().max() == 0


def test_detect_output_schema():
    hourly = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=100, freq="h"),
        "server_errors": [5] * 50 + [500] + [5] * 49,
        "requests": [1000] * 100,
        "server_error_rate": [0.005] * 100,
    })
    alerts = detect(hourly)
    assert set(alerts.columns) == {
        "ts", "server_errors", "requests", "server_error_rate",
        "baseline_med", "mad", "z", "severity",
    }
    assert len(alerts) >= 1


def test_detect_catches_planted_spike_on_real_data():
    p = CHECKPOINTS / "alerts.parquet"
    if not p.exists():
        pytest.skip("alerts.parquet missing — run `python src/phase3_alert.py` first")
    alerts = pd.read_parquet(p)
    # The deterministic sample plants a spike at 2024-06-04 15:00 UTC
    assert len(alerts) >= 1
    assert any(alerts["ts"] == pd.Timestamp("2024-06-04 15:00:00"))


def test_severity_labels_valid():
    p = CHECKPOINTS / "alerts.parquet"
    if not p.exists():
        pytest.skip("alerts.parquet missing")
    alerts = pd.read_parquet(p)
    valid = {"critical", "high", "elevated"}
    assert set(alerts["severity"]).issubset(valid)
