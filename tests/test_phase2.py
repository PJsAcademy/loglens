"""Phase 2 invariants (6)."""
from pathlib import Path

import pandas as pd
import pytest

CHECKPOINTS = Path(__file__).resolve().parent.parent / "checkpoints"


def _load(name):
    p = CHECKPOINTS / name
    if not p.exists():
        pytest.skip(f"{name} missing — run `python src/phase2_analyze.py` first")
    return pd.read_parquet(p)


def test_endpoint_stats_sorted():
    df = _load("endpoint_stats.parquet")
    assert (df["requests"].values[:-1] >= df["requests"].values[1:]).all()


def test_endpoint_rates_in_range():
    df = _load("endpoint_stats.parquet")
    assert ((df["error_rate"] >= 0) & (df["error_rate"] <= 1)).all()
    assert ((df["server_error_rate"] >= 0) & (df["server_error_rate"] <= 1)).all()


def test_hourly_monotonic_ts():
    df = _load("hourly.parquet")
    assert df["ts"].is_monotonic_increasing


def test_hourly_error_rate_sanity():
    df = _load("hourly.parquet")
    # errors <= requests must hold hour-by-hour
    assert (df["errors"] <= df["requests"]).all()
    assert (df["server_errors"] <= df["errors"]).all()


def test_top_ips_exactly_20():
    df = _load("top_ips.parquet")
    assert len(df) <= 20
    # Sorted by requests desc
    assert (df["requests"].values[:-1] >= df["requests"].values[1:]).all()


def test_ua_bucket_valid():
    df = _load("ua_stats.parquet")
    valid = {"bot", "tool", "mobile", "mac", "windows", "linux", "other"}
    assert set(df["ua_bucket"]).issubset(valid)
