"""Phase 3 — LogLens alerting via Median Absolute Deviation (MAD).

Robust-stats anomaly detection on hourly 5xx counts. MAD is the median of
abs(x - median(x)); the modified z-score is 0.6745 * (x - median) / MAD. Anything
with |z| > 3.5 is flagged. Robust = outliers don't move the baseline (unlike
mean/stdev).

Reads hourly.parquet, writes checkpoints/alerts.parquet with one row per
anomalous hour: ts, server_errors, baseline_med, mad, modified_z, severity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
CHECKPOINTS = HERE / "checkpoints"

MAD_THRESHOLD = 3.5  # standard cutoff for "outlier" under the modified z-score
MIN_BASELINE_HOURS = 24


def mad_zscores(x: pd.Series) -> pd.Series:
    """Modified z-score. Robust to outliers because it uses median + MAD.

    Degenerate cases:
      - MAD = 0 but IQR > 0: use IQR/1.349 as the scale (same expected value).
      - MAD = 0 AND IQR = 0 (near-constant baseline): any non-matching value IS
        the anomaly, so give it a huge z. This matches the detector's intent —
        "the baseline is flat; an outlier stands out trivially".
    """
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0:
        iqr = x.quantile(0.75) - x.quantile(0.25)
        if iqr == 0:
            # Degenerate baseline: anything off-median is a huge outlier.
            dev = x - med
            z = np.where(dev == 0, 0.0, np.sign(dev) * 100.0)
            return pd.Series(z, index=x.index)
        return 0.6745 * (x - med) / (iqr / 1.349)
    return 0.6745 * (x - med) / mad


def detect(hourly: pd.DataFrame) -> pd.DataFrame:
    df = hourly.sort_values("ts").reset_index(drop=True).copy()
    df["z"] = mad_zscores(df["server_errors"])
    alerts = df[df["z"].abs() > MAD_THRESHOLD].copy()
    alerts["baseline_med"] = df["server_errors"].median()
    alerts["mad"] = (df["server_errors"] - df["server_errors"].median()).abs().median()

    def severity(z):
        a = abs(z)
        if a >= 10: return "critical"
        if a >= 6:  return "high"
        return "elevated"
    alerts["severity"] = alerts["z"].map(severity)
    return alerts[["ts", "server_errors", "requests", "server_error_rate",
                   "baseline_med", "mad", "z", "severity"]].reset_index(drop=True)


def main() -> None:
    hourly = pd.read_parquet(CHECKPOINTS / "hourly.parquet")
    print(f"[phase3] loaded hourly: {len(hourly)} hours")
    if len(hourly) < MIN_BASELINE_HOURS:
        print(f"[phase3] not enough history (<{MIN_BASELINE_HOURS}h) for a baseline; skipping")
        return
    alerts = detect(hourly)
    alerts.to_parquet(CHECKPOINTS / "alerts.parquet", index=False)
    print(f"[phase3] detected {len(alerts)} anomalous hour(s):")
    if len(alerts):
        for _, a in alerts.iterrows():
            print(f"  {a['ts']}  5xx={int(a['server_errors']):>4}  "
                  f"z={a['z']:+.1f}  severity={a['severity']}")
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
