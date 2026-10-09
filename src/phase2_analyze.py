"""Phase 2 — LogLens analytics.

Reads silver.parquet; writes:
  - checkpoints/endpoint_stats.parquet : per endpoint traffic + error rate
  - checkpoints/hourly.parquet         : per hour requests + error rate + 5xx count
  - checkpoints/ua_stats.parquet       : per user-agent bucket counts
  - checkpoints/top_ips.parquet        : top-20 noisy IPs (candidate abusers)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
CHECKPOINTS = HERE / "checkpoints"


def main() -> None:
    silver = pd.read_parquet(CHECKPOINTS / "silver.parquet")
    print(f"[phase2] loaded silver: {len(silver):,} rows")

    endpoint_stats = (silver.groupby("path")
                            .agg(requests=("status", "size"),
                                 errors=("is_error", "sum"),
                                 server_errors=("is_server_error", "sum"),
                                 avg_bytes=("bytes_sent", "mean"))
                            .assign(error_rate=lambda d: d["errors"] / d["requests"],
                                    server_error_rate=lambda d: d["server_errors"] / d["requests"])
                            .sort_values("requests", ascending=False)
                            .reset_index())
    endpoint_stats.to_parquet(CHECKPOINTS / "endpoint_stats.parquet", index=False)
    print(f"[phase2] endpoint_stats: {len(endpoint_stats):,} rows")

    hourly = (silver.assign(ts=silver["timestamp"].dt.floor("h"))
                    .groupby("ts")
                    .agg(requests=("status", "size"),
                         errors=("is_error", "sum"),
                         server_errors=("is_server_error", "sum"))
                    .reset_index()
                    .assign(error_rate=lambda d: d["errors"] / d["requests"],
                            server_error_rate=lambda d: d["server_errors"] / d["requests"]))
    hourly.to_parquet(CHECKPOINTS / "hourly.parquet", index=False)
    print(f"[phase2] hourly: {len(hourly):,} rows")

    ua_stats = (silver.groupby("ua_bucket")
                      .agg(requests=("status", "size"),
                           error_rate=("is_error", "mean"))
                      .sort_values("requests", ascending=False)
                      .reset_index())
    ua_stats.to_parquet(CHECKPOINTS / "ua_stats.parquet", index=False)
    print(f"[phase2] ua_stats: {len(ua_stats):,} rows")

    top_ips = (silver.groupby("ip")
                     .agg(requests=("status", "size"),
                          errors=("is_error", "sum"))
                     .assign(error_rate=lambda d: d["errors"] / d["requests"])
                     .sort_values("requests", ascending=False)
                     .head(20)
                     .reset_index())
    top_ips.to_parquet(CHECKPOINTS / "top_ips.parquet", index=False)
    print(f"[phase2] top_ips: {len(top_ips):,} rows")


if __name__ == "__main__":
    main()
