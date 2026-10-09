"""Phase 1 — LogLens parse.

Reads raw Apache Common Log Format file if present (data/access.log), else
synthesises a deterministic 100k-request sample over 7 days with realistic
endpoints, status codes, and a planted anomaly window.
"""
from __future__ import annotations

import gzip
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "access.log"
CHECKPOINTS = HERE / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True)

# Apache Common Log Format: 127.0.0.1 - - [10/Oct/2000:13:55:36 -0700] "GET /apache_pb.gif HTTP/1.0" 200 2326
CLF = re.compile(
    r'(?P<ip>\S+)\s+'
    r'(?P<ident>\S+)\s+'
    r'(?P<user>\S+)\s+'
    r'\[(?P<ts>[^\]]+)\]\s+'
    r'"(?P<method>\w+)\s+(?P<path>\S+)\s+(?P<proto>[^"]+)"\s+'
    r'(?P<status>\d{3})\s+'
    r'(?P<bytes>\S+)'
    r'(?:\s+"(?P<referrer>[^"]*)")?'
    r'(?:\s+"(?P<ua>[^"]*)")?'
)

ENDPOINTS = [
    ("/", 0.18, 200),
    ("/api/products", 0.15, 200),
    ("/api/products/{id}", 0.10, 200),
    ("/api/cart", 0.08, 200),
    ("/api/checkout", 0.05, 200),
    ("/api/login", 0.06, 200),
    ("/api/signup", 0.03, 201),
    ("/api/health", 0.08, 200),
    ("/static/app.css", 0.10, 200),
    ("/static/app.js", 0.10, 200),
    ("/robots.txt", 0.02, 200),
    ("/admin", 0.03, 403),
    ("/missing", 0.02, 404),
]

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) AppleWebKit/605 Safari/605",
    "Mozilla/5.0 (Windows NT 10.0; Win64) Chrome/120 Safari/537",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0) AppleWebKit/605 Mobile/15E148",
    "Mozilla/5.0 (Linux; Android 14) Chrome/120 Mobile",
    "curl/8.4.0",
    "python-requests/2.31",
    "Googlebot/2.1 (+http://www.google.com/bot.html)",
    "Mozilla/5.0 (compatible; bingbot/2.0)",
]
METHODS = {"GET": 0.80, "POST": 0.15, "PUT": 0.03, "DELETE": 0.02}


def synthesise(n: int = 100_000, seed: int = 42) -> pd.DataFrame:
    """Deterministic 7-day CLF-shaped sample with:
    - diurnal traffic (sin wave)
    - realistic status distribution (0.90 2xx, 0.05 3xx, 0.03 4xx, 0.02 5xx baseline)
    - planted 500-error spike at day 4, 02:00-03:00 UTC (anomaly for Phase 3 to detect)
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-06-01")
    end = start + pd.Timedelta(days=7)

    # Build arrival times with diurnal weighting
    seconds = int((end - start).total_seconds())
    hours = np.arange(seconds) / 3600
    weight = 0.5 + 0.5 * np.sin(2 * np.pi * (hours / 24 - 0.3))
    weight = weight / weight.sum()
    offsets = rng.choice(seconds, size=n, replace=True, p=weight)
    offsets.sort()
    timestamps = start + pd.to_timedelta(offsets, unit="s")

    endpoint_probs = np.array([p for _, p, _ in ENDPOINTS])
    endpoint_probs /= endpoint_probs.sum()
    endpoint_idx = rng.choice(len(ENDPOINTS), size=n, p=endpoint_probs)
    paths = [ENDPOINTS[i][0] for i in endpoint_idx]
    base_status = np.array([ENDPOINTS[i][2] for i in endpoint_idx])

    # Random 500s baseline (~1%) + a planted spike at day 4 15:00-16:00 UTC (peak traffic)
    status = base_status.copy()
    rand = rng.random(n)
    status[rand < 0.01] = 500  # baseline 1% server errors
    spike_window = (timestamps >= start + pd.Timedelta(days=3, hours=15)) & \
                   (timestamps <  start + pd.Timedelta(days=3, hours=16))
    # In the spike window, 40% of requests become 500 (clear anomaly for Phase 3)
    spike_hits = rng.random(n) < 0.40
    status[spike_window & spike_hits] = 500

    # Occasional 404s / 403s layered in
    status[(rand >= 0.01) & (rand < 0.03)] = 404

    methods_list = list(METHODS.keys())
    method_probs = np.array(list(METHODS.values()))
    methods = rng.choice(methods_list, size=n, p=method_probs)

    uas = rng.choice(USER_AGENTS, size=n)

    # IPs: 500 unique clients with Pareto-ish repeat distribution
    client_count = 500
    client_rank = (rng.pareto(1.2, size=client_count) + 1).astype(int)
    client_pool = np.repeat(np.arange(client_count), client_rank)
    ip_ids = rng.choice(client_pool, size=n)
    ips = [f"10.{(i>>8)&255}.{i&255}.{(hash(str(i)) % 254) + 1}" for i in ip_ids]

    bytes_sent = np.where(status >= 400, 0, rng.integers(100, 10_000, size=n))

    return pd.DataFrame({
        "timestamp": timestamps,
        "ip": ips,
        "method": methods,
        "path": paths,
        "status": status,
        "bytes_sent": bytes_sent,
        "user_agent": uas,
    })


def parse_clf_line(line: str) -> dict | None:
    m = CLF.match(line)
    if not m:
        return None
    try:
        ts = pd.to_datetime(m.group("ts"), format="%d/%b/%Y:%H:%M:%S %z", utc=True).tz_localize(None)
    except Exception:
        return None
    bytes_sent = 0 if m.group("bytes") == "-" else int(m.group("bytes"))
    return {
        "timestamp": ts, "ip": m.group("ip"), "method": m.group("method"),
        "path": m.group("path"), "status": int(m.group("status")),
        "bytes_sent": bytes_sent, "user_agent": m.group("ua") or "-",
    }


def load_bronze() -> tuple[pd.DataFrame, dict[str, int]]:
    rejects = {"R1_malformed_line": 0, "R2_bad_timestamp": 0, "R3_bad_status": 0}
    if RAW.exists():
        print(f"[bronze] reading {RAW}")
        rows = []
        opener = gzip.open if RAW.suffix == ".gz" else open
        with opener(RAW, "rt") as f:
            for line in f:
                parsed = parse_clf_line(line.strip())
                if parsed is None:
                    rejects["R1_malformed_line"] += 1
                    continue
                rows.append(parsed)
        return pd.DataFrame(rows), rejects
    print(f"[bronze] {RAW} missing — synthesising 100k requests over 7 days (seed=42)")
    return synthesise(), rejects


def enrich_to_silver(bronze: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = bronze.copy()
    rejects = {
        "R1_bad_status": int(((df["status"] < 100) | (df["status"] > 599)).sum()),
        "R2_negative_bytes": int((df["bytes_sent"] < 0).sum()),
    }
    df = df[(df["status"] >= 100) & (df["status"] <= 599)]
    df = df[df["bytes_sent"] >= 0]

    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    df["status_class"] = (df["status"] // 100).map({2: "2xx", 3: "3xx", 4: "4xx", 5: "5xx"})
    df["is_error"] = df["status"] >= 400
    df["is_server_error"] = df["status"] >= 500

    def ua_bucket(ua: str) -> str:
        u = ua.lower()
        if "bot" in u or "spider" in u: return "bot"
        if "curl" in u or "python-requests" in u or "wget" in u: return "tool"
        if "iphone" in u or "android" in u or "mobile" in u: return "mobile"
        if "mac os x" in u: return "mac"
        if "windows" in u: return "windows"
        if "linux" in u: return "linux"
        return "other"

    df["ua_bucket"] = df["user_agent"].map(ua_bucket)
    return df.reset_index(drop=True), rejects


def main() -> None:
    bronze, parse_rejects = load_bronze()
    print(f"[bronze] {len(bronze):,} parsed rows")
    silver, enrich_rejects = enrich_to_silver(bronze)
    all_rejects = {**parse_rejects, **enrich_rejects}
    print(f"[silver] {len(silver):,} rows after enrichment")
    for rule, n in all_rejects.items():
        if n:
            print(f"  rejected {n:>5}  {rule}")
    silver.to_parquet(CHECKPOINTS / "silver.parquet", index=False)
    print(f"[silver] wrote {CHECKPOINTS / 'silver.parquet'}")


if __name__ == "__main__":
    main()
