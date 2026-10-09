# LogLens

Server-log analytics with Median Absolute Deviation (MAD) anomaly detection.
Python capstone of [Bits to Builds](https://bitstobuilds.com). CLF regex parser,
rich endpoint/UA/IP analytics, modified z-score alerting on hourly 5xx counts.

**Live demo:** (fill in after `publish.sh`)
**Source:** <https://github.com/PJsAcademy/loglens>

---

## What it does

| Phase | Deliverable |
|-------|-------------|
| 1. Parse   | Apache Common Log Format regex parser (synth fallback), 2 enrichment rules with reject counts |
| 2. Analyze | endpoint ranking, hourly series, UA bucket split, top-20 noisy IPs |
| 3. Alert   | MAD + modified z-score detector, severity labels (critical/high/elevated) |

## Run locally

```bash
pip install -r requirements.txt
python src/phase1_parse.py    # ~5s synth; writes silver.parquet
python src/phase2_analyze.py  # ~1s; writes 4 analytics tables
python src/phase3_alert.py    # ~0.5s; writes alerts.parquet
pytest tests/ -q              # 19 invariants should pass
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud (free)

```bash
REPO=loglens ../portfolio/publish.sh
```

Then at <https://share.streamlit.io>: Create app → pick this repo → main file
`streamlit_app.py` → Deploy.

## Honest limits

- 100k-request synthetic sample on the deployed version. Real logs → point
  `phase1_parse.py` at your `access.log` (gzip supported).
- MAD threshold 3.5 is global; a per-endpoint SLO would be the honest upgrade.
- UA bucketing is regex-based (fast, wrong on edge cases); use `ua-parser` for
  prod.
- No tz handling — timestamps are naive-UTC.

## Credits

Common Log Format regex adapted from Apache's standard. Built from the
[Bits to Builds](https://bitstobuilds.com) curriculum.
