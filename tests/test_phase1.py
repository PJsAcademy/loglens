"""Phase 1 invariants (7)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from phase1_parse import parse_clf_line  # noqa: E402

CHECKPOINTS = ROOT / "checkpoints"


@pytest.fixture(scope="module")
def silver() -> pd.DataFrame:
    path = CHECKPOINTS / "silver.parquet"
    if not path.exists():
        pytest.skip("silver.parquet missing — run `python src/phase1_parse.py` first")
    return pd.read_parquet(path)


def test_silver_nonempty(silver):
    assert len(silver) > 0


def test_schema(silver):
    required = {"timestamp", "ip", "method", "path", "status", "bytes_sent",
                "user_agent", "status_class", "is_error", "is_server_error", "ua_bucket"}
    assert required.issubset(set(silver.columns))


def test_status_codes_valid(silver):
    assert ((silver["status"] >= 100) & (silver["status"] <= 599)).all()


def test_no_negative_bytes(silver):
    assert (silver["bytes_sent"] >= 0).all()


def test_status_class_consistent(silver):
    expected_class = (silver["status"] // 100).map({2: "2xx", 3: "3xx", 4: "4xx", 5: "5xx"})
    assert (silver["status_class"] == expected_class).all()


def test_clf_parser_happy_path():
    line = (
        '192.168.1.1 - - [10/Oct/2023:13:55:36 +0000] '
        '"GET /index.html HTTP/1.1" 200 2326 "-" "Mozilla/5.0"'
    )
    parsed = parse_clf_line(line)
    assert parsed is not None
    assert parsed["ip"] == "192.168.1.1"
    assert parsed["method"] == "GET"
    assert parsed["path"] == "/index.html"
    assert parsed["status"] == 200
    assert parsed["bytes_sent"] == 2326


def test_clf_parser_rejects_garbage():
    assert parse_clf_line("not a log line") is None
    assert parse_clf_line("") is None
