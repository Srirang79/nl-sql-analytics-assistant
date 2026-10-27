"""Data loader, sample builder, evaluator and app smoke tests (all on the tiny fixture)."""
import sqlite3
import zipfile

import pandas as pd
import pytest

from src import config, data_loader, make_sample
from src.engine import SQLAssistant
from src.evaluate import evaluate, load_golden, results_match
from src.llm import DemoClient


def _export_like_kaggle(fixture_db, folder):
    """Write the fixture out as CSVs with the ORIGINAL Kaggle file names and misspelt column names."""
    folder.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(fixture_db)
    inv = {v: k for k, v in data_loader.FILES.items()}
    for table, fname in inv.items():
        if table == "geolocation":
            continue
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        df = df.rename(columns={v: k for k, v in data_loader.RENAMES.get(table, {}).items()})
        df.to_csv(folder / fname, index=False)


def test_loader_builds_database_with_renames_and_indexes(fixture_db, tmp_path):
    raw = tmp_path / "raw"
    _export_like_kaggle(fixture_db, raw)
    db = tmp_path / "olist.db"
    counts = data_loader.load(raw, db)
    assert counts["orders"] == 160 and "geolocation" not in counts
    conn = sqlite3.connect(db)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(products)")]
    assert "product_name_length" in cols and "product_name_lenght" not in cols
    assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name LIKE 'idx_orders_%'").fetchone()[0] >= 3
    with pytest.raises(FileExistsError):
        data_loader.load(raw, db)


def test_loader_extracts_kaggle_zip(fixture_db, tmp_path):
    staging = tmp_path / "staging"
    _export_like_kaggle(fixture_db, staging)
    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / "archive.zip", "w") as z:
        for f in staging.iterdir():
            z.write(f, f.name)
    counts = data_loader.load(raw, tmp_path / "o.db")
    assert counts["customers"] == 160


def test_loader_explains_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        data_loader.load(tmp_path / "empty", tmp_path / "x.db")
    assert "kaggle.com/datasets/olistbr/brazilian-ecommerce" in str(e.value)


def test_make_sample_is_consistent(fixture_db, tmp_path):
    out = tmp_path / "sample.db"
    counts = make_sample.make_sample(fixture_db, out, n_orders=40)
    assert 30 <= counts["orders"] <= 60
    conn = sqlite3.connect(out)
    orphans = conn.execute("SELECT COUNT(*) FROM order_items WHERE order_id NOT IN (SELECT order_id FROM orders)").fetchone()[0]
    missing_cust = conn.execute("SELECT COUNT(*) FROM orders WHERE customer_id NOT IN (SELECT customer_id FROM customers)").fetchone()[0]
    missing_prod = conn.execute("SELECT COUNT(*) FROM order_items WHERE product_id NOT IN (SELECT product_id FROM products)").fetchone()[0]
    assert orphans == missing_cust == missing_prod == 0


def test_every_reference_query_runs_on_the_schema(fixture_db):
    conn = sqlite3.connect(fixture_db)
    for g in load_golden():
        rows = conn.execute(g["reference_sql"]).fetchall()
        assert rows, g["id"]


def test_golden_set_shape():
    golden = load_golden()
    assert len(golden) >= 25 and len({g["id"] for g in golden}) == len(golden)
    assert {"aggregation", "join", "filter", "time", "multi-step"} <= {g["category"] for g in golden}


# ---- result comparison ---------------------------------------------------------------------
def test_results_match_rules():
    ref = pd.DataFrame({"state": ["SP", "RJ"], "n": [10, 5]})
    assert results_match(ref, pd.DataFrame({"s": ["RJ", "SP"], "count": [5.0, 10.0]}))                    # order + names ignored
    assert results_match(ref, pd.DataFrame({"state": ["SP", "RJ"], "n": [10, 5], "extra": [1, 2]}))       # extra column ok
    assert not results_match(ref, pd.DataFrame({"state": ["SP", "RJ"], "n": [10, 6]}))                   # wrong value
    assert not results_match(ref, pd.DataFrame({"state": ["SP"], "n": [10]}))                            # wrong row count
    assert not results_match(ref, pd.DataFrame({"state": ["RJ", "SP"], "n": [5, 10]}), ordered=True)     # ordered reference
    assert results_match(pd.DataFrame({"x": [1.234]}), pd.DataFrame({"y": [1.2]}))                       # float rounding


def test_evaluate_with_oracle_and_with_wrong_llm(fixture_db):
    golden = load_golden()[:8]
    oracle = DemoClient({g["question"]: g["reference_sql"] for g in golden})
    out = evaluate(fixture_db, SQLAssistant(fixture_db, oracle), golden, verbose=False)
    assert out["summary"]["execution_accuracy"] == 1.0 and out["summary"]["first_attempt_rate"] == 1.0

    class WrongLLM:
        name = "wrong"

        def complete(self, system, messages, max_tokens=1200):
            return "```sql\nSELECT 42 AS answer\n```"

    bad = evaluate(fixture_db, SQLAssistant(fixture_db, WrongLLM()), golden, verbose=False)
    assert bad["summary"]["execution_accuracy"] < 0.3 and bad["summary"]["valid_sql_rate"] == 1.0


# ---- app ------------------------------------------------------------------------------------
def test_app_runs_in_demo_mode(fixture_db, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SQL_ASSISTANT_DB", str(fixture_db))
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=90).run()
    assert not at.exception
    example = next(b for b in at.button if b.label.startswith("How many orders"))
    example.click().run()
    assert not at.exception and not at.error
    assert len(at.dataframe) >= 1


def test_app_explains_missing_database(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SQL_ASSISTANT_DB", str(tmp_path / "nope.db"))
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception and any("No database" in e.value for e in at.error)
