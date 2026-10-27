"""The validator is the safety and reliability layer: it must reject everything dangerous or wrong."""
import pytest

from src.database import connect_readonly
from src.schema import sqlglot_schema
from src.validator import SQLValidator


@pytest.fixture(scope="module")
def v(fixture_db):
    return SQLValidator(sqlglot_schema(connect_readonly(fixture_db)))


def test_accepts_simple_select_and_adds_limit(v):
    r = v.validate("SELECT COUNT(*) FROM orders;")
    assert r.ok and r.sql.rstrip().endswith("LIMIT 1000") and r.tables == ["orders"]


def test_keeps_existing_limit(v):
    r = v.validate("SELECT order_id FROM orders LIMIT 5")
    assert r.ok and r.sql.count("LIMIT") == 1


def test_accepts_cte_join_and_functions(v):
    sql = """WITH m AS (SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS month, o.order_id
                        FROM orders o WHERE o.order_status = 'delivered')
             SELECT month, COUNT(*) AS n, SUM(oi.price) AS rev FROM m
             JOIN order_items oi ON oi.order_id = m.order_id GROUP BY month ORDER BY month"""
    r = v.validate(sql)
    assert r.ok, r.errors


@pytest.mark.parametrize("sql", [
    "DROP TABLE orders",
    "DELETE FROM orders",
    "UPDATE orders SET order_status='x'",
    "INSERT INTO orders VALUES (1,2,3,4,5,6,7,8)",
    "PRAGMA table_info(orders)",
    "ATTACH DATABASE 'x.db' AS x",
    "CREATE TABLE t AS SELECT * FROM orders",
    "SELECT 1; DROP TABLE orders",
    "",
])
def test_rejects_non_select_and_multi_statement(v, sql):
    r = v.validate(sql)
    assert not r.ok and r.errors


def test_rejects_unknown_table_with_suggestion(v):
    r = v.validate("SELECT * FROM order_item")
    assert not r.ok and "order_items" in r.errors[0]


def test_rejects_system_table(v):
    r = v.validate("SELECT name FROM sqlite_master")
    assert not r.ok and "system table" in r.errors[0]


def test_rejects_hallucinated_column(v):
    r = v.validate("SELECT o.order_total FROM orders o")
    assert not r.ok and "order_total" in r.errors[0]


def test_rejects_ambiguous_column(v):
    r = v.validate("SELECT order_id FROM orders o JOIN order_items oi ON o.order_id = oi.order_id")
    assert not r.ok


def test_rejects_syntax_error(v):
    r = v.validate("SELEC COUNT(*) FROM orders")
    assert not r.ok


def test_warns_on_unusual_join(v):
    r = v.validate("SELECT COUNT(*) FROM orders o JOIN customers c ON o.order_id = c.customer_id")
    assert r.ok and any("Unusual join" in w for w in r.warnings)


def test_known_join_has_no_warning(v):
    r = v.validate("SELECT COUNT(*) FROM orders o JOIN customers c ON o.customer_id = c.customer_id")
    assert r.ok and not r.warnings


def test_warns_on_cartesian_join(v):
    r = v.validate("SELECT COUNT(*) FROM orders o JOIN customers c")
    assert any("no ON condition" in w for w in r.warnings)


def test_readonly_connection_blocks_writes(fixture_db):
    import sqlite3
    conn = connect_readonly(fixture_db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM orders")


def test_validator_accepts_every_golden_reference_query(v):
    """No false rejections: every hand-written correct query must pass validation without warnings."""
    from src.evaluate import load_golden

    for g in load_golden():
        r = v.validate(g["reference_sql"])
        assert r.ok, (g["id"], r.errors)
        assert not r.warnings, (g["id"], r.warnings)
