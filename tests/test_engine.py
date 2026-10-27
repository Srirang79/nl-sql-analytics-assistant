"""Engine tests use a scripted fake LLM, so they are deterministic and need no API key."""
import pytest

from src import config
from src.database import QueryTimeout, connect_readonly, run_query
from src.engine import SQLAssistant, extract_sql
from src.llm import DemoClient
from src.schema import build_schema_prompt


class FakeLLM:
    """Returns scripted replies in order; records what it was asked."""
    name = "fake"

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def complete(self, system, messages, max_tokens=1200):
        self.calls.append((system, messages))
        return self.replies.pop(0) if self.replies else "Explanation."


def sql(s):
    return f"```sql\n{s}\n```"


def test_extract_sql_variants():
    assert extract_sql("```sql\nSELECT 1\n```") == ("sql", "SELECT 1")
    assert extract_sql("Here you go:\n```\nSELECT 2\n```")[1] == "SELECT 2"
    assert extract_sql("WITH a AS (SELECT 1) SELECT * FROM a")[0] == "sql"
    assert extract_sql("UNANSWERABLE: no such data") == ("unanswerable", "no such data")
    assert extract_sql("CLARIFY: which year?") == ("clarify", "which year?")
    assert extract_sql("I think you want revenue")[0] == "unanswerable"


def test_happy_path(fixture_db):
    llm = FakeLLM(sql("SELECT COUNT(*) AS n FROM orders"), "There are 160 orders.")
    res = SQLAssistant(fixture_db, llm).ask("How many orders?")
    assert res.status == "ok" and res.attempts == 1 and res.df.iloc[0, 0] == 160
    assert res.explanation == "There are 160 orders."


def test_schema_prompt_is_sent_to_the_model(fixture_db):
    llm = FakeLLM(sql("SELECT 1 AS x FROM orders LIMIT 1"))
    SQLAssistant(fixture_db, llm).ask("anything")
    system = llm.calls[0][0]
    for needle in ("order_purchase_timestamp", "orders.customer_id = customers.customer_id", "delivered", "Business rules", "strftime"):
        assert needle in system


def test_repairs_hallucinated_column(fixture_db):
    llm = FakeLLM(sql("SELECT SUM(order_total) FROM orders"), sql("SELECT COUNT(*) AS n FROM orders"), "ok")
    res = SQLAssistant(fixture_db, llm).ask("count orders")
    assert res.status == "ok" and res.attempts == 2
    assert "order_total" in res.trace[0]
    repair_msg = llm.calls[1][1][-1]["content"]          # the error was fed back to the model
    assert "order_total" in repair_msg


def test_blocks_destructive_sql_and_gives_up(fixture_db):
    llm = FakeLLM(*[sql("DROP TABLE orders")] * 5)
    a = SQLAssistant(fixture_db, llm, max_repairs=2)
    res = a.ask("delete everything")
    assert res.status == "error" and res.attempts == 3 and res.df is None
    assert a.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 160   # table untouched


def test_clarify_and_unanswerable_pass_through(fixture_db):
    a = SQLAssistant(fixture_db, FakeLLM("CLARIFY: which year?", "UNANSWERABLE: we have no weather data"))
    r1, r2 = a.ask("revenue"), a.ask("will it rain?")
    assert (r1.status, r1.message) == ("clarify", "which year?")
    assert r2.status == "unanswerable"


def test_follow_up_uses_history(fixture_db):
    llm = FakeLLM(sql("SELECT COUNT(*) AS n FROM orders"), "e1", sql("SELECT COUNT(*) AS n FROM customers"), "e2")
    a = SQLAssistant(fixture_db, llm)
    first = a.ask("How many orders?")
    a.ask("and customers?", history=[first])
    sent = llm.calls[2][1]
    assert sent[0]["content"] == "Question: How many orders?" and "COUNT(*) AS n FROM orders" in sent[1]["content"]
    assert sent[-1]["content"] == "Question: and customers?"


def test_llm_failure_is_reported_not_raised(fixture_db):
    class Boom:
        name = "boom"

        def complete(self, *a, **k):
            raise RuntimeError("401 invalid key")

    res = SQLAssistant(fixture_db, Boom()).ask("hi")
    assert res.status == "error" and "401 invalid key" in res.message


def test_query_timeout(fixture_db):
    conn = connect_readonly(fixture_db)
    slow = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c WHERE x < 50000000) "
            "SELECT COUNT(*) FROM c")
    with pytest.raises(QueryTimeout):
        run_query(conn, slow, timeout_s=0.3)


def test_row_cap(fixture_db):
    conn = connect_readonly(fixture_db)
    df, truncated, _ = run_query(conn, "SELECT * FROM order_items", max_rows=10)
    assert len(df) == 10 and truncated


def test_schema_prompt_lists_real_values(fixture_db):
    text = build_schema_prompt(connect_readonly(fixture_db))
    assert "credit_card" in text and "order_status: " in text and "covers orders placed from 2017" in text


def test_demo_client_answers_examples_only(fixture_db):
    demo = DemoClient({"How many orders are there?": "SELECT COUNT(*) AS n FROM orders"})
    a = SQLAssistant(fixture_db, demo)
    assert a.ask("how many orders are there").status == "ok"
    assert a.ask("something else entirely").status == "unanswerable"
