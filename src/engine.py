"""The assistant: question -> schema-aware prompt -> LLM -> validate -> (repair) -> execute -> explain."""
from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import config, prompts
from .database import QueryTimeout, connect_readonly, run_query
from .llm import LLMClient
from .schema import build_schema_prompt, sqlglot_schema
from .validator import SQLValidator

_SQL_BLOCK = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.S | re.I)


@dataclass
class QueryResult:
    question: str
    status: str                      # ok | clarify | unanswerable | error
    sql: str = ""
    df: pd.DataFrame | None = None
    explanation: str = ""
    message: str = ""                # clarification question / reason / error text
    warnings: list[str] = field(default_factory=list)
    attempts: int = 0                # LLM SQL-generation calls used (1 = right first time)
    truncated: bool = False
    seconds: float = 0.0
    trace: list[str] = field(default_factory=list)  # what went wrong on failed attempts (for transparency)


def extract_sql(text: str) -> tuple[str, str]:
    """Parse an LLM reply -> (kind, payload) where kind is 'sql' | 'clarify' | 'unanswerable'."""
    t = (text or "").strip()
    m = re.match(r"^(UNANSWERABLE|CLARIFY)\s*:\s*(.*)$", t, re.S | re.I)
    if m:
        return m.group(1).lower(), m.group(2).strip()
    block = _SQL_BLOCK.search(t)
    if block:
        return "sql", block.group(1).strip()
    if re.match(r"^\s*(with|select)\b", t, re.I):
        return "sql", t
    return "unanswerable", t[:300] or "The model returned no SQL."


class SQLAssistant:
    def __init__(self, db_path: Path | str, llm: LLMClient, max_repairs: int = config.MAX_REPAIR_ATTEMPTS):
        self.db_path = Path(db_path)
        self.conn = connect_readonly(self.db_path)
        self.llm = llm
        self.max_repairs = max_repairs
        self.validator = SQLValidator(sqlglot_schema(self.conn))
        self.system_prompt = prompts.SQL_SYSTEM.format(schema=build_schema_prompt(self.conn))

    # ----------------------------------------------------------------------------------------
    def ask(self, question: str, history: list[QueryResult] | None = None) -> QueryResult:
        t0 = time.monotonic()
        res = QueryResult(question=question, status="error")
        messages = self._history_messages(history) + [{"role": "user", "content": f"Question: {question}"}]
        last_problem = ""

        for attempt in range(1, self.max_repairs + 2):
            res.attempts = attempt
            try:
                reply = self.llm.complete(self.system_prompt, messages)
            except Exception as e:  # network / auth / quota
                res.message = f"The language model call failed: {e}"
                break
            kind, payload = extract_sql(reply)
            if kind != "sql":
                res.status, res.message = kind, payload
                break

            v = self.validator.validate(payload)
            res.warnings = v.warnings
            if not v.ok:
                last_problem = " ".join(v.errors)
                res.trace.append(f"Attempt {attempt}: validation failed: {last_problem}")
            else:
                try:
                    self.conn.execute("EXPLAIN QUERY PLAN " + v.sql).fetchall()  # final dry-run inside SQLite
                    df, truncated, _ = run_query(self.conn, v.sql)
                    res.status, res.sql, res.df, res.truncated = "ok", v.sql, df, truncated
                    res.explanation = self._explain(question, v.sql, df, truncated)
                    break
                except QueryTimeout as e:
                    last_problem = str(e)
                    res.trace.append(f"Attempt {attempt}: {last_problem}")
                except sqlite3.Error as e:
                    last_problem = f"SQLite error: {e}"
                    res.trace.append(f"Attempt {attempt}: {last_problem}")
                res.sql = v.sql

            if attempt > self.max_repairs:
                res.status, res.message = "error", f"Could not produce a valid query after {attempt} attempts. Last problem: {last_problem}"
                break
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": prompts.REPAIR.format(problem=last_problem)}]

        if res.status == "error" and not res.message:
            res.message = last_problem or "Something went wrong."
        res.seconds = time.monotonic() - t0
        return res

    # ----------------------------------------------------------------------------------------
    @staticmethod
    def _history_messages(history: list[QueryResult] | None) -> list[dict]:
        msgs: list[dict] = []
        for h in (history or [])[-config.HISTORY_TURNS:]:
            if h.status == "ok" and h.sql:
                msgs.append({"role": "user", "content": f"Question: {h.question}"})
                msgs.append({"role": "assistant", "content": f"```sql\n{h.sql}\n```"})
        return msgs

    def _explain(self, question: str, sql: str, df: pd.DataFrame, truncated: bool) -> str:
        shown = df.head(config.MAX_ROWS_FOR_EXPLANATION)
        user = prompts.EXPLAIN_USER.format(
            question=question, sql=sql, n_rows=len(df),
            trunc=", truncated at the row limit" if truncated else "",
            table=shown.to_csv(index=False) if len(df) else "(no rows)",
        )
        try:
            return self.llm.complete(prompts.EXPLAIN_SYSTEM, [{"role": "user", "content": user}], max_tokens=400)
        except Exception as e:
            return f"(Could not generate an explanation: {e})"
