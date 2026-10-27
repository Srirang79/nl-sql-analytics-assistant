"""Query validation: everything a generated SQL string must pass BEFORE it touches the database.

Layers (each one fails with a message that is fed back to the LLM so it can repair the query):
  1. Parses as exactly ONE SQLite statement
  2. Is a read-only SELECT / WITH...SELECT / UNION  (no INSERT, UPDATE, DELETE, DROP, PRAGMA, ATTACH ...)
  3. Only references tables that exist (CTE names allowed), with "did you mean" suggestions
  4. Every column exists and is unambiguous (sqlglot qualifies the query against the real schema)
  5. Warns about joins that are not in the data model, and joins with no condition (cartesian products)
  6. Adds a LIMIT when the model forgot one
The executor adds two more guards at run time: a read-only connection and a query timeout.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError, SqlglotError
from sqlglot.optimizer.qualify import qualify

from . import config
from .schema import known_join_pairs

_FORBIDDEN_NAMES = (
    "Insert", "Update", "Delete", "Drop", "Create", "Alter", "Command", "Pragma", "Attach", "Detach",
    "Merge", "Transaction", "Commit", "Rollback", "Use", "Set", "TruncateTable", "Copy",
)
FORBIDDEN_NODES = tuple(getattr(exp, n) for n in _FORBIDDEN_NAMES if hasattr(exp, n))
SYSTEM_TABLE = re.compile(r"^sqlite_", re.I)


@dataclass
class ValidationResult:
    ok: bool
    sql: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)


class SQLValidator:
    def __init__(self, schema: dict[str, dict[str, str]], max_rows: int = config.MAX_ROWS_RETURNED):
        self.schema = schema
        self.tables_lower = {t.lower(): t for t in schema}
        self.max_rows = max_rows
        self.join_pairs = known_join_pairs()

    # ------------------------------------------------------------------------------------------
    def validate(self, sql: str) -> ValidationResult:
        sql = (sql or "").strip().rstrip(";").strip()
        if not sql:
            return ValidationResult(False, sql, ["Empty query."])

        # 1) exactly one statement that parses
        try:
            statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
        except ParseError as e:
            return ValidationResult(False, sql, [f"SQL syntax error: {str(e).splitlines()[0][:200]}"])
        except SqlglotError as e:
            return ValidationResult(False, sql, [f"Could not parse the SQL: {str(e)[:200]}"])
        if len(statements) != 1:
            return ValidationResult(False, sql, ["Only a single SQL statement is allowed (found several)."])
        stmt = statements[0]

        # 2) read-only
        if not isinstance(stmt, (exp.Select, exp.Union)):
            return ValidationResult(False, sql, [f"Only SELECT queries are allowed (got {type(stmt).__name__.upper()})."])
        bad = next(stmt.find_all(*FORBIDDEN_NODES), None) if FORBIDDEN_NODES else None
        if bad is not None:
            return ValidationResult(False, sql, [f"Forbidden operation inside the query: {type(bad).__name__.upper()}."])

        errors: list[str] = []
        warnings: list[str] = []

        # 3) tables
        cte_names = {c.alias_or_name.lower() for c in stmt.find_all(exp.CTE)}
        used_tables: list[str] = []
        for t in stmt.find_all(exp.Table):
            name = t.name
            if not name or name.lower() in cte_names:
                continue
            if SYSTEM_TABLE.match(name):
                errors.append(f"Table '{name}' is a system table and cannot be queried.")
            elif name.lower() not in self.tables_lower:
                hint = difflib.get_close_matches(name.lower(), list(self.tables_lower), n=2, cutoff=0.5)
                errors.append(f"Unknown table '{name}'." + (f" Did you mean {' or '.join(self.tables_lower[h] for h in hint)}?" if hint else
                              f" Available tables: {', '.join(self.schema)}."))
            else:
                used_tables.append(self.tables_lower[name.lower()])
        if errors:
            return ValidationResult(False, sql, errors, warnings, used_tables)

        # 4) columns (qualify against the real schema; catches hallucinated and ambiguous columns)
        try:
            qualify(stmt.copy(), schema=self.schema, dialect="sqlite", validate_qualify_columns=True)
        except OptimizeError as e:
            errors.append(self._column_error(str(e)))
        except SqlglotError as e:  # parser/optimizer quirk: do not block, the DB dry-run is the final check
            warnings.append(f"Column check skipped: {str(e)[:120]}")
        if errors:
            return ValidationResult(False, sql, errors, warnings, used_tables)

        # 5) join sanity
        warnings += self._join_warnings(stmt)

        # 6) LIMIT
        limit = stmt.args.get("limit")
        if limit is None:
            sql = f"{sql}\nLIMIT {self.max_rows}"
        return ValidationResult(True, sql, [], warnings, sorted(set(used_tables)))

    # ------------------------------------------------------------------------------------------
    def _column_error(self, msg: str) -> str:
        m = re.search(r"(?:Unknown column|Column) '?([\w.]+)'?", msg)
        col = m.group(1).split(".")[-1] if m else None
        if col:
            all_cols = {c for cols in self.schema.values() for c in cols}
            hint = difflib.get_close_matches(col, all_cols, n=3, cutoff=0.6)
            base = f"Unknown or ambiguous column '{col}'."
            return base + (f" Similar existing columns: {', '.join(hint)}." if hint else "") + " Check the schema and qualify columns with table aliases."
        return f"Column problem: {msg[:200]}"

    def _join_warnings(self, stmt: exp.Expression) -> list[str]:
        out: list[str] = []
        alias_to_table: dict[str, str] = {}
        for t in stmt.find_all(exp.Table):
            if t.name.lower() in self.tables_lower:
                alias_to_table[(t.alias_or_name or t.name).lower()] = self.tables_lower[t.name.lower()]

        for join in stmt.find_all(exp.Join):
            kind = str(join.args.get("kind") or "").upper()
            on = join.args.get("on")
            trivial_on = isinstance(on, exp.Boolean) and bool(on.this)  # sqlglot reads "JOIN t" (no ON) as ON TRUE
            if (on is None or trivial_on) and not join.args.get("using"):
                if kind == "CROSS":  # explicit CROSS JOIN or old-style comma join: fine only if a WHERE links the tables
                    sel = join.find_ancestor(exp.Select)
                    if sel is not None and not sel.args.get("where"):
                        out.append("Tables are cross-joined with no WHERE/ON condition (possible cartesian product).")
                else:
                    out.append("A join has no ON condition (possible cartesian product).")
            if on is None or trivial_on:
                continue
            for eq in on.find_all(exp.EQ):
                l, r = eq.left, eq.right
                if isinstance(l, exp.Column) and isinstance(r, exp.Column):
                    lt = alias_to_table.get((l.table or "").lower())
                    rt = alias_to_table.get((r.table or "").lower())
                    if lt and rt and lt != rt:
                        pair = frozenset({f"{lt}.{l.name}", f"{rt}.{r.name}"})
                        if pair not in self.join_pairs:
                            out.append(f"Unusual join {lt}.{l.name} = {rt}.{r.name}: not a known relationship in this data model.")
        return list(dict.fromkeys(out))
