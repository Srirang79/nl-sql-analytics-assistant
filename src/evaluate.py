"""Evaluate the assistant on a golden question set (execution accuracy).

For every question in evals/golden_questions.json the assistant's SQL is executed and its RESULT is
compared with the result of a hand-written reference query. Comparing results (not SQL text) is the
standard "execution accuracy" metric: many different queries are correct.

A generated result counts as a match when, ignoring row order (unless the reference is ordered) and
column names / order, it has the same rows. Extra columns are allowed (e.g. the model also returns a
count next to a name) as long as every reference column is present.

Usage:
    python -m src.evaluate --reference-only            # no LLM: check the reference SQL runs on this DB
    python -m src.evaluate --provider anthropic        # full evaluation (needs an API key)
    python -m src.evaluate --provider openai --model gpt-4o-mini --limit 10
    python -m src.evaluate --provider anthropic --update-readme   # also paste the table into README.md
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from . import config
from .database import connect_readonly, run_query
from .engine import QueryResult, SQLAssistant
from .llm import make_client


def load_golden(path: Path = config.GOLDEN_FILE) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _norm_value(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ("n", None)
    if isinstance(v, (int, float)):
        return ("n", round(float(v), 1))
    return ("s", str(v).strip().lower())


def _column_values(df: pd.DataFrame, i: int, ordered: bool):
    vals = [_norm_value(v) for v in df.iloc[:, i].tolist()]
    return vals if ordered else sorted(vals, key=repr)


def results_match(ref: pd.DataFrame, got: pd.DataFrame, ordered: bool = False) -> bool:
    """True if ``got`` contains every reference column (same values, same row count)."""
    if len(ref) != len(got):
        return False
    if ref.shape[1] == 0:
        return True
    got_cols = [_column_values(got, j, ordered) for j in range(got.shape[1])]
    for i in range(ref.shape[1]):
        want = _column_values(ref, i, ordered)
        if want not in got_cols:
            return False
    return True


def run_reference_only(db: Path, golden: list[dict]) -> int:
    conn = connect_readonly(db)
    bad = 0
    print(f"Checking {len(golden)} reference queries against {db}\n")
    for g in golden:
        try:
            df, _, secs = run_query(conn, g["reference_sql"])
            flag = "ok " if len(df) else "EMPTY"
            bad += 0 if len(df) else 1
            first = df.iloc[0].to_dict() if len(df) else {}
            print(f"[{flag}] {g['id']} {g['category']:<10} rows={len(df):<4} {secs*1000:5.0f} ms  {str(first)[:90]}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"[FAIL] {g['id']}: {e}")
    print(f"\n{len(golden) - bad}/{len(golden)} reference queries ran and returned rows.")
    return bad


def evaluate(db: Path, assistant: SQLAssistant, golden: list[dict], verbose: bool = True) -> dict:
    ref_conn = connect_readonly(db)
    rows = []
    for g in golden:
        ref_df, _, _ = run_query(ref_conn, g["reference_sql"])
        t0 = time.monotonic()
        res: QueryResult = assistant.ask(g["question"])
        ok_exec = res.status == "ok"
        correct = bool(ok_exec and results_match(ref_df, res.df, g.get("ordered", False)))
        rows.append(dict(id=g["id"], category=g["category"], question=g["question"], status=res.status,
                         correct=correct, attempts=res.attempts, first_try=ok_exec and res.attempts == 1,
                         seconds=round(time.monotonic() - t0, 2), generated_sql=res.sql, note=res.message or "; ".join(res.trace)))
        if verbose:
            print(f"[{'PASS' if correct else 'FAIL'}] {g['id']} ({res.attempts} attempt(s), {res.status}) {g['question'][:70]}")

    n = len(rows)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["correct"])
    summary = dict(
        model=assistant.llm.name, questions=n,
        execution_accuracy=sum(r["correct"] for r in rows) / n,
        valid_sql_rate=sum(r["status"] == "ok" for r in rows) / n,
        first_attempt_rate=sum(r["first_try"] for r in rows) / n,
        mean_attempts=statistics.mean(r["attempts"] for r in rows),
        median_seconds=statistics.median(r["seconds"] for r in rows),
        by_category={c: (sum(v), len(v)) for c, v in by_cat.items()},
        failures=Counter(r["status"] for r in rows if not r["correct"]),
    )
    return dict(summary=summary, rows=rows)


def to_markdown(summary: dict) -> str:
    s = summary
    lines = [
        f"Model: `{s['model']}` · {s['questions']} questions · execution accuracy compared with reference SQL results",
        "",
        "| Metric | Result |", "|---|---:|",
        f"| **Execution accuracy** (result matches reference) | **{s['execution_accuracy']:.0%}** |",
        f"| Valid, executable SQL produced | {s['valid_sql_rate']:.0%} |",
        f"| Correct on first attempt (no repair loop) | {s['first_attempt_rate']:.0%} |",
        f"| Mean LLM attempts per question | {s['mean_attempts']:.2f} |",
        f"| Median time per question | {s['median_seconds']:.1f}s |",
        "", "| Question type | Correct |", "|---|---:|",
    ]
    lines += [f"| {c} | {a}/{b} |" for c, (a, b) in s["by_category"].items()]
    return "\n".join(lines)


def update_readme(markdown: str, readme: Path = config.ROOT / "README.md") -> None:
    text = readme.read_text(encoding="utf-8")
    block = f"<!-- EVAL_RESULTS_START -->\n{markdown}\n<!-- EVAL_RESULTS_END -->"
    new = re.sub(r"<!-- EVAL_RESULTS_START -->.*?<!-- EVAL_RESULTS_END -->", lambda _: block, text, flags=re.S)
    readme.write_text(new, encoding="utf-8")
    print(f"Updated {readme}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(config.default_db_path()))
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument("--reference-only", action="store_true", help="just run the reference SQL (no LLM, no API key)")
    ap.add_argument("--update-readme", action="store_true")
    a = ap.parse_args()

    db = Path(a.db)
    golden = load_golden()[: a.limit]
    if a.reference_only:
        raise SystemExit(1 if run_reference_only(db, golden) else 0)

    try:
        llm = make_client(a.provider, model=a.model)
    except ValueError as e:  # e.g. missing API key
        raise SystemExit(f"{e}\nTip: python -m src.evaluate --reference-only needs no key.")
    assistant = SQLAssistant(db, llm)
    out = evaluate(db, assistant, golden)
    md = to_markdown(out["summary"])
    print("\n" + md)
    config.EVAL_DIR.mkdir(exist_ok=True)
    (config.EVAL_DIR / "results.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    (config.EVAL_DIR / "results.md").write_text(md + "\n", encoding="utf-8")
    if a.update_readme:
        update_readme(md)


if __name__ == "__main__":
    main()
