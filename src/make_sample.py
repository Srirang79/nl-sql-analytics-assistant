"""Cut a small, self-contained sample database out of the full Olist database.

The full database is ~100 MB and is not committed to Git. The sample (a few thousand orders plus every
row that belongs to them) is a few MB, so it CAN be committed: anyone cloning the repo gets a working app.

    python -m src.make_sample --orders 4000
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from . import config


def make_sample(src: Path = config.FULL_DB, dst: Path = config.SAMPLE_DB, n_orders: int = 4000) -> dict[str, int]:
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"{src} not found. Build it first with: python -m src.data_loader")
    if dst.exists():
        dst.unlink()
    conn = sqlite3.connect(dst)
    conn.execute("ATTACH DATABASE ? AS full", (str(src),))

    # the earliest-by-hash slice would bias dates; take every k-th order instead so all months are covered
    total = conn.execute("SELECT COUNT(*) FROM full.orders").fetchone()[0]
    step = max(1, total // n_orders)
    conn.execute(f"CREATE TEMP TABLE keep AS SELECT order_id FROM (SELECT order_id, ROW_NUMBER() OVER (ORDER BY order_id) AS rn FROM full.orders) WHERE rn % {step} = 0")

    def copy(table: str, where: str) -> None:
        ddl = conn.execute("SELECT sql FROM full.sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        if ddl is None:
            return
        conn.execute(ddl[0])
        conn.execute(f"INSERT INTO main.{table} SELECT * FROM full.{table} WHERE {where}")

    copy("orders", "order_id IN (SELECT order_id FROM keep)")
    copy("order_items", "order_id IN (SELECT order_id FROM keep)")
    copy("order_payments", "order_id IN (SELECT order_id FROM keep)")
    copy("order_reviews", "order_id IN (SELECT order_id FROM keep)")
    copy("customers", "customer_id IN (SELECT customer_id FROM main.orders)")
    copy("products", "product_id IN (SELECT product_id FROM main.order_items)")
    copy("sellers", "seller_id IN (SELECT seller_id FROM main.order_items)")
    copy("category_translation", "1=1")
    conn.commit()

    counts = {t: conn.execute(f"SELECT COUNT(*) FROM main.{t}").fetchone()[0]
              for (t,) in conn.execute("SELECT name FROM main.sqlite_master WHERE type='table'").fetchall()}
    conn.execute("DETACH DATABASE full")
    conn.execute("VACUUM")
    conn.close()
    print(f"Sample database written to {dst} ({dst.stat().st_size / 1e6:.1f} MB): {counts}")
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(config.FULL_DB))
    ap.add_argument("--dst", default=str(config.SAMPLE_DB))
    ap.add_argument("--orders", type=int, default=4000)
    a = ap.parse_args()
    try:
        make_sample(Path(a.src), Path(a.dst), a.orders)
    except FileNotFoundError as e:
        raise SystemExit(str(e))
