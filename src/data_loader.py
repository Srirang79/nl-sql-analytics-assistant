"""Load the Kaggle "Brazilian E-Commerce Public Dataset by Olist" CSVs into a SQLite database.

Dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce  (CC BY-NC-SA 4.0)
About 100,000 real (anonymised) orders placed on Brazilian marketplaces between 2016 and 2018,
split over 9 linked tables.

Usage:
    python -m src.data_loader                      # CSVs (or archive.zip) in data/raw/
    python -m src.data_loader --zip ~/Downloads/archive.zip
    python -m src.data_loader --include-geolocation   # also load the 1M-row zip-code table (large, rarely useful)

Tables are renamed to short names and two misspelt columns are fixed (see RENAMES). Dates stay as
ISO text ('YYYY-MM-DD HH:MM:SS'), which SQLite's strftime()/julianday() read directly.
"""
from __future__ import annotations

import argparse
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd

from . import config

# Kaggle file name -> table name
FILES = {
    "olist_orders_dataset.csv": "orders",
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_payments_dataset.csv": "order_payments",
    "olist_order_reviews_dataset.csv": "order_reviews",
    "olist_customers_dataset.csv": "customers",
    "olist_products_dataset.csv": "products",
    "olist_sellers_dataset.csv": "sellers",
    "product_category_name_translation.csv": "category_translation",
    "olist_geolocation_dataset.csv": "geolocation",
}
OPTIONAL = {"geolocation"}

# The source file misspells "length" as "lenght"
RENAMES = {
    "products": {
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length",
    }
}

INDEXES = [
    ("orders", "order_id"), ("orders", "customer_id"), ("orders", "order_status"),
    ("orders", "order_purchase_timestamp"),
    ("order_items", "order_id"), ("order_items", "product_id"), ("order_items", "seller_id"),
    ("order_payments", "order_id"), ("order_reviews", "order_id"),
    ("customers", "customer_id"), ("customers", "customer_unique_id"), ("customers", "customer_state"),
    ("products", "product_id"), ("products", "product_category_name"),
    ("sellers", "seller_id"), ("sellers", "seller_state"),
    ("category_translation", "product_category_name"),
]


def _extract_zip_if_needed(raw_dir: Path, zip_path: Path | None) -> None:
    candidates = [zip_path] if zip_path else sorted(raw_dir.glob("*.zip"))
    if all((raw_dir / f).exists() for f in FILES if FILES[f] not in OPTIONAL):
        return
    for z in candidates:
        if z and Path(z).exists():
            print(f"Extracting {z} -> {raw_dir}")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(raw_dir)
            return


def _missing_message(missing: list[str], raw_dir: Path) -> str:
    return (
        "Missing dataset files in " + str(raw_dir) + ":\n  - " + "\n  - ".join(missing) + "\n\n"
        "Download the dataset from Kaggle (free account needed):\n"
        f"  {config.KAGGLE_URL}\n"
        "Click 'Download', then either put the extracted CSVs or the downloaded archive.zip into data/raw/ "
        "and re-run, or pass --zip path/to/archive.zip.\n"
        "Or with the Kaggle CLI:\n"
        f"  kaggle datasets download -d {config.KAGGLE_SLUG} -p data/raw --unzip"
    )


def load(raw_dir: Path = config.RAW_DIR, db_path: Path = config.FULL_DB, zip_path: Path | None = None,
         include_geolocation: bool = False, force: bool = False) -> dict[str, int]:
    raw_dir, db_path = Path(raw_dir), Path(db_path)
    _extract_zip_if_needed(raw_dir, Path(zip_path) if zip_path else None)

    wanted = {f: t for f, t in FILES.items() if t not in OPTIONAL or include_geolocation}
    missing = [f for f in wanted if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(_missing_message(missing, raw_dir))

    if db_path.exists():
        if not force:
            raise FileExistsError(f"{db_path} already exists. Use --force to rebuild it.")
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    with sqlite3.connect(db_path) as conn:
        for fname, table in wanted.items():
            total = 0
            for i, chunk in enumerate(pd.read_csv(raw_dir / fname, chunksize=50_000)):
                chunk = chunk.rename(columns=RENAMES.get(table, {}))
                chunk.to_sql(table, conn, if_exists="replace" if i == 0 else "append", index=False)
                total += len(chunk)
            counts[table] = total
            print(f"  loaded {table:<22s} {total:>9,d} rows")
        existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table, col in INDEXES:
            if table in existing:
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{col} ON {table}({col})")
        conn.execute("ANALYZE")
        conn.commit()
    print(f"Database written to {db_path} ({db_path.stat().st_size / 1e6:.1f} MB)")
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=str(config.RAW_DIR))
    ap.add_argument("--db", default=str(config.FULL_DB))
    ap.add_argument("--zip", default=None, help="path to the archive.zip downloaded from Kaggle")
    ap.add_argument("--include-geolocation", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebuild the database if it exists")
    a = ap.parse_args()
    try:
        load(Path(a.raw_dir), Path(a.db), Path(a.zip) if a.zip else None, a.include_geolocation, a.force)
    except (FileNotFoundError, FileExistsError) as e:
        raise SystemExit(str(e))
