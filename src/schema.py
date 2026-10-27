"""Schema introspection + curated business documentation, turned into the LLM prompt context.

The model only writes good SQL if it knows (1) the exact tables and columns, (2) how they join,
(3) what the values look like and (4) the business rules (e.g. "revenue excludes freight",
"customer_id is per order, customer_unique_id is per person"). This module assembles all four
from the live database plus the hand-written notes below.
"""
from __future__ import annotations

import sqlite3

TABLE_DOCS: dict[str, dict] = {
    "orders": {
        "desc": "One row per customer order (the central table).",
        "cols": {
            "order_id": "primary key",
            "customer_id": "FK -> customers.customer_id (one per order, NOT per person)",
            "order_status": "delivered, shipped, canceled, unavailable, invoiced, processing, created, approved",
            "order_purchase_timestamp": "when the order was placed (text 'YYYY-MM-DD HH:MM:SS')",
            "order_approved_at": "payment approval time",
            "order_delivered_carrier_date": "handed to the carrier",
            "order_delivered_customer_date": "delivered to the customer (NULL if not delivered)",
            "order_estimated_delivery_date": "delivery date promised to the customer",
        },
    },
    "order_items": {
        "desc": "One row per item in an order. An order with 3 items has 3 rows.",
        "cols": {
            "order_id": "FK -> orders.order_id",
            "order_item_id": "item sequence number within the order (1, 2, 3...)",
            "product_id": "FK -> products.product_id",
            "seller_id": "FK -> sellers.seller_id",
            "shipping_limit_date": "seller's deadline to hand the item to the carrier",
            "price": "item price in BRL (excludes freight)",
            "freight_value": "shipping cost for this item in BRL",
        },
    },
    "order_payments": {
        "desc": "One row per payment on an order (an order can be paid with several methods/instalments).",
        "cols": {
            "order_id": "FK -> orders.order_id",
            "payment_sequential": "payment sequence number within the order",
            "payment_type": "credit_card, boleto, voucher, debit_card, not_defined",
            "payment_installments": "number of instalments chosen",
            "payment_value": "amount paid in BRL (includes freight)",
        },
    },
    "order_reviews": {
        "desc": "Customer satisfaction reviews, normally one per order.",
        "cols": {
            "review_id": "review identifier",
            "order_id": "FK -> orders.order_id",
            "review_score": "1 (worst) to 5 (best)",
            "review_comment_title": "optional free text (Portuguese)",
            "review_comment_message": "optional free text (Portuguese)",
            "review_creation_date": "survey sent date",
            "review_answer_timestamp": "survey answered time",
        },
    },
    "customers": {
        "desc": "Customer records. customer_id is generated per order; customer_unique_id identifies a real person.",
        "cols": {
            "customer_id": "primary key, FK target of orders.customer_id",
            "customer_unique_id": "stable id of a person (use COUNT(DISTINCT customer_unique_id) to count customers)",
            "customer_zip_code_prefix": "first 5 digits of the zip code",
            "customer_city": "city (lower-case, no accents)",
            "customer_state": "2-letter Brazilian state code, e.g. SP, RJ, MG",
        },
    },
    "products": {
        "desc": "Product catalogue.",
        "cols": {
            "product_id": "primary key",
            "product_category_name": "category in Portuguese (join category_translation for English)",
            "product_name_length": "characters in the product name",
            "product_description_length": "characters in the description",
            "product_photos_qty": "number of photos",
            "product_weight_g": "weight in grams",
            "product_length_cm": "package length", "product_height_cm": "package height",
            "product_width_cm": "package width",
        },
    },
    "sellers": {
        "desc": "Marketplace sellers.",
        "cols": {
            "seller_id": "primary key",
            "seller_zip_code_prefix": "first 5 digits of the zip code",
            "seller_city": "city", "seller_state": "2-letter state code",
        },
    },
    "category_translation": {
        "desc": "Maps Portuguese product category names to English.",
        "cols": {
            "product_category_name": "Portuguese name (joins products.product_category_name)",
            "product_category_name_english": "English name",
        },
    },
    "geolocation": {
        "desc": "Zip-code prefix to latitude/longitude (many rows per prefix). Rarely needed.",
        "cols": {
            "geolocation_zip_code_prefix": "zip code prefix", "geolocation_lat": "latitude",
            "geolocation_lng": "longitude", "geolocation_city": "city", "geolocation_state": "state",
        },
    },
}

# (table_a, col_a, table_b, col_b): the only joins the data model supports
RELATIONSHIPS = [
    ("orders", "customer_id", "customers", "customer_id"),
    ("order_items", "order_id", "orders", "order_id"),
    ("order_items", "product_id", "products", "product_id"),
    ("order_items", "seller_id", "sellers", "seller_id"),
    ("order_payments", "order_id", "orders", "order_id"),
    ("order_reviews", "order_id", "orders", "order_id"),
    ("products", "product_category_name", "category_translation", "product_category_name"),
    ("customers", "customer_zip_code_prefix", "geolocation", "geolocation_zip_code_prefix"),
    ("sellers", "seller_zip_code_prefix", "geolocation", "geolocation_zip_code_prefix"),
]

# Columns whose distinct values are listed in the prompt so the model filters on real spellings
ENUM_COLUMNS = [
    ("orders", "order_status"), ("order_payments", "payment_type"),
    ("order_reviews", "review_score"), ("customers", "customer_state"), ("sellers", "seller_state"),
]

BUSINESS_RULES = """\
- Currency is Brazilian reais (BRL). "Revenue"/"sales" = SUM(order_items.price) (freight excluded); \
use order_payments.payment_value for "total paid" (includes freight).
- Unless the question says otherwise, revenue and delivery analyses count only orders with \
order_status = 'delivered'.
- To count people use COUNT(DISTINCT customers.customer_unique_id); customers.customer_id is created per order.
- order_items has one row per item: joining it to orders multiplies order rows. Aggregate before joining \
or count DISTINCT order_id when counting orders.
- Dates are TEXT 'YYYY-MM-DD HH:MM:SS'. Use strftime('%Y-%m', col) for months, strftime('%Y', col) for years, \
julianday(a) - julianday(b) for day differences. Never use date('now'): the data is historical ("last year" or \
"latest month" means relative to the dataset's own date range given below).
- Category names are Portuguese; join category_translation to show English names.
- Order only by columns you select or by aggregates. Return readable column aliases."""


def introspect(conn: sqlite3.Connection) -> dict[str, list[tuple[str, str]]]:
    """{table: [(column, declared_type), ...]} for every user table in the database."""
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    return {t: [(r[1], r[2] or "TEXT") for r in conn.execute(f'PRAGMA table_info("{t}")')] for t in tables}


def sqlglot_schema(conn: sqlite3.Connection) -> dict[str, dict[str, str]]:
    return {t: {c: ty for c, ty in cols} for t, cols in introspect(conn).items()}


def known_join_pairs() -> set[frozenset]:
    """Allowed (table.col, table.col) equalities, as unordered pairs, for the join sanity check."""
    return {frozenset({f"{a}.{ca}", f"{b}.{cb}"}) for a, ca, b, cb in RELATIONSHIPS}


def _distinct_values(conn, table, col, limit=40):
    rows = conn.execute(f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL ORDER BY 1 LIMIT {limit + 1}').fetchall()
    vals = [str(r[0]) for r in rows]
    return vals[:limit], len(vals) > limit


def date_range(conn) -> tuple[str, str] | None:
    try:
        lo, hi = conn.execute("SELECT MIN(order_purchase_timestamp), MAX(order_purchase_timestamp) FROM orders").fetchone()
        return (lo[:10], hi[:10]) if lo and hi else None
    except sqlite3.Error:
        return None


def build_schema_prompt(conn: sqlite3.Connection, with_samples: bool = True) -> str:
    """The schema block placed in the system prompt."""
    schema = introspect(conn)
    out = ["## Tables and columns"]
    for table, cols in schema.items():
        doc = TABLE_DOCS.get(table, {})
        out.append(f"\n### {table}" + (f": {doc['desc']}" if doc.get("desc") else ""))
        for col, ty in cols:
            note = doc.get("cols", {}).get(col)
            out.append(f"- {col} ({ty})" + (f": {note}" if note else ""))
        if with_samples:
            cur = conn.execute(f'SELECT * FROM "{table}" LIMIT 2')
            names = [d[0] for d in cur.description]
            for row in cur.fetchall():
                shown = {n: (str(v)[:28] + "…" if v is not None and len(str(v)) > 28 else v) for n, v in zip(names, row)}
                out.append(f"  e.g. {shown}")

    out.append("\n## Joins (use only these)")
    for a, ca, b, cb in RELATIONSHIPS:
        if a in schema and b in schema:
            out.append(f"- {a}.{ca} = {b}.{cb}")

    enum_lines = []
    for table, col in ENUM_COLUMNS:
        if table in schema:
            vals, more = _distinct_values(conn, table, col)
            if vals:
                enum_lines.append(f"- {table}.{col}: {', '.join(vals)}" + (" …" if more else ""))
    if enum_lines:
        out.append("\n## Allowed values (exact spelling)")
        out += enum_lines

    rng = date_range(conn)
    out.append("\n## Business rules")
    out.append(BUSINESS_RULES)
    if rng:
        out.append(f"- The data covers orders placed from {rng[0]} to {rng[1]}.")
    return "\n".join(out)
