"""A TINY hand-built SQLite database with exactly the same schema as the real Olist tables.

It exists ONLY so the code (validator, engine, evaluator, app) can be tested without downloading
the 45 MB Kaggle dataset. It is not the project's dataset: the real data comes from Kaggle via
`python -m src.data_loader`.
"""
from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DDL = """
CREATE TABLE customers (customer_id TEXT, customer_unique_id TEXT, customer_zip_code_prefix INTEGER, customer_city TEXT, customer_state TEXT);
CREATE TABLE orders (order_id TEXT, customer_id TEXT, order_status TEXT, order_purchase_timestamp TEXT, order_approved_at TEXT,
    order_delivered_carrier_date TEXT, order_delivered_customer_date TEXT, order_estimated_delivery_date TEXT);
CREATE TABLE order_items (order_id TEXT, order_item_id INTEGER, product_id TEXT, seller_id TEXT, shipping_limit_date TEXT, price REAL, freight_value REAL);
CREATE TABLE order_payments (order_id TEXT, payment_sequential INTEGER, payment_type TEXT, payment_installments INTEGER, payment_value REAL);
CREATE TABLE order_reviews (review_id TEXT, order_id TEXT, review_score INTEGER, review_comment_title TEXT, review_comment_message TEXT,
    review_creation_date TEXT, review_answer_timestamp TEXT);
CREATE TABLE products (product_id TEXT, product_category_name TEXT, product_name_length INTEGER, product_description_length INTEGER,
    product_photos_qty INTEGER, product_weight_g INTEGER, product_length_cm INTEGER, product_height_cm INTEGER, product_width_cm INTEGER);
CREATE TABLE sellers (seller_id TEXT, seller_zip_code_prefix INTEGER, seller_city TEXT, seller_state TEXT);
CREATE TABLE category_translation (product_category_name TEXT, product_category_name_english TEXT);
"""

CATEGORIES = [("cama_mesa_banho", "bed_bath_table"), ("beleza_saude", "health_beauty"), ("esporte_lazer", "sports_leisure"),
              ("informatica_acessorios", "computers_accessories"), ("moveis_decoracao", "furniture_decor"), ("brinquedos", "toys")]
STATES = ["SP", "RJ", "MG", "RS", "PR", "BA"]
CITIES = ["sao paulo", "rio de janeiro", "belo horizonte", "porto alegre", "curitiba", "salvador"]
STATUSES = ["delivered"] * 8 + ["canceled", "shipped"]
FMT = "%Y-%m-%d %H:%M:%S"


def build_fixture_db(path: Path | str, n_orders: int = 160, seed: int = 7) -> Path:
    path = Path(path)
    if path.exists():
        path.unlink()
    rnd = random.Random(seed)
    conn = sqlite3.connect(path)
    conn.executescript(DDL)

    conn.executemany("INSERT INTO category_translation VALUES (?,?)", CATEGORIES)
    sellers = [(f"s{i:02d}", 10000 + i, CITIES[i % 6], STATES[i % 6]) for i in range(8)]
    conn.executemany("INSERT INTO sellers VALUES (?,?,?,?)", sellers)
    products = [(f"p{i:03d}", CATEGORIES[i % 6][0], 40, 300, 2, 500 + i, 20, 10, 15) for i in range(30)]
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?)", products)

    customers, orders, items, pays, reviews = [], [], [], [], []
    for i in range(n_orders):
        cid, uid = f"c{i:04d}", f"u{i % 120:04d}"            # some people order more than once
        st = i % 6
        customers.append((cid, uid, 20000 + st, CITIES[st], STATES[st]))
        start = datetime(2017, 1, 1) + timedelta(days=rnd.randint(0, 700), hours=rnd.randint(0, 23))
        status = rnd.choice(STATUSES)
        est = start + timedelta(days=rnd.randint(12, 25))
        deliv = start + timedelta(days=rnd.randint(5, 30)) if status == "delivered" else None
        orders.append((f"o{i:04d}", cid, status, start.strftime(FMT), (start + timedelta(hours=2)).strftime(FMT),
                       (start + timedelta(days=2)).strftime(FMT) if status != "canceled" else None,
                       deliv.strftime(FMT) if deliv else None, est.strftime(FMT)))
        total = 0.0
        for k in range(1, rnd.randint(1, 3) + 1):
            price, freight = round(rnd.uniform(15, 400), 2), round(rnd.uniform(5, 40), 2)
            total += price + freight
            items.append((f"o{i:04d}", k, products[rnd.randrange(30)][0], sellers[rnd.randrange(8)][0],
                          (start + timedelta(days=3)).strftime(FMT), price, freight))
        ptype = rnd.choice(["credit_card", "credit_card", "boleto", "voucher", "debit_card"])
        pays.append((f"o{i:04d}", 1, ptype, rnd.randint(1, 6) if ptype == "credit_card" else 1, round(total, 2)))
        if status != "canceled":
            reviews.append((f"r{i:04d}", f"o{i:04d}", rnd.choice([1, 2, 3, 4, 5, 5, 5]), None, None,
                            (start + timedelta(days=20)).strftime("%Y-%m-%d 00:00:00"), (start + timedelta(days=21)).strftime(FMT)))

    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers)
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)", orders)
    conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?,?,?)", items)
    conn.executemany("INSERT INTO order_payments VALUES (?,?,?,?,?)", pays)
    conn.executemany("INSERT INTO order_reviews VALUES (?,?,?,?,?,?,?)", reviews)
    conn.commit()
    conn.close()
    return path
