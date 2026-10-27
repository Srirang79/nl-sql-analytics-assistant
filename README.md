# Natural Language SQL Analytics Assistant

**Python · LLM · SQL (SQLite) · Streamlit**

Ask a business question in plain English, such as *"Which product categories had the highest revenue in 2017?"*,
and get back the answer, a chart, the SQL behind it and a short plain-language explanation. No SQL knowledge needed.

It runs on a **real dataset**: the Kaggle [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
about 100,000 anonymised orders placed on Brazilian marketplaces in 2016–2018, spread over 9 linked tables
(orders, items, payments, reviews, customers, products, sellers, categories, geolocation).

> **The hard part of "chat with your data" is trust.** A language model that writes SQL will sometimes invent a
> column, join the wrong tables, or double-count revenue. Most of this project is the machinery that catches those mistakes
> before a wrong number reaches a user: schema-aware prompts, a multi-layer query validator, a self-repair loop,
> read-only execution, and an evaluation set that measures accuracy instead of guessing.

---

## How it works

```
 question ──► schema-aware prompt ──► LLM writes SQL ──► validator ──► read-only run ──► plain-English answer
                 (tables, joins,            ▲            (6 checks)     (row cap +        + table + chart
                  allowed values,           │                │            timeout)        + the SQL, shown
                  business rules)           └── error fed back ┘
                                              (up to 2 repairs)
```

**1. Schema-aware prompts** (`src/schema.py`, `src/prompts.py`). Every request carries: all tables and columns read live
from the database, hand-written column meanings, the *only* allowed joins, the exact spelling of categorical values
(`order_status`, `payment_type`, state codes), sample rows, the dataset's date range, and the business rules a new
analyst would need, for example *revenue excludes freight*, *`customer_id` is per order but `customer_unique_id` is per person*,
*joining order items multiplies order rows*, *dates are text, so use `strftime`, never `now`*. Two worked examples are included.

**2. Query validation** (`src/validator.py`). Before anything runs, the SQL must:
1. parse as exactly **one** SQLite statement,
2. be a read-only `SELECT`/`WITH` (no `DROP`, `DELETE`, `UPDATE`, `INSERT`, `PRAGMA`, `ATTACH`...),
3. reference only **existing tables** (with "did you mean" hints),
4. reference only **existing, unambiguous columns** (checked against the real schema using sqlglot),
5. avoid suspicious joins: it warns on joins outside the data model (e.g. `orders.order_id = customers.customer_id`) and on cartesian products,
6. get a `LIMIT` added if the model forgot one.

**3. Self-repair loop** (`src/engine.py`). If validation or execution fails, the exact error is sent back to the model
and it tries again (up to 2 repairs). The UI shows when a query was self-corrected.

**4. Safe execution** (`src/database.py`). The database is opened **read-only** at the SQLite level (so even a validator bypass
could not write), results are capped at 1,000 rows, and queries are killed after 15 seconds.

**5. Grounded explanations.** A second model call explains the result using *only* the returned rows, states the
assumptions the query made (e.g. "delivered orders only"), and says so when results are empty or truncated.

**6. Honest failure modes.** If a question cannot be answered from the data, the assistant says so. If it is genuinely
ambiguous, it asks one clarifying question. Follow-up questions ("and in 2018?") use the previous turns.

## Quick start

```bash
git clone https://github.com/Srirang79/nl-sql-analytics-assistant.git
cd nl-sql-analytics-assistant
pip install -r requirements.txt
```

**1. Get the data (about 45 MB, free Kaggle account).** Open <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>, click
**Download**, and put the downloaded `archive.zip` into `data/raw/`. Or with the Kaggle CLI:
`kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip`.

**2. Build the database** (one command; it unzips and loads the 8 core tables into SQLite and prints row counts):

```bash
python -m src.data_loader
# add --include-geolocation to also load the 1M-row zip-code table (rarely useful, makes the DB much larger)
```

**3. Run the app:**

```bash
streamlit run app.py
```

**4. Choose how it answers:**
- **Live LLM** (any question): put an API key in the sidebar, or set it first (Anthropic by default, OpenAI also supported):
  - Git Bash / Mac / Linux: `export ANTHROPIC_API_KEY="your-key"`
  - PowerShell: `$env:ANTHROPIC_API_KEY="your-key"`
  - or in `.streamlit/secrets.toml`: `ANTHROPIC_API_KEY = "your-key"`
- **Demo mode** (no key): runs the pre-written SQL for the example questions, so you can try the interface, validation, charts and
  downloads without an account. It is clearly labelled and does not understand free-form questions.

Override the model with the sidebar field or `LLM_MODEL=...`. Your key is used only for the session and is never stored or logged.

*Optional: a committable sample.* `python -m src.make_sample --orders 4000` cuts a few-MB `data/sample.db` out of the full database.
Commit it and anyone cloning the repo can run the app without downloading anything (the app uses it automatically when the full DB is absent).

## Evaluation

Reliability is measured, not assumed. `evals/golden_questions.json` holds **26 business questions** across five types
(aggregation, joins, filters, time-based analysis, multi-step), each with a hand-written reference SQL query.

The metric is **execution accuracy**: the assistant's query is *run*, and its result is compared with the reference result
(row order and column names ignored unless the question is a ranking; extra columns allowed). Comparing results rather than SQL text is the
standard approach, because many different queries are correct.

```bash
python -m src.evaluate --reference-only                 # no key needed: checks all 26 reference queries run on your database
python -m src.evaluate --provider anthropic             # full run; writes evals/results.md and results.json
python -m src.evaluate --provider anthropic --update-readme   # also fills in the table below
```

<!-- EVAL_RESULTS_START -->
_Not run yet. Run `python -m src.evaluate --provider anthropic --update-readme` with an API key and this table fills itself in._
<!-- EVAL_RESULTS_END -->

## Project structure

```
├── app.py                     # Streamlit interface (Ask tab + Data & schema tab)
├── src/
│   ├── data_loader.py         # Kaggle CSVs / archive.zip  ->  SQLite (typed, indexed)
│   ├── schema.py              # live schema + business documentation -> prompt context
│   ├── prompts.py             # SQL-generation, repair and explanation prompts
│   ├── llm.py                 # Anthropic / OpenAI clients + key-free demo client
│   ├── validator.py           # 6-layer query validation
│   ├── database.py            # read-only connection, row cap, timeout
│   ├── engine.py              # question -> SQL -> validate -> repair -> run -> explain
│   ├── evaluate.py            # execution-accuracy evaluation
│   └── make_sample.py         # small committable sample database
├── evals/golden_questions.json
├── tests/                     # 44 tests
└── data/raw/                  # put the Kaggle download here (not committed)
```

## Tests

```bash
python -m pytest -q
```

44 tests run with **no API key and no download**. They use a scripted fake LLM and a tiny hand-built database with the same schema
as Olist (`tests/fixture_data.py`; a test fixture only, not the project's dataset). They cover the validator (dangerous SQL, hallucinated
tables and columns, ambiguous columns, join warnings, and that all 26 correct reference queries pass), the repair loop, follow-ups,
timeouts and row caps, the data loader (including the misspelt Kaggle column names and `archive.zip` extraction), the sample
builder, the result-matching rules and the Streamlit app.

## Limitations

- **Text-to-SQL is not guaranteed correct.** Validation catches invalid SQL, not a valid query that answers a slightly different
  question. That is why the SQL is always shown and the explanation states its assumptions.
- The evaluation questions were written by the author, so accuracy on them is a useful signal, not a guarantee for every question.
- **Data leaves your machine in live mode.** The schema, two sample rows per table and up to 30 result rows are sent to the LLM provider.
  The Olist data is anonymised, but do not point this at confidential data without checking your provider's terms.
- Single-database, single-user, no login. SQLite and the 15-second timeout suit a demo, not production traffic.
- The dataset covers 2016–2018 only, so "last year" means relative to the data, not today.

## Possible next steps

Few-shot retrieval of similar solved questions · a semantic layer of pre-defined metrics (revenue, late-delivery rate) the model
must use · a Postgres/BigQuery backend · result caching · saved questions and dashboards · user feedback (thumbs up/down) feeding
the evaluation set.

## Data source and license

Data: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) on Kaggle,
released under **CC BY-NC-SA 4.0** (attribution, non-commercial, share-alike). The raw files are not redistributed here: download them from Kaggle.
