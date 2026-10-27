"""Central configuration: paths, limits and model defaults."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FULL_DB = ROOT / "data" / "olist.db"      # built from the Kaggle CSVs by `python -m src.data_loader`
SAMPLE_DB = ROOT / "data" / "sample.db"   # small subset, safe to commit (`python -m src.make_sample`)
EVAL_DIR = ROOT / "evals"
GOLDEN_FILE = EVAL_DIR / "golden_questions.json"

KAGGLE_URL = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
KAGGLE_SLUG = "olistbr/brazilian-ecommerce"


def default_db_path() -> Path:
    """DB used by the app: $SQL_ASSISTANT_DB, else the full DB, else the committed sample."""
    env = os.environ.get("SQL_ASSISTANT_DB")
    if env:
        return Path(env)
    return FULL_DB if FULL_DB.exists() else SAMPLE_DB


# ---- safety limits ----------------------------------------------------------------------
MAX_ROWS_RETURNED = 1000      # hard cap on rows fetched from any query
MAX_ROWS_FOR_EXPLANATION = 30  # rows shown to the LLM when it explains a result
QUERY_TIMEOUT_S = 15          # a query running longer than this is aborted
MAX_REPAIR_ATTEMPTS = 2       # extra LLM attempts after a validation / execution error
HISTORY_TURNS = 3             # previous Q&A pairs sent along for follow-up questions

# ---- models (override with the LLM_MODEL environment variable or in the app sidebar) ----------
DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini"}
