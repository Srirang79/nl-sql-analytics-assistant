"""Streamlit interface:  streamlit run app.py"""
from __future__ import annotations

import os
import re

import pandas as pd
import streamlit as st

from src import config
from src.database import connect_readonly, table_counts
from src.engine import QueryResult, SQLAssistant
from src.evaluate import load_golden
from src.llm import DemoClient, make_client
from src.schema import TABLE_DOCS, introspect

st.set_page_config(page_title="Ask your data", page_icon="🔎", layout="wide")

GOLDEN = load_golden()
EXAMPLES = [g["question"] for g in GOLDEN]
DEMO_SQL = {g["question"]: g["reference_sql"] for g in GOLDEN}
SUGGESTED = [EXAMPLES[i] for i in (2, 7, 18, 19, 21, 22, 24, 8)]


# ------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_assistant(db_path: str, mode: str, provider: str, model: str, api_key: str) -> SQLAssistant:
    llm = DemoClient(DEMO_SQL) if mode == "demo" else make_client(provider, api_key or None, model or None)
    return SQLAssistant(db_path, llm)


def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "")
    except Exception:  # no secrets.toml
        return ""


def render_chart(df: pd.DataFrame) -> None:
    """Draw a simple chart when the result is 'a label column + numbers'."""
    if not (2 <= len(df) <= 60) or df.shape[1] < 2:
        return
    label, nums = df.columns[0], [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
    if not nums or pd.api.types.is_numeric_dtype(df[label]) and df[label].nunique() > 12:
        return
    data = df.set_index(label)[nums[:1]]
    is_time = df[label].astype(str).str.match(r"^\d{4}-\d{2}$").all()
    (st.line_chart if is_time else st.bar_chart)(data, **({} if is_time else {"sort": False}))


def render_result(res: QueryResult) -> None:
    if res.status == "ok":
        if res.explanation:
            st.markdown(res.explanation)
        if res.df is not None and len(res.df):
            render_chart(res.df)
            st.dataframe(res.df, width="stretch", hide_index=True)
            st.download_button("Download CSV", res.df.to_csv(index=False), file_name="result.csv", key=f"dl_{id(res)}")
        elif res.df is not None:
            st.info("The query ran but returned no rows.")
        if res.truncated:
            st.caption(f"Showing the first {config.MAX_ROWS_RETURNED:,} rows only.")
        with st.expander("How this answer was produced (SQL)"):
            st.code(res.sql, language="sql")
            meta = f"{res.seconds:.1f}s · {res.attempts} model attempt(s)"
            if res.trace:
                meta += " · self-corrected after: " + " | ".join(res.trace)
            st.caption(meta)
            for w in res.warnings:
                st.warning(w)
    elif res.status == "clarify":
        st.warning(f"I need one detail to answer this: {res.message}")
    elif res.status == "unanswerable":
        st.info(res.message)
    else:
        st.error(res.message)
        if res.sql:
            with st.expander("Last query attempted"):
                st.code(res.sql, language="sql")


# ------------------------------------------------------------------------------------------
db_path = config.default_db_path()
if not db_path.exists():
    st.title("🔎 Ask your data")
    st.error("No database found yet.")
    st.markdown(
        f"""
This app queries the Kaggle **[Olist Brazilian E-commerce dataset]({config.KAGGLE_URL})**.
Build the database once:

```bash
# 1. download archive.zip from Kaggle and put it in data/raw/
# 2. build the SQLite database
python -m src.data_loader
```
Then restart the app."""
    )
    st.stop()

with st.sidebar:
    st.header("Settings")
    mode_label = st.radio("Mode", ["Live LLM", "Demo (no API key)"], index=0 if (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY") or secret("ANTHROPIC_API_KEY") or secret("OPENAI_API_KEY")) else 1,
                          help="Demo mode runs pre-written SQL for the example questions only. Live mode uses an LLM to understand any question.")
    mode = "demo" if mode_label.startswith("Demo") else "live"
    provider, model, api_key = "anthropic", "", ""
    if mode == "live":
        provider = st.selectbox("Provider", ["anthropic", "openai"])
        model = st.text_input("Model (optional)", placeholder=config.DEFAULT_MODELS[provider])
        env_key = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        api_key = st.text_input(f"{provider.title()} API key", type="password",
                                value=os.environ.get(env_key, "") or secret(env_key),
                                help="Used only for this session; never stored.")
    else:
        st.info("Demo mode: only the example questions work. Switch to Live LLM to ask anything.")
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()
    st.caption(f"Database: `{db_path.name}`")

st.title("🔎 Ask your data in plain English")
st.caption("Ask a business question about ~100k real Brazilian e-commerce orders. "
           "The assistant writes SQL, validates it (read-only), runs it and explains the result.")

if "history" not in st.session_state:
    st.session_state.history = []

tab_ask, tab_data = st.tabs(["Ask", "Data & schema"])

with tab_ask:
    if not st.session_state.history:
        st.markdown("**Try one of these:**")
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED):
            if cols[i % 2].button(q, key=f"ex{i}", width="stretch"):
                st.session_state.pending = q
                st.rerun()

    for h in st.session_state.history:
        with st.chat_message("user"):
            st.write(h.question)
        with st.chat_message("assistant"):
            render_result(h)

    typed = st.chat_input("e.g. Which product categories had the highest revenue in 2017?")
    question = typed or st.session_state.pop("pending", None)
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            try:
                assistant = build_assistant(str(db_path), mode, provider, model, api_key)
            except Exception as e:  # noqa: BLE001 - e.g. missing API key
                st.error(str(e))
                st.stop()
            with st.spinner("Writing and validating the SQL..."):
                res = assistant.ask(question, st.session_state.history)
            render_result(res)
        st.session_state.history.append(res)

with tab_data:
    conn = connect_readonly(db_path)
    counts = table_counts(conn)
    st.subheader("Tables")
    st.dataframe(pd.DataFrame({"table": list(counts), "rows": list(counts.values()),
                               "description": [TABLE_DOCS.get(t, {}).get("desc", "") for t in counts]}),
                 hide_index=True, width="stretch")
    pick = st.selectbox("Inspect a table", list(counts))
    schema = introspect(conn)[pick]
    st.dataframe(pd.DataFrame([(c, t, TABLE_DOCS.get(pick, {}).get("cols", {}).get(c, "")) for c, t in schema],
                              columns=["column", "type", "meaning"]), hide_index=True, width="stretch")
    st.dataframe(pd.read_sql_query(f'SELECT * FROM "{pick}" LIMIT 5', conn), hide_index=True, width="stretch")
    st.caption(f"Source: [Kaggle: Brazilian E-Commerce Public Dataset by Olist]({config.KAGGLE_URL}) (CC BY-NC-SA 4.0).")
