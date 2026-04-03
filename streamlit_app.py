from __future__ import annotations

import logging
import json
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import pandas as pd
import streamlit as st

from iit_jammu.app.db import run_query, get_database_stats
from iit_jammu.app.planner import plan_question, plan_sql_and_chart, fix_sql_and_chart, explain_concept
from iit_jammu.app.visuals import build_figure, build_anomaly_figure
from iit_jammu.app.analytics import run_anomaly_analysis
from iit_jammu.config import GROQ_MODEL, SQLITE_PATH, CHROMA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_conversation_context(memory: list[dict], max_turns: int = 4) -> str:
    if not memory:
        return "No prior conversation context."
    
    recent = memory[-max_turns:]
    chunks = []

    for i, turn in enumerate(recent, start = 1):
        chunks.append(
            "\n".join(
                [
                    f"Turn {i}",
                    f"User: {turn.get('question', '')}",
                    f"Intent: {turn.get('intent', '')}",
                    f"Interpretation: {turn.get('interpretation', '')}",
                    f"Resolved plan: {json.dumps(turn.get('plan', {}), ensure_ascii=False)}",
                    f"Outcome: {turn.get('outcome', '')}"
                ]
            )
        )
    return "\n\n".join(chunks)

def _record_turn(question: str, intent: str, interp: str, plan: dict, outcome: str) -> None:
    st.session_state.planner_memory.append({
        "question": question,
        "intent": intent,
        "interpretation": interp,
        "plan": plan,
        "outcome": outcome
    })
    # keep memory bounded to last 20 turns
    st.session_state.planner_memory = st.session_state.planner_memory[-20:]

st.set_page_config(page_title = "FloatChat", layout = "wide")
st.title("FloatChat - ARGO Ocean Data Explorer")
st.caption(f"LLM: {GROQ_MODEL}")

vector_ready = CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())
db_ready = SQLITE_PATH.exists()


with st.sidebar:
    st.header("System status")

    st.write(f"SQLite: {'Connected' if db_ready else 'Not found'}")
    st.write(f"Vector Store: {'Ready' if vector_ready else 'Not built'}")

    if not db_ready:
        st.warning("Run `ingest/load_sqlite.py` to load data.")
    if not vector_ready:
        st.warning("Run `ingest/build_vector_store.py` to build the vector store.")
    
    if db_ready:
        @st.cache_data(ttl=300)
        def _cached_stats():
            return get_database_stats()
        
        stats = _cached_stats()
        if stats:
            st.divider()
            st.subheader("Dataset coverage")
            col1, col2 = st.columns(2)
            col1.metric("Profiles", f"{stats['n_profiles']:,}")
            col2.metric("Floats", f"{stats['n_floats']:,}")
            st.caption(
                f"**Dates:** {str(stats['date_min'])[:10]} → {str(stats['date_max'])[:10]}"
            )
            st.caption(
                f"**Lat:** {stats['lat_min']}° to {stats['lat_max']}°  \n"
                f"**Lon:** {stats['lon_min']}° to {stats['lon_max']}°"
            )

    st.divider()
    st.markdown("**Example prompts**")
    st.markdown("- Show trajectories of floats in Jan 2024")
    st.markdown("- Plot temperature vs pressure for profile 1901910_207_20240109T233537_A")
    st.markdown("- Compare salinity and temperature for float 1901910")
    st.markdown("- How many profiles are there and what is the date range?")
    st.markdown("- Based only on the dataset, what anomalies are present in temperature readings?")
    
    if st.button("🗑 Clear conversation"):
        st.session_state.history = []
        st.session_state.planner_memory = []
        st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []

if "planner_memory" not in st.session_state:
    st.session_state.planner_memory = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

# Main Chat Loop

question = st.chat_input("Ask about ARGO data...")

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        conversation_context = build_conversation_context(
            st.session_state.planner_memory, max_turns = 4)

        try:
            with st.spinner("Understanding your question..."):
                query_plan, context = plan_question(
                    question, conversation_context = conversation_context)
        except RuntimeError as exc:
            msg = str(exc)
            st.error(msg)
            st.session_state.history.append(("assistant", f"⚠️ {msg}"))
            _record_turn(question, "error", "", {}, msg)
            st.stop()

            
        interp = query_plan.user_goal or question
        st.markdown(f"**Intent:** {query_plan.intent}")
        st.markdown(f"**Interpretation:** {interp}")

        if query_plan.intent == "unsupported":
            msg = (
                "This question doesn't appear to be answerable from the ARGO dataset. "
                "Try asking about temperature, salinity, float trajectories, or anomaly detection."
            )
            st.info(msg)
            st.session_state.history.append(("assistant", msg))
            _record_turn(question, "unsupported", interp, query_plan.model_dump(), msg)
            st.stop()

        if query_plan.intent == "concept_explanation":
            answer = explain_concept(question, context)
            st.markdown(answer)

            assistant_md = "\n\n".join(
                [
                    f"**Intent:** {query_plan.intent}",
                    f"**Interpretation:** {interp}",
                    answer
                ]
            )
            st.session_state.history.append(("assistant", assistant_md))
            _record_turn(
                question,
                query_plan.intent,
                interp,
                query_plan.model_dump(),
                answer
            )

            st.stop()

            
        if query_plan.intent == "anomaly_detection":
            with st.spinner("Running anomaly analysis..."):
                analysis = run_anomaly_analysis(query_plan)
            
            st.markdown(analysis.explanation)
                
            if not analysis.summary_df.empty:
                st.markdown("**Analysis summary**")
                st.dataframe(analysis.summary_df, use_container_width=True)
                
            value_col = query_plan.parameter or "temperature"
            fig = build_anomaly_figure(analysis.full_df, value_col=value_col)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            elif not analysis.full_df.empty:
                st.warning("Could not render the anomaly chart. Raw data is shown below.")
                
            if not analysis.anomalies_df.empty:
                st.markdown("**Top anomalous rows**")
                st.dataframe(analysis.anomalies_df, use_container_width=True)
            else:
                st.info("No anomalies detected for the selected variable and filters.")
                
            assistant_md = "\n\n".join(
                [
                    f"**Intent:** {query_plan.intent}",
                    f"**Interpretation:** {interp}",
                    analysis.explanation
                ]
            )
            st.session_state.history.append(("assistant", assistant_md))
            _record_turn(question, query_plan.intent, interp, query_plan.model_dump(), analysis.explanation)

        else:
            try:
                with st.spinner("Generating query..."):
                    sql_plan = plan_sql_and_chart(question, 
                                                query_plan, 
                                                context,
                                                conversation_context=conversation_context
                                                )
            except RuntimeError as exc:
                msg = str(exc)
                st.error(msg)
                st.session_state.history.append(("assistant", f"⚠️ {msg}"))
                _record_turn(question,
                            query_plan.intent,
                            interp,
                            query_plan.model_dump(),
                            msg)
                st.stop()

            with st.expander("Generated SQL"):
                st.code(sql_plan.sql, language="sql")
            
            df = None
            try:
                df = run_query(sql_plan.sql)
            except ValueError as first_exc:
                st.warning(f"First query attempt failed: {first_exc} \nRetrying with auto-fix...")
                logger.warning("First SQL attempt failed: %s", first_exc)

                try:
                    with st.spinner("Auto-fixing query..."):
                        sql_plan = fix_sql_and_chart(
                            question,
                            failed_sql=sql_plan.sql,
                            error=str(first_exc)
                        )
                    with st.expander("Fixed SQL"):
                        st.code(sql_plan.sql, language="sql")
                    df = run_query(sql_plan.sql)
                except (ValueError, RuntimeError) as second_exc:
                    msg = (
                        f"The query could not be executed even after auto-correction: {second_exc}. "
                        "Try rephrasing your question or narrowing the filters."
                    )
                    st.error(msg)
                    st.session_state.history.append(("assistant", f"⚠️ {msg}"))
                    _record_turn(question,
                                    query_plan.intent,
                                    interp,
                                    query_plan.model_dump(),
                                    msg)
                    st.stop()
                
            if df is not None and not df.empty:
                for col in df.columns:
                    if "timestamp" in col.lower():
                        try:
                            df[col] = pd.to_datetime(df[col])
                        except Exception:
                            pass
            st.markdown(sql_plan.explanation)

            fig = build_figure(df, sql_plan)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            elif df is not None and not df.empty:
                st.warning(
                    "Could not render the requested chart type. "
                    "The raw data is shown below."
                )
                
            if df is None or df.empty:
                st.info(
                    "No matching rows were found. "
                    "Try broadening the data range, removing geographic filters, or rephrasing."
                )
            else:
                st.dataframe(df, use_container_width=True)
                
            outcome = sql_plan.explanation + f" (returned {len(df) if df is not None else 0} rows)"
            assistant_md = "\n\n".join([
                f"**Intent:** `{query_plan.intent}`",
                f"**Interpretation:** {interp}",
                sql_plan.explanation,
                f"_Returned {len(df) if df is not None else 0} rows._"
            ])
            st.session_state.history.append(("assistant", assistant_md))
            _record_turn(question,
                            query_plan.intent,
                            interp,
                            query_plan.model_dump(),
                            outcome)
