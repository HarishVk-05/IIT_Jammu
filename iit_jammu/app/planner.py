from __future__ import annotations
import logging
import re
from typing import Tuple
from langchain_core.prompts import ChatPromptTemplate
from iit_jammu.app.db import get_sql_db
from iit_jammu.app.llm import get_llm
from iit_jammu.app.models import QueryPlan, SqlAndChartPlan
from iit_jammu.app.retriever import safe_retrieve

logger = logging.getLogger(__name__)

CONCEPT_GLOSSARY = {
    "salinity": (
        "**Salinity** is the amount of dissolved salts in seawater.\n\n"
        "In FloatChat, `salinity` is a measurement column in the `measurements` table. "
        "It is recorded by ARGO floats at different pressures and timestamps."
    ),
    "temperature": (
        "**Temperature** is the measured seawater temperature.\n\n"
        "In FloatChat, `temperature` is a measurement column in the `measurements` table."
    ),
    "pressure": (
        "**Pressure** is the dataset's depth-like vertical coordinate.\n\n"
        "In FloatChat, users may ask about depth, but the ARGO data stores vertical position as `pressure`."
    ),
    "argo": (
        "**ARGO** is a global ocean observing system of profiling floats.\n\n"
        "In this project, ARGO floats provide measurements such as temperature and salinity across time, "
        "location, and pressure levels."
    ),
    "profile": (
        "**A profile** is one vertical sampling event from a float.\n\n"
        "In your dataset, a profile groups measurements taken by a float at one timestamp/cycle across different pressures."
    ),
    "qc": (
        "**QC** means quality control.\n\n"
        "Your dataset includes QC columns such as `temperature_qc`, `salinity_qc`, and `pressure_qc`."
    )
}

CONCEPT_EXPLANATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You explain ARGO and dataset concepts for FloatChat.
        
        Rules:
        Answer in plain language.
        Explain both the general meaning and what it means in this dataset.
        keep the answer concise and useful.
        """
    ),
    (
        "human",
        "Retrieved context: \n{context}\n\nUser question:\n{question}"
    )
])

QUERY_PLAN_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an ARGO ocean data query planner.
        Convert the user question into a structured plan.

        Rules:
        Prefer trajectory_map for paths of floats over time.
        Prefer profile_plot only for explicit vertical-profile questions such as:
        "temperature vs pressure", "salinity vs pressure", "depth profile",
        "vertical structure", "plot with pressure", or clearly profile-specific plots.
        Prefer time_trend when the user asks how a variable behaves, changes, evolves, or trends over time.
        Prefer time_trend when a time period is mentioned together with a depth band or mid-depth / deep / shallow wording.
        Prefer comparison_plot for comparing two parameters or multiple floats when the main goal is comparison rather than temporal trend.
        Prefer anomaly detection for questions about anomalies, outliers, unusual readings, spikes, suspicious values, or abnormal behavior in the dataset.
        Prefer concept_explanation for basic concept, meaning, purpose, role, or use-case questions such as "what is salinity", "what is the use of the ARGO dataset", "what purpose does the ARGO dataset serve", "what is ARGO used for", or "what does QC mean"; do not classify these as dataset_summary, table, or time_trend.
        Prefer dataset_summary for count, coverage, or time-range overview questions.
        Use table for general filtered tabular answers.
        Use unsupported only when the question truly cannot be answered from the ARGO dataset.
        Do not invent IDs.

        Interpretation rules:
        If the user mentions depth, depth range, mid-depth, shallow, or deep water, treat that as a vertical filter concept.
        The dataset stores vertical position as pressure, but this planner should capture the user meaning as depth_min_m / depth_max_m when possible.
        If the user asks for behavior over time in a depth band, do not classify it as profile_plot.
        If the user says "vs pressure", that strongly indicates profile_plot.
        
        Multi-turn behavior:
        Use recent conversation context when the current message is a follow-up.
        Resolve references like "same", "that", "those", "again", "now", "instead", "for this one", or "only" using the recent conversation context.
        Preserve prior filters, entity references, depth bands, time ranges, and intent when the new message clearly depends on them.
        If the current message explicitly changes one thing, keep the rest from the recent conversation context unless contradicted.
        When the current user message explicitly names a measurement variable such as temperature, salinity, or pressure, that variable must override the variable from prior conversation context.
        Do not carry forward a prior measurement variable if the current user message explicitly changes it.

        Important normalization rules:
        - If the user says 'depth', map it to 'pressure'.
        - latitude_min, latitude_max, longitude_min, longitude_max must be numeric values, not quoted strings.
        """,

    ),
    ("human", 
     "Recent conversation context:\n{conversation_context}\n\nRetrieved context:\n{context}\n\nUser question:\n{question}")
])

SQL_PLAN_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an expert SQLite planner for an ARGO ocean data.
        Return a SQL query and chart plan.
        
        You must output:
        one valid SQLite SELECT query only
        one chart plan matching the query result

        Hard constraints:
        Output exactly one SELECT query only.
        Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA, or multiple statements.
        Available tables: profiles_metadata, measurements.
        Never reference columns that do not exist.
        Use only SQLite-compatible SQL.

        Database usage rules:
        Use profiles_metadata for profile-level, float-level, geospatial, trajectory, and summary questions.
        Use measurements for depth-based, temperature, salinity, and pressure analysis.
        profile_id links profile-level and measurement-level data.
        measurements contains: profile_id, float_id, cycle_number, timestamp, direction, latitude, longitude, pressure, temperature, salinity, position_qc, pressure_qc, temperature_qc, salinity_qc, data_mode, source_file
        profiles_metadata contains: profile_id, float_id, cycle_number, timestamp, direction, latitude, longitude, data_mode, n_points, min_pressure, max_pressure, has_temp, has_salinity, source_file

        Forbidden mistake:
        Do not use measurements for float trajectory/path/drift/map questions unless the user explicitly asks for depth-resolved measurement points.
        
        Important interpretation rules:
        The dataset stores vertical position as pressure, not geometric depth.
        If the user mentions depth or depth range, interpret it as a pressure filter.
        Do not plot pressure on an axis unless the user is explicitly asking for a profile plot.
        For depth ranges, map depth_min_m / depth_max_m to pressure bounds.
        For "at depth X", do not use pressure BETWEEN X AND X unless the user explicitly asks for exact pressure.
        Prefer a small pressure band around the requested depth, such as ±10.
        Prefer dataset-supported geographic bounds from retrieved context instead of inventing broad ocean bounds.
        If the user asks for a trend at a given depth, depth should act as a filter and time should be the x-axis.
        If the user asks a broad behavioral question such as "how does X behave", "how does X change", "how does X evolve", or similar, and a time period is mentioned, prefer a temporal summary rather than a profile plot.
        Use pressure/depth as a filter unless the user explicitly asks for a profile plot or says "vs pressure".
        If the query is vague but includes a time period and a depth band, prefer aggregating the measured variable over time instead of plotting raw measurements.

        Date and time rules:
        Use half-open date intervals: timestamp >= start_date AND timestamp < next_day_or_next_month_boundary.
        If GROUP BY DATE(timestamp) is used, SELECT DATE(timestamp) AS day, not raw timestamp.
        For temporal trends, ORDER BY the grouped time field.

        Chart selection rules:
        1. trajectory_map
            Query from profiles_metadata.
            Select longitude, latitude, timestamp, float_id, and optionally profile_id.
            Use chart_type = "map".
        
        2. profile_plot
            Use measurements.
            This is for questions like temperature vs pressure or salinity vs pressure for a float/profile/profile set.
            The measured variable (for example temperature or salinity) MUST be the x-axis.
            Pressure must be the y-axis.
            Never put pressure on the x-axis for profile_plot.
            Use chart_type = "line" for a single profile or clearly ordered profile curve.
            Add ORDER BY pressure ASC.
        
        3. time trend / temporal trend
            If the user says words like "trend", "over time", "during Jan", "daily", "monthly", "throughout", or asks how a variable changes over time, the x-axis should be time.
            The SELECT must include timestamp or a time bucket such as DATE(timestamp).
            For aggregated trends, prefer:
                DATE(timestamp) AS day
                AVG(parameter) AS avg_parameter
            then GROUP BY DATE(timestamp)
            Add ORDER BY the time field.
            Use chart_type = "line" for aggregated temporal trends.
            If returning many raw observations from multiple floats without aggregation, prefer chart_type = "scatter", optionally color by float_id.
            Prefer daily or monthly aggregation when the user asks about behavior across a period rather than a single profile.
            Do not use a line chart over raw pressure values for broad time-period behavior questions.

        4. comparison_plot
            Use measurements unless clearly profile-summary-only.
            Compare variables across time, floats, or profiles.
            Use scatter for raw many-point comparisons.
            Use line only when there is a meaningful ordered x-axis.
        
        5. table
            Use for general filtered lookup answers when no chart is clearly appropriate.

        Multi-turn behavior:
        Use recent conversation context when the current question is a follow-up.
        If the user says "same", "do the same", "now", "instead", or similar, preserve the prior resolved filters and entities unless the new question explicitly changes them.
        Prefer the structured plan as the main source of truth, but use recent conversation context to resolve ambiguous follow-ups.
        Conflict resolution priority:
        1. current user question
        2. structured plan
        3. recent conversation context
        4. retrieved context
        If the current user question explicitly names a variable such as salinity or temperature, the SQL must use that variable even in a follow-up request.
        In follow-up requests, preserve prior filters, entities, and intent unless changed, but do not preserve a prior variable when the current user question explicitly names a different one.
        
        Output rules:
            title should be concise and match the chart.
            explanation should briefly explain what the query returns and how the chart should be interpreted.
        """
    ),
    (
        "human",
        "Recent conversation context:\n{conversation_context}\n\nSchema:\n{schema}\n\nRetrieved context:\n{context}\n\nStructured plan:\n{plan}\n\nUser question:\n{question}"
    )
])

SQL_FIX_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an expert SQLite query fixer for ARGO ocean data.
        The previous SQL query failed. Fix it and return a corrected SQL query and chart plan.
        
        Hard constraints:
        Output exactly one SELECT query only.
        Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA, or multiple statements.
        Available tables: profiles_metadata, measurements.
        Never reference columns that do not exist.
        Use only SQLite-compatible SQL.

        measurements contains: profile_id, float_id, cycle_number, timestamp, direction, latitude, longitude,
        pressure, temperature, salinity, position_qc, pressure_qc, temperature_qc, salinity_qc, data_mode, source_file

        profiles_metadata contains: profile_id, float_id, cycle_number, timestamp, direction, latitude, longitude,
        data_mode, n_points, min_pressure, max_pressure, has_temp, has_salinity, source_file
        """
    ),
    (
        "human",
        "Schema:\n{schema}\n\n"
        "Original question:\n{question}\n\n"
        "Failed SQL:\n{failed_sql}\n\n"
        "Error message:\n{error}\n\n"
        "Fix the SQL so it runs correctly on the schema above."
    )
])

def explain_concept(question: str, context: str = "") -> str:
    q = question.lower().strip()

    for term, answer in CONCEPT_GLOSSARY.items():
        if re.search(rf"\b{re.escape(term)}\b", q):
            return answer
    try:
        llm = get_llm(temperature=0.1)
        chain = CONCEPT_EXPLANATION_PROMPT | llm
        resp = chain.invoke(
            {
                "question": question,
                "context": context or "(no retrieved context)"
            }
        )
        return resp.content if hasattr(resp, "content") else str(resp)
    
    except Exception as exc:
        logger.warning("Concept explanation failed: %s", exc)
        return(
            "I understood this as a concept question, not a data query. "
            "Please try asking something like `What is salinity in this dataset?`"
        )
    
def _get_schema() -> str:
    try:
        return get_sql_db().get_table_info(["profiles_metadata", "measurements"])
    except Exception as exc:
        logger.warning("Could not fetch schema: %s", exc)
        return "(schema unavailable)"

def plan_question(question: str, conversation_context: str = "") -> Tuple[QueryPlan, str]:
    _docs, context = safe_retrieve(question, k = 4)
    llm = get_llm().with_structured_output(QueryPlan)
    chain = QUERY_PLAN_PROMPT | llm
    
    try:
        plan = chain.invoke(
            {
            "question": question, 
            "context" : context or "(no retrieved context)",
            "conversation_context": conversation_context or "No prior conversation context."
            }
            )
    except Exception as exc:
        logger.error("Query planning failed: %s", exc)
        raise RuntimeError(
            "The AI planner could not interpret your question. "
            "Try rephrasing it or being more specific."
        ) from exc
    
    return plan, context

def plan_sql_and_chart(
        question: str, 
        query_plan: QueryPlan, 
        context: str,
        conversation_context: str = ""
        ) -> SqlAndChartPlan:
    schema = _get_schema()
    llm = get_llm().with_structured_output(SqlAndChartPlan)
    chain = SQL_PLAN_PROMPT | llm

    try:
        sql_plan = chain.invoke(
            {
                "schema": schema,
                "context": context or "(no retrieved context)",
                "plan": query_plan.model_dump_json(indent=2),
                "question": question,
                "conversation_context": conversation_context or "No prior conversation context."
            }
        )
    except Exception as exc:
        logger.error("SQL planning failed: %s", exc)
        raise RuntimeError(
            "The AI planner could not generate a query for your question. "
            "Try rephrasing it."
        ) from exc
    
    return sql_plan

def fix_sql_and_chart(
        question: str,
        failed_sql: str,
        error: str
) -> SqlAndChartPlan:
    schema = _get_schema()
    llm = get_llm().with_structured_output(SqlAndChartPlan)
    fix_chain = SQL_FIX_PROMPT | llm

    try:
        fixed_plan = fix_chain.invoke(
            {
                "schema": schema,
                "question": question,
                "failed_sql": failed_sql,
                "error": error
            }
        )
        logger.info("SQl auto-fixed successfully.")
        return fixed_plan
    except Exception as exc:
        logger.error("SQL fix attempt failed: %s", exc)
        raise ValueError(
            "The query could not be auto-corrected. Try rephrasing your question."
        ) from exc
    