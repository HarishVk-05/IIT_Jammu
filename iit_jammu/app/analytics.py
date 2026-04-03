from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from iit_jammu.app.db import ENGINE
from iit_jammu.app.models import QueryPlan

logger = logging.getLogger(__name__)

QC_COLUMN_MAP = {
    "temperature": "temperature_qc",
    "salinity": "salinity_qc",
    "pressure": "pressure_qc",
    "latitude": "position_qc",
    "longitude": "position_qc"
}

ACCEPTED_QC = {"1", "2"}

def _normalize_qc(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return s

def _qc_columns_for_value(value_col: str) -> list[str]:
    cols: list[str] = []

    value_qc_col = QC_COLUMN_MAP.get(value_col)
    if value_qc_col:
        cols.append(value_qc_col)
    
    # pressure is used for depth bins, so carry its QC too
    if "pressure_qc" not in cols:
        cols.append("pressure_qc")
    
    return cols

def _apply_qc_filters(
        df: pd.DataFrame,
        value_col: str,
        accepted_qc: set[str] = ACCEPTED_QC
    ) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    qc_stats: dict = {}

    value_qc_col = QC_COLUMN_MAP.get(value_col)
    # hard filter rows whos QC is not acceptable
    if value_qc_col and value_qc_col in out.columns:
        out[value_qc_col] = _normalize_qc(out[value_qc_col])
        before = len(out)
        out = out[out[value_qc_col].isin(accepted_qc)].copy()
        qc_stats["rows_removed_by_value_qc"] = before - len(out)
    
    # if pressure QC is bad, keep the row but dont trust pressure for depth binning
    if "pressure_qc" in out.columns and "pressure" in out.columns:
        out["pressure_qc"] = _normalize_qc(out["pressure_qc"])
        bad_pressure_mask = out["pressure"].notna() & (~out["pressure_qc"].isin(accepted_qc))
        qc_stats["rows_with_bad_pressure_qc"] = int(bad_pressure_mask.sum())
        out.loc[bad_pressure_mask, "pressure"] = np.nan
    
    return out, qc_stats

@dataclass
class AnalysisResult:
    title: str
    explanation: str
    full_df: pd.DataFrame
    anomalies_df: pd.DataFrame
    summary_df: pd.DataFrame

def _robust_zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors = "coerce")
    median = s.median()
    mad = np.median(np.abs(s - median))

    if pd.isna(mad) or mad == 0:
        std = s.std()
        if pd.isna(std) or std == 0:
            return pd.Series(np.zeros(len(s)), index = s.index, dtype = float)
        return (s - s.mean()) / std
    
    return 0.6745 * (s - median) / mad


def _make_depth_bins(pressure: pd.Series, bin_size: int = 100) -> pd.Series:
    p = pd.to_numeric(pressure, errors = "coerce")
    return (np.floor(p / bin_size) * bin_size).astype("Int64")

def _group_robust_z(
        df: pd.DataFrame,
        value_col: str,
        group_cols: list[str],
        min_group_size: int = 15
) -> pd.Series:
    if any(col not in df.columns for col in group_cols):
        return pd.Series(np.nan, index = df.index, dtype = float)
    
    group_sizes = df.groupby(group_cols, dropna = False)[value_col].transform("size")
    z = df.groupby(group_cols, dropna=False)[value_col].transform(_robust_zscore)
    z = pd.to_numeric(z, errors="coerce")
    z[group_sizes < min_group_size] = np.nan
    return z

def _group_median(
        df: pd.DataFrame,
        value_col: str,
        group_cols: list[str],
        min_group_size: int = 15
) -> pd.Series:
    if any(col not in df.columns for col in group_cols):
        return pd.Series(np.nan, index = df.index, dtype = float)

    group_sizes = df.groupby(group_cols, dropna = False)[value_col].transform("size")
    med = df.groupby(group_cols, dropna = False)[value_col].transform("median")
    med = pd.to_numeric(med, errors = "coerce")
    med[group_sizes < min_group_size] = np.nan
    return med

def _default_abs_deviation_threshold(value_col: str) -> float:
    return {
        "temperature": 1.5,
        "salinity": 0.4,
        "pressure": 50.0
    }.get(value_col, 1.0)

def _contextual_anomaly_scores(
        df:pd.DataFrame,
        value_col: str,
        scope: str,
        depth_bin_size: int = 100,
        min_group_size: int = 15
) -> pd.DataFrame:
    out = df.copy()

    if "pressure" in out.columns:
        out["depth_bin"] = _make_depth_bins(out["pressure"], bin_size=depth_bin_size)
    else:
        out["depth_bin"] = pd.Series([pd.NA] * len(out), index = out.index, dtype = "Int64")
    
    out["global_robust_z"] = _robust_zscore(out[value_col])
    out["global_median"] = pd.to_numeric(out[value_col], errors = "coerce").median()


    if scope == "per_float" and "float_id" in out.columns:
        primary_groups = ["float_id", "depth_bin"]
        fallback_groups = [["depth_bin"], ["float_id"]]
    elif scope == "per_profile" and "profile_id" in out.columns:
        primary_groups = ["profile_id", "depth_bin"]
        fallback_groups = [["depth_bin"], ["profile_id"]]
    else:
        primary_groups = ["depth_bin"]
        fallback_groups = []
    
    context_z = _group_robust_z(
        out,
        value_col=value_col,
        group_cols=primary_groups,
        min_group_size=min_group_size
    )

    context_median = _group_median(
        out,
        value_col = value_col,
        group_cols = primary_groups,
        min_group_size = min_group_size
    )

    for groups in fallback_groups:
        missing_z = context_z.isna()
        if missing_z.any():
            fallback_z = _group_robust_z(
                out,
                value_col=value_col,
                group_cols=groups,
                min_group_size=min_group_size
            )
            context_z.loc[missing_z] = fallback_z.loc[missing_z]
        
        missing_med = context_median.isna()
        if missing_med.any():
            fallback_med = _group_median(
                out,
                value_col=value_col,
                group_cols=groups,
                min_group_size=min_group_size
            )
            context_median.loc[missing_med] = fallback_med.loc[missing_med]
    context_z = context_z.fillna(out["global_robust_z"])
    context_median = context_median.fillna(out["global_median"])

    out["context_robust_z"] = context_z
    out["context_median"] = context_median
    out["abs_deviation"] = (pd.to_numeric(out[value_col], errors = "coerce") - out["context_median"]).abs()
    out["anomaly_score"] = out["context_robust_z"].abs()

    return out

def _build_measurement_sql(plan: QueryPlan, value_col: str) -> tuple[str, dict]:
    cols = ["timestamp", "float_id", "profile_id", "pressure", value_col]
    cols.extend(_qc_columns_for_value(value_col))
    cols = list(dict.fromkeys(cols))  # preserve order, remove duplicates

    sql = f"SELECT {', '.join(cols)} FROM measurements WHERE {value_col} IS NOT NULL"
    params: dict = {}

    if plan.start_date:
        sql += " AND timestamp >= :start_date"
        params["start_date"] = plan.start_date
    
    if plan.end_date:
        sql += " AND timestamp < :end_date"
        params["end_date"] = plan.end_date

    if plan.depth_min_m is not None:
        sql += " AND pressure >= :depth_min"
        params["depth_min"] = float(plan.depth_min_m)
    
    if plan.depth_max_m is not None:
        sql += " AND pressure <= :depth_max"
        params["depth_max"] = float(plan.depth_max_m)
    
    if plan.latitude_min is not None:
        sql += " AND latitude >= :lat_min"
        params["lat_min"] = float(plan.latitude_min)
    
    if plan.latitude_max is not None:
        sql += " AND latitude <= :lat_max"
        params["lat_max"] = float(plan.latitude_max)
    
    if plan.longitude_min is not None:
        sql += " AND longitude >= :lon_min"
        params["lon_min"] = float(plan.longitude_min)
    
    if plan.longitude_max is not None:
        sql += " AND longitude <= :lon_max"
        params["lon_max"] = float(plan.longitude_max)
    
    if plan.float_id:
        sql += " AND float_id = :float_id"
        params["float_id"] = plan.float_id
    
    if plan.profile_id:
        sql += " AND profile_id = :profile_id"
        params["profile_id"] = plan.profile_id
    
    sql += " ORDER BY timestamp ASC"
    return sql, params

def run_anomaly_analysis(plan: QueryPlan) -> AnalysisResult:

    if ENGINE is None:
        return AnalysisResult(
            title = "Database unavailable",
            explanation="The SQLite database is not connected. Run ingest/load_sqlite.py first.",
            full_df=pd.DataFrame(),
            anomalies_df=pd.DataFrame(),
            summary_df=pd.DataFrame()
        )
    
    value_col = plan.parameter or "temperature"
    scope = plan.analysis_scope or "per_float"
    sql, params = _build_measurement_sql(plan, value_col)
    
    try:
        with ENGINE.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)
    except OperationalError as exc:
        logger.error("Anomaly query failed: %s", exc)
        return AnalysisResult(
            title = "Query failed",
            explanation = f"Could not retrieve data for anomaly analysis: {exc}",
            full_df = pd.DataFrame(),
            anomalies_df = pd.DataFrame(),
            summary_df = pd.DataFrame()
        )
    if df.empty:
        return AnalysisResult(
            title = f"No {value_col} anomalies found",
            explanation="No rows matched the filters, so anomaly detection could not be performed.",
            full_df=pd.DataFrame(),
            anomalies_df=pd.DataFrame(),
            summary_df=pd.DataFrame()
        )
    
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors = "coerce")

    if value_col in df.columns:
        df[value_col] = pd.to_numeric(df[value_col], errors = "coerce")
    if "pressure" in df.columns:
        df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")
    df, qc_stats = _apply_qc_filters(df, value_col= value_col)
    df = df.dropna(subset=[value_col]).copy()

    if df.empty:
        return AnalysisResult(
            title=f"No {value_col} anomalies found",
            explanation=(
                f"No QC-valid rows matched the filters for {value_col}, "
                "so anomaly detection could not be performed."
            ),
            full_df=pd.DataFrame(),
            anomalies_df=pd.DataFrame(),
            summary_df=pd.DataFrame(),
        )

    try:
        df = _contextual_anomaly_scores(
            df,
            value_col=value_col,
            scope = scope,
            depth_bin_size = 50,
            min_group_size = 15
        )
    except Exception as exc:
        logger.error("Anomaly scoring failed: %s", exc)
        return AnalysisResult(
            title = "Anomaly scoring failed",
            explanation=f"An error occurred during anomaly scoring: {exc}",
            full_df=df,
            anomalies_df=pd.DataFrame(),
            summary_df=pd.DataFrame()
        )

    abs_dev_threshold = _default_abs_deviation_threshold(value_col)

    df["is_anomaly"] = (
        (df["anomaly_score"] >= 5.0) &
        (df["abs_deviation"] >= abs_dev_threshold)
    )

    anomaly_total = int(df["is_anomaly"].sum())
    top_k = max(1, int(plan.top_k or 20))

    anomalies_df = (
        df[df["is_anomaly"]]
        .sort_values("anomaly_score", ascending=False)
        .head(top_k)
        .copy()
    )

    summary = {
        "variable": value_col,
        "scope": scope,
        "total_rows_analyzed": int(len(df)),
        "anomaly_count": anomaly_total,
        "anomaly_fraction": float(df["is_anomaly"].mean()) if len(df) > 0 else 0.0,
        "min_value": float(df[value_col].min()),
        "max_value": float(df[value_col].max()),
        "median_value": float(df[value_col].median())
    }

    summary_df = pd.DataFrame([summary])

    if anomaly_total == 0:
        explanation = (
            f"No strong anomalies were detected in {value_col} after QC filtering using contextual robust z-score analysis "
            f"with scope={scope}. Readings were compared within similar depth bands and entities."
        )
    else:
        explanation = (
            f"Detected {anomaly_total} high-suspicion {value_col} anomalies after QC filtering using contextual robust z-score analysis "
            f"with scope={scope}. Readings were compared within similar depth bands and entities, "
            f"and were required to exceed an absolute deviation threshold of {abs_dev_threshold}. "
            f"Showing the top {min(top_k, anomaly_total)} most extreme rows."
        )
    title = f"{value_col.capitalize()} anomalies"

    return AnalysisResult(
        title = title,
        explanation = explanation,
        full_df=df,
        anomalies_df= anomalies_df,
        summary_df= summary_df
    )