from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as  go 

from iit_jammu.app.models import SqlAndChartPlan

logger = logging.getLogger(__name__)

def build_figure(df: pd.DataFrame, plan: SqlAndChartPlan):
    """
    Build a Plotly figure from a query result and chart plan.
    """
    if df is None or df.empty:
        return None
    
    try:
        if plan.chart_type == "map":
            return _build_map(df, plan)
        if plan.chart_type in {"line", "scatter"}:
            return _build_line_scatter(df, plan)
        return _build_table(df, plan)
    except Exception as exc:
        logger.warning("Figure build failed (chart_type=%s): %s", plan.chart_type, exc)
        # fallback: always try to render a table
        try:
            return _build_table(df, plan)
        except Exception:
            return None

def _build_map(df: pd.DataFrame, plan: SqlAndChartPlan):
    if not {"latitude", "longitude"}.issubset(df.columns):
        logger.warning("Map requested but latitude/longitude columns are missing.")    
        return None
    
    map_df = df.copy()
    if "timestamp" in map_df.columns:
        try:
            map_df["timestamp"] = pd.to_datetime(map_df["timestamp"])
            map_df = map_df.sort_values(["float_id", "timestamp"] if "float_id" in map_df.columns else ["timestamp"])
            map_df["timestamp"] = map_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
        
    color = plan.color_col if plan.color_col in map_df.columns else None
    hover_cols = [c for c in ["float_id", "profile_id", "timestamp"] if c in map_df.columns]
    # prefer trajectory lines when float_id is available
    center_lat = map_df["latitude"].mean()
    center_lon = map_df["longitude"].mean()
    
    if "float_id" in map_df.columns and "timestamp" in map_df.columns:
        fig = px.line_mapbox(
            map_df,
            lat = "latitude",
            lon = "longitude",
            line_group = "float_id",
            color = color or "float_id",
            hover_data = hover_cols,
            zoom = 3.5,
            height = 650,
            title = plan.title
        )
        # add markers so short/single point trajectories are visible
        fig.update_traces(mode="lines+markers", marker=dict(size=5))
    else:
        fig = px.scatter_mapbox(
            map_df,
            lat = "latitude",
            lon = "longitude",
            color = color,
            hover_data = hover_cols,
            zoom = 3.5,
            height = 650,
            title = plan.title
        )
    fig.update_layout(
        mapbox = dict(
            style = "open-street-map",
            center=dict(lat=center_lat, lon = center_lon)
        ),
        margin = {"l":0, "r":0, "t":50, "b":0},
        legend = dict(
            title = "Float ID",
            orientation = "v",
            x = 1.01,
            y = 1,
            font = dict(size = 9),
            itemsizing="constant",
            tracegroupgap = 2
        )
    )
    return fig

def _build_line_scatter(df: pd.DataFrame, plan: SqlAndChartPlan):
    if not plan.x_col or not plan.y_col:
        logger.warning("Line/scatter requested but x_col or y_col is missing in plan.")
        return _build_table(df, plan)
    missing = [c for c in [plan.x_col, plan.y_col] if c not in df.columns]
    if missing:
        logger.warning("Columns %s not found in query result (available %s).", missing, list(df.columns))
        return _build_table(df, plan)
    
    color = plan.color_col if plan.color_col in df.columns else None
    
    if plan.chart_type == "line":
        fig = px.line(df, x = plan.x_col, y = plan.y_col, color = color, title = plan.title)
    else:
        fig = px.scatter(df, x = plan.x_col, y = plan.y_col, color = color, title = plan.title)
        
    if plan.y_col == "pressure":
        fig.update_yaxes(autorange = "reversed")
        
    return fig
    
def _build_table(df: pd.DataFrame, plan: SqlAndChartPlan):
    fig = go.Figure(
        data = [
            go.Table(
                header = dict(values = list(df.columns)),
                cells = dict(values = [df[col] for col in df.columns])
            )
        ]
    )
    fig.update_layout(title = plan.title, height = 500)
    return fig

def build_anomaly_figure(df: pd.DataFrame, value_col: str = "temperature"):
    """
    Scatter plot highlighting anomaly readings.
    """
    if df is None or df.empty:
        return None
    if value_col not in df.columns:
        logger.warning("build_anomaly_figure: column '%s' not in DataFrame.", value_col)
        return None
    if "is_anomaly" not in df.columns:
        logger.warning("build_anomaly_figure: 'is_anomaly' column missing.")
        return None
    
    try:
        plot_df = df.copy()
        if "timestamp" in plot_df.columns:
            plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"], errors = "coerce")
        
        normal_df = plot_df[~plot_df["is_anomaly"]].copy()
        anomaly_df = plot_df[plot_df["is_anomaly"]].copy()

        # sample normal points to avoid overplotting
        if len(normal_df) > 1000:
            normal_df = normal_df.sample(1000, random_state=42)
            sampled_notes = " (1 000 normal points sampled for clarity)"
        else:
            sampled_notes = ""
        
        x_col = "timestamp" if "timestamp" in plot_df.columns else plot_df.index
        hover_base = [c for c in ["float_id", "profile_id", "pressure"] if c in normal_df.columns]
        
        fig = px.scatter(
        normal_df,
        x = x_col,
        y = value_col,
        opacity=0.18,
        hover_data=hover_base,
        title = f"{value_col.capitalize()} with anomalies highlighted{sampled_notes}"
        )

        if not anomaly_df.empty:
            anomaly_hover = hover_base + [c for c in ["anomaly_score"] if c in anomaly_df.columns]
            anomaly_fig = px.scatter(
                anomaly_df,
                x = x_col,
                y = value_col,
                hover_data=anomaly_hover
            )

            for trace in anomaly_fig.data:
                trace.name = "anomaly"
                trace.showlegend = True
                trace.marker.color = "red"
                fig.add_trace(trace)

        for trace in fig.data:
            if trace.name == "":
                trace.name = "normal"
                trace.showlegend = True    

        return fig
    except Exception as exc:
        logger.warning("build_anomaly_figure failed: %s", exc)
        return None