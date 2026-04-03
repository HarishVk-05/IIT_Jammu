from typing import Literal, Optional
from pydantic import BaseModel, Field

AnalysisScope = Literal["global", "per_float", "per_profile"]

class QueryPlan(BaseModel):
    intent: Literal[
        "dataset_summary",
        "trajectory_map",
        "profile_plot",
        "time_trend",
        "comparison_plot",
        "anomaly_detection",
        "table",
        "concept_explanation",
        "unsupported"
    ] = Field(description="Best-fit response type")

    parameter: Optional[Literal[
        "temperature", "salinity", "pressure", "latitude", "longitude", "timestamp"
        ]] = None
    parameter_2: Optional[Literal[
        "temperature", "salinity", "pressure", "latitude", "longitude", "timestamp"
        ]] = None
    
    float_id: Optional[str] = None
    profile_id: Optional[str] = None

    start_date: Optional[str] = Field(default=None, description="YYYY-MM-DD if present")
    end_date: Optional[str] = Field(default=None, description="YYYY-MM-DD if present")
    
    latitude_min: Optional[float] = None
    latitude_max: Optional[float] = None
    longitude_min: Optional[float] = None
    longitude_max: Optional[float] = None

    depth_min_m: Optional[float] = None
    depth_max_m: Optional[float] = None

    analysis_scope: Optional[AnalysisScope] = None
    top_k: int = 20

    limit: int = 200
    user_goal: str = Field(default="", description="Short plain English restatement")

class SqlAndChartPlan(BaseModel):
    sql: str = Field(description="Single SELECT query only")
    chart_type: Literal["map", "line", "scatter", "table"]
    x_col: Optional[str] = None
    y_col: Optional[str] = None
    color_col: Optional[str] = None
    title: str
    explanation: str

