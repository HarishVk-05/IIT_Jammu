from __future__ import annotations

import shutil
import json
import pandas as pd

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface  import HuggingFaceEmbeddings

from iit_jammu.config import PROFILES_PATH, MEASUREMENTS_PATH, CHROMA_DIR, EMBED_MODEL

SCHEMA_DOCS = [
    Document(
        page_content = (
            "Table: profiles_metadata\n"
            "Columns: profile_id, float_id, cycle_number, timestamp, direction, latitude, longitude, "
            "data_mode, n_points, min_pressure, max_pressure, has_temp, has_salinity, source_file\n"
            "Use this table to answer profile-level, float-level, date-range, geospatial, and trajectory questions."
        ),
        metadata = {"source": "schema", "table": "profiles_metadata"}
    ),
    Document(
        page_content=(
            "Table: measurements\n"
            "Columns: profile_id, float_id, cycle_number, timestamp, direction, latitude, longitude, pressure, "
            "temperature, salinity, position_qc, pressure_qc, temperature_qc, salinity_qc, data_mode, source_file\n"
            "Use this table for depth profiles and parameter analysis like temperature or salinity vs pressure."
        ),
        metadata = {"source": "schema", "table": "measurements"}
    ),
    Document(
        page_content=(
            "Domain notes: pressure is depth-like and increases downward. For profile plots, pressure should usually be shown on the y-axis and inverted. "
            "Temperature column is named temperature. Salinity column is named salinity. Latitude and longitude are in degrees."
        ),
        metadata = {"source": "schema", "table": "domain"}
    )
]

GLOSSARY_DOCS = [
    Document(
        page_content=(
            "Salinity is the amount of dissolved salts in seawater. "
            "In this dataset, salinity is stored in the measurements table as the salinity column. "
            "It can be analyzed across time, float, profile, and pressure."
        ),
        metadata={"source": "glossary", "term": "salinity"}
    ),
    Document(
        page_content=(
            "Temperature is the seawater temperature measurement stored in the measurements table. "
            "It can be analyzed over time or against pressure for vertical profiles."
        ),
        metadata={"source": "glossary", "term": "temperature"}
    ),
    Document(
        page_content=(
            "Pressure is the depth-like vertical coordinate in ARGO data. "
            "For profile plots, pressure is usually shown on the y-axis and interpreted as increasing downward."
        ),
        metadata={"source": "glossary", "term": "pressure"}
    ),
    Document(
        page_content=(
            "ARGO is a global ocean observing system of profiling floats. "
            "These floats collect measurements such as temperature and salinity across time and location."
        ),
        metadata={"source": "glossary", "term": "argo"}
    ),
    Document(
        page_content=(
            "A profile is one vertical sampling event from a float. "
            "A profile groups measurements taken at different pressures for a float at a given timestamp or cycle."
        ),
        metadata={"source": "glossary", "term": "profile"}
    ),
    Document(
        page_content=(
            "QC means quality control. "
            "Columns like temperature_qc, salinity_qc, pressure_qc, and position_qc indicate data-quality status."
        ),
        metadata={"source": "glossary", "term": "qc"}
    )
]

def build_profile_docs(profiles: pd.DataFrame) -> list[Document]:
    docs: list[Document] = []
    for _, row in profiles.iterrows():
        text = (
            f"Profile {row.profile_id} from float {row.float_id}, cycle {row.cycle_number}, timestamp {row.timestamp}, "
            f"direction {row.direction}, location ({row.latitude}, {row.longitude}), data mode {row.data_mode}, "
            f"points {row.n_points}, pressure range {row.min_pressure} to {row.max_pressure}, "
            f"has_temp={row.has_temp}, has_salinity={row.has_salinity}."
        )
        docs.append(
            Document(
                page_content=text,
                metadata = {
                    "source": "profile_metadata",
                    "profile_id": str(row.profile_id),
                    "float_id": str(row.float_id),
                    "timestamp": str(row.timestamp)
                }
            )
        )
    return docs

def build_stats_doc(profiles: pd.DataFrame, measurements: pd.DataFrame) -> Document:
    payload = {
        "n_profiles": int(len(profiles)),
        "n_measurements": int(len(measurements)),
        "n_unique_floats": int(profiles["float_id"].astype(str).nunique()),
        "timestamp_min": str(profiles["timestamp"].min()),
        "timestamp_max": str(profiles["timestamp"].max()),
        "latitude_min": float(profiles["latitude"].min()),
        "latitude_max": float(profiles["latitude"].max()),
        "longitude_min": float(profiles["longitude"].min()),
        "longitude_max": float(profiles["longitude"].max()),
    }
    return Document(
        page_content="Dataset summary: " + json.dumps(payload, indent = 2),
        metadata = {"source": "dataset_summary"}
    )

def vector_store_exists() -> bool:
    return CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())

def get_vectordb() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name = EMBED_MODEL)
    return Chroma(
        collection_name = "Argo_profiles",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
def main(force_rebuild: bool = False) -> None:
    if vector_store_exists() and not force_rebuild:
        print(f"Vector store already exists at {CHROMA_DIR}. Loading existing store and skipping rebuild.")
        vectordb = get_vectordb()
        print(vectordb)
        return 
    
    if not PROFILES_PATH.exists() or not MEASUREMENTS_PATH.exists():
        raise FileNotFoundError("Run ingest/parse_netcdf.py first")
    
    profiles = pd.read_parquet(PROFILES_PATH)
    measurements = pd.read_parquet(MEASUREMENTS_PATH)

    docs = []
    docs.extend(SCHEMA_DOCS)
    docs.extend(GLOSSARY_DOCS)
    docs.append(build_stats_doc(profiles, measurements))
    docs.extend(build_profile_docs(profiles))

    if force_rebuild and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    vectordb = get_vectordb()
    vectordb.add_documents(docs)

    print(f"Indexed {len(docs)} documents into {CHROMA_DIR}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rebuild", action = "store_true")
    args = parser.parse_args()

    main(force_rebuild=args.force_rebuild)