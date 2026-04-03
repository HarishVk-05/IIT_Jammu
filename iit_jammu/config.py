from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw_netcdf"
PROCESSED_DIR = DATA_DIR / "processed"
SQLITE_DIR = DATA_DIR / "sqlite"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CHROMA_DIR = ARTIFACTS_DIR / "chroma"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SQLITE_DIR.mkdir(parents=True, exist_ok = True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

MEASUREMENTS_PATH = PROCESSED_DIR / "measurements.parquet"
PROFILES_PATH = PROCESSED_DIR / "profiles_metadata.parquet"
SQLITE_PATH = SQLITE_DIR / "argo.db"

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"