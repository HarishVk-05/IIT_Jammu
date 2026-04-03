from sqlalchemy import create_engine, text
import pandas as pd
from iit_jammu.config import MEASUREMENTS_PATH, PROFILES_PATH, SQLITE_PATH

def main() -> None:
    if not MEASUREMENTS_PATH.exists() or not PROFILES_PATH.exists():
        raise FileNotFoundError ("Run ingest/parse_netcdf.py first")
    
    measurements = pd.read_parquet(MEASUREMENTS_PATH)
    profiles = pd.read_parquet(PROFILES_PATH)

    engine = create_engine(f"sqlite:///{SQLITE_PATH}")
    with engine.begin() as conn:
        measurements.to_sql("measurements", conn, if_exists="replace", index = False)
        profiles.to_sql("profiles_metadata", conn, if_exists="replace", index = False)

        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_measurements_profile_id ON measurements(profile_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_measurements_float_id ON measurements(float_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_measurements_timestamp ON measurements(timestamp)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_profiles_profile_id ON profiles_metadata(profile_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_profiles_float_id ON profiles_metadata(float_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_profiles_timestamp ON profiles_metadata(timestamp)"))
    
    print(f"SQLite DB ready at {SQLITE_PATH}")
    print(f"Measurements rows: {len(measurements)}")
    print(f"Profiles rows: {len(profiles)}")

if __name__ == "__main__":
    main()