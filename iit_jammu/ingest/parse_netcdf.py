from iit_jammu.config import RAW_DIR, MEASUREMENTS_PATH, PROFILES_PATH
import pandas as pd
import xarray as xr


def build_profile_id(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(df["timestamp"], errors = "coerce")
    return (
        df["float_id"].astype("string").fillna("NA") + "_"
        + df["cycle_number"].astype("string").fillna("NA") + "_"
        + ts.dt.strftime("%Y%m%dT%H%M%S").fillna("NA") + "_"
        + df["direction"].astype("string").fillna("NA")
    )

def parse_one_file(nc_path):
    with xr.open_dataset(nc_path) as ds:
        df = ds.to_dataframe().reset_index(drop = True)
    
    rename_map = {
        "PLATFORM_NUMBER": "float_id",
        "CYCLE_NUMBER": "cycle_number",
        "TIME": "timestamp",
        "DIRECTION": "direction",
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
        "PRES": "pressure",
        "TEMP": "temperature",
        "PSAL": "salinity",
        "POSITION_QC": "position_qc",
        "PRES_QC": "pressure_qc",
        "TEMP_QC": "temperature_qc",
        "PSAL_QC": "salinity_qc",
        "DATA_MODE": "data_mode"
    }

    df = df.rename(columns = rename_map)
    keep_cols = [
        "float_id", "cycle_number", "timestamp", "direction",
        "latitude", "longitude", "pressure", "temperature", "salinity",
        "position_qc", "pressure_qc", "temperature_qc", "salinity_qc",
        "data_mode"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors = "coerce")
    str_cols = [
        "float_id", "cycle_number", "direction", "data_mode",
        "position_qc", "pressure_qc", "temperature_qc", "salinity_qc"
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")
    
    numeric_cols = ["latitude", "longitude", "pressure", "temperature", "salinity"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors = "coerce")
        
    df["source_file"] = nc_path.name
    df["profile_id"] = build_profile_id(df)
    measurements_df = df.copy()

    profiles_df = (
        df.groupby("profile_id", dropna=False).agg(
            float_id = ("float_id", "first"),
            cycle_number = ("cycle_number", "first"),
            timestamp = ("timestamp", "first"),
            direction = ("direction", "first"),
            latitude = ("latitude", "first"),
            longitude = ("longitude", "first"),
            data_mode = ("data_mode", "first"),
            n_points = ("profile_id", "size"),
            min_pressure = ("pressure", "min"),
            max_pressure = ("pressure", "max"),
            has_temp = ("temperature", lambda s: bool(s.notna().any())),
            has_salinity = ("salinity", lambda s: bool(s.notna().any())),
            source_file = ("source_file", "first")
        ).reset_index()
    )
    return measurements_df, profiles_df

def main() -> None:
    all_measurements = []
    all_profiles = []
    files = sorted(RAW_DIR.glob("*.nc"))

    if not files:
        raise FileNotFoundError(f"No .nc files found in {RAW_DIR}")
    
    for path in files:
        print(f"Parsing {path.name} ...")
        meas_df, prof_df = parse_one_file(path)
        all_measurements.append(meas_df)
        all_profiles.append(prof_df)
    
    measurements_df = pd.concat(all_measurements, ignore_index=True)
    profiles_df = pd.concat(all_profiles, ignore_index=True)


    measurements_df.to_parquet(MEASUREMENTS_PATH, index = False)
    profiles_df.to_parquet(PROFILES_PATH, index = False)

    print("\nSaved: ")
    print(MEASUREMENTS_PATH, len(measurements_df))
    print(PROFILES_PATH, len(profiles_df))

    print("\nSample profiles:")
    print(profiles_df.head())
    
    print("\nSample measurements:")
    print(measurements_df.head())

if __name__ == "__main__":
    main()