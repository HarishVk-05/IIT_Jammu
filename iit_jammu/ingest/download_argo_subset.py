
from iit_jammu.config import RAW_DIR
from argopy import DataFetcher

# [lon_min, lon_max, lat_min, lat_max, depth_min, depth_max, date_min, date_max]
BOX = [60, 90, -10, 20, 0, 500, "2024-01-01", "2024-03-01"]

def main() -> None:
    fetcher = DataFetcher(src = "erddap").region(BOX)
    ds = fetcher.load().data
    out_file = RAW_DIR / "indian_ocean_argo_2024_jan_feb.nc"
    ds.to_netcdf(out_file)
    print(f"Saved to: {out_file}")
    print(ds)

if __name__ == "__main__":
    main()