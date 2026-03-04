#!/usr/bin/env python3
"""Download NOAA CO-OPS tide gauge data for Ab Astris Table 9.

Uses the NOAA CO-OPS API to fetch hourly water level observations
and harmonic predictions for 6 stations spanning diverse tidal regimes.

Stations (see Table 9 of the paper):
    9414290  San Francisco, CA   (mixed semidiurnal)
    8518750  The Battery, NY     (semidiurnal)
    8658120  Wilmington, NC      (semidiurnal)
    8724580  Key West, FL        (mixed, mainly diurnal)
    1612340  Honolulu, HI        (mixed semidiurnal)
    9447130  Seattle, WA         (mixed semidiurnal)

Requirements:
    pip install noaa-coops pandas
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    import noaa_coops
except ImportError:
    raise ImportError(
        "noaa-coops is required: pip install noaa-coops"
    )


STATIONS = {
    "9414290": "San Francisco, CA",
    "8518750": "The Battery, NY",
    "8658120": "Wilmington, NC",
    "8724580": "Key West, FL",
    "1612340": "Honolulu, HI",
    "9447130": "Seattle, WA",
}

DATA_DIR = Path(__file__).parent / "data"


def download_station(station_id: str, year: int = 2023) -> None:
    """Download hourly observations and predictions for one station."""
    DATA_DIR.mkdir(exist_ok=True)

    station = noaa_coops.Station(station_id)
    begin = f"{year}0101"
    end = f"{year}1231"

    # Hourly water level observations
    obs_path = DATA_DIR / f"{station_id}_hourly_height_{year}_{year}.csv"
    if not obs_path.exists():
        print(f"  Downloading observations for {station_id}...")
        obs = station.get_data(
            begin_date=begin, end_date=end,
            product="hourly_height", datum="MSL",
            units="metric", time_zone="gmt",
        )
        obs.to_csv(obs_path)
    else:
        print(f"  [skip] {obs_path.name} (already exists)")

    # Harmonic predictions (ground truth)
    pred_path = DATA_DIR / f"{station_id}_predictions_{year}_{year}.csv"
    if not pred_path.exists():
        print(f"  Downloading predictions for {station_id}...")
        pred = station.get_data(
            begin_date=begin, end_date=end,
            product="predictions", datum="MSL",
            units="metric", time_zone="gmt",
        )
        pred.to_csv(pred_path)
    else:
        print(f"  [skip] {pred_path.name} (already exists)")


def main() -> None:
    print("Downloading NOAA CO-OPS tide gauge data...\n")
    for sid, name in STATIONS.items():
        print(f"Station {sid} — {name}")
        try:
            download_station(sid)
        except Exception as e:
            print(f"  ERROR: {e}")
    print(f"\nDone. Files saved to {DATA_DIR}/")


if __name__ == "__main__":
    main()
