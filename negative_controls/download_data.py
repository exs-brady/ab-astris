#!/usr/bin/env python3
"""Download datasets for Ab Astris negative control experiments (Table 10).

Three negative controls are used to validate the CV boundary:
    1. Cryptocurrency prices (BTC, ETH) — CoinGecko / Yahoo Finance
    2. Heart rate variability (HRV) — PhysioNet MIT-BIH Normal Sinus Rhythm
    3. Sunspot numbers — SILSO (Royal Observatory of Belgium)

Requirements:
    pip install yfinance wfdb pandas
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def download_crypto() -> None:
    """Download daily BTC and ETH prices via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("  [skip] yfinance not installed (pip install yfinance)")
        return

    DATA_DIR.mkdir(exist_ok=True)

    for ticker, name in [("BTC-USD", "btc"), ("ETH-USD", "eth")]:
        path = DATA_DIR / f"{name}_daily.csv"
        if path.exists():
            print(f"  [skip] {path.name} (already exists)")
            continue
        print(f"  Downloading {ticker}...")
        df = yf.download(ticker, start="2020-01-01", end="2024-01-01",
                         progress=False)
        df.to_csv(path)


def download_hrv() -> None:
    """Download MIT-BIH Normal Sinus Rhythm records from PhysioNet."""
    try:
        import wfdb
    except ImportError:
        print("  [skip] wfdb not installed (pip install wfdb)")
        return

    DATA_DIR.mkdir(exist_ok=True)
    hrv_dir = DATA_DIR / "nsrdb"
    hrv_dir.mkdir(exist_ok=True)

    # First 5 records used in the paper
    records = ["16265", "16272", "16273", "16420", "16483"]
    for rec in records:
        path = hrv_dir / f"{rec}.dat"
        if path.exists():
            print(f"  [skip] {rec} (already exists)")
            continue
        print(f"  Downloading PhysioNet record {rec}...")
        try:
            wfdb.dl_database("nsrdb", dl_dir=str(hrv_dir), records=[rec])
        except Exception as e:
            print(f"  ERROR downloading {rec}: {e}")


def download_sunspots() -> None:
    """Download SILSO monthly sunspot numbers."""
    import urllib.request

    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "silso_monthly.csv"

    if path.exists():
        print(f"  [skip] {path.name} (already exists)")
        return

    url = "https://www.sidc.be/SILSO/INFO/snmtotcsv.php"
    print("  Downloading SILSO monthly sunspot data...")
    urllib.request.urlretrieve(url, path)


def main() -> None:
    print("Downloading negative control datasets...\n")

    print("1. Cryptocurrency (BTC, ETH)")
    download_crypto()

    print("\n2. Heart Rate Variability (PhysioNet)")
    download_hrv()

    print("\n3. Sunspot numbers (SILSO)")
    download_sunspots()

    print(f"\nDone. Files saved to {DATA_DIR}/")


if __name__ == "__main__":
    main()
