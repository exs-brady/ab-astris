"""
Tremorscope Bulk Data Download
==============================
Downloads continuous day-long waveforms for all 4 volcanoes as miniSEED files.
Run this on your local machine. Requires: pip install obspy

Usage:
    python tremorscope_download.py              # Download all volcanoes
    python tremorscope_download.py kilauea      # Download one volcano
    python tremorscope_download.py --check      # Check what's already downloaded

Output structure:
    tremorscope_data/
    ├── kilauea_2018/
    │   ├── HV.UWE.HHZ.2018-04-01.mseed
    │   ├── HV.UWE.HHZ.2018-04-02.mseed
    │   └── ...
    ├── msh_2004/
    │   ├── UW.HSR.EHZ.2004-08-15.mseed
    │   └── ...
    ├── pavlof_2016/
    │   └── ...
    └── augustine_2006/
        └── ...

Each file is one calendar day of 100 Hz seismic data (~35 MB uncompressed).
Total download: ~2000 day-files, ~50-70 GB. Takes 2-4 hours depending on connection.

If a file already exists, it's skipped — safe to restart if interrupted.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import time

try:
    from obspy.clients.fdsn import Client
    from obspy import UTCDateTime
except ImportError:
    print("ERROR: ObsPy not installed. Run: pip install obspy")
    sys.exit(1)

# Fix SSL certificate issues on macOS
import certifi
os.environ.setdefault('SSL_CERT_FILE', certifi.where())


# ═════════════════════════════════════════════════════════════════════════════
# VOLCANO CONFIGURATIONS
# ═════════════════════════════════════════════════════════════════════════════

VOLCANOES = {
    'kilauea': {
        'name': 'Kīlauea 2018',
        'network': 'HV',
        'station': 'UWE',
        'channel': 'HHZ',
        'start': '2018-04-01',
        'end': '2018-08-15',
        'folder': 'kilauea_2018',
    },
    'msh': {
        'name': 'Mt St Helens 2004',
        'network': 'UW',
        'station': 'HSR',
        'channel': 'EHZ',
        'start': '2004-09-01',
        'end': '2005-04-01',
        'folder': 'msh_2004',
    },
    'pavlof': {
        'name': 'Pavlof 2016',
        'network': 'AV',
        'station': 'PS4A',
        'channel': 'EHZ',
        'start': '2016-02-15',
        'end': '2016-04-30',
        'folder': 'pavlof_2016',
    },
    'augustine': {
        'name': 'Augustine 2006',
        'network': 'AV',
        'station': 'AUH',
        'channel': 'EHZ',
        'start': '2005-11-15',
        'end': '2006-04-01',
        'folder': 'augustine_2006',
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# DOWNLOAD ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def download_volcano(key, config, base_dir, client):
    """Download all day-files for one volcano."""
    vol_dir = base_dir / config['folder']
    vol_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.strptime(config['start'], '%Y-%m-%d')
    end = datetime.strptime(config['end'], '%Y-%m-%d')
    total_days = (end - start).days

    net = config['network']
    sta = config['station']
    cha = config['channel']

    print(f"\n{'='*65}")
    print(f"  {config['name']}")
    print(f"  {net}.{sta}.{cha}")
    print(f"  {config['start']} → {config['end']} ({total_days} days)")
    print(f"  Output: {vol_dir}/")
    print(f"{'='*65}")

    downloaded = 0
    skipped = 0
    failed = 0
    failed_dates = []

    for day_offset in range(total_days):
        date = start + timedelta(days=day_offset)
        date_str = date.strftime('%Y-%m-%d')
        filename = f"{net}.{sta}.{cha}.{date_str}.mseed"
        filepath = vol_dir / filename

        # Progress
        pct = (day_offset + 1) / total_days * 100
        status_line = f"  [{pct:5.1f}%] {date_str}"

        # Skip if exists
        if filepath.exists() and filepath.stat().st_size > 1000:
            skipped += 1
            if day_offset % 20 == 0:  # Print every 20th skip to show progress
                print(f"{status_line} — skipped (exists)")
            continue

        # Download
        t1 = UTCDateTime(date_str + "T00:00:00")
        t2 = t1 + 86400  # 24 hours

        retries = 3
        for attempt in range(retries):
            try:
                st = client.get_waveforms(net, sta, '*', cha, t1, t2)
                st.merge(method=1, fill_value='interpolate')
                st.write(str(filepath), format='MSEED')
                downloaded += 1
                size_mb = filepath.stat().st_size / 1e6
                print(f"{status_line} — OK ({size_mb:.1f} MB)")
                break
            except Exception as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    print(f"{status_line} — retry {attempt+1} ({e})")
                    time.sleep(wait)
                else:
                    failed += 1
                    failed_dates.append(date_str)
                    print(f"{status_line} — FAILED: {e}")

        # Be polite to the server
        time.sleep(0.3)

    print(f"\n  Done: {downloaded} downloaded, {skipped} skipped, {failed} failed")
    if failed_dates:
        print(f"  Failed dates: {', '.join(failed_dates[:10])}")
        if len(failed_dates) > 10:
            print(f"  ... and {len(failed_dates)-10} more")

    return downloaded, skipped, failed


def check_status(base_dir):
    """Show download status for all volcanoes."""
    print(f"\n{'='*65}")
    print(f"  DOWNLOAD STATUS")
    print(f"  Base directory: {base_dir}")
    print(f"{'='*65}")

    for key, config in VOLCANOES.items():
        vol_dir = base_dir / config['folder']
        start = datetime.strptime(config['start'], '%Y-%m-%d')
        end = datetime.strptime(config['end'], '%Y-%m-%d')
        total_days = (end - start).days

        if vol_dir.exists():
            files = list(vol_dir.glob('*.mseed'))
            valid = [f for f in files if f.stat().st_size > 1000]
            size_gb = sum(f.stat().st_size for f in valid) / 1e9
            print(f"\n  {config['name']:<25} {len(valid):>4}/{total_days} days "
                  f"({len(valid)/total_days*100:.0f}%)  [{size_gb:.2f} GB]")

            # Find gaps
            existing_dates = set()
            for f in valid:
                parts = f.stem.split('.')
                if len(parts) >= 4:
                    existing_dates.add(parts[3])

            gaps = []
            for d in range(total_days):
                date = start + timedelta(days=d)
                if date.strftime('%Y-%m-%d') not in existing_dates:
                    gaps.append(date.strftime('%Y-%m-%d'))

            if gaps and len(gaps) <= 10:
                print(f"    Gaps: {', '.join(gaps)}")
            elif gaps:
                print(f"    Gaps: {len(gaps)} missing days")
        else:
            print(f"\n  {config['name']:<25}    0/{total_days} days (not started)")


def main():
    base_dir = Path('tremorscope_data')
    base_dir.mkdir(exist_ok=True)

    # Parse args
    args = sys.argv[1:]

    if '--check' in args:
        check_status(base_dir)
        return

    # Which volcanoes to download?
    if args:
        targets = {}
        for arg in args:
            arg = arg.lower().strip('-')
            if arg == 'check':
                continue
            # Fuzzy match
            matched = None
            for key in VOLCANOES:
                if arg in key or key in arg:
                    matched = key
                    break
            if matched:
                targets[matched] = VOLCANOES[matched]
            else:
                print(f"Unknown volcano: {arg}")
                print(f"Available: {', '.join(VOLCANOES.keys())}")
                return
    else:
        targets = VOLCANOES

    # Connect — use service_mappings for cross-version ObsPy compatibility
    # _discover_services=False is required when using custom service_mappings
    print("Connecting to EarthScope FDSN...")
    client = Client(
        base_url='https://service.earthscope.org',
        service_mappings={
            'dataselect': 'https://service.earthscope.org/fdsnws/dataselect/1',
            'station': 'https://service.earthscope.org/fdsnws/station/1',
        },
        _discover_services=False
    )
    print("Connected.")

    total_stats = {'downloaded': 0, 'skipped': 0, 'failed': 0}
    t_start = time.time()

    for key, config in targets.items():
        d, s, f = download_volcano(key, config, base_dir, client)
        total_stats['downloaded'] += d
        total_stats['skipped'] += s
        total_stats['failed'] += f

    elapsed = time.time() - t_start

    print(f"\n{'='*65}")
    print(f"  ALL COMPLETE")
    print(f"  Downloaded: {total_stats['downloaded']}")
    print(f"  Skipped:    {total_stats['skipped']}")
    print(f"  Failed:     {total_stats['failed']}")
    print(f"  Time:       {elapsed/60:.1f} minutes")
    print(f"{'='*65}")

    # Show final status
    check_status(base_dir)

    print(f"\n  Next step: Upload the tremorscope_data/ folder (or zip it)")
    print(f"  and we'll run the timeline analysis.")


if __name__ == '__main__':
    main()