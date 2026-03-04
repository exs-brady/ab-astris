#!/usr/bin/env bash
# Download CWRU Bearing Data Center files used in Ab Astris Table 5.
#
# Source: Case Western Reserve University Bearing Data Center
#         https://engineering.case.edu/bearingdatacenter
#
# The experiment uses normal baseline + inner race fault recordings
# at 12 kHz sampling rate (drive-end accelerometer).
#
# Files downloaded (Matlab .mat format):
#   - Normal baseline: 97.mat (0 HP load)
#   - Inner race faults: 105.mat (0.007"), 169.mat (0.014"), 209.mat (0.021")
#   - Ball faults: 118.mat (0.007"), 185.mat (0.014"), 222.mat (0.021")
#   - Outer race faults: 130.mat (0.007"), 197.mat (0.014"), 234.mat (0.021")

set -euo pipefail

DATA_DIR="$(dirname "$0")/data"
BASE_URL="https://engineering.case.edu/sites/default/files"

mkdir -p "$DATA_DIR"

# File IDs used in the paper experiments
FILES=(
    "97.mat"    # Normal baseline
    "105.mat"   # IR fault 0.007"
    "169.mat"   # IR fault 0.014"
    "209.mat"   # IR fault 0.021"
    "118.mat"   # Ball fault 0.007"
    "185.mat"   # Ball fault 0.014"
    "222.mat"   # Ball fault 0.021"
    "130.mat"   # OR fault 0.007" (centered)
    "197.mat"   # OR fault 0.014" (centered)
    "234.mat"   # OR fault 0.021" (centered)
)

echo "Downloading CWRU bearing fault data..."
for f in "${FILES[@]}"; do
    if [ -f "$DATA_DIR/$f" ]; then
        echo "  [skip] $f (already exists)"
    else
        echo "  [download] $f"
        curl -sL "$BASE_URL/$f" -o "$DATA_DIR/$f"
    fi
done

echo "Done. Files saved to $DATA_DIR/"
