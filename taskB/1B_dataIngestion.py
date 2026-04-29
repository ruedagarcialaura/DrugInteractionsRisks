"""
TASK B: DATA INGESTION SCRIPT
------------------------------
What it does: Reads the 9 FAERS quarterly JSON files directly from the source
zip archive and consolidates them into a single parquet file.
Goal: Produce consolidated_data.parquet (>100k records) shared with Task A,
without extracting the zip to disk.
"""

import pandas as pd
import json
import zipfile
import sys
import os


def consolidate_from_zip(zip_path):
    """
    Reads all JSON files inside a zip archive, normalizes them, and saves to parquet.
    """
    all_reports = []
    total_records = 0

    print(f"Opening zip: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        json_names = [name for name in zf.namelist() if name.endswith('.json')]
        print(f"Found {len(json_names)} JSON files.")

        for name in json_names:
            try:
                with zf.open(name) as f:
                    data = json.loads(f.read().decode('utf-8'))
                reports = data.get('results', [])
                count = len(reports)
                if count > 0:
                    all_reports.extend(reports)
                    total_records += count
                    print(f"  {os.path.basename(name)} | Records: {count}")
                else:
                    print(f"  Warning: {os.path.basename(name)} has no results.")
            except Exception as e:
                print(f"  Error reading {name}: {e}")

    print("-" * 40)
    print(f"Total Consolidated Records: {total_records:,}")

    if total_records >= 100_000:
        print("Status: SUCCESS - Dataset meets the >100k requirement.")
    else:
        print(f"Status: FAILED - Need {100_000 - total_records:,} more records.")

    df = pd.json_normalize(all_reports)
    output_file = "consolidated_data.parquet"
    df.to_parquet(output_file, engine='pyarrow')
    print(f"Saved: {output_file}  ({len(df):,} rows x {len(df.columns)} columns)")
    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python taskB/1B_dataIngestion.py <path_to_zip>")
        print("Example: python taskB/1B_dataIngestion.py \"taskB/9 json files - no tocar.zip\"")
    else:
        consolidate_from_zip(sys.argv[1])
