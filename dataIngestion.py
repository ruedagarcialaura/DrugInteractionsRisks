import pandas as pd
import json
import glob
import os
import sys

def consolidate_fda_data(folder_path):
    """
    Reads all JSON files in a folder, counts records, and merges them.
    """
    # Use glob to find all JSON files in the specified directory
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    all_reports = []
    total_records = 0
    
    print(f"Starting data ingestion...")
    print(f"Found {len(json_files)} files.")
    
    for file in json_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Extract reports from the 'results' key
                reports = data.get('results', [])
                count = len(reports)
                
                if count > 0:
                    all_reports.extend(reports)
                    total_records += count
                    print(f"File: {os.path.basename(file)} | Records: {count}")
                else:
                    print(f"Warning: {os.path.basename(file)} has no results.")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    print("-" * 30)
    print(f"Total Consolidated Records: {total_records}")
    
    # Assignment Requirement Check 
    if total_records >= 100000:
        print("Status: SUCCESS - Dataset meets the >100k requirement.")
    else:
        print(f"Status: FAILED - You need {100000 - total_records} more records.")

    # Convert to DataFrame
    # json_normalize flattens the nested JSON structure into a table [cite: 159]
    df = pd.json_normalize(all_reports)

    # Save to Parquet to preserve nested lists for Drug Interaction Mining [cite: 110]
    output_file = "consolidated_data.parquet"
    df.to_parquet(output_file, engine='pyarrow')
    print(f"File saved successfully as: {output_file}")
    
    return df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("To run the script,specify the folder with the json files! -> Type: python dataIngestion.py <path_to_folder_with_json_files>")
    else:
        path = sys.argv[1]
        consolidate_fda_data(path)
   