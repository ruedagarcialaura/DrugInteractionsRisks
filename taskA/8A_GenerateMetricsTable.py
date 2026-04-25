"""
8A: EVALUATION METRICS EXTRACTOR
--------------------------------
What it does: Scans the filtered association rules CSVs, calculates key 
evaluation metrics (Total Rules, Max Lift, Avg Confidence, Unique Reactions), 
and outputs a Markdown table for the final report.
"""

import pandas as pd
import glob
import os

def generate_comparison_table(folder_path):
    print(f"Scanning directory: {folder_path} for CSV files...\n")
    
    # Use glob to find all CSV files in the filtered rules directory
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print("Error: No CSV files found. Please check the folder path.")
        return

    metrics_data = []

    for file_path in csv_files:
        filename = os.path.basename(file_path).upper()
        
        # 1. Extract Metadata from filename
        if "APRIORI" in filename:
            algo = "Apriori"
        elif "FP" in filename:
            algo = "FP-Growth"
        else:
            algo = "Unknown Algorithm"
            
        if "DRUG_NAMES" in filename:
            focus = "Drug Names"
        elif "ACTIVE_SUBSTANCES" in filename or "ACTIVESUBSTANCE" in filename:
            focus = "Active Substances"
        else:
            focus = "Unknown Focus"
            
        # Extract Support threshold
        try:
            support_val = filename.split('_')[-1].replace('.CSV', '').replace('_', '.')
        except:
            support_val = "N/A"
            
        model_configuration = f"{focus} ({algo}) - {support_val}"

        # 2. Read Data and Calculate Metrics
        df = pd.read_csv(file_path)
        
        if df.empty:
            total_rules = 0
            max_lift = 0.0
            avg_confidence = 0.0
            unique_reactions = 0
        else:
            total_rules = len(df)
            max_lift = df['lift'].max()
            avg_confidence = df['confidence'].mean()
            unique_reactions = df['consequents'].nunique()

        # 3. Store Results
        metrics_data.append({
            "Model Configuration": model_configuration,
            "Total Rules Found": total_rules,
            "Max Lift": round(max_lift, 2),
            "Avg. Confidence": round(avg_confidence, 4),
            "Unique Reactions": unique_reactions
        })

    # 4. Create a DataFrame for clean formatting
    results_df = pd.DataFrame(metrics_data)
    
    # Sort the dataframe to group by Focus Area and Algorithm
    results_df = results_df.sort_values(by="Model Configuration")

    # 5. Print the Markdown Table
    print("### Quantitative Evaluation of Filtered Rules")
    print(results_df.to_markdown(index=False))
    
    # Optionally, save to a CSV for your records
    os.makedirs("taskA/evaluation", exist_ok=True)
    results_df.to_csv("taskA/evaluation/final_metrics_table.csv", index=False)
    print("\nMetrics table successfully saved to taskA/evaluation/final_metrics_table.csv")

if __name__ == "__main__":
    # Point this to the directory where your 5 filtered CSVs are saved
    generate_comparison_table("taskA/filtered_association_rules")