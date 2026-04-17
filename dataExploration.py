"""
DATA EXPLORATION SCRIPT
-----------------------
What it does: Generates statistical visualizations (histograms, pie charts, bar plots).
Goal: To understand the dataset's demographics, most frequent medications, and 
seriousness levels. It saves all charts automatically to the /plots folder.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_plots(df):
    """
    Creates visual reports and saves them to the 'plots' folder.
    """
    # 1. Setup the plots directory
    output_folder = "plots"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    sns.set_theme(style="whitegrid")
    
    # ---------------------------------------------------------
    # 1. Seriousness and Sex (Combined Figure)
    # ---------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
    if 'serious' in df.columns:
        serious_map = {'1': 'Serious', '2': 'Non-Serious'}
        serious_data = df['serious'].map(serious_map).fillna('Unknown')
        sns.countplot(x=serious_data, ax=axes1[0], palette="viridis")
        axes1[0].set_title('General Seriousness Distribution')

    if 'patient.patientsex' in df.columns:
        sex_map = {'1': 'Male', '2': 'Female', '0': 'Unknown'}
        sex_data = df['patient.patientsex'].map(sex_map).fillna('Not Reported')
        sns.countplot(x=sex_data, ax=axes1[1], palette="magma")
        axes1[1].set_title('Patient Sex Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "seriousness_and_sex.png"))
    plt.show()

    # ---------------------------------------------------------
    # 2. PIE PLOT - Severity Indicators
    # ---------------------------------------------------------
    severity_cols = [
        'seriousnesshospitalization', 'seriousnesscongenitalanomali', 
        'seriousnessdeath', 'seriousnessdisabling', 
        'seriousnesslifethreatening', 'seriousnessother'
    ]
    
    severity_counts = {}
    for col in severity_cols:
        if col in df.columns:
            # Cleaning label for the chart
            label = col.replace('seriousness', '').capitalize()
            severity_counts[label] = (df[col] == '1').sum()

    if severity_counts and sum(severity_counts.values()) > 0:
        plt.figure(figsize=(8, 8))
        plt.pie(severity_counts.values(), labels=severity_counts.keys(), 
                autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Paired"))
        plt.title('Breakdown of Severity Indicators')
        plt.savefig(os.path.join(output_folder, "severity_indicators_pie.png"))
        plt.show()

    # ---------------------------------------------------------
    # 3. HISTOGRAMS - Top 15 Drug Names & Active Substances
    # ---------------------------------------------------------
    if 'patient.drug' in df.columns:
        drugs_exploded = df.explode('patient.drug')
        
        # Extracting names with helper functions for safety
        drug_names = drugs_exploded['patient.drug'].apply(
            lambda x: x.get('medicinalproduct') if isinstance(x, dict) else None
        ).dropna()
        
        active_substances = drugs_exploded['patient.drug'].apply(
            lambda x: x.get('activesubstance', {}).get('activesubstancename') 
            if isinstance(x, dict) and x.get('activesubstance') else None
        ).dropna()

        # Plot 1: Drug Names
        plt.figure(figsize=(12, 6))
        sns.barplot(x=drug_names.value_counts().head(15).values, 
                    y=drug_names.value_counts().head(15).index, 
                    palette="Blues_r")
        plt.title('Top 15 Most Frequent Drug Names')
        plt.xlabel('Frequency')
        plt.ylabel('Drug Name')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "top_drug_names.png"))
        plt.show() 

        # Plot 2: Active Substances
        plt.figure(figsize=(12, 6))
        sns.barplot(x=active_substances.value_counts().head(15).values, 
                    y=active_substances.value_counts().head(15).index, 
                    palette="Greens_r")
        plt.title('Top 15 Most Frequent Active Substance Names')
        plt.xlabel('Frequency')
        plt.ylabel('Active Substance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "top_active_substances.png"))
        plt.show() 

    # ---------------------------------------------------------
    # 4. Distribution of Number of Drugs per Report
    # ---------------------------------------------------------

    if 'patient.drug' in df.columns:
        # --- 4A. Using Drug Names (Brand Names) ---
        drug_name_counts = df['patient.drug'].dropna().apply(
            lambda x: len(set([d.get('medicinalproduct') for d in x if isinstance(d, dict)]))
        )

        plt.figure(figsize=(12, 6))
        sns.histplot(drug_name_counts, bins=range(1, 16), kde=False, color="skyblue", edgecolor="white")
        plt.title('Distribution: Number of UNIQUE Brand Names per Report')
        plt.xlabel('Number of Unique Brands')
        plt.ylabel('Frequency')
        plt.xlim(1, 15)
        plt.xticks(range(1, 16))
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "dist_drug_names_per_report.png"))
        plt.show()

        # --- 4B. Using Active Substances (Chemical Components) ---        
        def extract_active_substances(drug_list):
            substances = []
            for d in drug_list:
                if not isinstance(d, dict): continue
                
                # Active substance is usually inside 'activesubstance'
                active_info = d.get('activesubstance')
                
                # Case 1: It's a list (common in FAERS)
                if isinstance(active_info, list) and len(active_info) > 0:
                    name = active_info[0].get('activesubstancename')
                    if name: substances.append(name.upper())
                
                # Case 2: It's a direct dictionary
                elif isinstance(active_info, dict):
                    name = active_info.get('activesubstancename')
                    if name: substances.append(name.upper())
            
            return len(set(substances)) # Count unique names only

        active_substance_counts = df['patient.drug'].dropna().apply(extract_active_substances)

        plt.figure(figsize=(12, 6))
        sns.histplot(active_substance_counts, bins=range(1, 16), kde=False, color="salmon", edgecolor="white")
        plt.title('Distribution: Number of UNIQUE Active Substances per Report')
        plt.xlabel('Number of Unique Active Substances')
        plt.ylabel('Frequency')
        plt.xlim(1, 15)
        plt.xticks(range(1, 16))
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "dist_active_substances_per_report.png"))
        plt.show()

def run_data_exploration(parquet_path):
    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Total Records: {len(df)}")
    
    generate_plots(df)
    return df

if __name__ == "__main__":
    parquet_file = "consolidated_data.parquet"
    try:
        df_final = run_data_exploration(parquet_file)
        print("\nAll plots have been saved/overwritten in the 'plots' folder.")
    except FileNotFoundError:
        print("Error: Run consolidation script first.")