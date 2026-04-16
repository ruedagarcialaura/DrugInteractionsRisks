import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots(df):
    """
    Creates visual reports for data imbalance and demographics.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Target Variable Imbalance (Critical for Classification Task)
    if 'serious' in df.columns:
        # Map values for better readability: 1 is Serious, 2 is Non-Serious
        serious_map = {'1': 'Serious', '2': 'Non-Serious'}
        serious_data = df['serious'].map(serious_map).fillna('Unknown')
        
        sns.countplot(x=serious_data, ax=axes[0], palette="viridis")
        axes[0].set_title('Target Variable: Seriousness Distribution')
        axes[0].set_xlabel('Outcome')
        axes[0].set_ylabel('Number of Reports')

    # Plot 2: Patient Sex Distribution
    if 'patient.patientsex' in df.columns:
        # FAERS Codes: 1=Male, 2=Female
        sex_map = {'1': 'Male', '2': 'Female', '0': 'Unknown'}
        sex_data = df['patient.patientsex'].map(sex_map).fillna('Not Reported')
        
        sns.countplot(x=sex_data, ax=axes[1], palette="magma")
        axes[1].set_title('Demographics: Patient Sex')
        axes[1].set_xlabel('Sex')
        axes[1].set_ylabel('Number of Reports')

    plt.tight_layout()
    plt.show()

    # Plot 3: Distribution of Number of Drugs per Report
    if 'patient.drug' in df.columns:
        plt.figure(figsize=(10, 5))
        drug_counts = df['patient.drug'].dropna().apply(len)
        
        # Limit to 15 drugs for better visibility of the "long tail"
        sns.histplot(drug_counts, bins=range(1, 16), kde=False, color="skyblue")
        plt.title('Distribution: Number of Drugs per Adverse Event Report')
        plt.xlabel('Number of Drugs')
        plt.ylabel('Frequency')
        plt.xlim(1, 15)
        plt.show()

def run_data_exploration(parquet_path):
    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    print("-" * 30)
    print("--- DATASET OVERVIEW ---")
    print(f"Total Records: {len(df)}")
    
    # 1. Target Imbalance Analysis 
    print("\n--- TARGET VARIABLE ANALYSIS ---")
    if 'serious' in df.columns:
        counts = (df['serious'].value_counts(normalize=True) * 100).round(2).astype(str) + '%'
        print(f"Imbalance Ratio:\n{counts}")

    # 2. Demographic Inspection [cite: 80, 101]
    print("\n--- DEMOGRAPHICS CHECK ---")
    for field in ['patient.patientonsetage', 'patient.patientsex']:
        if field in df.columns:
            print(f"Missing {field}: {df[field].isnull().sum()}")

    # 3. Trigger Visualizations
    generate_plots(df)

    return df

if __name__ == "__main__":
    parquet_file = "consolidated_data.parquet"
    try:
        df_final = run_data_exploration(parquet_file)
    except FileNotFoundError:
        print("Error: Run consolidation script first.")