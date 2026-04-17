"""
TASK A: PREPROCESSING SCRIPT
----------------------------
What it does: Extracts and cleans drug names, active substances, and patient 
reactions from the raw nested dictionaries. It then groups them by report ID.
Goal: To create two separate transactional datasets ("Brand Names + Reactions" 
and "Active Substances + Reactions") to compare which one yields better 
association rules.
"""

import pandas as pd

def preprocess_for_association(parquet_path):

    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # 1. Filter only necessary columns for Task A
    cols_needed = ['safetyreportid', 'patient.drug', 'patient.reaction']
    df_a = df[cols_needed].copy()

    print("Exploding drug and reaction lists...")

    # 2. Explode the 'patient.drug' list to get individual drug dictionaries
    df_drugs = df_a.explode('patient.drug')
    
    # Extract the 'medicinalproduct' name and the active substance name from the dictionary
    # We use .str.get() because each entry is now a dictionary
    df_drugs['drug_name'] = df_drugs['patient.drug'].str.get('medicinalproduct')
    df_drugs['activesubstance_name'] = df_drugs['patient.drug'].str.get('activesubstance').str.get('activesubstancename')

    # 3. Explode the 'patient.reaction' list
    df_reactions = df_a.explode('patient.reaction')
    df_reactions['reaction_name'] = df_reactions['patient.reaction'].str.get('reactionmeddrapt')

    # 4. Clean up names (uppercase and strip whitespace)
    df_drugs['drug_name'] = df_drugs['drug_name'].str.strip().str.upper()
    df_drugs['activesubstance_name'] = df_drugs['activesubstance_name'].str.strip().str.upper()
    df_reactions['reaction_name'] = df_reactions['reaction_name'].str.strip().str.upper()

    # 5. Create the Transactional Format
    # Group by report ID to get a list of all drugs and reactions for that specific case
    transactions_drugs = df_drugs.groupby('safetyreportid')['drug_name'].apply(list)
    transactions_reac = df_reactions.groupby('safetyreportid')['reaction_name'].apply(list)
    transactions_activesubstance = df_drugs.groupby('safetyreportid')['activesubstance_name'].apply(list)

    # Merge them into final transaction lists while removing duplicates
    df_final_drug_names = pd.concat([transactions_drugs, transactions_reac, transactions_activesubstance], axis=1)
    df_final_drug_names['all_items'] = (df_final_drug_names['drug_name'] + df_final_drug_names['reaction_name']).apply(lambda x: list(set(x)))

    df_final_activesubstance = pd.concat([transactions_activesubstance, transactions_reac], axis=1)
    df_final_activesubstance['all_items'] = (df_final_activesubstance['activesubstance_name'] + df_final_activesubstance['reaction_name']).apply(lambda x: list(set(x)))

    # 6. Save the preprocessed data
    output_path_drug_names = "taskA/task_a_transactions_drug_names.parquet"
    df_final_drug_names[['all_items']].to_parquet(output_path_drug_names)
    print(f"{'='*50}")
    print(f"Success! Preprocessed {len(df_final_drug_names)} drug names transactions.")
    print(f"Sample transaction: {df_final_drug_names['all_items'].iloc[0]}")
    print(f"{'='*50}\n")
    
    output_path_activesubstance = "taskA/task_a_transactions_activesubstance.parquet"
    df_final_activesubstance[['all_items']].to_parquet(output_path_activesubstance)
    print(f"{'='*50}")
    print(f"Success! Preprocessed {len(df_final_activesubstance)} active substance transactions.")
    print(f"Sample transaction: {df_final_activesubstance['all_items'].iloc[0]}")
    print(f"{'='*50}\n")
   
    return df_final_drug_names, df_final_activesubstance


if __name__ == "__main__":
    preprocess_for_association("consolidated_data.parquet")