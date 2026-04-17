"""
6A: DEEP INTERACTION FILTERING
------------------------------
What it does: Removes "obvious" clinical noise (e.g., Drug A -> Condition A) 
and isolates multi-drug combinations linked to actual adverse reactions.
Goal: To discover genuine drug-drug interactions that are not just symptoms 
of the underlying disease.
"""

import pandas as pd
import os

def filter_true_interactions(csv_input):
    if not os.path.exists(csv_input):
        print(f"Error: {csv_input} not found. Please run the FP-Growth script first.")
        return

    print(f"Reading rules from {csv_input}...")
    df = pd.read_csv(csv_input)

    # 1. DEFINE NOISE TERMS (Indication Bias & Administrative terms)
    # We remove these from the 'consequents' because they are not side effects.
    noise_terms = [
        'NO ADVERSE EVENT', 'PRODUCT RECALL', 'RECALLED', 'CONDITION AGGRAVATED',
        'DERMATITIS ATOPIC', 'DIABETES', 'PSORIASIS', 'RHEUMATOID ARTHRITIS',
        'OFF LABEL USE', 'UNDERDOSE', 'PRODUCT USED FOR UNKNOWN INDICATION',
        'DRUG INEFFECTIVE', 'PRODUCT QUALITY ISSUE'
    ]

    # 2. FILTERING LOGIC
    # We want to keep rules where the consequent is NOT in our noise list
    def is_medical_reaction(x):
        x_upper = str(x).upper()
        return not any(term in x_upper for term in noise_terms)

    # Apply the filter to the 'consequents' column
    df_clean = df[df['consequents'].apply(is_medical_reaction)].copy()

    # 3. ISOLATE MULTI-DRUG INTERACTIONS
    # We only look for rules where the antecedent contains 2 or more items (the combination)
    # Since CSV stores them as strings, we check for commas which indicate multiple items
    true_interactions = df_clean[df_clean['antecedents'].apply(lambda x: len(str(x).split(',')) >= 2)].copy()

    # 4. SORT BY LIFT
    # High Lift in this filtered list usually points to a significant adverse signal
    true_interactions = true_interactions.sort_values('lift', ascending=False)

    print(f"\n--- Analysis Complete: {len(true_interactions)} True Interactions Found ---")
    
    if not true_interactions.empty:
        # Save to a new dedicated file
        output_path = "taskA/filtered_medical_interactions.csv"
        true_interactions.to_csv(output_path, index=False)
        
        print("\nTOP 15 GENUINE MEDICAL INTERACTIONS:")
        print(true_interactions[['antecedents', 'consequents', 'confidence', 'lift']].head(15))
        print(f"\n✅ Results saved to: {output_path}")
    else:
        print("No complex interactions found after filtering noise.")
        print("TIP: Try lowering your 'min_support' to 0.0005 in your FP-Growth script to find rarer interactions.")

if __name__ == "__main__":
    # Point this to your FP-Growth results
    filter_true_interactions("taskA/final_association_rules.csv")