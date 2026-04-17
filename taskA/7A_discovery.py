"""
8A: CLINICAL SIGNAL DISCOVERY (0.002 Support Optimized)
------------------------------------------------------
Goal: Extract non-obvious interactions from high-support data by
stripping away administrative noise and known clinical protocols.
"""

import pandas as pd

def find_clinical_signals(csv_input):
    df = pd.read_csv(csv_input)

    # 1. THE "NOISE" BLACKLIST
    # We remove anything that is a known protocol or administrative fact
    blacklist = [
        'RECALLED', 'NO ADVERSE EVENT', 'PRODUCT', 'EXPOSURE', 'ADMINISTERED',
        'CYCLOPHOSPHAMIDE', 'DOXORUBICIN', 'VINCRISTINE', 'RITUXIMAB', # Cancer protocol
        'DUPILUMAB', 'DERMATITIS', 'DIABETES', 'RAMIPRIL'
    ]

    def is_obvious(x):
        x_upper = str(x).upper()
        return any(term in x_upper for term in blacklist)

    # Filter out rules where ANY side (cause or effect) contains noise
    df_discovery = df[~df['antecedents'].apply(is_obvious) & 
                      ~df['consequents'].apply(is_obvious)].copy()

    # 2. FOCUS ON INTERACTIONS (2+ Drugs)
    df_interactions = df_discovery[df_discovery['antecedents'].apply(lambda x: len(str(x).split(',')) >= 2)].copy()

    # 3. LOOK FOR "HIDDEN" STRENGTH
    # We want Lift between 1.5 and 15 (Real interactions live here)
    final_discovery = df_interactions[(df_interactions['lift'] > 1.5) & (df_interactions['lift'] < 15)]

    print(f"\n--- Analysis of Non-Obvious Interactions (Support 0.002) ---")
    if not final_discovery.empty:
        # Clean text for the report
        final_discovery['antecedents'] = final_discovery['antecedents'].str.replace("frozenset({", "", regex=False).str.replace("})", "", regex=False).str.replace("'", "", regex=False)
        final_discovery['consequents'] = final_discovery['consequents'].str.replace("frozenset({", "", regex=False).str.replace("})", "", regex=False).str.replace("'", "", regex=False)
        
        print(final_discovery[['antecedents', 'consequents', 'confidence', 'lift']].head(20))
        final_discovery.to_csv("taskA/discovery_results.csv", index=False)
    else:
        print("No hidden signals found at this support level. The dataset is dominated by common protocols.")

if __name__ == "__main__":
    find_clinical_signals("taskA/final_association_rules.csv")