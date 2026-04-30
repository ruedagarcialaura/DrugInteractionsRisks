"""
TASK A: APRIORI ALGORITHM & RULES
---------------------------------
What it does: Executes the Apriori algorithm to discover frequent groups of 
items and generates rules based on Support, Confidence, and Lift.
Goal: To identify complex multi-item relationships, such as drug-drug 
interactions ("Drug A + Drug B -> Reaction X") or multi-symptom 
profiles caused by specific medications.
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os
import argparse

def run_association_mining(file_path, min_support=0.01):
    print(f"\n--- Loading Matrix: {file_path} ---")
    df = pd.read_parquet(file_path)
    
    # --- PRUNING  ---
    # we calculate the frequency of each item (column) across all transactions (rows) to determine its support. 
    # The support is the proportion of transactions that contain the item. 
    # We then filter out items that do not meet the minimum support threshold.
    item_support = df.mean(axis=0)
    
    # We filter: Only keep items that meet the minimum support threshold
    # Note: if the support is 0.01, the item must be present in at least 1% of the reports
    frequent_cols = item_support[item_support >= min_support].index
    df_filtered = df[frequent_cols]
    
    print(f"Original items: {df.shape[1]}")
    print(f"Items after pruning (support >= {min_support}): {df_filtered.shape[1]}")
    
    if df_filtered.empty:
        print("No items met the support threshold. Try a lower min_support.")
        return

    # --- APRIORI ---
    print("Running Apriori algorithm...")
    frequent_itemsets = apriori(df_filtered, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        print("No frequent itemsets found.")
        return

    # --- GENERATING RULES ---
    # we use lift as the metric to find interesting associations. A lift > 1 indicates a positive association between antecedent and consequent.
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0,
                              num_itemsets=len(frequent_itemsets))
    
    # Order by lift descending to get the most interesting rules at the top
    rules = rules.sort_values('lift', ascending=False)
    
    print(f"Found {len(rules)} association rules.")
    return rules

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Apriori association mining")
    parser.add_argument("--file_path", default="taskA/active_substances_encoded.parquet", help="Path to encoded parquet input file")
    parser.add_argument("--min_support", type=float, default=0.005, help="Minimum support threshold")
    args = parser.parse_args()

    results = run_association_mining(args.file_path, min_support=args.min_support)
    
    if results is not None:
        print("\nTOP 10 RULES FOUND:")
        print(results[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

        input_name = os.path.splitext(os.path.basename(args.file_path))[0]
        input_name_replaced = input_name.replace("_encoded", "")
        min_support_tag = f"{args.min_support:g}".replace(".", "_")
        
        output_dir = "taskA/association_rules"
        os.makedirs(output_dir, exist_ok=True)
        
        
        output_path = os.path.join(
            output_dir,
            f"association_rules_APRIORI_{input_name_replaced}_{min_support_tag}.csv"
        )
        results.to_csv(output_path, index=False)

        # Filter rules where the antecedent contains more than one item
        # This helps identify drug-drug interactions (Drug A + Drug B -> Reaction)
        interactions = results[results['antecedents'].apply(lambda x: len(x) > 1)]

        print("\nDETECTED INTERACTIONS (2+ ITEMS):")
        if not interactions.empty:
            print(interactions[['antecedents', 'consequents', 'lift']].head(10))
        else:
            print("No multi-item interactions found at this support level. Try lowering min_support to 0.001.")

    