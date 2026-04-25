"""
TASK A: FP-GROWTH ALGORITHM & RULES
-----------------------------------
What it does: Executes the FP-Growth (Frequent Pattern Growth) algorithm.
Goal: To handle large datasets and low support thresholds more efficiently than Apriori.
This script overcomes the "MemoryError" by using a compressed tree structure 
to find frequent itemsets and complex interactions (Drug A + Drug B -> Reaction).
"""

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
import os
import argparse
from tqdm import tqdm
import time

def run_fpgrowth_mining(file_path, min_support=0.001):
    # 1. Progress bar for loading the data
    print("\n--- Initializing Data Mining Process ---")
    with tqdm(total=3, desc="Mining Pipeline", colour="cyan") as pbar:
        
        # Step 1: Loading
        df = pd.read_parquet(file_path)
        pbar.update(1)
        pbar.set_description("Pruning items...")

        # --- PRUNING ---
        item_support = df.mean(axis=0)
        frequent_cols = item_support[item_support >= min_support].index
        df_filtered = df[frequent_cols]
        
        pbar.update(1)
        pbar.set_description("Building FP-Tree (Processing)...")

        if df_filtered.empty:
            print("No items met the support threshold.")
            return None

        # --- FP-GROWTH ---
        # Note: fpgrowth doesn't support tqdm natively, 
        # but the pbar will stay visible in cyan while it works.
        frequent_itemsets = fpgrowth(df_filtered, min_support=min_support, use_colnames=True)
        
        pbar.update(1)
        pbar.set_description("Complete!")

    if frequent_itemsets.empty:
        print("No frequent itemsets found.")
        return None

    # --- GENERATING RULES ---
    print("Generating association rules...")
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    rules = rules.sort_values('lift', ascending=False)
    
    print(f"Found {len(rules)} association rules.")
    return rules

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FP-Growth association mining")
    parser.add_argument("--file_path", default="taskA/active_substances_encoded.parquet", help="Path to encoded parquet input file")
    parser.add_argument("--min_support", type=float, default=0.002, help="Minimum support threshold")
    args = parser.parse_args()

    results = run_fpgrowth_mining(args.file_path, min_support=args.min_support)
    
    if results is not None:
        # (Rest of your printing logic)
        print("\nTOP 10 RULES FOUND (GENERAL):")
        print(results[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
        input_name = os.path.splitext(os.path.basename(args.file_path))[0]
        input_name_replaced = input_name.replace("_encoded", "")
        min_support_tag = f"{args.min_support:g}".replace(".", "_")

        output_dir = "taskA/association_rules"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(
            output_dir,
            f"association_rules_FP_GROWTH_{input_name_replaced}_{min_support_tag}.csv"
        )
        results.to_csv(output_path, index=False)

        interactions = results[results['antecedents'].apply(lambda x: len(x) > 1)]
        print("\nDETECTED INTERACTIONS (2+ ITEMS):")
        if not interactions.empty:
            print(interactions[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(15))
        else:
            print("No multi-item interactions found at this support level.")
        