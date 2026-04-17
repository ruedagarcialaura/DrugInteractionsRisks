"""
TASK A: TRANSACTION ENCODING
----------------------------
What it does: Converts lists of text (drugs and reactions) into a binary matrix 
of 0s and 1s (One-Hot Encoding).
Goal: To transform human-readable data into a mathematical format that the 
Apriori algorithm can process.
"""

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import os

def encode_transactions(input_name, output_name):
    print(f"\n--- Processing: {input_name} ---")
    
    # 1. Load your preprocessed lists
    input_path = f"taskA/{input_name}.parquet"
    df = pd.read_parquet(input_path)
    
    # Ensure we are working with the 'all_items' column 
    print("Cleaning transactions from null values...")
    dataset = [
        [str(item) for item in transaction if item is not None] 
        for transaction in df['all_items'].tolist()
    ]

    # 2. Initialize the Encoder
    te = TransactionEncoder()
    
    print("Fitting encoder and transforming data to binary matrix...")
    # Create anumpy boolean array
    te_ary = te.fit(dataset).transform(dataset)
    
    # 3. Convert to DataFrame
    # Columns will be the unique drug and reaction names
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # 4. Save as Parquet 
    output_path = f"taskA/{output_name}.parquet"
    df_encoded.to_parquet(output_path, engine='pyarrow')
    
    print(f"Success!")
    print(f"Matrix Shape: {df_encoded.shape} (Reports x Unique Items)")
    print(f"Encoded file saved as: {output_path}")
    
    # Show a small preview of the columns (the "Items")
    print(f"Preview of first 10 items: {list(df_encoded.columns[:10])}")

if __name__ == "__main__":
    # do both to allow for the Comparison of Solutions 
    try:
        encode_transactions("drug_names_transactions", "drug_names_encoded")
        encode_transactions("active_substances_transactions", "active_substances_encoded")
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure you ran the preprocessing script first.")