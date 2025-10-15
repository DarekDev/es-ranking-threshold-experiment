#!/usr/bin/env python3
"""
Quick script to verify data types are consistent after fixes.
"""

import sys
sys.path.append('.')
from stage_rank.io_utils import load_csv
from pathlib import Path

DATASET_DIR = Path("../DATASET")

print("Verifying data types in all CSV files...\n")

# Check all test files
test_files = list(DATASET_DIR.glob("test_data_session_*.csv"))

for test_file in test_files:
    print(f"Checking {test_file.name}...")
    df = load_csv(str(test_file))
    
    # Check item_category column
    if 'item_category' in df.columns:
        unique_types = df['item_category'].apply(type).unique()
        print(f"   item_category types: {unique_types}")
        print(f"   item_category sample values: {df['item_category'].head(3).tolist()}")
        
        # Check if all are strings
        all_strings = df['item_category'].apply(lambda x: isinstance(x, str)).all()
        if all_strings:
            print(f"   ✅ All values are strings")
        else:
            print(f"   ❌ Mixed types detected!")
    print()

# Check train file
print(f"Checking train_data_recsys.csv...")
df = load_csv(str(DATASET_DIR / "train_data_recsys.csv"))
if 'item_category' in df.columns:
    unique_types = df['item_category'].apply(type).unique()
    print(f"   item_category types: {unique_types}")
    all_strings = df['item_category'].apply(lambda x: isinstance(x, str)).all()
    if all_strings:
        print(f"   ✅ All values are strings")
    else:
        print(f"   ❌ Mixed types detected!")

print("\n✅ Data type verification complete!")
