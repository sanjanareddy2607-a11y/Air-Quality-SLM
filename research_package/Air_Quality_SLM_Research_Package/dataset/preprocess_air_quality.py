"""
Air Quality Dataset Preprocessing Pipeline
=========================================
Author: Sanjana Reddy (BTech CSE AI/ML)
Project: SLM Air Quality Domain Dataset for Hugging Face
Source: UCI Machine Learning Repository (Air Quality Dataset)

Description:
This script loads the raw UCI Air Quality dataset (semicolon-delimited, decimal comma),
removes trailing artifact columns and empty rows, normalizes missing values (-200 -> NaN),
verifies chronological ordering, performs a temporal split (70% train, 10% validation, 20% test)
to prevent data leakage, and outputs clean CSV splits along with summary metrics.
"""

import os
import sys
import pandas as pd
import numpy as np

def preprocess_air_quality(input_csv="AirQualityUCI.csv", output_dir="HF_Air_Quality_Dataset"):
    print(f"[*] Starting Air Quality Preprocessing...")
    print(f"[*] Reading source dataset: {input_csv}")
    
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file '{input_csv}' not found.")
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Read source dataset handling European formatting (sep=';', decimal=',')
    raw_df = pd.read_csv(input_csv, sep=';', decimal=',')
    initial_shape = raw_df.shape
    print(f"[+] Loaded raw dataset with shape: {initial_shape}")
    
    # 2. Prune trailing empty columns and rows
    df = raw_df.dropna(how='all')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(subset=['Date', 'Time']).copy()
    cleaned_shape = df.shape
    print(f"[+] Cleaned artifact rows/columns. New shape: {cleaned_shape}")
    
    # 3. Clean numeric columns and replace sentinel -200 with NaN
    numeric_cols = [c for c in df.columns if c not in ['Date', 'Time']]
    missing_summary = {}
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        m200_count = int((df[col] == -200.0).sum())
        missing_summary[col] = m200_count
        df[col] = df[col].replace(-200.0, np.nan)
        
    print(f"[+] Missing values (-200) replaced with NaN across {len(numeric_cols)} numeric columns.")
    
    # 4. Verify and enforce chronological ordering
    df['parsed_dt'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'].str.replace('.', ':', regex=False),
        format='%d/%m/%Y %H:%M:%S'
    )
    is_monotonic = df['parsed_dt'].is_monotonic_increasing
    print(f"[+] Chronological timestamp ordering check: {is_monotonic}")
    df = df.sort_values('parsed_dt').reset_index(drop=True)
    df = df.drop(columns=['parsed_dt'])
    
    # 5. Chronological Splits (70% train, 10% validation, 20% test)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # 6. Save splits
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "validation.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"[✓] Saved {train_path} ({len(train_df)} rows, {len(train_df)/n*100:.2f}%)")
    print(f"[✓] Saved {val_path} ({len(val_df)} rows, {len(val_df)/n*100:.2f}%)")
    print(f"[✓] Saved {test_path} ({len(test_df)} rows, {len(test_df)/n*100:.2f}%)")
    
    return {
        "initial_shape": initial_shape,
        "cleaned_shape": cleaned_shape,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "total_rows": n,
        "missing_summary": missing_summary
    }

if __name__ == "__main__":
    preprocess_air_quality()
