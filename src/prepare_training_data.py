"""
prepare_training_data.py
Prepares instruction and measurement text datasets from processed CSV files
and encodes them into binary memmap files for fast training of the Air Quality SLM.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TRAINING_DIR = DATA_DIR / "training"
BIN_DIR = PROJECT_ROOT / "training" / "data_bin"

def clean_val(val, unit=""):
    """Format float or missing values cleanly for natural text."""
    if pd.isna(val) or val is None:
        return "Unavailable"
    if isinstance(val, (int, np.integer)):
        return f"{val}{unit}"
    if isinstance(val, (float, np.floating)):
        return f"{val:.2f}{unit}".rstrip('0').rstrip('.') if '.' in f"{val:.2f}" else f"{val:.2f}{unit}"
    return str(val)

def row_to_texts(row):
    """
    Generate diverse instruction and descriptive text samples from a single air quality record.
    Ensures domain-specific vocabulary, values, and tasks are learned.
    """
    date = str(row.get('Date', 'Unknown'))
    time = str(row.get('Time', 'Unknown'))
    co = clean_val(row.get('CO(GT)'))
    s1 = clean_val(row.get('PT08.S1(CO)'))
    nmhc = clean_val(row.get('NMHC(GT)'))
    c6h6 = clean_val(row.get('C6H6(GT)'))
    s2 = clean_val(row.get('PT08.S2(NMHC)'))
    nox = clean_val(row.get('NOx(GT)'))
    s3 = clean_val(row.get('PT08.S3(NOx)'))
    no2 = clean_val(row.get('NO2(GT)'))
    s4 = clean_val(row.get('PT08.S4(NO2)'))
    s5 = clean_val(row.get('PT08.S5(O3)'))
    t = clean_val(row.get('T'))
    rh = clean_val(row.get('RH'))
    ah = clean_val(row.get('AH'))

    texts = []

    # 1. Full environmental record
    t1 = (
        f"Air quality measurement report for {date} at {time}: "
        f"Carbon Monoxide CO(GT)={co} mg/m3 (sensor PT08.S1={s1}), "
        f"Non-Methane Hydrocarbons NMHC(GT)={nmhc} ug/m3 (sensor PT08.S2={s2}), "
        f"Benzene C6H6(GT)={c6h6} ug/m3, Nitrogen Oxides NOx(GT)={nox} ppb (sensor PT08.S3={s3}), "
        f"Nitrogen Dioxide NO2(GT)={no2} ug/m3 (sensor PT08.S4={s4}), "
        f"Ozone sensor PT08.S5={s5}, Temperature={t} C, Relative Humidity={rh} %, Absolute Humidity={ah} g/m3.\n<|endoftext|>\n"
    )
    texts.append(t1)

    # 2. Instruction: Predict Absolute Humidity (matching notebook demonstration)
    t2 = (
        f"Task: Predict the absolute humidity given the following parameters: "
        f"Date: {date}, Time: {time}, CO(GT): {co}, NOx(GT): {nox}, NO2(GT): {no2}, Temperature: {t} C, Relative Humidity: {rh} %. "
        f"Response: The absolute humidity is {ah} g/m3.\n<|endoftext|>\n"
    )
    texts.append(t2)

    # 3. Instruction: Temperature and Humidity Query
    t3 = (
        f"Question: What were the ambient temperature and relative humidity recorded on {date} at {time}? "
        f"Answer: On {date} at {time}, the ambient temperature was {t} degrees Celsius and the relative humidity was {rh}%.\n<|endoftext|>\n"
    )
    texts.append(t3)

    # 4. Instruction: Toxic gas concentrations
    t4 = (
        f"Query: What are the primary pollutant concentrations (CO, NOx, NO2) on {date} at {time}? "
        f"Result: Recorded CO is {co} mg/m3, NOx is {nox} ppb, and NO2 is {no2} ug/m3 with benzene level at {c6h6} ug/m3.\n<|endoftext|>\n"
    )
    texts.append(t4)

    return texts

def get_or_load_tokenizer():
    """Load Qwen tokenizer with fallbacks."""
    for model_id in ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-7B", "Qwen/Qwen3-8B"]:
        try:
            print(f"Loading tokenizer: {model_id}...")
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            print(f"Loaded tokenizer successfully! Vocab size: {tok.vocab_size}")
            return tok, model_id
        except Exception as e:
            print(f"Could not load {model_id}: {e}")

    # Fallback to gpt2 tokenizer if Qwen is unreachable
    print("Falling back to gpt2 tokenizer...")
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, "gpt2"

def prepare_splits(tokenizer, force_rebuild=False):
    """Convert train, validation, and test CSVs into tokenized memmap bin files."""
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        'train': PROCESSED_DIR / 'train.csv',
        'val': PROCESSED_DIR / 'validation.csv',
        'test': PROCESSED_DIR / 'test.csv',
    }

    metadata = {}

    for split_name, csv_path in splits.items():
        bin_path = BIN_DIR / f"{split_name}.bin"
        jsonl_path = TRAINING_DIR / f"{split_name}_instructions.jsonl"

        if bin_path.exists() and not force_rebuild:
            print(f"Using existing binary split: {bin_path}")
            metadata[split_name] = {
                'bin_path': str(bin_path),
                'size_bytes': bin_path.stat().st_size,
            }
            continue

        if not csv_path.exists():
            raise FileNotFoundError(f"Source split file not found: {csv_path}")

        print(f"\nProcessing {split_name} split from {csv_path}...")
        df = pd.read_csv(csv_path)

        all_texts = []
        with open(jsonl_path, 'w', encoding='utf-8') as jf:
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Formatting {split_name}"):
                texts = row_to_texts(row)
                all_texts.extend(texts)
                for t in texts:
                    jf.write(json.dumps({"text": t.strip()}) + "\n")

        print(f"Encoding {len(all_texts)} text samples for {split_name}...")
        all_ids = []
        batch_size = 500
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            enc = tokenizer(batch, add_special_tokens=False, return_attention_mask=False)
            for ids in enc['input_ids']:
                all_ids.extend(ids)

        arr = np.array(all_ids, dtype=np.uint32 if tokenizer.vocab_size > 65536 else np.uint16)
        print(f"Writing {len(arr):,} tokens to {bin_path}...")
        
        dtype = np.uint32 if tokenizer.vocab_size > 65536 else np.uint16
        mm = np.memmap(bin_path, dtype=dtype, mode='w+', shape=(len(arr),))
        mm[:] = arr[:]
        mm.flush()
        del mm

        metadata[split_name] = {
            'bin_path': str(bin_path),
            'num_tokens': len(arr),
            'dtype': str(dtype.__name__),
            'num_text_samples': len(all_texts),
            'source_rows': len(df),
        }
        print(f"Successfully created {bin_path} ({len(arr):,} tokens).")

    # Save data summary metadata
    meta_path = BIN_DIR / "dataset_metadata.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    return metadata

if __name__ == "__main__":
    print("=" * 60)
    print("AIR QUALITY SLM - DATASET TOKENIZATION & BINARY PREPARATION")
    print("=" * 60)
    tok, model_id = get_or_load_tokenizer()
    meta = prepare_splits(tok, force_rebuild=False)
    print("\nDataset Preparation Complete:")
    print(json.dumps(meta, indent=2))
