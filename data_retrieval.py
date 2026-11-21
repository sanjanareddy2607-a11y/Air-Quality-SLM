"""data_retrieval.py
Historical air-quality data retrieval from the UCI Air Quality dataset.

Looks up records by date and time using exact matching on the UCI dataset CSV files.
Returns original dataset values with correct measurement units.
Does not invent missing values.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime


# ── Dataset paths ───────────────────────────────────────────────────────────
# Try to locate the dataset relative to the current working directory first,
# then fall back to the project structure.

_CWD = Path.cwd()
_POSSIBLE_ROOTS = [
    _CWD,  # current working directory
    _CWD / "Air-Quality-SLM",  # common repo name
    _CWD.parent,  # parent of cwd
]
_PROJECT_ROOT = None
for root in _POSSIBLE_ROOTS:
    if (root / "data" / "processed" / "train.csv").exists():
        _PROJECT_ROOT = root
        break
if _PROJECT_ROOT is None:
    # Last resort: use the directory of this file going up 3 levels
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_CSV = _PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_CSV = _PROJECT_ROOT / "data" / "processed" / "validation.csv"
TEST_CSV = _PROJECT_ROOT / "data" / "processed" / "test.csv"


# ── Load datasets once ─────────────────────────────────────────────────────

_train_df = None
_val_df = None
_test_df = None


def _get_dataset(split: str):
    """Return the DataFrame for the requested split ('train', 'val', 'test')."""
    global _train_df, _val_df, _test_df
    if split == "train":
        if _train_df is None:
            _train_df = pd.read_csv(TRAIN_CSV)
        return _train_df
    elif split == "val":
        if _val_df is None:
            _val_df = pd.read_csv(VAL_CSV)
        return _val_df
    elif split == "test":
        if _test_df is None:
            _test_df = pd.read_csv(TEST_CSV)
        return _test_df
    raise ValueError(f"Unknown split: {split}")


# ── Normalise date/time for matching ───────────────────────────────────────

DATE_FMT_IN = "%d/%m/%Y"
TIME_FMT_IN = "%H.%M.%S"


def normalise_date(date_str: str) -> str:
    """Normalise a date string to DD/MM/YYYY format."""
    d = date_str.strip()
    # Already in expected format
    try:
        datetime.strptime(d, DATE_FMT_IN)
        return d
    except ValueError:
        pass
    # Try other common formats
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(d, fmt).strftime(DATE_FMT_IN)
        except ValueError:
            continue
    return d  # best effort


def normalise_time(time_str: str) -> str:
    """Normalise a time string to HH.MM.SS format."""
    t = time_str.strip()
    try:
        datetime.strptime(t, TIME_FMT_IN)
        return t
    except ValueError:
        pass
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p"):
        try:
            return datetime.strptime(t, fmt).strftime(TIME_FMT_IN)
        except ValueError:
            continue
    return t  # best effort


# ── Core lookup ─────────────────────────────────────────────────────────────

def find_record(date_str: str, time_str: str, split: str = "val") -> dict | None:
    """Find a dataset record matching the given date and time.

    Parameters
    ----------
    date_str : str
        Date in DD/MM/YYYY format or similar (will be normalised)
    time_str : str
        Time in HH.MM.SS format or similar (will be normalised)
    split : str
        One of 'train', 'val', 'test'

    Returns
    -------
    dict | None
        Dictionary of row values if found, None if no matching record exists.
    """
    df = _get_dataset(split)
    norm_date = normalise_date(date_str)
    norm_time = normalise_time(time_str)

    # Exact match on both Date and Time columns
    mask = (df["Date"].astype(str).str.strip() == norm_date) & \
           (df["Time"].astype(str).str.strip() == norm_time)

    matches = df[mask]
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        # Return the first match; duplicates can exist in the dataset
        return matches.iloc[0].to_dict()

    return matches.iloc[0].to_dict()


# ── Public API: historical retrieval ───────────────────────────────────────

def get_pollutant_concentration(date: str, time: str, pollutant: str,
                                 split: str = "val") -> str | None:
    """Get a pollutant concentration for a given date/time from the dataset.

    Parameters
    ----------
    date : str
        Date in DD/MM/YYYY format
    time : str
        Time in HH.MM.SS format
    pollutant : str
        One of: 'CO', 'NOx', 'NO2', 'C6H6', 'NMHC'
    split : str
        Dataset split to search ('train', 'val', 'test')

    Returns
    -------
    str | None
        String like "166.0 ug/m3" or None if not found
    """
    record = find_record(date, time, split)
    if record is None:
        return None

    col_map = {
        "CO": "CO(GT)",
        "NOx": "NOx(GT)",
        "NO2": "NO2(GT)",
        "C6H6": "C6H6(GT)",
        "NMHC": "NMHC(GT)",
    }

    col = col_map.get(pollutant)
    if col is None or col not in record:
        return None

    val = record[col]
    # Format: value unit
    if pd.isna(val):
        return None

    return f"{val} ug/m3"


def get_temperature(date: str, time: str, split: str = "val") -> str | None:
    """Get temperature for a given date/time from the dataset."""
    record = find_record(date, time, split)
    if record is None:
        return None

    val = record.get("T")
    if pd.isna(val):
        return None

    return f"{val} degrees Celsius"


def get_relative_humidity(date: str, time: str, split: str = "val") -> str | None:
    """Get relative humidity for a given date/time from the dataset."""
    record = find_record(date, time, split)
    if record is None:
        return None

    val = record.get("RH")
    if pd.isna(val):
        return None

    return f"{val} %"


def get_absolute_humidity(date: str, time: str, split: str = "val") -> str | None:
    """Get absolute humidity for a given date/time from the dataset."""
    record = find_record(date, time, split)
    if record is None:
        return None

    val = record.get("AH")
    if pd.isna(val):
        return None

    # Return with calibrated unit
    return f"{val} g/m3"


def get_all_sensor_values(date: str, time: str, split: str = "val") -> dict | None:
    """Get all available PT08 sensor values and meteorological data for a given date/time."""
    record = find_record(date, time, split)
    if record is None:
        return None

    # Return key fields
    result = {}
    sensor_keys = [
        "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)",
        "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)",
        "T", "RH", "AH"
    ]
    for key in sensor_keys:
        if key in record:
            result[key] = record[key]
    return result if result else None


# ── Example-driven lookup ──────────────────────────────────────────────────

def example_lookup(date: str, time: str) -> dict | None:
    """Look up a record from the training set as an example.

    Used for demonstration/example prompts.
    """
    return find_record(date, time, "train")