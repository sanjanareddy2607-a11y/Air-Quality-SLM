# Dataset Documentation — UCI Air Quality (prepared for the Air Quality SLM)

All counts below were re-computed from the files in this package for this audit; they are not copied
from earlier prose. The script used is described in §11.

---

## 1. Identity and source

| Property | Value | Verified from |
|---|---|---|
| Dataset name | Air Quality (also "Air Quality Time-Series" in the prepared release) | UCI repository page |
| Original repository | UCI Machine Learning Repository, dataset id 360 | `https://archive.ics.uci.edu/dataset/360/air+quality` |
| DOI | 10.24432/C59Q63 | repository page |
| Licence | CC BY 4.0 | repository page |
| Collected by | ENEA, Portici Research Center, Italy | dataset card (`docs/README.md`) |
| Primary publication | De Vito, Massera, Piga, Martinotto, Di Francia (2008), *Sensors and Actuators B* 129(2) 750–757 | dataset card |
| Raw file in workspace | `data/raw/AirQualityUCI.csv`, 785,065 bytes, SHA256 `13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996` | measured |
| Prepared splits in workspace | `data/processed/{train,validation,test}.csv` | measured |

**Acquisition method.** The CSV was downloaded manually from the UCI repository (no downloader script
exists in the workspace) and committed to the repository. There is no automated fetch step and no
recorded download timestamp; this is recorded in `MISSING_EVIDENCE.md`.

## 2. Size and record counts

| Quantity | Value |
|---|---:|
| Raw file data rows | 9,471 |
| Raw columns | 17 (15 real + 2 empty artifact columns `Unnamed: 15`, `Unnamed: 16`) |
| Records after cleaning | **9,357** |
| Features after cleaning | **15** (13 numeric + `Date`, `Time`) |
| Distinct calendar dates | 391 |
| Duplicate `(Date, Time)` pairs | **0** |
| Total missing cells (numeric) | 16,701 |

The raw file has 9,472 newline-terminated lines: one header plus 9,471 data lines, of which the last
114 are empty (they are the artifact rows removed by `df.dropna(how='all')`).

## 3. Time coverage and sampling

* **Start:** 10/03/2004 18:00:00
* **End:** 04/04/2005 14:00:00
* **Sampling frequency:** hourly averages (one record per hour), 13 consecutive months
* **Ordering:** strictly chronological (verified monotonic in `docs/VALIDATION_REPORT.md`, check 9)

The timestamps are stored as two string columns: `Date` in `DD/MM/YYYY`, `Time` in `HH.MM.SS`
(dots, not colons — this format is reproduced in every prompt and every lookup).

## 4. Features, units and measurement origin

| # | Feature | Unit | Origin | Missing (%) |
|---|---|---|---|---:|
| 1 | `Date` | — | timestamp | 0 |
| 2 | `Time` | — | timestamp | 0 |
| 3 | `CO(GT)` | mg/m³ | certified reference analyser | 17.99 |
| 4 | `PT08.S1(CO)` | sensor response (Ω nominal) | metal-oxide sensor | 3.91 |
| 5 | `NMHC(GT)` | µg/m³ | certified reference analyser | **90.23** |
| 6 | `C6H6(GT)` | µg/m³ | certified reference analyser | 3.91 |
| 7 | `PT08.S2(NMHC)` | sensor response | metal-oxide sensor | 3.91 |
| 8 | `NOx(GT)` | ppb | certified reference analyser | 17.52 |
| 9 | `PT08.S3(NOx)` | sensor response | metal-oxide sensor | 3.91 |
| 10 | `NO2(GT)` | µg/m³ | certified reference analyser | 17.55 |
| 11 | `PT08.S4(NO2)` | sensor response | metal-oxide sensor | 3.91 |
| 12 | `PT08.S5(O3)` | sensor response | metal-oxide sensor | 3.91 |
| 13 | `T` | °C | meteorological probe | 3.91 |
| 14 | `RH` | % | meteorological probe | 3.91 |
| 15 | `AH` | g/m³ | meteorological probe (dataset scale, see §9) | 3.91 |

**Sensor information.** Five tin-oxide (metal-oxide) chemical sensors `PT08.S1…S5` co-located with the
reference analysers; their readings are raw resistance responses, non-linearly cross-sensitive to
other gases and to temperature/humidity. `(GT)` columns are the reference ("ground truth") values.
See figure `figures/dataset/fig01_feature_distributions.png` for the empirical distributions of all
13 numeric columns on the training split.

## 5. Missing values and their distribution

Measured over the full 9,357 records:

| Column group | Missing records | % |
|---|---:|---:|
| `PT08.S1…S5`, `C6H6(GT)`, `T`, `RH`, `AH` | 366 | 3.91 |
| `CO(GT)` | 1,683 | 17.99 |
| `NOx(GT)` | 1,639 | 17.52 |
| `NO2(GT)` | 1,642 | 17.55 |
| `NMHC(GT)` | 8,443 | 90.23 |

Per split (missing rows for `T`/`AH`, `CO(GT)`, `NMHC(GT)`):

| Split | Rows | `T`/`AH` missing | `CO(GT)` missing | `NMHC(GT)` missing |
|---|---:|---:|---:|---:|
| train | 6,549 | 148 | 1,483 | 5,635 |
| validation | 936 | 132 | 109 | **936 (100 %)** |
| test | 1,872 | 86 | 91 | **1,872 (100 %)** |

**`NMHC(GT)` is entirely absent from validation and test.** Any NMHC metric on those splits is
undefined; this is a property of the source data (the reference instrument was retired), not a
preprocessing error. See figure `figures/dataset/fig02_missing_values.png`.

## 6. Data-cleaning procedure (as implemented)

From `src/preprocess_air_quality.py` (identical copy at `data/processed/preprocess_air_quality.py`):

1. `pd.read_csv(input_csv, sep=';', decimal=',')` — the source uses semicolons and decimal commas.
2. `df.dropna(how='all')` — removes the 114 trailing blank rows (9,471 → 9,357).
3. Drop every column whose name matches `^Unnamed` — removes the two trailing empty artifact columns.
4. `df.dropna(subset=['Date','Time'])` — removes any row without a timestamp.
5. For each numeric column: `pd.to_numeric(..., errors='coerce')`, count `-200` values, then
   `replace(-200.0, np.nan)`. **Verified:** the raw file contains exactly 16,701 occurrences of
   `-200` and the processed files contain exactly 16,701 missing numeric cells — i.e. every sentinel
   was converted and **no `-200` remains**.
6. Parse `Date + Time` into a datetime, assert monotonic ordering, sort ascending, drop the helper.
7. Split contiguously: first 70 % train, next 10 % validation, final 20 % test; write three CSVs.

**Duplicate handling:** no duplicates were removed because none exist; `(Date, Time)` is unique in
all three splits (0 duplicates measured). No row-level de-duplication rule was applied beyond the
timestamp requirement.

**What was deliberately *not* done.** Missing values are **not** imputed, interpolated or forward
filled; rows with missing target values are kept in the CSVs and simply dropped per-quantity at
scoring time. No outlier removal, no smoothing and no resampling were applied.

## 7. Feature selection

All 13 numeric columns are retained. **No feature-selection step was performed**: every numeric
column is written into the textual representation of each record. `NMHC(GT)` is included despite its
90 % missingness, and appears in text as the literal string `Unavailable` where missing
(`prepare_training_data.clean_val`).

## 8. Preprocessing and normalisation for the model

* **No scaling, standardisation, log transform or min-max normalisation is applied anywhere.**
  `src/prepare_training_data.py` inserts the raw floating-point values into text verbatim
  (`f"{val:.2f}"`, trailing zeros stripped), and the tokenizer sees digits only.
* The only transformations applied to the original dataset are: delimiter/decimal parsing, artifact
  removal, sentinel→NaN, chronological sorting, and the contiguous split.
* Textualisation (the real "feature engineering" step) produces **4 templates per record**:

  1. **Full report** — `Air quality measurement report for {date} at {time}: Carbon Monoxide
     CO(GT)=… mg/m3 (sensor PT08.S1=…), Non-Methane Hydrocarbons NMHC(GT)=… ug/m3 (sensor PT08.S2=…),
     Benzene C6H6(GT)=… ug/m3, Nitrogen Oxides NOx(GT)=… ppb (sensor PT08.S3=…), Nitrogen Dioxide
     NO2(GT)=… ug/m3 (sensor PT08.S4=…), Ozone sensor PT08.S5=…, Temperature=… C, Relative
     Humidity=… %, Absolute Humidity=… g/m3.` + `<|endoftext|>`
  2. **Absolute-humidity instruction** — `Task: Predict the absolute humidity given the following
     parameters: Date: …, Time: …, CO(GT): …, NOx(GT): …, NO2(GT): …, Temperature: … C, Relative
     Humidity: … %. Response: The absolute humidity is … g/m3.` + `<|endoftext|>`
  3. **Temperature/relative-humidity question** — `Question: What were the ambient temperature and
     relative humidity recorded on … at …? Answer: On … at …, the ambient temperature was … degrees
     Celsius and the relative humidity was …%.` + `<|endoftext|>`
  4. **Pollutant query** — `Query: What are the primary pollutant concentrations (CO, NOx, NO2) on …
     at …? Result: Recorded CO is … mg/m3, NOx is … ppb, and NO2 is … ug/m3 with benzene level at …
     ug/m3.` + `<|endoftext|>`

  Missing values become the token sequence `Unavailable`, which teaches an explicit "no reading"
  form rather than a fabricated number.

## 9. Absolute humidity: a dataset-scale quantity (important)

The `AH` column is **not** standard g/m³ on the usual physical scale. A Magnus/Tetens calculation
from the same row's `T` and `RH` reproduces it almost perfectly in *shape* but not in *scale*:
Pearson r = 0.9996 with the dataset column ~7.4× smaller than the standard formula output
(`docs/OLLAMA_INTEGRATION.md` §7b reports a median ratio of 0.1310; a least-squares scale fitted on
train only is 0.1351678, `docs/eval_v2/numeric_diagnostics.json`).

Closed form used for verification (scale fitted on **train only**):

```
e_s = 6.112 * exp(17.67*T / (T + 243.5))                  # hPa
AH  = 0.1351678 * 216.7 * (e_s * RH / 100) / (T + 273.15) # g/m3 (dataset scale)
```

| Split | n | MAE | RMSE | r | constant baseline MAE |
|---|---:|---:|---:|---:|---:|
| train | 6,401 | 0.01881 | 0.02327 | 0.99807 | 0.29173 |
| val | 804 | 0.02296 | 0.02388 | 0.99979 | 0.45874 |
| test | 1,786 | 0.01880 | 0.02024 | 0.99956 | 0.55756 |

Consequence for the paper: **always score AH against the dataset column**, never against a
recomputed physical g/m³ value. Figure `figures/evaluation/fig10_ah_formula_reconstruction.png`.

Ranges (all 9,357 records): `AH` 0.1847 – 2.231 g/m³ (mean 1.0255); `T` −1.9 – 44.6 °C.

## 10. Training / validation / test split

| Split | File | Rows | Share | Start | End |
|---|---|---:|---:|---|---|
| train | `data/processed/train.csv` | 6,549 | 69.99 % | 10/03/2004 18:00 | 08/12/2004 14:00 |
| validation | `data/processed/validation.csv` | 936 | 10.00 % | 08/12/2004 15:00 | 16/01/2005 14:00 |
| test | `data/processed/test.csv` | 1,872 | 20.01 % | 16/01/2005 15:00 | 04/04/2005 14:00 |
| **total** | — | **9,357** | 100 % | 10/03/2004 18:00 | 04/04/2005 14:00 |

The split is **strictly chronological and contiguous** — no shuffling, so no future information leaks
into training. Verification from `docs/VALIDATION_REPORT.md`: 0 overlapping timestamps between any
pair of splits, 0 duplicate timestamps inside any split, monotonic ordering preserved (checks 6–9).

**Boundary caveat (measured).** The split boundary falls mid-day: train ends on 08/12/2004 at
14:00 and validation begins on the same date at 15:00. `docs/eval_v2/baseline.json` therefore records
`date_overlap_with_train = 1` for validation (i.e. one shared *calendar date*) and `0` for test.
Timestamp overlap is zero; only that one date string is shared. The evaluation report states
"date_overlap_with_train = 0" for both splits, which is true for timestamps but imprecise for dates —
flagged in `MISSING_EVIDENCE.md`. It does not affect any reported metric (scoring is per record).

## 11. Transformations applied after splitting

1. **Textualisation** (4 templates per record) — `src/prepare_training_data.py`.
2. **Tokenisation** — Qwen2.5-0.5B BPE, `add_special_tokens=False`, samples concatenated into one
   stream per split:
   `train.bin` 3,041,432 tokens · `val.bin` 431,740 · `test.bin` 870,987 · total **4,344,159**.
   Sample counts: 26,196 / 3,744 / 7,488 = **37,428** instruction samples.
3. **Instruction JSONL copies** — `data/training/{train,val,test}_instructions.jsonl` (2.2 MB /
   7.9 MB / 1.1 MB) written in the same loop for inspection and reuse.
4. **Windowed batching at training time** — random 64-token windows (v1) or 128-token windows with a
   shuffled permutation per epoch (v2). Not a dataset transformation; no new files are written.

Note: the `.bin` token files are referenced by configuration but **are not present in the audited
workspace** (`training/data_bin/` is absent); the counts above come from the recorded configuration
and training summaries. See `MISSING_EVIDENCE.md`.

## 12. Reproduction of the statistics in this document

```bash
# from the project root, with the project virtual environment
./venv/Scripts/python.exe -c "
import pandas as pd
parts = {s: pd.read_csv(f'data/processed/{s}.csv') for s in ['train','validation','test']}
allr = pd.concat(parts.values(), ignore_index=True)
print(allr.shape)
print(allr.isna().mean().mul(100).round(2))
print('dup timestamps:', allr.duplicated(['Date','Time']).sum())
print('unique dates:', allr['Date'].nunique())
"
```

Cleaning can be re-derived with `python data/processed/preprocess_air_quality.py` (expects
`AirQualityUCI.csv` in the working directory), and the 12-point integrity suite with
`python src/validate_dataset.py`.
