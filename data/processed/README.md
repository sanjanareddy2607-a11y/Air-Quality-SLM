---
annotations_creators:
- no-annotation
language_creators:
- found
language:
- en
license:
- cc-by-4.0
multilinguality:
- monolingual
size_categories:
- 1K<n<10K
source_datasets:
- original
task_categories:
- tabular-regression
- time-series-forecasting
task_ids:
- multivariate-time-series-forecasting
- environmental-monitoring
pretty_name: Air Quality Time-Series (UCI Air Quality Dataset)
tags:
- environmental-science
- time-series
- air-quality
- chemical-sensors
- tabular
- slm-research
- urban-pollution
dataset_info:
  features:
  - name: Date
    dtype: string
  - name: Time
    dtype: string
  - name: CO(GT)
    dtype: float64
  - name: PT08.S1(CO)
    dtype: float64
  - name: NMHC(GT)
    dtype: float64
  - name: C6H6(GT)
    dtype: float64
  - name: PT08.S2(NMHC)
    dtype: float64
  - name: NOx(GT)
    dtype: float64
  - name: PT08.S3(NOx)
    dtype: float64
  - name: NO2(GT)
    dtype: float64
  - name: PT08.S4(NO2)
    dtype: float64
  - name: PT08.S5(O3)
    dtype: float64
  - name: T
    dtype: float64
  - name: RH
    dtype: float64
  - name: AH
    dtype: float64
  splits:
  - name: train
    num_bytes: 576735
    num_examples: 6549
  - name: validation
    num_bytes: 77949
    num_examples: 936
  - name: test
    num_bytes: 164863
    num_examples: 1872
  download_size: 819547
  dataset_size: 819547
configs:
- config_name: default
  data_files:
  - split: train
    path: train.csv
  - split: validation
    path: validation.csv
  - split: test
    path: test.csv
---

# Air Quality Time-Series Dataset (UCI)

*Prepared for Domain-Specific Small Language Model (SLM) & Time-Series Research*

---

## 1. Dataset Overview

The **Air Quality Time-Series Dataset** is a structured, benchmark tabular dataset containing 9,357 hourly averaged records of urban atmospheric pollutants and meteorological parameters. Recorded in an Italian city between **March 10, 2004 and April 4, 2005**, the data was captured using an embedded array of five metal oxide chemical sensors co-located with a certified reference analyzer operated by the local environmental agency.

This repository provides a standardized, preprocessed, and strictly chronologically split version of the canonical UCI Air Quality dataset. It is organized into **training (70%)**, **validation (10%)**, and **testing (20%)** subsets. This dataset serves as the empirical tabular data layer for the academic research project:

> **Air Quality SLM – Domain-Specific Small Language Model for Air Quality Data**  
> *Objective*: Investigating methods to ground domain-specific language models on empirical environmental sensor telemetry, tabular time-series forecasting, and calibrated abstention/refusal behavior for missing observations.

---

## 2. Dataset Description

- **Measurement Modality**: Continuous hourly multi-sensor environmental telemetry.
- **Geographic Setting**: Road-level sampling point in a heavily polluted urban zone in an Italian city (Mediterranean temperate urban environment).
- **Deployment Period**: March 10, 2004, 18:00:00 through April 4, 2005, 14:00:00 (approx. 13 continuous months).
- **Measurement Architecture**:
  1. **Reference certified analyzers** providing true ground-truth concentrations (`*GT*`) for Carbon Monoxide ($\text{CO}$), Non-Metanic Hydrocarbons ($\text{NMHC}$), Benzene ($\text{C}_6\text{H}_6$), Total Nitrogen Oxides ($\text{NO}_x$), and Nitrogen Dioxide ($\text{NO}_2$).
  2. **Solid-state metal oxide chemical multisensors** (`PT08.S1` through `PT08.S5`) yielding electrical resistance responses sensitive to chemical oxidizing/reducing reactions.
  3. **Meteorological probes** recording ambient temperature ($T$), relative humidity ($RH$), and absolute humidity ($AH$).

---

## 3. Source and Citation

The raw data was collected by the **ENEA (Italian National Agency for New Technologies, Energy and Sustainable Economic Development)** Portici Research Center and donated to the UCI Machine Learning Repository.

- **Primary Repository**: [UCI Machine Learning Repository – Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality)
- **Dataset DOI**: [10.24432/C59Q63](https://doi.org/10.24432/C59Q63)
- **License**: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)).

### Academic Citation
```bibtex
@article{devito2008field,
  title={On-field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario},
  author={De Vito, Saverio and Massera, Ettore and Piga, Marco and Martinotto, Luca and Di Francia, Girolamo},
  journal={Sensors and Actuators B: Chemical},
  volume={129},
  number={2},
  pages={750--757},
  year={2008},
  publisher={Elsevier},
  doi={10.1016/j.snb.2007.09.060}
}
```

---

## 4. Dataset Structure

The repository contains three primary tabular data files in comma-separated value (`.csv`) format:

```text
.
├── README.md               # Dataset Card with Hugging Face YAML metadata
├── train.csv               # 6,549 rows (69.99%) – 2004-03-10 18:00 to 2004-12-08 14:00
├── validation.csv          #   936 rows (10.00%) – 2004-12-08 15:00 to 2005-01-16 14:00
└── test.csv                # 1,872 rows (20.01%) – 2005-01-16 15:00 to 2005-04-04 14:00
```

### Split Proportions and Record Counts

| Split | File Name | Observations | Proportion | Start Timestamp | End Timestamp | File Size |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Train** | `train.csv` | 6,549 | 69.99% | 2004-03-10 18:00:00 | 2004-12-08 14:00:00 | 576.7 KB |
| **Validation** | `validation.csv` | 936 | 10.00% | 2004-12-08 15:00:00 | 2005-01-16 14:00:00 | 77.9 KB |
| **Test** | `test.csv` | 1,872 | 20.01% | 2005-01-16 15:00:00 | 2005-04-04 14:00:00 | 164.9 KB |
| **Total** | — | **9,357** | **100.00%** | **2004-03-10 18:00:00** | **2005-04-04 14:00:00** | **819.5 KB** |

---

## 5. Data Fields and Features

The dataset contains **15 verified features**. Below is the technical specification distinguishing ground-truth reference measurements from metal oxide sensor responses:

| Feature Name | Data Type | Physical Meaning | Verified Units | Measurement Origin |
|:---|:---:|:---|:---:|:---|
| `Date` | `string` | Calendar date of observation (`DD/MM/YYYY`) | — | Timestamp metadata |
| `Time` | `string` | Time of observation (`HH.MM.SS`, 24-hour hourly cycle) | — | Timestamp metadata |
| `CO(GT)` | `float64` | True hourly averaged Carbon Monoxide ($\text{CO}$) concentration | $\text{mg/m}^3$ | Co-located reference certified analyzer |
| `PT08.S1(CO)` | `float64` | Sensor resistance response (nominally targeting $\text{CO}$) | Sensor resistance ($\Omega$) | Metal oxide solid-state chemical sensor |
| `NMHC(GT)` | `float64` | True hourly averaged Non-Metanic HydroCarbons ($\text{NMHC}$) concentration | $\mu\text{g/m}^3$ | Co-located reference certified analyzer |
| `C6H6(GT)` | `float64` | True hourly averaged Benzene ($\text{C}_6\text{H}_6$) concentration | $\mu\text{g/m}^3$ | Co-located reference certified analyzer |
| `PT08.S2(NMHC)` | `float64` | Sensor resistance response (nominally targeting $\text{NMHC}$) | Sensor resistance ($\Omega$) | Metal oxide solid-state chemical sensor |
| `NOx(GT)` | `float64` | True hourly averaged Nitrogen Oxides ($\text{NO}_x$) concentration | $\text{ppb}$ | Co-located reference certified analyzer |
| `PT08.S3(NOx)` | `float64` | Sensor resistance response (nominally targeting $\text{NO}_x$) | Sensor resistance ($\Omega$) | Metal oxide solid-state chemical sensor |
| `NO2(GT)` | `float64` | True hourly averaged Nitrogen Dioxide ($\text{NO}_2$) concentration | $\mu\text{g/m}^3$ | Co-located reference certified analyzer |
| `PT08.S4(NO2)` | `float64` | Sensor resistance response (nominally targeting $\text{NO}_2$) | Sensor resistance ($\Omega$) | Metal oxide solid-state chemical sensor |
| `PT08.S5(O3)` | `float64` | Sensor resistance response (nominally targeting Ozone $\text{O}_3$) | Sensor resistance ($\Omega$) | Metal oxide solid-state chemical sensor |
| `T` | `float64` | Ambient air temperature | $^\circ\text{C}$ | Integrated meteorological sensor |
| `RH` | `float64` | Relative humidity | $\%$ | Integrated meteorological sensor |
| `AH` | `float64` | Absolute humidity | $\text{g/m}^3$ | Integrated meteorological sensor |

> **Important Distinction**: Columns containing `(GT)` represent **Ground Truth** physical chemical concentrations measured by high-precision, certified municipal analyzers. Columns containing `PT08.S*` represent **raw sensor resistance responses** from low-cost metal oxide transducers, which are non-linearly dependent on target gases, cross-interfering pollutants, temperature, and humidity.

---

## 6. Data Preprocessing Methodology

The original raw file from the UCI repository (`AirQualityUCI.csv`) required systematic data engineering to reach machine-learning and Hugging Face hub standards:

1. **Delimiter and Numeric Notation Normalization**:
   - The source file was exported using standard European formatting with semicolons (`;`) as column delimiters and commas (`,`) as decimal separators.
   - Semicolons and decimal commas were systematically parsed, converting all numeric columns into standard IEEE floating-point numbers.
2. **Artifact Truncation & Blank Record Removal**:
   - Trailing semicolon delimiters in the raw CSV created two completely empty artifact columns (`Unnamed: 15` and `Unnamed: 16`), which were removed.
   - 114 blank trailing rows at the end of the file (indices 9,357 to 9,470) were pruned, reducing the raw record count from **9,471** to **9,357** clean records.
3. **Missing-Value Sentinel Conversion (`-200` $\rightarrow$ `NaN`)**:
   - In the raw dataset, instrument failure and missing values were encoded as the negative integer sentinel `-200`.
   - Because physical pollutant concentrations, sensor resistances, and Kelvin/Celsius ambient temperatures cannot equal $-200$, all occurrences were converted to standard IEEE floating-point `NaN` (saved as empty fields in CSV).
   - *Audit result*: Exactly 0 instances of `-200` remain in the processed splits.
4. **Sparsity Characterization**:
   - `NMHC(GT)`: Missing in **90.23%** of records (8,443 rows). The reference instrument for non-metanic hydrocarbons was discontinued after May 2004. These entries are preserved as `NaN` rather than dropping the rows, preserving over 8,400 valid readings across all other pollutants.
   - `CO(GT)`, `NOx(GT)`, `NO2(GT)`: Missing in approximately 17.5%–18.0% of records due to reference analyzer recalibration cycles.
   - `PT08.S1` through `PT08.S5`, `T`, `RH`, `AH`: Missing in **3.91%** of records (366 rows) corresponding to total multisensor hardware power outages.
5. **Leakage-Free Chronological Split**:
   - Rather than random shuffling—which causes temporal lookahead leakage in time-series data—records were sorted chronologically by timestamp and split into strictly contiguous blocks:
     - **Train**: Indices 0 to 6,548 (First 70.0%)
     - **Validation**: Indices 6,549 to 7,484 (Next 10.0%)
     - **Test**: Indices 7,485 to 9,356 (Final 20.0%)

---

## 7. Loading the Dataset

### A. Using the Hugging Face `datasets` Library

```python
from datasets import load_dataset

# Load from Hugging Face Hub:
# Replace <your-username> with your actual Hugging Face profile name
dataset = load_dataset("<your-username>/air-quality-timeseries")

# Or load directly from local split CSVs:
dataset = load_dataset("csv", data_files={
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv"
})

print(dataset)
print("\nFirst training sample:")
print(dataset["train"][0])
```

#### Output:
```text
DatasetDict({
    train: Dataset({
        features: ['Date', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'],
        num_rows: 6549
    })
    validation: Dataset({
        features: ['Date', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'],
        num_rows: 936
    })
    test: Dataset({
        features: ['Date', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'],
        num_rows: 1872
    })
})
```

### B. Using Pandas

```python
import pandas as pd

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

print(f"Train records: {len(train_df)}")
print(f"Validation records: {len(val_df)}")
print(f"Test records: {len(test_df)}")
```

---

## 8. Intended Uses

- **In-Field Low-Cost Sensor Calibration**: Learning non-linear transfer functions mapping multi-sensor resistance arrays (`PT08.S*`) and environmental covariates ($T, RH, AH$) to true ground-truth concentrations (`*GT*`).
- **Time-Series Forecasting**: Benchmarking autoregressive and deep sequence models (e.g., ARIMA, LSTM, Transformer-based forecasters) on multi-pollutant prediction over continuous time windows.
- **Multivariate Tabular Regression**: Predicting individual pollutant concentrations (e.g., Benzene $\text{C}_6\text{H}_6$ or $\text{CO}$) from chemical sensor arrays.
- **Foundational Data Layer for Small Language Models (SLMs)**:
  - Synthetic instruction-tuning dataset generation (converting numerical sensor readings into structured text reasoning pairs).
  - Training models to perform numeric-to-text environmental summaries.
  - Teaching small language models to accurately refuse or report "no reading available" on missing sensors (`NaN`) rather than hallucinating plausible concentration values.

> **Scope Note**: This dataset consists strictly of cleaned numerical tabular records. It is not an end-to-end conversational text corpus. Language-model researchers must first apply an instruction/prompt-serialization template before fine-tuning causal LLMs.

---

## 9. Limitations and Considerations

- **Single Sampling Location**: Data reflects the micro-climate and emission profile of a single road-level monitoring station in an Italian urban setting; spatial generalization requires domain transfer or fine-tuning.
- **Temporal Window**: Measurements were collected during 2004–2005. While physical chemical relationships remain invariant, municipal emission baselines reflect the vehicle fleet and regulations of that period.
- **NMHC Ground Truth Retirement**: `NMHC(GT)` is missing in 90.23% of rows. Any model requiring NMHC ground truth should restrict training to the initial months or treat NMHC as a target with high missingness.
- **Sensor Drift & Thermal Cross-Sensitivity**: Metal oxide semiconductor sensors exhibit drift over time and significant non-linear sensitivity to relative and absolute humidity. Models that omit temperature and humidity covariates will likely demonstrate degraded predictive validity.
- **Not an Official AQI Feed**: The dataset contains raw gas concentrations and sensor resistances, not categorized Air Quality Index (AQI) values or official health alerts.

---

## 10. Dataset Maintenance and Versioning

- **Current Version**: `v1.0.0` (November 2026) – Baseline preprocessed release with chronological 70/10/20 train/validation/test splits, sentinel `-200` normalization, and artifact column pruning.
- **Future Work**:
  - `v1.1.0` (Planned): Release of pre-tokenized and instruction-paired `.jsonl` derivatives for direct fine-tuning of Small Language Models (e.g., Qwen / Bonsai architectures).
- **Maintenance**: Maintained as part of academic coursework for Semester 3 BTech CSE (AI/ML).

---

## 11. Acknowledgements and References

We gratefully acknowledge the researchers at **ENEA (Portici Research Center, Italy)** for collecting and publishing the original field measurements, and the **UCI Machine Learning Repository** for maintaining the benchmark dataset.

1. **Saverio De Vito, Ettore Massera, Marco Piga, Luca Martinotto, and Girolamo Di Francia** (2008). *On-field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario*. Sensors and Actuators B: Chemical, 129(2), 750–757.
2. **UCI Machine Learning Repository**: [Air Quality Dataset Entry](https://archive.ics.uci.edu/dataset/360/air+quality).
