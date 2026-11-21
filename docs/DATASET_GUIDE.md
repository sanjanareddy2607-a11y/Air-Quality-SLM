# Dataset Guide

Everything you need to understand the data before reading the rest of the documentation. Statistics were
recomputed from `data/raw/air+quality/AirQualityUCI.csv` and the processed splits; the command that
reproduces them is at the end.

---

## 1. What the dataset is, in plain language

For one year, a small cabinet of chemical sensors sat at road level in a polluted Italian city. Every
hour it sniffed the air and reported how strongly each of its five sensors responded. Next to it, a
certified reference analyser measured the *actual* concentrations of five pollutants. Weather
instruments recorded temperature, relative humidity and absolute humidity at the same time.

The result is a spreadsheet where each row is **one hour** and each column is **one measurement**. It is
a historical record, not a live feed: the newest row is from April 2005.

That framing matters for this project. The model was trained on these rows, so it can only ever talk
about hours that are in this table.

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository, dataset 360 ("Air Quality") |
| Donor / authors | S. De Vito, E. Massera, M. Piga, L. Martinotto, G. Di Francia |
| DOI | 10.24432/C59K5F |
| Licence | CC BY 4.0 (research use) |
| Published | 2008 (dataset donated to UCI 2016) |
| File in this project | `data/raw/air+quality/AirQualityUCI.csv` (semicolon-separated, decimal comma) |
| Rows in the file | 9,358 records including the header row; **9,357 data rows** after dropping trailing blank rows |
| Columns | 15 (2 time columns + 13 measurements) |
| Sampling | hourly averages |
| Measurement window | 10 March 2004 18:00 → 4 April 2005 14:00 |
| Unique calendar dates | 391 |

## 2. The fifteen columns

| # | Column | Meaning | Unit | Type |
|---|---|---|---|---|
| 1 | `Date` | Measurement date | DD/MM/YYYY | time |
| 2 | `Time` | Measurement time | HH.MM.SS | time |
| 3 | `CO(GT)` | Carbon monoxide, reference analyser | **mg/m³** | ground truth |
| 4 | `PT08.S1(CO)` | Tin-oxide sensor response, nominally targeting CO | sensor counts (~ -200…2040) | sensor |
| 5 | `NMHC(GT)` | Non-methane hydrocarbons, reference analyser | µg/m³ | ground truth |
| 6 | `C6H6(GT)` | Benzene, reference analyser | µg/m³ | ground truth |
| 7 | `PT08.S2(NMHC)` | Titania sensor response, nominally targeting NMHC | sensor counts | sensor |
| 8 | `NOx(GT)` | Total nitrogen oxides, reference analyser | **ppb** | ground truth |
| 9 | `PT08.S3(NOx)` | Tungsten-oxide sensor response, nominally targeting NOx | sensor counts | sensor |
| 10 | `NO2(GT)` | Nitrogen dioxide, reference analyser | µg/m³ | ground truth |
| 11 | `PT08.S4(NO2)` | Tungsten-oxide sensor response, nominally targeting NO₂ | sensor counts | sensor |
| 12 | `PT08.S5(O3)` | Indium-oxide sensor response, nominally targeting ozone | sensor counts | sensor |
| 13 | `T` | Temperature | °C | environment |
| 14 | `RH` | Relative humidity | % | environment |
| 15 | `AH` | Absolute humidity | g/m³ | environment (derived by the dataset authors) |

Four families, and the distinction matters when you read the model's output:

* **Ground truth** (`*GT`) — what a calibrated analyser measured. These are the labels.
* **Chemical sensor responses** (`PT08.*`) — raw instrument signals, not concentrations. They drift and
  cross-react; converting them to a concentration requires calibration the dataset does not provide.
* **Environmental** (`T`, `RH`, `AH`) — measured or computed weather variables.
* **Derived** (`AH`) — computed by the dataset authors from `T` and `RH`; it is a physical formula
  output, not an independent measurement. This is why the project's humidity task is really a
  formula-reconstruction task.

## 3. Missing data — the single most important property

The file uses the sentinel value **`-200`** wherever a measurement is absent. There are **16,701 such
cells**, which is 12.0 % of all measurement cells.

| Column | Missing cells | % of 9,357 rows |
|---|---:|---:|
| `NMHC(GT)` | 8,443 | **90.23 %** |
| `CO(GT)` | 1,683 | 17.99 % |
| `NO2(GT)` | 1,642 | 17.55 % |
| `NOx(GT)` | 1,639 | 17.52 % |
| `C6H6(GT)` | 366 | 3.91 % |
| `PT08.S1(CO)` | 366 | 3.91 % |
| `PT08.S2(NMHC)` | 366 | 3.91 % |
| `PT08.S3(NOx)` | 366 | 3.91 % |
| `PT08.S4(NO2)` | 366 | 3.91 % |
| `PT08.S5(O3)` | 366 | 3.91 % |
| `T` | 366 | 3.91 % |
| `RH` | 366 | 3.91 % |
| `AH` | 366 | 3.91 % |

Two consequences that are easy to miss:

1. **`NMHC(GT)` is unusable.** It is missing in 90 % of rows and **100 % of both the validation and test
   splits**, so no NMHC metric can be computed on held-out data at all.
2. **The 366 all-column gaps are the same 366 rows** — a block of missing hours, not scattered noise.

**The project does not impute anything.** `-200` becomes `NaN` in the CSVs and the literal word
`Unavailable` in the text templates. This is a deliberate choice: inventing numbers to fill gaps would
train the model to hallucinate.

## 4. Ranges and sanity checks (whole dataset)

| Column | Min | Max | Mean |
|---|---:|---:|---:|
| `CO(GT)` | 0.1 | 11.9 | 2.15 |
| `NOx(GT)` | 17 | 1,479 | — |
| `NO2(GT)` | 17 | 340 | — |
| `C6H6(GT)` | 0.1 | 63.7 | — |
| `T` (°C) | −1.9 | 44.6 | — |
| `RH` (%) | 9.2 | 88.7 | — |
| `AH` (g/m³) | 0.1847 | 2.231 | 1.0255 |

Sanity checks performed (`src/validate_dataset.py`, `docs/VALIDATION_REPORT.md`):

* 0 duplicate `(Date, Time)` pairs.
* Dates are strictly increasing when the file is read in order.
* No value outside the documented physical range after sentinel removal.
* The `-200` count equals the `NaN` count exactly — no other encoding of missingness exists.

## 5. How the data was split

The split is **chronological, not random**, because a monitoring record must be evaluated on the future:

| Split | Rows | From | To |
|---|---:|---|---|
| train | 6,549 (70 %) | 10/03/2004 18:00 | 08/12/2004 14:00 |
| validation | 936 (10 %) | 08/12/2004 15:00 | 16/01/2005 14:00 |
| test | 1,872 (20 %) | 16/01/2005 15:00 | 04/04/2005 14:00 |

One calendar date (`08/12/2004`) appears in both train and validation because the boundary falls mid-day.
**No `(Date, Time)` pair appears in two splits.** Some automated checks count overlap by date and therefore
report "1"; that is a boundary artefact, not leakage (see `KNOWN_LIMITATIONS.md`).

Files: `data/processed/train.csv`, `validation.csv`, `test.csv` — 15 columns each, same schema as the raw
file with `-200` replaced by empty cells.

## 6. What the model is trained to say about a row

Each row is turned into four natural-language texts (see `TRAINING_GUIDE.md` §3): a full report, an
absolute-humidity instruction, a temperature/RH question, and a pollutant question. The numbers are
serialised verbatim — **no scaling, no normalisation, no binning** — so the model sees raw physical
values with their units written next to them.

That is also why a "prediction" from this model is really a lookup it was asked to memorise: the answer is
the same row's value, restated in a sentence.

## 7. Known data-quality concerns to state in any paper

1. **Sensor drift and cross-sensitivity** — documented by the dataset authors; the PT08 responses are not
   stable proxies for concentration.
2. **`NMHC(GT)` is 90 % missing** and completely absent from validation and test.
3. **The period is 2004–2005**, so nothing here describes present-day air quality.
4. **One city, one roadside site** — no spatial coverage.
5. **`AH` is derived, not measured** — so an "AH prediction" error mixes model error with formula choice.
6. **Hourly averages hide peaks** — short pollution events are smoothed away.
7. **The dataset has no health, weather-forecast or policy columns**, which is why out-of-domain
   questions genuinely cannot be answered from it.

## 8. Reproduce the statistics

```bash
./venv/Scripts/python.exe - <<'PY'
import pandas as pd
raw = pd.read_csv("data/raw/air+quality/AirQualityUCI.csv", sep=";", decimal=",")
raw = raw.dropna(how="all", axis=1).dropna(how="all")
num = raw.drop(columns=["Date", "Time"])
print("rows", len(raw))
print("sentinel cells", int((num == -200).sum().sum()))
print(num.replace(-200, pd.NA).isna().sum())
print(raw.head(3))
PY
```

## 9. Citation

```
S. De Vito, E. Massera, M. Piga, L. Martinotto, G. Di Francia, "On field calibration of an
electronic nose for benzene estimation in an urban pollution monitoring scenario,"
Sensors and Actuators B: Chemical, 129(2), 750-757, 2008.
Dataset: UCI Machine Learning Repository, dataset 360, doi:10.24432/C59K5F, CC BY 4.0.
```
