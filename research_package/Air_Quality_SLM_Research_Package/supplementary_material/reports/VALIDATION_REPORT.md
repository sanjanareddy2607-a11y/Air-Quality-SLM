# Air Quality Dataset Validation Report

**Dataset Name**: `air-quality-timeseries` (UCI Air Quality)  
**Execution Timestamp**: 2026-11-12 14:20:51  
**Validation Suite**: Automated 12-Point Machine Learning & Hugging Face Standard Verification  
**Final Validation Status**: **PASSED**

---

## 1. Executive Summary Table

| Metric / Dimension | Raw Source Dataset | Cleaned & Processed Dataset |
|:---|:---:|:---:|
| **Total Rows** | 9,471 | 9,357 |
| **Total Columns** | 17 (includes 2 unnamed artifacts) | 15 (clean structured features) |
| **Date Range** | 10/03/2004 to 04/04/2005 | 10/03/2004 18:00:00 to 04/04/2005 14:00:00 |
| **Missing Value Sentinels (-200)** | Present across all numeric features | 0 (Cleanly converted to IEEE standard `NaN`) |
| **Duplicate Timestamps** | 0 | 0 |
| **Split Leakage** | N/A | 0 overlap across all split boundaries |

---

## 2. Dataset Split Proportions

| Split | File Name | Record Count | Proportion (%) | Start Timestamp | End Timestamp |
|:---|:---|:---:|:---:|:---:|:---:|
| **Train** | `train.csv` | 6,549 | 69.99% | 2004-03-10 18:00:00 | 2004-12-08 14:00:00 |
| **Validation** | `validation.csv` | 936 | 10.00% | 2004-12-08 15:00:00 | 2005-01-16 14:00:00 |
| **Test** | `test.csv` | 1,872 | 20.01% | 2005-01-16 15:00:00 | 2005-04-04 14:00:00 |
| **Total** | — | **9,357** | **100.00%** | **2004-03-10 18:00:00** | **2005-04-04 14:00:00** |

---

## 3. Comprehensive 12-Check Verification Results

| # | Verification Check | Status | Details |
|:---:|:---|:---:|:---|
| 1 | CSV Files Readable | **PASSED** | Reopened train (6549), val (936), test (1872) |
| 2 | Column Schema Integrity | **PASSED** | All 15 expected columns present in all 3 splits: True |
| 3 | Data Type Correctness | **PASSED** | Date and Time are strings; all 13 sensor and weather features are float64 |
| 4 | Non-Empty Records | **PASSED** | train: 6549, validation: 936, test: 1872 |
| 5 | Missing Value Handling | **PASSED** | 0 remaining -200 sentinels. Missing values represented as IEEE NaN (total NaNs: 16701) |
| 6 | Duplicate Check | **PASSED** | Duplicate timestamps: train=0, val=0, test=0 |
| 7 | Split Disjointness (No Overlap) | **PASSED** | Overlap counts: train/val=0, val/test=0, train/test=0 |
| 8 | Approximate 70/10/20 Proportions | **PASSED** | train: 69.99%, validation: 10.00%, test: 20.01% (approx 70/10/20) |
| 9 | Chronological Ordering Preserved | **PASSED** | Within splits monotonic: True. Strict temporal boundary: True |
| 10 | README Consistency | **PASSED** | Counts in README match exactly: 6549 train, 936 val, 1872 test, 9357 total |
| 11 | Hugging Face datasets Loader | **PASSED** | Successfully loaded DatasetDict with splits: ['train', 'validation', 'test'] |
| 12 | Sample Record Retrieval | **PASSED** | Printed sample record from each of train, validation, and test |

---

## 4. Sample Record Verification (via `datasets`)

### Training Split Example (`train[0]`):
```json
{'Date': '10/03/2004', 'Time': '18.00.00', 'CO(GT)': 2.6, 'PT08.S1(CO)': 1360.0, 'NMHC(GT)': 150.0, 'C6H6(GT)': 11.9, 'PT08.S2(NMHC)': 1046.0, 'NOx(GT)': 166.0, 'PT08.S3(NOx)': 1056.0, 'NO2(GT)': 113.0, 'PT08.S4(NO2)': 1692.0, 'PT08.S5(O3)': 1268.0, 'T': 13.6, 'RH': 48.9, 'AH': 0.7578}
```

### Validation Split Example (`validation[0]`):
```json
{'Date': '08/12/2004', 'Time': '15.00.00', 'CO(GT)': 1.2, 'PT08.S1(CO)': 997.0, 'NMHC(GT)': None, 'C6H6(GT)': 4.6, 'PT08.S2(NMHC)': 742.0, 'NOx(GT)': 179.0, 'PT08.S3(NOx)': 911.0, 'NO2(GT)': 82.0, 'PT08.S4(NO2)': 1180.0, 'PT08.S5(O3)': 899.0, 'T': 16.7, 'RH': 54.7, 'AH': 1.0311}
```

### Testing Split Example (`test[0]`):
```json
{'Date': '16/01/2005', 'Time': '15.00.00', 'CO(GT)': 0.7, 'PT08.S1(CO)': 833.0, 'NMHC(GT)': None, 'C6H6(GT)': 2.0, 'PT08.S2(NMHC)': 584.0, 'NOx(GT)': 107.0, 'PT08.S3(NOx)': 1144.0, 'NO2(GT)': 80.0, 'PT08.S4(NO2)': 821.0, 'PT08.S5(O3)': 463.0, 'T': 11.3, 'RH': 32.5, 'AH': 0.4334}
```

---

## 5. Final Quality Audit

- **Data Integrity**: PASSED
- **Temporal Consistency**: PASSED (100% strictly chronological, zero leakage)
- **Hugging Face Hub Compatibility**: PASSED (Compliant with Hugging Face Hub dataset viewer & `datasets` library)
