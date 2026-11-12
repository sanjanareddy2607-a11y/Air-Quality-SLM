# Air Quality LLM Dataset Builder

## Contents

- **`air_quality_llm_dataset_builder.ipynb`** — the notebook. Open it, set
  `INPUT_CSV` to your `AirQualityUCI.csv` path, and run all cells.
- **`air_quality_api_docs/`** — the Google Air Quality API documentation used
  to ground pollutant names/units, the "no calibrated AQI in raw data" caveat,
  and the health-recommendation population list referenced by the notebook's
  prompt generators.
- **`sample_output_preview.jsonl`** — 15 example lines from an actual run of
  the notebook against the full 9,357-row UCI dataset, so you can see the
  output quality before running it yourself.

## What changed vs. the original `Data_Convertion.ipynb`

| | Original | This notebook |
|---|---|---|
| Rows used | first 100 of 9,357 | all 9,357 (configurable) |
| Instruction templates | 1, repeated verbatim | 7 intent types × multiple paraphrases each |
| `-200` missing-value sentinel | passed through as a literal reading | cleaned to `NaN`; generates explicit "no reading available" examples |
| Pollutants not in the dataset (PM2.5, SO2, ...) | never addressed | explicit "this dataset doesn't include that" examples |
| Sensor vs. reference-analyzer columns | not distinguished | `PT08.*` sensor columns explicitly labeled as raw sensor response, not concentration |
| Output | single ~15KB JSONL (100 lines) | ~29,850 examples across train/val JSONL + cleaned JSON, plus compressed `.jsonl.zst`/`.gz` |

## Why this reduces false positives

The three things above — abstaining on genuinely missing readings instead of
reporting `-200`, abstaining on pollutants the dataset never measured instead
of inventing a number, and never overclaiming an official safety verdict from
raw sensor data — are all explicit training examples in the generated
dataset. A model fine-tuned on only positive factual examples has no signal
for when *not* to answer; this generator bakes "I don't have that" responses
in at roughly 20–35% of the dataset so the model learns both behaviors.

## Compression

The notebook's last cell compresses the JSONL files with `zstandard`
(`.jsonl.zst`) if available, or falls back to `gzip` (`.jsonl.gz`)
automatically — e.g. the full training file compressed from 6.6 MB down to
568 KB (gzip) in testing.
