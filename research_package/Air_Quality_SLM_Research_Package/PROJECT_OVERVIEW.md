# Project Overview

## 1. Problem statement

Air-quality monitoring produces dense, highly structured numerical records: hourly concentrations
of CO, NOx, NO2, benzene and non-methane hydrocarbons, metal-oxide sensor resistances, and
meteorological variables. These records are consumed by specialists through dashboards and APIs,
but the everyday question asked of them — *"what was the NOx level on 8 December 2004 at 15:00?"*,
*"how humid was the air?"* — is asked in natural language.

General-purpose language models can phrase such answers fluently, but they cannot know the value of
a measurement taken at a specific place and time; when asked, they produce a plausible number.
This project investigates the narrow, practical question:

> Can a very small language model, trained from scratch on a *single* tabular air-quality dataset,
> learn the vocabulary, units and sentence forms of that domain well enough to serve as the natural
> language front end of an air-quality assistant — and if so, what exactly does it learn, and what
> does it still fail at?

## 2. Research motivation

* **Domain grounding over scale.** A 20.4 M-parameter decoder trained only on 3.04 M tokens of
  air-quality text is a controlled test of how much domain behaviour comes from data formatting
  rather than from general language ability.
* **Negative results are informative.** Constant numerical output, hallucinated timestamps and
  correct-but-empty formatting are the failure modes practitioners actually meet when they point a
  small model at a measurement table. They deserve measurement, not concealment.
* **Deployment realism.** A model that only exists as a PyTorch checkpoint is not usable by a
  non-programmer. The project therefore carries the model through GGUF conversion into Ollama
  Desktop and measures the difference between "it runs" and "it is correct".

## 3. Objectives

| # | Objective | Outcome in this project |
|---|---|---|
| O1 | Prepare a reproducible air-quality dataset | 9,357 cleaned records, 15 features, chronological 70/10/20 split, 12-point validation passed (`docs/VALIDATION_REPORT.md`) |
| O2 | Convert the table into instruction-form text | 4 templates per record, 37,428 samples, 4,344,159 tokens |
| O3 | Train a compact domain SLM from scratch | Qwen3-architecture decoder, 20,397,568 parameters, 150 steps on CPU |
| O4 | Evaluate language modelling and numerical accuracy separately | Full-split loss/perplexity, teacher-forced answer loss, regex numeric scoring against dataset labels, constant-predictor baselines |
| O5 | Improve the model where measurement shows a defect | Continued training at 128-token context (v2); diagnosis of the 64-token truncation that hid the instruction |
| O6 | Deploy through Ollama with a reproducible artefact | Two GGUF exports, three registered tags, deterministic decoding parameters |
| O7 | Make the deployed assistant answer correctly | Deterministic Python layer: exact dataset lookup, closed-form absolute humidity, out-of-domain refusal |

## 4. Proposed methodology (summary)

1. **Dataset**: UCI Air Quality (De Vito et al., 2008) — European-format CSV, semicolon-delimited,
   decimal comma, `-200` sentinel for missing values.
2. **Cleaning**: drop artifact columns and 114 blank rows (9,471 → 9,357 records), convert `-200`
   to `NaN`, keep missing values as missing, sort chronologically, split contiguously 70/10/20.
3. **Textualisation**: each record is rendered as four texts (a full measurement report, an
   absolute-humidity instruction, a temperature/relative-humidity question, a pollutant query) with
   values inserted verbatim and units written out.
4. **Tokenisation**: Qwen2.5-0.5B BPE tokenizer (151,665 tokens) applied to the concatenated stream;
   uint32 memory-mapped binaries per split.
5. **Training**: decoder-only Qwen3-style transformer (RMSNorm, GQA with per-head Q/K norm, RoPE,
   SwiGLU, tied embeddings, d_model 128, 4 layers, d_ff 512), next-token cross-entropy, AdamW,
   warmup + cosine decay, CPU only.
6. **Evaluation**: one harness for all checkpoints — non-overlapping full-split loss, greedy
   generations from the four templates, numeric extraction compared with the dataset's own labels,
   structure/format metrics, memorisation checks, constant-predictor baselines.
7. **Conversion**: custom exporter writing a `qwen3`-architecture F32 GGUF with the verified tensor
   mapping (input embeddings scaled by √128, unscaled tied output head, Q/K norm tensors preserved).
8. **Deployment**: `ollama create` with greedy decoding, `num_ctx` matched to the trained context,
   `num_predict` sized to the longest template answer.
9. **Application layer**: `domain_guard.py` routes greetings, out-of-domain questions, lookups and
   humidity calculations; numbers come from Python, wording from the model.

## 5. Contributions

1. **A reproducible, split-safe instruction corpus** derived from a public benchmark table, with the
   preprocessing verified by 12 automated checks and zero timestamp overlap between splits.
2. **A from-scratch 20.4 M-parameter domain SLM** and a second checkpoint produced by documented
   continued training (context 64 → 128, 150 → 1,350 total steps), with both checkpoints preserved
   separately so the comparison is auditable.
3. **A two-axis evaluation protocol** that separates language-modelling quality from numerical
   prediction quality, including constant-predictor baselines, per-template teacher-forced loss and
   output-structure metrics. This protocol is what makes the central finding visible: v2 is a
   near-perfect formatter and a constant-number speaker.
4. **A quantified diagnosis of the 64-token context defect**: the smallest complete training example
   is 83 tokens, so v1 never saw an instruction and its answer in the same window; Ollama silently
   reduced an 86-token prompt to its last 34 tokens.
5. **A verified PyTorch→GGUF mapping** for a non-standard Qwen3 variant, validated token-for-token
   against llama.cpp (13/13 tokenizer samples, identical greedy text).
6. **A deployed architecture that is honest by construction**: the model supplies language, Python
   supplies measurements, and the closed-form humidity formula is shown to reproduce the dataset's
   own `AH` column with MAE 0.019 g/m³ (r = 0.99956) — evidence that arithmetic, not information,
   was the model's bottleneck.

## 6. Scope and non-goals

* The model is **not** an air-quality predictor, a forecasting system, a regulatory tool or a
  real-time monitor. Its outputs are not measurements.
* Only one monitoring station and one 13-month period (March 2004 – April 2005) are covered.
* No hyper-parameter search, no quantisation study and no GPU training were performed; the CPU-only
  constraint is documented rather than hidden.
* Retrieval and humidity calculation are deterministic Python components; the language model is not
  used for them. Confusing the two is the most likely error a reader of this project can make, and
  `SYSTEM_ARCHITECTURE.md` marks the boundary explicitly.

## 7. Where the project stands

| Component | Evidence status |
|---|---|
| Dataset preparation | Complete and independently verifiable from the CSVs |
| Tokenisation | Counts recorded; `.bin` files referenced but not present in the workspace (see `MISSING_EVIDENCE.md`) |
| v1 training | Complete; loss progression and runtime recorded in `training_summary.json` |
| v2 continued training | Complete to 1,200 steps (0.202 epoch); plateau and early-stop logic documented |
| v3 | Registered Modelfile only — no distinct weights, no distinct measurement |
| Full evaluation | Complete for v1 and v2 on validation and test splits |
| GGUF conversion | Complete, with checksums and tensor-mapping verification |
| Ollama deployment | Complete and running (Ollama 0.33.2, F32 GGUF, `num_ctx` 64/128) |
| Python application | Implemented (routing, retrieval, humidity, guard), with known unit-labelling defects |
| Screenshots | **None exist** in the workspace — must be captured manually (`MISSING_EVIDENCE.md`) |
