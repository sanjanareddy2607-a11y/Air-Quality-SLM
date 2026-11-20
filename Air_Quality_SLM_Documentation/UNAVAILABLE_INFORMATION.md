# Unavailable Information

Everything a reader might expect and will not find. Each entry states what is missing, how that was
established, and what it affects. Nothing here was filled in by assumption.

---

## 1. Screenshots of the Ollama Desktop session

**Missing.** No image file of the project exists anywhere. A recursive search for `*.png`, `*.jpg`,
`*.jpeg`, `*.webp` outside `venv/` returns only two fixtures inside the vendored `markitdown` package.

**Affects.** The two screenshots the brief asks for (greeting; refusal + humidity) cannot be included.

**Closure.** Capture them following `docs/assets/screenshots/README.md`, which gives the exact prompts, the
expected replies, and a warning that the replies are produced by `ollama_proxy.py`, not by the model.

## 2. The dataset download event

**Missing.** No URL, mirror, date or checksum recorded at download time. The raw file's last-write time is
2026-08-05, which is the only date evidence.

**Affects.** "Reproducible from first principles" stops at the CSV; a reader must fetch it from UCI.

**Closure.** Record `https://archive.ics.uci.edu/dataset/360/air+quality` plus SHA-256 `13277ae5…`.

## 3. A dependency manifest

**Missing.** No `requirements.txt`, `pyproject.toml`, `environment.yml`, `Pipfile` or lockfile.

**Affects.** Versions in this documentation were read from the live venv on 2026-11-20. They are accurate
for this machine, not pinned by the repository.

**Closure.** `pip freeze > requirements.txt`, then commit it.

## 4. A v3 evaluation

**Missing.** No loss, no perplexity, no generation dump, no report for `air-quality-slm-v3:latest`.

**Affects.** v3 cannot be described as better or worse than v2 in any measured sense.

**Closure.** Run `src/evaluate_v2_full.py` against the v3 tag and store the dump.

## 5. Any metric for precision, recall, F1, latency or human preference

**Missing.** None were computed.

**Affects.** These must not appear in a results table. Latency is anecdotal ("well under a second").

**Closure.** These metrics require a different experimental design; see `docs/EVALUATION_REPORT.md` §11.

## 6. A quantised model

**Missing.** No `Q4_K_M` or any other quantised GGUF exists.

**Affects.** No size, speed or quality claim about quantisation is possible.

**Closure.** Quantise with llama.cpp and evaluate with the same harness.

## 7. Hugging Face repository contents

**Unverified.** `src/upload_to_huggingface.py` exists and the model card names the dataset
`sanju7/air-quality-uci`, but no upload log, commit record or repository listing was found. The documentation
does not claim a Hugging Face model repository exists.

**Closure.** Check the account and record the URL, or state that no upload was performed.

## 8. Screenshots, plots or figures produced before this audit

**Missing.** No experiment figure existed in the repository before this documentation work. All 24 figures
in `docs/assets/` were generated during it from stored data (see `FIGURES_GENERATED.md`).

**Affects.** There are no "original" figures to compare against, and no historical plot can be cited.

## 9. A record of intermediate experiments, 12–15 November

**Missing.** Nothing in the workspace is dated between the processed CSVs (12 Nov) and the token streams
(15 Nov 22:31).

**Affects.** The timeline in `docs/PROJECT_HISTORY.md` has a gap. It is stated as a gap, not filled.

## 10. The `training/logs/` directory referenced by early reports

**Missing.** `docs/CONTINUED_TRAINING_REPORT.md` refers to `training/logs/loss_curve.csv`; the file actually
lives at `docs/loss_curve.csv`. No `training/logs/` directory exists.

**Affects.** A path in an older report is wrong; the data itself is present.

## 11. Optimiser state for v2

**Missing.** The exported v1 weights contained no optimiser/scheduler state, so v2 began with a fresh AdamW.

**Affects.** v2 is *continued training*, not a bit-exact resumption. Stated wherever v2 is described.

## 12. Exact reproducibility of the v1 loss trace

**Missing.** v1 samples batches with replacement and depends on thread order.

**Affects.** A re-run will not match 1.3390/1.3651 to the last digit. The stored JSON is the record.

## 13. Archived logs from the earliest service runs

**Missing.** `proxy.log.prev`, `backend.log.prev` and `watchdog.boot.log.prev` are truncated predecessors; the
earliest part of the service history is gone.

**Affects.** The service narrative starts at 2026-11-20 00:21.

## 14. A v3 system prompt actually sent by the application

**Missing.** `Modelfile.v3` documents an intended system text, and `ollama_client.py` supports a
`system_prompt` argument, but `app.py` does not pass one, and `MODEL_NAME` still points at v2.

**Affects.** v3's `{{ .System }}` branch is unexercised. Do not claim the system prompt is in use.

## 15. Commit history for the evaluation and documentation work

**Missing.** The newest commit is `e967ef9` (proxy hardening). All evaluation dumps, reports, figures and both
packages are uncommitted working-tree files.

**Affects.** A clone of the repository will not reproduce this documentation.

## 16. Airflow/weather/health/policy columns

**Missing.** The dataset has none, so questions about them genuinely cannot be answered. That is why
`domain_guard.py` refuses them, not a limitation of the model.
