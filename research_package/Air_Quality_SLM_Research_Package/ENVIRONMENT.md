# Environment

Everything here was read from the machine that produced the artefacts in this package, on
2026-11-20, using `sys.version`, `platform.platform()`, `pip list` and `torch.cuda.is_available()`.
Nothing is inferred from documentation or from an earlier report.

---

## 1. Hardware

| Item | Value | How obtained |
|---|---|---|
| Operating system | Windows 11, build `10.0.26200` | `platform.platform()` → `Windows-11-10.0.26200-SP0` |
| CPU | AMD Ryzen 5 5600H with Radeon Graphics | `wmic cpu get name` |
| Logical processors | 12 | `wmic cpu get NumberOfLogicalProcessors` |
| Physical memory | 7,887,097,856 B = **7.35 GiB** | `(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory` |
| GPU used for training | **none** | `torch.cuda.is_available()` → `False` |
| PyTorch thread count | `6` reported by `torch.get_num_threads()`; the evaluation scripts also pass `--threads 6` | `torch.get_num_threads()` and `scripts/run_eval_v2.sh` |

**Consequence, stated plainly:** every training run and every evaluation in this package was executed
on CPU in float32. There is no CUDA result anywhere in this project, and the GPU training branch in
`training_config.json` (`training_gpu`, `model_gpu`) was never exercised. Any table that reports a
GPU number would be fabricated; none does.

## 2. Software

### 2.1 Interpreter

| Item | Value |
|---|---|
| Python | 3.14.4 (`tags/v3.14.4:23116f9`, 2026-04-07, MSC v.1944 64-bit) |
| System interpreter | `C:\Python314\python.exe` |
| Virtual environment | `venv/` in the project root; all commands in this package use `./venv/Scripts/python.exe` |
| Packaging | plain `pip` into the venv; no `requirements.txt`, `pyproject.toml` or lockfile exists in the repository (see `MISSING_EVIDENCE.md` §6) |

### 2.2 Packages actually installed in `venv/`

| Package | Version | Used by |
|---|---|---|
| `torch` | 2.14.1 (CPU build, `2.14.1+cpu` per `training_config.json`) | training, evaluation, GGUF export |
| `transformers` | 5.23.0 | tokenizer loading only (the model class is custom) |
| `tokenizers` | 0.23.3 | BPE tokenizer behind `transformers` |
| `numpy` | 2.5.3 | token streams, metrics |
| `pandas` | 3.1.1 | dataset cleaning, retrieval, metrics |
| `safetensors` | 0.8.0 | installed; checkpoints are stored as `.bin`/`.pt`, not safetensors |
| `datasets` | 5.2.0 | installed for the Hugging Face helpers in `src/upload_to_huggingface.py` |
| `requests` | 2.34.2 | proxy/HTTP checks |
| `pillow` | 12.3.0 | installed with matplotlib; not used directly by the pipeline |
| `gguf` | 0.19.0 | writing GGUF metadata and tensors |
| `tqdm` | 4.70.1 | progress bars |
| `ollama` (Python client) | 0.6.3 | `ollama_client.py` |
| `matplotlib` | 3.11.2 | **figure generation only** (see §3) |
| `torchvision` | 0.29.1 | pulled in by the torch install; unused |

### 2.3 External binaries

| Item | Value |
|---|---|
| Ollama server | 0.33.2 (`ollama --version`) |
| llama.cpp build used by that Ollama | `b12503` (shipped inside `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\`) |
| Git | present; the repository history used for the development timeline is in `.git/` |

## 3. Disclosed deviation: matplotlib was installed for this package

`matplotlib` 3.11.2 was **not** part of the original training/evaluation environment. It was installed
into `venv/` during the preparation of this evidence package, solely so that
`figures/scripts/make_figures.py` and `make_diagrams.py` could render PNG and PDF figures. No model,
dataset, checkpoint or recorded metric was produced with it. Any future reproduction of the pipeline
itself does not need matplotlib; only regenerating the figures does.

## 4. Determinism and reproducibility of the environment

| Source of variation | Status |
|---|---|
| Random seed | `42` for v1 (`training_config.json` → `environment.seed`), `1234` for v2 (`src/continue_training.py`) |
| v1 batch sampling | random **with replacement** (`torch.randint`), so v1's exact loss trace is not bit-reproducible across runs |
| v2 batch sampling | one shuffled permutation per epoch, seeded; more stable than v1 but still not guaranteed bit-identical across torch versions |
| Thread count | affects floating-point reduction order; the recorded numbers used 6 threads |
| Greedy decoding | temperature 0, top-k 1, so generation is deterministic for a fixed build and GGUF |
| Torch/transformers version | pinned only by what is installed; a different version can change results at the last digit |

The package therefore treats the **stored JSON artefacts** as the record of truth and does not promise
that a re-run reproduces the last digit. It does promise that a re-run reproduces the *conclusions*,
because the effects reported are large (loss 1.45 → 0.56; 1–5 distinct outputs per 936 prompts).

## 5. Minimal environment to re-run the pipeline

```
Windows 11 (or any OS with Python 3.10+ and the same wheels)
Python 3.12–3.14, 64-bit
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers tokenizers numpy pandas gguf tqdm ollama requests
# figures only:
pip install matplotlib
# external:
Ollama 0.33.x (bundles llama.cpp b12503)
```

No CUDA toolkit, no compiler, no `llama.cpp` source build is required: GGUF files were written with the
`gguf` Python package and consumed by the llama.cpp that Ollama ships.
