# Application Layer

The Python programme a user actually talks to. **Every number it returns comes from Python, not from
the model.** The model contributes natural-language continuation only.

```
user input
   │
   ├─ domain_guard.is_greeting        → WELCOME_MESSAGE (no model call)
   ├─ domain_guard.is_out_of_domain   → refusal text    (no model call)
   ├─ detect_intent == "retrieval"    → data_retrieval.find_record → formatted values
   ├─ detect_intent == "ah_calculation" → humidity_calculator.calculate_absolute_humidity
   └─ otherwise                       → handle_unknown / ollama_client.generate_response
```

| File | Role |
|---|---|
| `app.py` | Interactive shell: welcome text, routing, date/time and T/RH parsing, answer formatting |
| `domain_guard.py` | Regex/keyword routing to greeting, retrieval, AH calculation, out-of-domain |
| `data_retrieval.py` | Exact `(Date, Time)` lookup in `data/processed/{train,validation,test}.csv`; returns raw dataset values |
| `humidity_calculator.py` | Tetens saturation vapour pressure plus a dataset-derived scale factor |
| `ollama_client.py` | Thin wrapper over `ollama.generate()`; `MODEL_NAME = "air-quality-slm-improved:latest"`, `num_ctx 128`, `num_predict 48`, greedy |
| `ollama_proxy.py` | Two-port proxy: 11434 (what Ollama Desktop and clients call) → 11435 (real server) |
| `proxy_watchdog.py` | Supervises proxy + server, restarts on failure, logs to `proxy.log` / `watchdog.log` |
| `INTEGRATION_OPERATIONS.md` | Start/stop/check procedure, `.bat` wrappers, recovery behaviour |

## Verified quirks to be aware of

* `data_retrieval.get_pollutant_concentration` and `app.handle_retrieval` label **CO and NOx as `ug/m3`**
  although the dataset stores CO in mg/m³ and NOx in ppb. The value is correct; the unit string is not.
  Recorded in `../MISSING_EVIDENCE.md` §5.
* `humidity_calculator.magnus_saturation_vp` documents `T + 243.5` but executes `T + 237.3`.
  `get_scale_factor()` measures a one-row ratio at runtime (0.12874776 on this machine) rather than
  using the documented 0.976 fit. `T = 16.7 °C`, `RH = 54.7 %` therefore returns **1.0284 g/m³**.
  Recorded in `../MISSING_EVIDENCE.md` §4.
* `ollama_client.MODEL_NAME` points at the **v2** tag, not v3, even though `Modelfile.v3` expects a
  system prompt. Switching the application to v3 is a one-line change that was not applied.
* `app.py` inserts `parent.parent` into `sys.path`; run it from the project root so that
  `domain_guard`, `data_retrieval` and `humidity_calculator` resolve.

## Run

```bash
./venv/Scripts/python.exe app.py
```

If Ollama is unreachable the shell still starts and answers from Python lookups and calculations,
printing `! Ollama not available - will use Python lookups/calculations only`.
