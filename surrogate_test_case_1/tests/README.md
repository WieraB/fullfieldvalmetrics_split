# FMU verification tests

Tests that the Wine-piped `FullSurrogate.fmu` is operating correctly,
judged against the vendor's own reference FEA in
`../train-valid-data/` (209 training + 24 validation points, 18 inputs,
40 outputs).

## Run

```bash
cd surrogate_test_case_1
python3 -m pytest tests/ -q          # full suite (~40 s, needs Wine)
python3 tests/accuracy_report.py     # per-output accuracy tables (CSV optional)
```

Prerequisites are the ones from `../README_surrogate.md` (`wine64` on PATH
or `FMI3_WINE`, the Wine prefix, `C:\pyemb\python.exe`).  Data/VML tests do
not touch Wine and run even without it — except the fixtures that evaluate
the FMU, which fail fast with the setup message if Wine is unusable.

## What is asserted

| module | question answered |
|---|---|
| `test_data_integrity.py` | are the loaded CSVs complete, finite, aligned, duplicate-free, in-domain, and in the expected design box? |
| `test_model_description.py` | does the FMU XML declare FMI 3.0 with exactly the 18/40 variables named in the CSVs (same order), Windows-only binaries, the encrypted payload, and the vendor quirks we depend on (all-zero start values, **no `fmi3SetupExperiment` export**)? |
| `test_wine_pipeline.py` | is the *pipe* sound: deterministic within and across processes, batch == single, order-independent, reused instance == fresh instance, unknown inputs rejected, all outputs finite? |
| `test_interpolation.py` | does the FMU reproduce `TrainingResults` at the 209 training points (Kriging must interpolate them **exactly**), and bound its error on the 24 unseen validation points? |
| `test_physics_invariants.py` | do structural invariants hold (`Sample Maximum ≥ every TC`, `Sample Average ≤ Sample Maximum`) and does every strong input→output influence point in the same direction as the FEA data (Spearman sign)? |

## Tolerance rationale (calibrated on observed runs)

| check | limit | observed |
|---|---|---|
| training max abs err | 1e-5 | **5.0e-8** (= CSV round-trip precision → exact interpolation) |
| training NRMSE (by output range) | 1e-3 | 2.2e-4 |
| validation R² (worst output) | 0.95 | 0.9645 |
| validation NRMSE | 0.06 | 0.0405 |
| validation max abs err | 10 | 5.73 °C on `Sample Maximum Temperature` |
| validation mean abs err | 0.5 | 0.059 |
| sensitivity sign agreement (strong pairs) | 95 % | 100 % (92/92) |

Interpretation of a failure:

* **training tests fail** → the pipe or data is wrong (inputs mis-mapped,
  wrong file, wrong FMU).  The surrogate interpolates its training points
  by construction, so there is no legitimate reason for error > 1e-5.
* **validation tests fail while training passes** → the FMU and data are
  wired correctly, but surrogate accuracy degraded (or files were swapped).
* **`test_wine_pipeline` fails** → plumbing bug (determinism, instance
  reuse, batching), independent of any reference data.
* **`test_model_description.test_dll_omits_setup_experiment` fails** →
  SmartUQ re-exported the FMU *with* `fmi3SetupExperiment`; upgrade
  `fmi3_wine.py` to a full SetupExperiment/DoStep flow (the native fmpy
  path in `surrogate_fmu.py` then becomes preferable).

## Notes

* All heavy sweeps are session-scoped fixtures — the 233 FMU evaluations
  cost ~2 Wine sessions, the whole suite ~15.
* Wine-specific tests are marked `wine` and **skip** (not fail) if the FMU
  ever ships `binaries/x86_64-linux/` and the wrapper auto-switches to
  native fmpy; the data/description tests still run.
* Validation residuals are *expected* — that is the surrogate's modelling
  error vs FEA, which is exactly what an accuracy study should quantify
  (use `accuracy_report.py` for the per-output numbers).
