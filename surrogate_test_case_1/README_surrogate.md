# Running Windows-exported `FullSurrogate.fmu` on Linux

`FullSurrogate.fmu` is a **SmartUQ** surrogate exported as an **FMI 3.0
Co-Simulation** FMU with **only a Windows binary**. Using the directions provided in this folder, it is possible to make it run on Linux. The notebook can be amended to use a native Linux export if it exists.

| File | Purpose |
|---|---|
| `read_surrogate.ipynb` | Main notebook: inspect the FMU, then evaluate it |
| `surrogate_fmu.py` | Linux-side wrapper (`Surrogate` class, pandas API) |
| `fmi3_wine.py` | Stdlib-only FMI 3.0 driver; runs *inside* the Windows Python under Wine |
| `train-valid-data/` | Vendor reference data: 209 training + 24 validation points (18 inputs, 40 outputs) |
| `tests/` | Pytest suite verifying the Wine pipe against that data; `accuracy_report.py` for per-output metrics |

## Why not fmpy

1. `fmpy.simulate_fmu()` on Linux raises *"The FMU cannot be simulated on the
   current platform (linux64)"* — there is no `binaries/x86_64-linux/`.
2. fmpy's Linux→Wine remoting bridge ships **FMI 2.0** client/server stubs
   (44 `fmi2*` symbols, zero `fmi3*`), so it structurally cannot carry an
   FMI 3.0 FMU.
3. The surrogate payload `resources/Model.smartuq` is **AES-encrypted** (the
   HDF5 root carries an `iv` attribute; every numeric dataset uses a custom
   HDF5 filter registered by SmartUQ's `csmartuq.dll`), so the model cannot be
   reimplemented natively. The structure is a Kriging model: 18 inputs,
   40 outputs, 209 training points.

Importing numpy/fmpy inside the Windows Python also crashes this Wine build
(`unimplemented function ucrtbase.dll.crealf`) — one more reason the driver
uses `ctypes` from the standard library only.

## One-time Wine setup

Prerequisites: 64-bit Wine (`wine64`), and the ability to download two files
from python.org.

```bash
# 1. A wine64 shim on PATH (Ubuntu's Wine does not ship one)
mkdir -p ~/.local/bin
printf '#!/bin/sh\nexec /usr/lib/wine/wine64 "$@"\n' > ~/.local/bin/wine64
chmod +x ~/.local/bin/wine64
export PATH="$HOME/.local/bin:$PATH"

# 2. A stable 64-bit prefix (NOT under /tmp — it must survive reboots)
export WINEPREFIX="$HOME/.local/share/fullfieldvalmetrics/wineprefix"
export WINEARCH=win64
wine64 wineboot --init

# 3. Embeddable Windows Python 3.11 (no installer, no GUI needed)
curl -LO https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip
unzip -q python-3.11.9-embed-amd64.zip -d "$WINEPREFIX/drive_c/pyemb"

# 4. Enable site-packages in the embeddable distribution
cd "$WINEPREFIX/drive_c/pyemb"
sed -i 's/^#import site/import site/' python311._pth
echo 'Lib\\site-packages' >> python311._pth

# 5. Bootstrap pip (numpy/fmpy are NOT needed — the driver is stdlib-only)
curl -LO https://bootstrap.pypa.io/get-pip.py
wine64 'C:\pyemb\python.exe' get-pip.py

# 6. Sanity check
echo '{"inputs":[{"I_coil_rms (A)":150,"F_coil (Hz)":50000,
"T_cool_sample (degC)":25,"HTC_sample":100,
"St_alpha_scaling":1,"St_sigma_scaling":1,"St_k_scaling":1,"St_E_scaling":1,
"St_nu_scaling":1,"St_cp_scaling":1,"Cu_sigma_scaling":1,"Cu_k_scaling":1,
"Cu_rho_scaling":1}]}' \
 | wine64 'C:\pyemb\python.exe' Z:\\path\\to\\fmi3_wine.py Z:\\path\\to\\FullSurrogate.fmu --batch
```

The prefix is ~1.3 GB (Wine's builtin DLLs). Overrides if you install
elsewhere: `FMI3_WINE`, `FMI3_WINEPREFIX`, `FMI3_WINPY`.

**Note on stdin:** Wine cannot handle file redirection (`< file.json`, raises
`WinError 6`) but works fine with pipes — `surrogate_fmu.py` uses
`subprocess` with `input=`, which is a pipe.

## Usage

```python
from surrogate_fmu import Surrogate

s = Surrogate("FullSurrogate.fmu")
s.input_names, s.output_names          # 18 inputs, 40 outputs (no Wine needed)

# single point — unspecified inputs default to NOMINAL_INPUTS
s.evaluate({"I_coil_rms (A)": 150.0, "F_coil (Hz)": 50_000.0})

# batch — ONE Wine process / ONE FMU instance for all points
df = s.evaluate_batch([{"I_coil_rms (A)": I} for I in range(50, 300, 25)])
```

Measured throughput: **~250 ms per point** including Wine + Python start-up
(a 1000-point sweep is a few minutes).

Every input has FMU `start = 0`, which is a degenerate operating point (zero
conductivity scaling, zero coolant temperature). `NOMINAL_INPUTS` in
`surrogate_fmu.py` supplies a default — **note that its values (150 A,
50 kHz, 25 °C, HTC 100) lie *outside* the training design box below, so
results at the defaults are extrapolations. Edit it to match the operating
point of your study.**

## Verification tests

```bash
python3 -m pytest tests/ -q          # 38 tests, ~40 s (Wine required)
python3 tests/accuracy_report.py     # per-output MAE/RMSE/R2 tables
```

`tests/` checks the whole stack against the vendor's reference FEA in
`train-valid-data/`: data integrity, FMU XML ↔ CSV agreement, pipe
behaviour (determinism, batching, instance reuse), **exact interpolation
of all 209 training points** (observed error 5e-8 = CSV precision), bounded
error on the 24 validation points (R² ≥ 0.964), physical invariants and
sensitivity-direction agreement. See `tests/README.md` for the tolerance
rationale and how to interpret a failure.

## Vendor quirks worth knowing

These are deviations from the FMI 3.0 spec that the driver works around:

* `fmi3SetupExperiment` is **not exported** (the string does not even occur in
  the binary). The FMU's stop time is therefore never set, so `fmi3DoStep`
  fails with *"stop time is 6.08e-310"* (uninitialised memory). Since the model
  is a static algebraic surrogate, outputs are already computed at the end of
  initialisation mode, so `fmi3DoStep` is simply not used.
* `resourcePath` is concatenated with `Model.smartuq` and passed straight to
  `H5Fopen()`, which requires a plain filesystem path. Passing the spec's
  `file:///...` URI form makes HDF5 fail with errno 2, so the driver passes a
  raw path to the `resources/` directory with a trailing separator.
* Inputs may be set again *after* initialisation mode; verified bit-identical
  to setting them inside initialisation mode on a fresh instance, which is what
  makes the fast batch path correct.

## To remember

* **Stay inside the surrogate's design space.** The bounds *inside* the FMU
  are encrypted, but `train-valid-data/TrainingDOE.csv` gives them (and the
  test suite pins them in `tests/metrics.py`):

  | input | training range |
  |---|---|
  | `I_coil_rms` | 1409 – 1691 A |
  | `F_coil` | 123.0 – 124.1 kHz |
  | `T_cool_sample` | 140 – 155 °C |
  | `HTC_sample` | 37 140 – 50 090 |
  | `alpha/beta_tb`, `dx/dy/dz` | ±~2.5 (mm / deg) |
  | all 9 material scalings | 0.90 – 1.10 |

  A Kriging model extrapolates smoothly-but-wrongly outside that box (the
  "temperature falls as current rises" artefact seen at 150 A / 50 kHz is
  exactly that).
* **The clean fix is a re-export.** Ask SmartUQ to include `binaries/x86_64-linux/fmi3.so` in the FMU export. Nothing in this folder has to change: `surrogate_fmu.py` detects a native Linux binary and uses fmpy instead of Wine.
