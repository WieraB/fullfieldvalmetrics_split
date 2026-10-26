# fullfieldvalmetrics_split

Repository for the **nuisance/transferrable epistemic
uncertainty split** for the validation process using MAVM,
applied to the pulse-25X thermocouple and coil-voltage point sensors.

It was isolated from `fullfieldvalmetrics`. The experimental data and the code that generates the paper figures [1] are included unchanged.

[1] - "Towards virtual qualification in nuclear fusion: demonstrating probabilistic model validation on a high heat flux component"

Scrambled Sobol sampling is used to sample all uncertainty dimensions.

## The idea

In the paper [1]'s pipeline every epistemic input is pooled into a single epistemic
axis and every aleatory input into another, so the surrogate is sampled as `E x A`. When `E` contains HIVE-specific uncertainty (the coil current is only known to +/-94.2 A) the pooled epistemic spread is so wide that the simulation p-box swallows the experimental box, and `d+`/`d-` approach zero. Here, the partition splits the epistemic inputs by *context of
use*:

* **E_t — transferrable**: epistemic uncertainty that still applies at the use context (material property scalings, heat-transfer coefficient).
* **E_n — nuisance**: HIVE-specific uncertainties not applied at the use context (coil current set-point, coolant temperature, coil frequency).
* **A — aleatory**: irreducible uncertainties (`alpha_tb`, `beta_tb`, `dx`, `dy`, `dz`).

The surrogate is sampled on a tensor of shape `E_n x E_t x A`. The MAVM is computed *within* each nuisance block (so `E_n` never enters a CDF and is not double counted), and the metric of interest is `max over E_n of d+ / d-`.

## Layout

```
fullfieldvalmetrics_split/
├── README.md                             this file
├── .gitignore
├── plotter_log.log                       local log (untracked)
├── surrogate_test_case_1/                Wine-wrapped SmartUQ surrogate
│   ├── FullSurrogate.fmu                 the FMU (Windows-only binary; run via Wine)
│   ├── surrogate_fmu.py                  Linux-side wrapper: Surrogate, evaluate_batch
│   ├── fmi3_wine.py                      stdlib-only FMI 3.0 driver (runs under Windows Python)
│   ├── train-valid-data/                 vendor reference FEA: 209 train + 24 validation points
│   ├── tests/                            pytest suite (38 tests) pinning the design box & accuracy
│   ├── read_surrogate.ipynb              FMU inspection / evaluation notebook
│   └── README_surrogate.md               Wine setup, quirks, caveats   <-- read this first
├── scripts/
│   ├── mavm_nuisance_split.py            *** the partitioned-metric pipeline ***
│   ├── plot_mavm_nuisance.py             *** figures for a results directory ***
│   ├── validate_against_paper.py         *** checks the metric vs the paper's published tables ***
│   ├── paper_scale_check.py              *** surrogate at the paper's sampling density + sizing study ***
│   ├── valmetrics.py                     MAVM implementation (vm.mavm; Whiting area-stepping)
│   ├── topyvale_tcs_pulse25X.py          reference paper-MAVM pipeline (pooled E x A)
│   ├── main_thermocouples_pulses25X.py   code that generates the paper point-sensor figures
│   ├── mavm_nuisance_split/              partitioned-metric results (one dir per design)
│   └── paper_scale_check/                paper-scale check results (one dir per design)
├── STC_Exp_TCs_25X/                      experimental TC + CV pulse data (paper input)
├── STC_ProbSim_FieldsFull_25X_v4/        FEA samples used to generate the paper figures
├── STC_ProbSim_MetaData_25X/             summary CSVs incl. experimental epistemic errors
└── images_pointsensors_pulse25X_v4/      paper output: figures + summary CSVs (regenerable)
```

All five data directories are identical to
`fullfieldvalmetrics`. `scripts/main_thermocouples_pulses25X.py` reads
`STC_Exp_TCs_25X`, `STC_ProbSim_FieldsFull_25X_v4` and
`STC_ProbSim_MetaData_25X`, and rewrites `images_pointsensors_pulse25X_v4`.

## Environment
`pyvale` and `valmetrics` dependencies are needed. So, it needs to be run using an environment containing Pyvale. Conda environment is used here.

The surrogate needs Wine (Windows-only FMU binary). One-time setup is documented in `surrogate_test_case_1/README_surrogate.md`. The prefix is outside the repo at `~/.local/share/fullfieldvalmetrics/wineprefix` (override with `FMI3_WINE`, `FMI3_WINEPREFIX`, `FMI3_WINPY`).

## Running

### 1. Verify the Wine surrogate pipe

```bash
cd surrogate_test_case_1
python3 -m pytest tests/ -q          # 38 tests, ~50 s
```

### 2. Reproduce the paper's pooled-box MAVM

```bash
cd <repo root>
~/anaconda3/envs/pyvale/bin/python scripts/topyvale_tcs_pulse25X.py
```

Reads `STC_ProbSim_FieldsReduced_25X` if present, otherwise
`STC_ProbSim_FieldsFull_25X_v4`. Writes `topyvale_mavm_pulse25X/`.

### 3. Reproduce the paper's point-sensor figures

```bash
cd <repo root>
~/anaconda3/envs/pyvale/bin/python scripts/main_thermocouples_pulses25X.py
```

Regenerates everything in `images_pointsensors_pulse25X_v4/`
(51 files: per-sensor CDFs, d-extreme weighted CDFs, MAVM maps, plus
`pointsensors_mavm.csv`, `pointsensors_dextremes.csv`,
`pointsensors_total_uncertainty.csv`, `sim_nom_mean_temperature.csv`,
`axis_limits.json`).

### 4. Check the metric against the paper's numbers

```bash
cd <repo root>
~/anaconda3/envs/pyvale/bin/python scripts/validate_against_paper.py
```

Rebuilds the paper's [1] bound rows from its own FEA samples, recomputes the eight-row MAVM table and compares it with `pointsensors_mavm.csv`
(max |difference| 3.6e-15), then checks the model-form selection against `pointsensors_total_uncertainty_temperature.csv` (max |difference| 5.6e-17).
Both must stay below 1e-9.

### 5. Run the nuisance-split MAVM

```bash
cd <repo root>
~/anaconda3/envs/pyvale/bin/python scripts/mavm_nuisance_split.py \
    [--n-en 8] [--n-et 16] [--n-a 100] [--seed 20261021] \
    [--convention model_form|reference] [--force]
```

`--n-en` is the number of *non-centre* nuisance points; the design is the
best-estimate centre plus `n_en` scrambled-Sobol points, so the tensor has
`(n_en+1) x n_et x n_a` surrogate evaluations. Each `E_n` block is evaluated
as one `evaluate_batch` call and cached in `block_XX.csv`, so an interrupted
run resumes where it stopped (`--force` re-evaluates).

`--convention` selects the bound-selection rule of the reported metric:
`model_form` (default) is the rule used in the paper [1], `reference` is the max over all four bound combinations. The two are written to different files, so neither overwrites the other.

Results land in `scripts/mavm_nuisance_split/nen*_net*_na*/`:

| file | contents |
|---|---|
| `block_XX.csv` | one E_n block: inputs + all 40 outputs, `n_et x n_a` rows |
| `design_meta.json` | seed, sizes, partition, design-space containment check |
| `mavm_profile.csv` | `d+`/`d-` per sensor x E_n block (the profile), in the reported convention, with the winning bound combination |
| `mavm_summary.csv` | `max` over E_n and the merged-box baseline per sensor, same convention, plus a `convention` column |
| `mavm_*_reference.csv` | the same two tables in the reference convention (only if `--convention reference` was run) |

### 6. Plot a results directory

```bash
~/anaconda3/envs/pyvale/bin/python scripts/plot_mavm_nuisance.py \
    [--res-dir scripts/mavm_nuisance_split/nen9_net16_na100] \
    [--convention model_form|reference] [--sensors TC2 TC3]
```

Writes six figures into `<res-dir>/plots/`:

1. `01_profile_per_sensor.png` — `d+`/`d-` across the E_n blocks, one panel per sensor
2. `02_dpm_vs_current.png` — the same against coil current (the dominant nuisance axis)
3. `03_ecdfs.png` — p-box picture: per-block sub-boxes, merged box, experiment box (chosen sensors)
4. `04_distributions.png` — per-block aleatory spreads vs the experiment band (chosen sensors)
5. `05_split_vs_paper.png` — split metric vs the MAVM values from the paper [1] (`images_pointsensors_pulse25X_v4/pointsensors_total_uncertainty_temperature.csv`, rows `model_form_plus`/`model_form_minus`), with the same-sample merged value as the middle bar

All figures use model-form convention by default.

### 7. The paper-scale partitioned case

```bash
cd <repo root>
~/anaconda3/envs/pyvale/bin/python scripts/mavm_nuisance_split.py \
    --n-et 250 --n-a 400
~/anaconda3/envs/pyvale/bin/python scripts/plot_mavm_nuisance.py \
    --res-dir scripts/mavm_nuisance_split/nen9_net250_na400
```

The same partitioned pipeline, but at the paper's sampling size
inside every E_n block [1]: 250 transferrable-epistemic points x 400 aleatory
draws = 100 000 surrogate evaluations per block, 9 blocks = 900 000
evaluations at 1.67 ms/point, ~25 min. The block samples (~1.1 GB) are
gitignored — the metric tables and the figures of the run are tracked.

Result (model-form convention, `scripts/mavm_nuisance_split/nen9_net250_na400/`),
next to the paper's [1] published values:

| sensor | split d⁺ | merged d⁺ | paper d⁺ | split d⁻ | merged d⁻ | paper d⁻ |
|---|---|---|---|---|---|---|
| TC2 | 3.054 | 0 | 0 | 7.052 | 0 | 0.00094 |
| TC3 | 41.05 | 9.752 | 8.613 | 0 | 0 | 0 |
| TC5 | 3.428 | 0 | 0 | 3.970 | 0.0008 | 0.0016 |
| TC6 | 9.237 | 0 | 0 | 10.30 | 0 | 0.000095 |
| TC8 | 8.988 | 0.887 | 0.554 | 0.231 | 0 | 0 |
| TC9 | 10.94 | 0.022 | 0.0258 | 4.196 | 0 | 0 |
| TC10 | 0 | 0 | 0 | 17.79 | 8.319 | 7.761 |
| CV | 9.568 | 2.112 | 0.561 | 1.171 | 0 | 0 |

Two things to read off it:

* the **merged** column is the paper-style single epistemic box on the very
  same samples, and it lands on the published numbers (TC2/TC5/TC6/TC10
  exactly 0, TC3 +13 %, TC9 −15 %, TC10 +7 %; CV is the exception, +277 %);
* the **split** column is what the partition buys: the pooled box swallows
  the experiment for TC2/TC5/TC6/TC10 and the metric collapses to 0, while
  the per-block boxes keep a non-zero, resolution-independent measure.

Compared with the pilot (`--n-et 16 --n-a 100`) the split values drop by
10–30 % (TC2 4.66 → 3.05, TC3 44.3 → 41.0, TC6 11.6 → 9.24): with 250 E_t
rows the bound selection has many more candidates, so the per-block p-box is
better determined and the earlier small-sample optimism disappears.

## Two MAVM conventions

| convention | function | definition |
|---|---|---|
| **paper model-form** (default, reported) | `model_form_bounds` | `d+` against the **sim-max** bound row only, `d-` against the **sim-min** row only; of the two experimental bounds the paper keeps the one making the total bound most extreme |
| reference (topyvale) | `mavm_over_bounds` | max over `sim{min,max} x exp{min,max}`; also picks up the gap between the experiment and the **inner** simulation bound |

`model_form_bounds` is what `main_thermocouples_pulses25X.py` writes as `model_form_plus`/`model_form_minus` in `images_pointsensors_pulse25X_v4/pointsensors_total_uncertainty_temperature.csv`, i.e. the numbers in the paper.

## The epistemic uncertainty split

Taken from Table 1 of the paper [1]:

* `E_n = {I_coil_rms, F_coil, T_cool_sample}` — rig loading / set-points
  (F_coil in the FMU is in Hz: 123 500 +/- 500; the paper table [1] is in kHz)
* `E_t = {9 material scalings, HTC_sample}` — includes the Dittus-Boelter
  correlation error
* `A   = {alpha_tb, beta_tb, dx, dy, dz}` — aleatory-dominant normals,
  +/-3 sigma, clipped to the surrogate's training box

## To remember

* The FMU exposes `Cu_sigma_scaling` (18 inputs) while the paper's Table 1
  marks "Electrical conductivity, copper" as **No** (17 epistemic "Yes"
  inputs). TODO
* `E_n` blocks are ordered by the Sobol draw, not by current.
* The paper-scale run uses **250** E_t points, which is not a power of two, so scipy warns that the scrambled-Sobol sequence is not perfectly balance ("balance properties of Sobol' points require n to be a power of 2"). The same 250-point sequence reproduces the paper's published values to 1.6 % in `scripts/paper_scale_check.py`, so the imbalance is not important for this.
