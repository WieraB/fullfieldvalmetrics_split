"""Evaluate the SmartUQ ``FullSurrogate.fmu`` on Linux via Wine.

``FullSurrogate.fmu`` ships **only** a Windows binary
(``binaries/x86_64-windows/fmi3.dll``) and its surrogate payload
(``resources/Model.smartuq``) is AES-encrypted, so there is no way to
evaluate it natively on Linux without a re-export from SmartUQ.

This module runs the FMU under Wine by shelling out to ``fmi3_wine.py``
(a stdlib-only FMI 3.0 driver) inside a Windows Python.  It is deliberately
independent of fmpy/numpy on the Windows side, because importing those
crashes this Wine build.

Requirements (set up once -- see ``README_surrogate.md``):

* ``wine64`` on ``PATH`` (a shim in ``~/.local/bin`` is fine)
* a 64-bit Wine prefix with the *embeddable* Windows Python 3.11 installed
  at ``C:\\pyemb``

Environment overrides::

    FMI3_WINE       path to wine64          (default: wine64 on PATH)
    FMI3_WINEPREFIX the Wine prefix         (default: ~/.local/share/fullfieldvalmetrics/wineprefix)
    FMI3_WINPY      the Windows python.exe  (default: C:\\pyemb\\python.exe)

Typical use::

    from surrogate_fmu import Surrogate

    s = Surrogate("FullSurrogate.fmu")
    print(s.input_names)

    one = s.evaluate({"I_coil_rms (A)": 150.0, "F_coil (Hz)": 50_000.0})
    df  = s.evaluate_batch([{"I_coil_rms (A)": i} for i in (100, 150, 200)])

If a ``binaries/x86_64-linux`` folder ever appears in the FMU, this module
uses native fmpy instead of Wine automatically.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

_HERE = Path(__file__).resolve().parent
_DRIVER = _HERE / "fmi3_wine.py"


def read_model_description(fmu_path):
    """Read the model description straight from the .fmu (pure Python,
    no Wine, no fmpy -- the XML is not encrypted, only the surrogate data).

    Returns a dict with ``model_name``, ``fmi_version``, ``generation_tool``,
    ``default_experiment``, ``inputs`` and ``outputs``.
    """
    with zipfile.ZipFile(fmu_path) as zf:
        root = ET.fromstring(zf.read("modelDescription.xml"))

    de = root.find("DefaultExperiment")
    default_experiment = {}
    if de is not None:
        default_experiment = {
            k: float(de.get(k))
            for k in ("startTime", "stopTime", "stepSize")
            if de.get(k) is not None
        }

    def collect(causality):
        out = []
        mv = root.find("ModelVariables")
        if mv is None:
            return out
        for v in mv:
            if v.tag != "Float64" or v.get("causality") != causality:
                continue
            out.append({
                "name": v.get("name"),
                "value_reference": int(v.get("valueReference")),
                "start": float(v.get("start") or 0.0),
            })
        return out

    return {
        "model_name": root.get("modelName"),
        "fmi_version": root.get("fmiVersion"),
        "generation_tool": root.get("generationTool"),
        "default_experiment": default_experiment,
        "inputs": collect("input"),
        "outputs": collect("output"),
    }

# A sane "nominal operating point" for the induction-heating surrogate.
# Every input in the FMU has start=0, which is a degenerate operating point
# (zero conductivity scaling, zero coolant temperature, ...), so we default
# scaling factors to 1.0 and set coil/loading/coolant to the values used by
# the paper/design in this repo (see scripts/mavm_nuisance_split.py).
NOMINAL_INPUTS = {
    "I_coil_rms (A)": 1550.0,
    "F_coil (Hz)": 123500.0,
    "alpha_tb (deg)": 0.70,
    "beta_tb (deg)": -0.32,
    "dx (mm)": -1.42,
    "dy (mm)": 0.50,
    "dz (mm)": -0.23,
    "T_cool_sample (degC)": 149.0,
    "HTC_sample": 43600.0,
    "St_alpha_scaling": 1.0,
    "St_sigma_scaling": 1.0,
    "St_k_scaling": 1.0,
    "St_E_scaling": 1.0,
    "St_nu_scaling": 1.0,
    "St_cp_scaling": 1.0,
    "Cu_sigma_scaling": 1.0,
    "Cu_k_scaling": 1.0,
    "Cu_rho_scaling": 1.0,
}


def _wine_env():
    env = dict(os.environ)
    env["WINEDEBUG"] = "-all"  # silence Wine's console noise
    env.setdefault("WINEARCH", "win64")
    env.setdefault(
        "WINEPREFIX",
        str(Path.home() / ".local/share/fullfieldvalmetrics/wineprefix"),
    )
    return env


def _to_wine_path(path: Path) -> str:
    """Translate a Linux path into the drive letter Wine maps it to.

    Wine maps ``/`` to ``Z:`` by default.  Anything outside that mapping
    raises, rather than passing a path the FMU cannot open.
    """
    path = path.resolve()
    posix = str(path)
    if not posix.startswith("/"):
        return str(path)  # already Windows-style
    return "Z:" + posix.replace("/", "\\")


class Surrogate:
    """Lazy, batch-friendly handle to the SmartUQ surrogate FMU."""

    def __init__(self, fmu_path, wine_prefix=None, wine=None, winpy=None,
                 extract_dir=None, timeout=600):
        self.fmu_path = Path(fmu_path).resolve()
        if not self.fmu_path.is_file():
            raise FileNotFoundError(self.fmu_path)

        self.wine = (
            wine
            or os.environ.get("FMI3_WINE")
            or shutil.which("wine64")
            # common location for a user-made shim
            or (str(Path.home() / ".local/bin/wine64")
                if (Path.home() / ".local/bin/wine64").exists()
                else None)
        )
        self.wine_prefix = (
            wine_prefix
            or os.environ.get("FMI3_WINEPREFIX")
            or Path.home() / ".local/share/fullfieldvalmetrics/wineprefix"
        )
        self.winpy = winpy or os.environ.get("FMI3_WINPY") or r"C:\pyemb\python.exe"
        self.timeout = timeout

        # Cache the extracted FMU next to the .fmu so repeat calls are fast.
        self.extract_dir = (
            Path(extract_dir) if extract_dir
            else self.fmu_path.parent / ".fmu_extracted" / self.fmu_path.stem
        )

        self._info = None
        self._backend = None

    # -- backend selection -------------------------------------------------
    @property
    def backend(self) -> str:
        """``"native-fmpy"`` if the FMU ships a Linux binary, else ``"wine"``.

        Detection is cached.  Wine is preferred over a broken fmpy install:
        if ``import fmpy`` fails, we silently fall back to Wine.
        """
        if self._backend is None:
            self._backend = self._detect_backend()
        return self._backend

    def _detect_backend(self):
        if platform.system() != "Linux":
            return "wine"
        with zipfile.ZipFile(self.fmu_path) as zf:
            names = zf.namelist()
        has_linux = any(n.startswith("binaries/x86_64-linux/") for n in names)
        if not has_linux:
            return "wine"
        try:
            import fmpy  # noqa: F401
        except Exception:
            return "wine"
        return "native-fmpy"

    # -- setup checks ------------------------------------------------------
    def check_wine(self) -> str:
        """Raise a helpful error if the Wine toolchain is unusable."""
        if platform.system() != "Linux":
            return "not Linux"
        if not self.wine:
            raise RuntimeError(
                "wine64 not found. Install Wine 64-bit, or point FMI3_WINE at it.\n"
                "  e.g.  export PATH=\"$HOME/.local/bin:$PATH\"  (if you made a shim)"
            )
        if not Path(self.wine_prefix).is_dir():
            raise RuntimeError(
                f"Wine prefix missing: {self.wine_prefix}\n"
                "See README_surrogate.md for the one-time setup."
            )
        return self.wine

    def _extract(self):
        """Unzip the FMU once; reuse the cache afterwards."""
        stamp = self.extract_dir / ".stamp"
        want = f"{self.fmu_path.stat().st_size}:{self.fmu_path.stat().st_mtime_ns}"
        if stamp.exists() and stamp.read_text() == want:
            return self.extract_dir
        if self.extract_dir.exists():
            shutil.rmtree(self.extract_dir)
        self.extract_dir.mkdir(parents=True)
        with zipfile.ZipFile(self.fmu_path) as zf:
            zf.extractall(self.extract_dir)
        stamp.write_text(want)
        return self.extract_dir

    # -- model description (no Wine needed) --------------------------------
    @property
    def info(self):
        """Model description: input/output names and defaults (no Wine)."""
        if self._info is None:
            self._info = read_model_description(self.fmu_path)
        return self._info

    @property
    def input_names(self):
        return [v["name"] for v in self.info["inputs"]]

    @property
    def output_names(self):
        return [v["name"] for v in self.info["outputs"]]

    # -- the actual driver call -------------------------------------------
    def _run_driver(self, extra_args, stdin=None):
        self.check_wine()
        extracted = self._extract()
        cmd = [
            self.wine, self.winpy,
            _to_wine_path(_DRIVER),
            _to_wine_path(self.fmu_path),
            "--unzip-dir", _to_wine_path(extracted),
            *extra_args,
        ]
        proc = subprocess.run(
            cmd,
            input=stdin,
            capture_output=True,
            text=True,
            env=_wine_env(),
            timeout=self.timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "FMU evaluation failed under Wine.\n"
                f"command: {' '.join(cmd)}\n"
                f"stderr:\n{proc.stderr[-4000:]}"
            )
        return proc.stdout

    def evaluate(self, inputs=None, nominal=True):
        """Evaluate at one point.  Unspecified inputs default to
        ``NOMINAL_INPUTS`` (or to the FMU's own start values if
        ``nominal=False``)."""
        if nominal:
            point = dict(NOMINAL_INPUTS)
        else:
            point = {v["name"]: v.get("start", 0.0) for v in self.info["inputs"]}
        if inputs:
            unknown = set(inputs) - set(point)
            if unknown:
                raise KeyError(
                    f"Unknown input(s): {sorted(unknown)}\n"
                    f"Valid inputs: {sorted(point)}"
                )
            point.update({k: float(v) for k, v in inputs.items()})
        return self.evaluate_batch([point]).iloc[0].to_dict()

    def evaluate_batch(self, points):
        """Evaluate many points in ONE Wine process (one FMU instance).

        Parameters
        ----------
        points : iterable of dict
            Each dict maps input names to values.  Unspecified inputs take
            their nominal value.

        Returns
        -------
        pandas.DataFrame
            One row per point, one column per output.  Input columns are
            included too, so the frame is ready for downstream analysis.
        """
        rows = []
        for p in points:
            row = dict(NOMINAL_INPUTS)
            unknown = set(p) - set(row)
            if unknown:
                raise KeyError(
                    f"Unknown input(s): {sorted(unknown)}\n"
                    f"Valid inputs: {sorted(row)}"
                )
            row.update({k: float(v) for k, v in p.items()})
            rows.append(row)

        if self.backend == "native-fmpy":
            out = self._evaluate_native(rows)
        else:
            payload = json.dumps({"inputs": rows})
            out = json.loads(self._run_driver(["--batch"], stdin=payload))

        import pandas as pd
        df_in = pd.DataFrame(rows)
        df_out = pd.DataFrame(out)
        return pd.concat([df_in, df_out], axis=1)

    def _evaluate_native(self, rows):
        """Evaluate via fmpy directly (used once a Linux binary exists).

        Note: this path is exercised only if SmartUQ re-exports the FMU with
        ``binaries/x86_64-linux``.  It calls ``fmi3SetupExperiment``, which the
        current Windows DLL does not export, so it is expected to work only on
        a spec-compliant re-export.
        """
        from fmpy import simulate_fmu
        import numpy as np

        outputs = [v["name"] for v in self.info["outputs"]]
        out = []
        for row in rows:
            result = simulate_fmu(
                str(self.fmu_path),
                start_values=row,
                output=outputs,
            )
            out.append({k: float(np.asarray(result[k])[-1]) for k in outputs})
        return out

    # -- convenience -------------------------------------------------------
    def __repr__(self):
        try:
            n_in, n_out = len(self.input_names), len(self.output_names)
        except Exception:
            n_in = n_out = "?"
        return (f"<Surrogate {self.fmu_path.name}: {n_in} inputs, "
                f"{n_out} outputs, backend={self.backend}>")
