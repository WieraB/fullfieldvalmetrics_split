#!/usr/bin/env python3
"""Run a Windows-only FMI 3.0 Co-Simulation FMU under Wine.

This script runs inside a *Windows* Python (e.g. the embeddable CPython
launched with ``wine64``) and drives the FMU through the raw FMI 3.0 C
interface using ``ctypes`` from the standard library only.

Why not fmpy?
-------------
* On Linux, fmpy refuses to load an FMU that has no ``binaries/x86_64-linux``
  folder.
* fmpy's Linux -> Wine remoting bridge ships FMI **2.0** client/server stubs
  (44 ``fmi2*`` symbols, no ``fmi3*``), so it cannot carry an FMI 3.0 FMU.
* Importing numpy/fmpy crashes this Wine build (missing ``ucrtbase.crealf``).

Vendor quirks discovered while building this (SmartUQ, FMU generated
2026-02-06, FMI 3.0):

* ``fmi3SetupExperiment`` is **not exported**, even though the FMI 3.0 spec
  makes it mandatory.  Consequently the FMU's stop time is never set, and
  ``fmi3DoStep`` fails ("stop time is 6.08e-310" = uninitialised memory).
  Since the model is a static (algebraic) surrogate, its outputs are already
  computed at the end of initialisation, so ``fmi3DoStep`` is not needed.
* ``resourcePath`` is concatenated with ``Model.smartuq`` and handed straight
  to ``H5Fopen()``, which only accepts a plain filesystem path.  The FMI spec
  says it is a URI (``file:///...``); passing the spec form makes HDF5 fail
  with errno 2.  We therefore pass a raw Windows path to the ``resources``
  directory, with a trailing separator.

Usage::

    # list inputs/outputs
    python fmi3_wine.py FullSurrogate.fmu --list

    # one evaluation (outputs as JSON on stdout)
    python fmi3_wine.py FullSurrogate.fmu --input "I_coil_rms (A)=150"

    # batch evaluation: JSON on stdin, one process, one FMU instance
    #   stdin : {"inputs": [{"I_coil_rms (A)": 150}, {...}, ...]}
    #   stdout: [{"<output name>": <value>, ...}, ...]
    python fmi3_wine.py FullSurrogate.fmu --batch
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from ctypes import (
    CFUNCTYPE,
    POINTER,
    c_bool,
    c_char_p,
    c_double,
    c_int,
    c_size_t,
    c_uint32,
    c_void_p,
)

# ---------------------------------------------------------------------------
# FMI 3.0 C types
# ---------------------------------------------------------------------------
fmi3Status = c_int
fmi3Warning = 1
fmi3Error = 3

_STATUS_NAMES = {0: "ok", 1: "warning", 2: "discard", 3: "error", 4: "fatal"}

# callback prototypes (FMI 3.0)
fmi3LogMessageCallback = CFUNCTYPE(None, c_void_p, fmi3Status, c_char_p, c_char_p)
fmi3IntermediateUpdateCallback = CFUNCTYPE(
    None,
    c_void_p,   # instanceEnvironment
    c_double,   # intermediateUpdateTime
    c_bool,     # intermediateValuesNeeded
    c_bool,     # intermediateStepMode
    c_bool,     # intermediateVariableGetRequired
    c_bool,     # intermediateVariableSetRequired
    c_void_p,
    c_void_p,
)


# ---------------------------------------------------------------------------
# Model description
# ---------------------------------------------------------------------------
class Variable:
    def __init__(self, name, value_reference, causality, start):
        self.name = name
        self.value_reference = value_reference
        self.causality = causality
        self.start = start

    @property
    def start_float(self):
        return float(self.start) if self.start is not None else 0.0


def read_model_description(unzip_dir):
    """Parse modelDescription.xml without fmpy."""
    path = os.path.join(unzip_dir, "modelDescription.xml")
    root = ET.parse(path).getroot()

    cs = root.find("CoSimulation")
    if cs is None:
        raise RuntimeError(
            "This FMU is not a Co-Simulation FMU "
            f"(child elements: {[c.tag for c in root]})"
        )

    default_experiment = {}
    de = root.find("DefaultExperiment")
    if de is not None:
        default_experiment = {
            k: float(de.get(k))
            for k in ("startTime", "stopTime", "stepSize")
            if de.get(k) is not None
        }

    variables = []
    mv = root.find("ModelVariables")
    if mv is not None:
        for v in mv:
            if v.tag != "Float64":
                continue
            variables.append(
                Variable(
                    name=v.get("name"),
                    value_reference=int(v.get("valueReference")),
                    causality=v.get("causality"),
                    start=v.get("start"),
                )
            )

    return {
        "model_name": root.get("modelName"),
        "fmi_version": root.get("fmiVersion"),
        "instantiation_token": root.get("instantiationToken"),
        "generation_tool": root.get("generationTool"),
        "model_identifier": cs.get("modelIdentifier"),
        "variables": variables,
        "default_experiment": default_experiment,
    }


# ---------------------------------------------------------------------------
# FMU wrapper
# ---------------------------------------------------------------------------
class FMU3:
    """Minimal FMI 3.0 Co-Simulation client built on ctypes (stdlib only)."""

    def __init__(self, unzip_dir, log=True):
        self.unzip_dir = os.path.abspath(unzip_dir)
        self.log = log
        self.instance = None
        self._keepalive = []  # stop callbacks being garbage collected

        self.md = read_model_description(self.unzip_dir)
        self._load_library()
        self._bind()

        self.log_cb = fmi3LogMessageCallback(self._log_message)
        self.iu_cb = fmi3IntermediateUpdateCallback(self._intermediate_update)
        self._keepalive += [self.log_cb, self.iu_cb]

    # -- library loading ---------------------------------------------------
    def _load_library(self):
        binaries = os.path.join(self.unzip_dir, "binaries")
        ident = self.md["model_identifier"]

        preferred = os.path.join(binaries, "x86_64-windows", ident + ".dll")
        if os.path.isfile(preferred):
            dll_path = preferred
        else:
            found = []
            if os.path.isdir(binaries):
                for plat in sorted(os.listdir(binaries)):
                    cand = os.path.join(binaries, plat, ident + ".dll")
                    if os.path.isfile(cand):
                        found.append(cand)
            if not found:
                raise FileNotFoundError(f"No binary for the FMU in {binaries}")
            dll_path = found[0]

        # The FMU's own DLLs (csmartuq.dll, hdf5.dll, blosc.dll, ...) live in
        # resources/ and are loaded by the FMU at run time, so make sure they
        # resolve.
        resources = os.path.join(self.unzip_dir, "resources")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(os.path.dirname(dll_path))
                os.add_dll_directory(resources)
            except (OSError, FileNotFoundError):
                pass
        os.environ["PATH"] = (
            os.path.dirname(dll_path) + os.pathsep + resources
            + os.pathsep + os.environ.get("PATH", "")
        )

        self.dll_path = dll_path
        self.dll = ctypes.CDLL(dll_path)

    # -- callbacks ---------------------------------------------------------
    def _log_message(self, env, status, category, message):
        if not self.log:
            return
        cat = category.decode("utf-8", "replace") if category else ""
        msg = message.decode("utf-8", "replace") if message else ""
        print(
            f"[FMU {_STATUS_NAMES.get(int(status), status)}] {cat}: {msg}",
            file=sys.stderr,
        )

    @staticmethod
    def _intermediate_update(*args):
        return None

    # -- binding -----------------------------------------------------------
    def _bind(self):
        d = self.dll
        self._missing = set()

        def optional(name, restype, argtypes):
            """Bind an FMI function, tolerating exports the vendor omitted."""
            try:
                fn = getattr(d, name)
            except AttributeError:
                self._missing.add(name)
                return None
            fn.restype = restype
            fn.argtypes = argtypes
            return fn

        optional("fmi3GetVersion", c_char_p, [])

        optional("fmi3InstantiateCoSimulation", c_void_p, [
            c_char_p,           # instanceName
            c_char_p,           # instantiationToken
            c_char_p,           # resourcePath
            POINTER(c_bool),    # visible
            POINTER(c_bool),    # loggingOn
            POINTER(c_bool),    # eventModeUsed
            POINTER(c_bool),    # earlyReturnAllowed
            POINTER(c_uint32),  # requiredIntermediateVariables
            c_size_t,           # nRequiredIntermediateVariables
            c_void_p,           # instanceEnvironment
            c_void_p,           # logMessage
            c_void_p,           # intermediateUpdate
        ])

        optional("fmi3FreeInstance", None, [c_void_p])
        optional("fmi3EnterInitializationMode", fmi3Status, [c_void_p])
        optional("fmi3ExitInitializationMode", fmi3Status, [c_void_p])
        optional("fmi3EnterStepMode", fmi3Status, [c_void_p])
        optional("fmi3Terminate", fmi3Status, [c_void_p])
        optional("fmi3Reset", fmi3Status, [c_void_p])

        # Optional: SmartUQ's static surrogate does not export this at all
        # (verified: the string does not even occur in the binary).
        optional(
            "fmi3SetupExperiment",
            fmi3Status,
            [c_void_p, c_bool, c_double, c_double, c_bool, c_double],
        )

        optional(
            "fmi3DoStep",
            fmi3Status,
            [
                c_void_p,           # instance
                c_double,           # currentCommunicationPoint
                c_double,           # communicationStepSize
                c_bool,             # noSetFMUStatePriorToCurrentPoint
                POINTER(c_bool),    # eventHandlingNeeded
                POINTER(c_bool),    # terminateSimulation
                POINTER(c_bool),    # earlyReturn
                POINTER(c_double),  # lastSuccessfulTime
            ],
        )

        optional(
            "fmi3SetFloat64",
            fmi3Status,
            [c_void_p, POINTER(c_uint32), c_size_t,
             POINTER(c_double), c_size_t],
        )

        optional(
            "fmi3GetFloat64",
            fmi3Status,
            [c_void_p, POINTER(c_uint32), c_size_t,
             POINTER(c_double), c_size_t],
        )

    # -- helpers -----------------------------------------------------------
    def _check(self, status, what):
        if status >= fmi3Error:
            raise RuntimeError(
                f"{what} failed with FMI status "
                f"{_STATUS_NAMES.get(int(status), status)}"
            )
        return status

    @property
    def inputs(self):
        return [v for v in self.md["variables"] if v.causality == "input"]

    @property
    def outputs(self):
        return [v for v in self.md["variables"] if v.causality == "output"]

    # -- lifecycle ---------------------------------------------------------
    def instantiate(self):
        visible = c_bool(False)
        logging_on = c_bool(self.log)
        event_mode = c_bool(False)
        early_return = c_bool(False)

        # NOTE: the FMI spec says resourcePath is a URI, but SmartUQ's DLL
        # concatenates it with "Model.smartuq" and hands it straight to
        # H5Fopen(), which only accepts a plain filesystem path.  Passing a
        # "file:///" URI makes HDF5 fail with errno 2, so pass a raw Windows
        # path with a trailing separator.
        #
        # FMI's resourcePath is the *resources* folder of the unzipped FMU
        # (not the FMU root) -- Model.smartuq lives in resources/.
        resource_path = os.path.join(self.unzip_dir, "resources")
        if not resource_path.endswith(("\\", "/")):
            resource_path += "\\"

        if self.log:
            version = self.dll.fmi3GetVersion()
            print(f"[FMU] fmi3GetVersion -> {version!r}", file=sys.stderr)

        self.instance = self.dll.fmi3InstantiateCoSimulation(
            b"FullSurrogate",
            (self.md["instantiation_token"] or "").encode("utf-8"),
            resource_path.encode("utf-8"),
            ctypes.byref(visible),
            ctypes.byref(logging_on),
            ctypes.byref(event_mode),
            ctypes.byref(early_return),
            None,
            0,
            None,
            self.log_cb,
            self.iu_cb,
        )
        if not self.instance:
            raise RuntimeError("fmi3InstantiateCoSimulation returned NULL")
        return self.instance

    def setup_experiment(self, start_time=0.0, stop_time=None, tolerance=None):
        de = self.md["default_experiment"]
        if stop_time is None:
            stop_time = de.get("stopTime", 1.0)
        self.stop_time = stop_time
        fn = getattr(self.dll, "fmi3SetupExperiment", None)
        if fn is None:
            # Vendor omitted it (see module docstring).  Harmless for a
            # static surrogate whose outputs are ready after initialisation.
            return
        self._check(
            fn(
                self.instance,
                c_bool(tolerance is not None),
                float(tolerance or 0.0),
                float(start_time),
                c_bool(stop_time is not None),
                float(stop_time or 0.0),
            ),
            "fmi3SetupExperiment",
        )

    def enter_initialization_mode(self):
        self._check(
            self.dll.fmi3EnterInitializationMode(self.instance),
            "fmi3EnterInitializationMode",
        )

    def exit_initialization_mode(self):
        self._check(
            self.dll.fmi3ExitInitializationMode(self.instance),
            "fmi3ExitInitializationMode",
        )

    def enter_step_mode(self):
        """FMI 3.0 requires Step Mode before fmi3DoStep for hybrid CS."""
        fn = getattr(self.dll, "fmi3EnterStepMode", None)
        if fn is None:
            return False
        return fn(self.instance) < fmi3Error

    def set_float64(self, value_references, values):
        n = len(value_references)
        vr = (c_uint32 * n)(*value_references)
        val = (c_double * n)(*values)
        self._check(
            self.dll.fmi3SetFloat64(self.instance, vr, n, val, n),
            "fmi3SetFloat64",
        )

    def get_float64(self, value_references):
        n = len(value_references)
        vr = (c_uint32 * n)(*value_references)
        val = (c_double * n)()
        self._check(
            self.dll.fmi3GetFloat64(self.instance, vr, n, val, n),
            "fmi3GetFloat64",
        )
        return list(val)

    def do_step(self, current=0.0, step=1.0):
        event_needed = c_bool(False)
        terminate = c_bool(False)
        early_return = c_bool(False)
        last_time = c_double(current)
        status = self.dll.fmi3DoStep(
            self.instance,
            c_double(current),
            c_double(step),
            c_bool(True),
            ctypes.byref(event_needed),
            ctypes.byref(terminate),
            ctypes.byref(early_return),
            ctypes.byref(last_time),
        )
        self._check(status, "fmi3DoStep")
        return {
            "eventHandlingNeeded": bool(event_needed.value),
            "terminateSimulation": bool(terminate.value),
            "earlyReturn": bool(early_return.value),
            "lastSuccessfulTime": float(last_time.value),
        }

    def close(self):
        if self.instance:
            fn = getattr(self.dll, "fmi3Terminate", None)
            if fn is not None:
                try:
                    fn(self.instance)
                except Exception:
                    pass
            self.dll.fmi3FreeInstance(self.instance)
            self.instance = None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
class SurrogateSession:
    """One FMU instance, evaluated at many points.

    FMI 3.0 forbids re-entering initialization mode, so the first point is
    applied inside initialization mode and later points are applied with
    plain ``fmi3SetFloat64`` calls after initialization.  Verified against
    fresh instances: both routes give bit-identical outputs, and repeated
    evaluations are deterministic.
    """

    def __init__(self, unzip_dir, log=False):
        self.fmu = FMU3(unzip_dir, log=log)
        self.fmu.instantiate()
        self.fmu.setup_experiment(
            start_time=self.fmu.md["default_experiment"].get("startTime", 0.0),
            stop_time=self.fmu.md["default_experiment"].get("stopTime", 1.0),
        )
        self._refs_in = [v.value_reference for v in self.fmu.inputs]
        self._names_in = [v.name for v in self.fmu.inputs]
        self._refs_out = [v.value_reference for v in self.fmu.outputs]
        self._names_out = [v.name for v in self.fmu.outputs]
        self._started = False

    def evaluate(self, inputs=None):
        values = {n: 0.0 for n in self._names_in}
        for v in self.fmu.inputs:
            values[v.name] = v.start_float
        if inputs:
            unknown = set(inputs) - set(values)
            if unknown:
                raise KeyError(
                    f"Unknown input(s): {sorted(unknown)}. "
                    f"Valid inputs: {sorted(values)}"
                )
            values.update({k: float(v) for k, v in inputs.items()})

        vec = [values[n] for n in self._names_in]

        if not self._started:
            self.fmu.enter_initialization_mode()
            self.fmu.set_float64(self._refs_in, vec)
            self.fmu.exit_initialization_mode()
            self._started = True
        else:
            self.fmu.set_float64(self._refs_in, vec)

        out = self.fmu.get_float64(self._refs_out)
        return dict(zip(self._names_out, out))

    def close(self):
        self.fmu.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def evaluate(fmu_path, inputs=None, unzip_dir=None, log=False):
    """Evaluate the FMU once and return inputs + outputs as a dict."""
    temp_created = None
    if unzip_dir is None:
        temp_created = tempfile.mkdtemp(prefix="fmu_")
        unzip_dir = temp_created
        with zipfile.ZipFile(fmu_path, "r") as zf:
            zf.extractall(unzip_dir)

    try:
        with SurrogateSession(unzip_dir, log=log) as session:
            outputs = session.evaluate(inputs)
            values = {n: 0.0 for n in session._names_in}
            for v in session.fmu.inputs:
                values[v.name] = v.start_float
            if inputs:
                values.update({k: float(v) for k, v in inputs.items()})
            return {
                "model_name": session.fmu.md["model_name"],
                "fmi_version": session.fmu.md["fmi_version"],
                "inputs": values,
                "outputs": outputs,
            }
    finally:
        if temp_created:
            shutil.rmtree(temp_created, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _extract(fmu_path, unzip_dir):
    if unzip_dir is None:
        unzip_dir = tempfile.mkdtemp(prefix="fmu_")
        cleanup = unzip_dir
        with zipfile.ZipFile(fmu_path, "r") as zf:
            zf.extractall(unzip_dir)
    else:
        os.makedirs(unzip_dir, exist_ok=True)
        cleanup = None
        if not os.path.isfile(os.path.join(unzip_dir, "modelDescription.xml")):
            with zipfile.ZipFile(fmu_path, "r") as zf:
                zf.extractall(unzip_dir)
    return unzip_dir, cleanup


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("fmu", help="Path to the .fmu file")
    parser.add_argument(
        "--input", action="append", default=[],
        help='Input as "name=value" (repeatable)',
    )
    parser.add_argument(
        "--inputs-json", help="JSON file mapping input names to values"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List inputs/outputs as JSON and exit (no instantiation)",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Read {'inputs': [{...}, ...]} from stdin, reuse one FMU "
             "instance, write a list of output dicts to stdout",
    )
    parser.add_argument(
        "--unzip-dir",
        help="Reuse an already-extracted FMU directory (skips extraction)",
    )
    parser.add_argument("--log", action="store_true",
                        help="Show the FMU's log messages")
    args = parser.parse_args(argv)

    unzip_dir, cleanup = _extract(args.fmu, args.unzip_dir)

    try:
        if args.list:
            md = read_model_description(unzip_dir)
            payload = {
                "model_name": md["model_name"],
                "fmi_version": md["fmi_version"],
                "generation_tool": md["generation_tool"],
                "default_experiment": md["default_experiment"],
                "inputs": [
                    {"name": v.name, "start": v.start_float}
                    for v in md["variables"] if v.causality == "input"
                ],
                "outputs": [
                    {"name": v.name}
                    for v in md["variables"] if v.causality == "output"
                ],
            }
            print(json.dumps(payload, indent=2))
            return 0

        if args.batch:
            request = json.load(sys.stdin)
            results = []
            with SurrogateSession(unzip_dir, log=args.log) as session:
                for point in request.get("inputs", []):
                    results.append(session.evaluate(point))
            print(json.dumps(results))
            return 0

        inputs = {}
        if args.inputs_json:
            with open(args.inputs_json) as fh:
                inputs.update(json.load(fh))
        for item in args.input:
            name, _, value = item.partition("=")
            inputs[name.strip()] = float(value)

        print(json.dumps(evaluate(args.fmu, inputs=inputs,
                                  unzip_dir=unzip_dir, log=args.log),
                         indent=2))
        return 0
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
