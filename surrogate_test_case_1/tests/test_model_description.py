"""Model description and package layout of ``FullSurrogate.fmu``.

All checks here read the unencrypted XML / PE binary straight from the zip
(no Wine) and cross-check them against the vendor CSV headers, so they prove
the datasets really belong to *this* FMU.
"""

from __future__ import annotations

import zipfile

import pytest

# Symbols the Wine driver relies on.  fmi3SetupExperiment is deliberately
# ABSENT from the vendor DLL -- see test_dll_omits_setup_experiment.
REQUIRED_EXPORTS = [
    b"fmi3Instantiate",
    b"fmi3EnterInitializationMode",
    b"fmi3ExitInitializationMode",
    b"fmi3SetFloat64",
    b"fmi3GetFloat64",
    b"fmi3DoStep",
]


def test_fmi_version_and_tool(surrogate):
    info = surrogate.info
    assert info["fmi_version"] == "3.0", info["fmi_version"]
    assert info["generation_tool"] == "SmartUQ"
    assert info["model_name"] == "FullSurrogate"


def test_inputs_match_training_doe_headers(surrogate, input_names, train_doe):
    """Names *and order* must match: the DOE CSV is the surrogate's own
    input vector."""
    assert input_names == list(train_doe.columns)


def test_outputs_match_results_headers(surrogate, output_names, train_res,
                                       valid_res):
    assert output_names == list(train_res.columns)
    assert output_names == list(valid_res.columns)


def test_value_references_well_formed(surrogate):
    ins = surrogate.info["inputs"]
    outs = surrogate.info["outputs"]
    in_refs = [v["value_reference"] for v in ins]
    out_refs = [v["value_reference"] for v in outs]
    assert len(set(in_refs)) == len(in_refs), "duplicate input valueReference"
    assert len(set(out_refs)) == len(out_refs), "duplicate output valueReference"
    assert not set(in_refs) & set(out_refs), (
        "a valueReference is used for both an input and an output")


def test_start_values_are_all_zero(surrogate):
    """Vendor quirk: every input ships start=0 (a degenerate operating
    point: zero conductivity scaling, zero coolant temperature, ...).
    This is exactly why the wrapper fills in NOMINAL_INPUTS."""
    starts = {v["name"]: v["start"] for v in surrogate.info["inputs"]}
    nonzero = {k: v for k, v in starts.items() if v != 0.0}
    assert not nonzero, (
        f"inputs now carry non-zero starts {nonzero} -- review "
        "NOMINAL_INPUTS in surrogate_fmu.py, the default point may be stale")


def test_binary_layout_is_windows_only(fmu_path):
    with zipfile.ZipFile(fmu_path) as zf:
        names = zf.namelist()
    assert "binaries/x86_64-windows/fmi3.dll" in names
    assert not any(n.startswith("binaries/x86_64-linux/") for n in names), (
        "a Linux binary appeared! The Wine-specific tests will skip and the "
        "wrapper should auto-switch to native fmpy.")


def test_backend_selection_follows_binary_layout(surrogate, fmu_path):
    with zipfile.ZipFile(fmu_path) as zf:
        has_linux = any(n.startswith("binaries/x86_64-linux/")
                        for n in zf.namelist())
    if has_linux:
        assert surrogate.backend == "native-fmpy"
    else:
        assert surrogate.backend == "wine"


def test_surrogate_payload_present(fmu_path):
    with zipfile.ZipFile(fmu_path) as zf:
        names = zf.namelist()
        assert "resources/Model.smartuq" in names
        payload = zf.read("resources/Model.smartuq")
    assert len(payload) > 1024
    # AES-encrypted SmartUQ payload: no plaintext dataset headers visible.
    assert b"kriging" not in payload.lower()


def test_dll_omits_setup_experiment(fmu_path):
    """Vendor quirk (regression guard for the driver's design).

    The DLL does not export ``fmi3SetupExperiment`` at all, which is why
    ``fmi3_wine.py`` runs the static surrogate straight through init mode
    instead of a full ``fmi3DoStep`` simulation (there is no supported way
    to communicate the stop time to this binary).

    If SmartUQ ever re-exports the FMU *with* this symbol, this test fails
    on purpose: the driver should then be upgraded to a spec-compliant
    SetupExperiment + DoStep flow (and the native fmpy path becomes the
    preferred one).
    """
    with zipfile.ZipFile(fmu_path) as zf:
        dll = zf.read("binaries/x86_64-windows/fmi3.dll")

    for sym in REQUIRED_EXPORTS:
        assert dll.count(sym) >= 1, f"DLL no longer exports {sym.decode()}"

    assert dll.count(b"fmi3SetupExperiment") == 0, (
        "fmi3SetupExperiment is now exported -- upgrade fmi3_wine.py to a "
        "full SetupExperiment/DoStep flow and re-validate this suite")


def test_default_experiment_declares_stop_time(surrogate):
    """The XML *does* declare startTime/stopTime/stepSize even though the
    DLL cannot be told about them (see previous test) -- documents that the
    missing DoStep is a binary quirk, not missing metadata."""
    de = surrogate.info["default_experiment"]
    assert "startTime" in de and de["startTime"] == 0.0
    assert "stopTime" in de and de["stopTime"] > 0
    assert "stepSize" in de and de["stepSize"] > 0
