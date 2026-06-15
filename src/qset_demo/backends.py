"""Build Qiskit backends from Braket-style calibration JSON files.

Each file in hardware_specs/ ("<Name> calibration.json", schema
``braket.device_schema.standardized_gate_model_qpu_device_properties``)
becomes a compile-only BackendV2 whose Target carries the real calibration
data (T1/T2, gate and readout errors), so the transpiler's noise-aware
layout selection works against actual device data.

Two schema flavors are handled:
  - version 1: per-qubit oneQubitProperties / per-pair twoQubitProperties
    (superconducting devices; connectivity comes from the pair keys)
  - version 3: device-level aggregates only (trapped-ion devices;
    all-to-all connectivity, qubit count supplied by the registry below)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from qiskit.circuit import Measure, Parameter
from qiskit.circuit.library import CZGate, RGate, RXXGate, RZGate
from qiskit.providers import BackendV2, Options, QubitProperties
from qiskit.transpiler import InstructionProperties, Target

SPECS_DIR = Path(os.environ.get("QSET_HARDWARE_SPECS", "hardware_specs"))


@dataclass
class DeviceSpec:
    kind: str  # "superconducting" | "trapped_ion"
    num_qubits: int | None = None  # required for device-level (v3) schemas


# The calibration files don't carry architecture info, so it lives here.
# Forte (IonQ) publishes only device-level aggregates; its qubit count is
# not in the file.
DEVICES = {
    "Cepheus": DeviceSpec(kind="superconducting"),
    "Garnet": DeviceSpec(kind="superconducting"),
    "Forte": DeviceSpec(kind="trapped_ion", num_qubits=36),
}


def available_backends() -> list[str]:
    return sorted(DEVICES)


class CalibrationBackend(BackendV2):
    """Compile-only backend: has a Target, cannot execute circuits."""

    def __init__(self, name: str, target: Target):
        super().__init__(name=name)
        self._target = target

    @property
    def target(self) -> Target:
        return self._target

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls):
        return Options()

    def run(self, run_input, **options):
        raise NotImplementedError(f"{self.name} is compile-only (no simulator attached)")


def _fidelity(entries: list[dict], *preferred: str) -> float | None:
    """Pick a fidelity value by fidelityType name, in order of preference."""
    by_type = {e["fidelityType"]["name"]: e["fidelity"] for e in entries}
    for name in preferred:
        if name in by_type:
            return by_type[name]
    return None


def _error(fidelity: float | None) -> float | None:
    return None if fidelity is None else max(0.0, 1.0 - fidelity)


def _build_superconducting(data: dict) -> Target:
    one_q = data["oneQubitProperties"]
    # Qubit ids may be non-contiguous (dead qubits) or 1-based; remap to
    # dense 0-based indices.
    index = {qid: i for i, qid in enumerate(sorted(one_q, key=int))}

    qubit_props = []
    for qid in sorted(one_q, key=int):
        props = one_q[qid]
        qubit_props.append(
            QubitProperties(t1=props["T1"]["value"], t2=props["T2"]["value"])
        )
    target = Target(num_qubits=len(index), qubit_properties=qubit_props)

    r_props, rz_props, measure_props = {}, {}, {}
    for qid, i in index.items():
        fids = one_q[qid].get("oneQubitFidelity", [])
        gate_err = _error(
            _fidelity(fids, "RANDOMIZED_BENCHMARKING", "SIMULTANEOUS_RANDOMIZED_BENCHMARKING")
        )
        r_props[(i,)] = InstructionProperties(error=gate_err)
        rz_props[(i,)] = InstructionProperties(error=0.0)  # virtual Z
        measure_props[(i,)] = InstructionProperties(error=_error(_fidelity(fids, "READOUT")))

    cz_props = {}
    for pair, props in data["twoQubitProperties"].items():
        a, b = pair.split("-")
        if a not in index or b not in index:
            continue
        err = _error(
            next(
                (g["fidelity"] for g in props["twoQubitGateFidelity"] if g["gateName"] == "CZ"),
                None,
            )
        )
        cz_props[(index[a], index[b])] = InstructionProperties(error=err)
        cz_props[(index[b], index[a])] = InstructionProperties(error=err)

    target.add_instruction(RGate(Parameter("theta"), Parameter("phi")), r_props)
    target.add_instruction(RZGate(Parameter("lam")), rz_props)
    target.add_instruction(CZGate(), cz_props)
    target.add_instruction(Measure(), measure_props)
    return target


def _build_trapped_ion(data: dict, num_qubits: int) -> Target:
    t1 = data["T1"]["value"]
    t2 = data["T2"]["value"]
    target = Target(
        num_qubits=num_qubits,
        qubit_properties=[QubitProperties(t1=t1, t2=t2)] * num_qubits,
    )

    gate_1q_duration = data["singleQubitGateDuration"]["value"]
    gate_2q_duration = data["twoQubitGateDuration"]["value"]
    gate_2q_error = _error(
        _fidelity(data["twoQubitGateFidelity"], "RANDOMIZED_BENCHMARKING")
    )
    readout_error = _error(_fidelity(data["readoutFidelity"], "RANDOMIZED_BENCHMARKING"))
    readout_duration = data["readoutDuration"]["value"]

    qubits = range(num_qubits)
    target.add_instruction(
        RGate(Parameter("theta"), Parameter("phi")),
        {(i,): InstructionProperties(duration=gate_1q_duration) for i in qubits},
    )
    target.add_instruction(
        RZGate(Parameter("lam")),
        {(i,): InstructionProperties(error=0.0, duration=0.0) for i in qubits},
    )
    # Fully connected: the Mølmer-Sørensen gate is available on every pair.
    rxx_props = {}
    for i, j in combinations(qubits, 2):
        props = InstructionProperties(error=gate_2q_error, duration=gate_2q_duration)
        rxx_props[(i, j)] = props
        rxx_props[(j, i)] = props
    target.add_instruction(RXXGate(Parameter("theta")), rxx_props)
    target.add_instruction(
        Measure(),
        {
            (i,): InstructionProperties(error=readout_error, duration=readout_duration)
            for i in qubits
        },
    )
    return target


def get_backend(name: str, specs_dir: Path | None = None) -> CalibrationBackend:
    if name not in DEVICES:
        raise SystemExit(
            f"Unknown backend {name!r}. Available: {', '.join(available_backends())}"
        )
    spec = DEVICES[name]
    path = (specs_dir or SPECS_DIR) / f"{name} calibration.json"
    data = json.loads(path.read_text())

    if spec.kind == "superconducting":
        target = _build_superconducting(data)
    else:
        target = _build_trapped_ion(data, spec.num_qubits)
    return CalibrationBackend(name=name, target=target)
