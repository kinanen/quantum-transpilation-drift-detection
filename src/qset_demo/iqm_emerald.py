"""Build an IQM (qiskit-on-iqm) backend on project Emerald's exact coupling map.

Used only when ``QSET_TRANSPILER=iqm``; requires ``iqm-client[qiskit]``. The
backend routes on Emerald's 85-pair topology (a subset of IQM's Crystal-54),
reusing Aphrodite's calibration error profile trimmed to those pairs, so the
IQM transpiler and the vanilla Qiskit pipeline compile against the *identical*
device — drift then reflects the transpiler, not the topology.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SPECS_DIR = Path(os.environ.get("QSET_HARDWARE_SPECS", "hardware_specs"))


def _connectivity(name: str) -> list[tuple[str, str]]:
    """Emerald's coupling map as IQM ``QB<n>`` name pairs (from calibration)."""
    data = json.loads((SPECS_DIR / f"{name} calibration.json").read_text())
    one_q = sorted(data["oneQubitProperties"], key=int)
    idmap = {qid: f"QB{i + 1}" for i, qid in enumerate(one_q)}
    pairs = []
    for pair in data["twoQubitProperties"]:
        a, b = pair.split("-")
        pairs.append((idmap[a], idmap[b]))
    return pairs


def build_iqm_backend(name: str = "Emerald"):
    """Return an ``IQMFakeBackend`` on ``name``'s 85-pair topology.

    The connectivity is a subset of Crystal-54, so Aphrodite's error profile
    (T1/T2, gate/readout errors, durations) covers every pair after trimming.
    """
    from iqm.qiskit_iqm import IQMFakeAphrodite
    from iqm.qiskit_iqm.fake_backends.iqm_fake_backend import (
        IQMErrorProfile,
        IQMFakeBackend,
    )
    from iqm.station_control.interface.models import StaticQuantumArchitecture

    pairs = _connectivity(name)
    pairset = {frozenset(p) for p in pairs}
    arch = StaticQuantumArchitecture(
        dut_label=f"{name}85",
        qubits=[f"QB{i}" for i in range(1, 55)],
        computational_resonators=[],
        connectivity=pairs,
    )
    ep = IQMFakeAphrodite().error_profile  # superset (90 pairs)

    def trim(gate_dict):
        return {
            g: {pr: err for pr, err in d.items() if frozenset(pr) in pairset}
            for g, d in gate_dict.items()
        }

    trimmed = IQMErrorProfile(
        t1s=ep.t1s,
        t2s=ep.t2s,
        single_qubit_gate_depolarizing_error_parameters=ep.single_qubit_gate_depolarizing_error_parameters,
        two_qubit_gate_depolarizing_error_parameters=trim(
            ep.two_qubit_gate_depolarizing_error_parameters
        ),
        single_qubit_gate_durations=ep.single_qubit_gate_durations,
        two_qubit_gate_durations=ep.two_qubit_gate_durations,
        readout_errors=ep.readout_errors,
        name=f"{name}85",
    )
    return IQMFakeBackend(arch, trimmed, name=name)
