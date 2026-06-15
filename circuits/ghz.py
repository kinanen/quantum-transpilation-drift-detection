"""A 5-qubit GHZ state circuit."""

from qiskit import QuantumCircuit

N_QUBITS =10


def build_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS, N_QUBITS, name="ghz")
    qc.h(0)
    for i in range(N_QUBITS - 1):
        qc.cx(i, i + 1)
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc
