"""Compile a Qiskit circuit against multiple backends and log results to MLflow.

A "Qiskit file" is either:
  - a .py file that defines ``build_circuit() -> QuantumCircuit`` (or a
    module-level ``circuit`` variable), or
  - an OpenQASM 2 file (.qasm).

For every (backend, optimization level) pair the circuit is transpiled and
one MLflow run is recorded with the compilation parameters, metrics
(depth, gate counts, transpile time, ...) and the transpiled circuit as an
artifact.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
import time
from pathlib import Path

import mlflow
import qiskit
from qiskit import QuantumCircuit, qasm2, qasm3
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qset_demo.backends import available_backends, get_backend

DEFAULT_OPT_LEVELS = [0, 1, 2, 3]


def load_circuit(path: Path) -> QuantumCircuit:
    """Load a QuantumCircuit from a .py or .qasm file."""
    if path.suffix == ".qasm":
        return qasm2.load(str(path))
    if path.suffix == ".py":
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "build_circuit"):
            circuit = module.build_circuit()
        elif hasattr(module, "circuit"):
            circuit = module.circuit
        else:
            raise ValueError(
                f"{path} must define build_circuit() or a module-level 'circuit'"
            )
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError(f"{path} produced {type(circuit)}, expected QuantumCircuit")
        return circuit
    raise ValueError(f"Unsupported file type: {path} (expected .py or .qasm)")


def two_qubit_gate_count(circuit: QuantumCircuit) -> int:
    return sum(1 for inst in circuit.data if inst.operation.num_qubits == 2)


def compile_and_log(
    circuit: QuantumCircuit,
    circuit_name: str,
    source_file: str,
    backend_name: str,
    optimization_level: int,
) -> dict:
    """Transpile the circuit for one backend/level and record an MLflow run."""
    backend = get_backend(backend_name)

    pass_manager = generate_preset_pass_manager(
        optimization_level=optimization_level, backend=backend
    )
    start = time.perf_counter()
    transpiled = pass_manager.run(circuit)
    transpile_seconds = time.perf_counter() - start

    run_name = f"{circuit_name}-{backend_name}-O{optimization_level}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "circuit": circuit_name,
                "backend": backend_name,
                "git_sha": os.environ.get("GITHUB_SHA", "local"),
                "git_ref": os.environ.get("GITHUB_REF_NAME", "local"),
            }
        )
        mlflow.log_params(
            {
                "circuit": circuit_name,
                "source_file": source_file,
                "backend": backend_name,
                "backend_num_qubits": backend.num_qubits,
                "basis_gates": ",".join(sorted(backend.operation_names)),
                "optimization_level": optimization_level,
                "qiskit_version": qiskit.__version__,
            }
        )

        metrics = {
            "depth": transpiled.depth(),
            "total_ops": transpiled.size(),
            "two_qubit_gates": two_qubit_gate_count(transpiled),
            "qubits_used": len(transpiled.layout.final_index_layout())
            if transpiled.layout
            else transpiled.num_qubits,
            "transpile_seconds": round(transpile_seconds, 4),
        }
        mlflow.log_metrics(metrics)
        for gate, count in transpiled.count_ops().items():
            mlflow.log_metric(f"gate_count.{gate}", count)

        with tempfile.TemporaryDirectory() as tmp:
            qasm_path = Path(tmp) / f"{run_name}.qasm3"
            qasm_path.write_text(qasm3.dumps(transpiled))
            mlflow.log_artifact(str(qasm_path), artifact_path="transpiled")

    return {"backend": backend_name, "opt_level": optimization_level, **metrics}


def write_github_summary(circuit_name: str, results: list[dict]) -> None:
    """Append a results table to the GitHub Actions job summary, if available."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return
    lines = [
        f"### Compilation results: `{circuit_name}`",
        "",
        "| Backend | Opt level | Depth | Total ops | 2q gates | Transpile (s) |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['backend']} | O{r['opt_level']} | {r['depth']} "
            f"| {r['total_ops']} | {r['two_qubit_gates']} | {r['transpile_seconds']} |"
        )
    with open(summary_file, "a") as f:
        f.write("\n".join(lines) + "\n\n")


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Used locally to point runs at the hosted MLflow server without exporting
    secrets by hand. Existing environment variables are never overwritten, so
    in CI the GitHub-provided secrets always take precedence over any .env.
    """
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("circuit_file", type=Path, help="Path to a .py or .qasm circuit file")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=available_backends(),
        help=f"Backends from hardware_specs/ (default: {available_backends()})",
    )
    parser.add_argument(
        "--opt-levels",
        nargs="+",
        type=int,
        default=DEFAULT_OPT_LEVELS,
        choices=[0, 1, 2, 3],
        help=f"Transpiler optimization levels to sweep (default: {DEFAULT_OPT_LEVELS})",
    )
    parser.add_argument(
        "--experiment",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "qiskit-compilation"),
        help="MLflow experiment name",
    )
    args = parser.parse_args(argv)

    circuit = load_circuit(args.circuit_file)
    circuit_name = circuit.name if circuit.name != "circuit" else args.circuit_file.stem

    mlflow.set_experiment(args.experiment)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Circuit: {circuit_name} ({circuit.num_qubits} qubits, depth {circuit.depth()})")

    results = []
    for backend_name in args.backends:
        for level in args.opt_levels:
            result = compile_and_log(
                circuit, circuit_name, str(args.circuit_file), backend_name, level
            )
            print(
                f"  {backend_name} O{level}: depth={result['depth']} "
                f"ops={result['total_ops']} 2q={result['two_qubit_gates']} "
                f"({result['transpile_seconds']}s)"
            )
            results.append(result)

    write_github_summary(circuit_name, results)
    print(f"Logged {len(results)} runs to experiment '{args.experiment}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
