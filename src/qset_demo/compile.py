"""Compile a Qiskit circuit against multiple backends and log results to MLflow.

A "Qiskit file" is either:
  - a .py file that defines ``build_circuit() -> QuantumCircuit`` (or a
    module-level ``circuit`` variable), or
  - an OpenQASM 2 file (.qasm).

For every (backend, optimization level) pair the circuit is transpiled and
one MLflow run is recorded with the compilation parameters (backend, basis
gates, coupling-map id, transpiler + version, optimization level, seed,
drift thresholds, CI linkage), metrics (depth, gate counts, transpile time,
structural drift, ...) and artifacts: the source circuit, the transpiled
circuit (OpenQASM 3), the backend target description, a metrics JSON and a
gate-count plot.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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

# Fixed transpiler seed so Sabre layout/routing is reproducible. Without it,
# O2/O3 pick a random initial layout each run, producing spurious structural
# drift even when nothing changed. Override with QSET_TRANSPILER_SEED.
DEFAULT_TRANSPILER_SEED = 20240622


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


# Count-valued structural features F(c, t, r) = <depth, gates, twoQ, layout>
# compared against the baseline run when scoring structural drift.
STRUCTURAL_METRICS = ("depth", "total_ops", "two_qubit_gates", "qubits_used")
DRIFT_EPSILON = 1.0  # avoids division by zero for zero-valued baseline metrics
DRIFT_WARN_DEFAULT = 0.15
DRIFT_FAIL_DEFAULT = 0.40


def structural_drift(
    current: dict, baseline: dict, epsilon: float = DRIFT_EPSILON
) -> float:
    """Mean relative absolute change of the count-valued structural metrics
    versus a baseline run:

        D = (1 / K) * sum_k |m_k(run) - m_k(base)| / max(m_k(base), epsilon)

    This is a CI drift signal indicating how much the compiled realization
    changed relative to the approved baseline, not a formal circuit distance.
    """
    total = sum(
        abs(current[k] - baseline[k]) / max(baseline[k], epsilon)
        for k in STRUCTURAL_METRICS
    )
    return total / len(STRUCTURAL_METRICS)


def drift_status(drift: float, warn: float, fail: float) -> str:
    """pass / warn / fail from configurable thresholds."""
    if drift >= fail:
        return "fail"
    if drift >= warn:
        return "warn"
    return "pass"


def fetch_baseline_metrics(
    circuit_name: str, backend_name: str, optimization_level: int
) -> tuple[dict, str] | None:
    """Count metrics and run_id of the most recent prior run for the same
    circuit, backend, and optimization level — the structural-drift baseline.
    Returns None when there is no comparable prior run (so drift defaults to 0).
    """
    filter_string = (
        f"tags.circuit = '{circuit_name}' "
        f"and params.backend = '{backend_name}' "
        f"and params.optimization_level = '{optimization_level}'"
    )
    try:
        runs = mlflow.search_runs(
            filter_string=filter_string,
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
    except Exception:
        return None
    if runs.empty:
        return None
    row = runs.iloc[0]
    baseline = {}
    for key in STRUCTURAL_METRICS:
        value = row.get(f"metrics.{key}")
        if value is None or value != value:  # missing column or NaN
            return None
        baseline[key] = value
    return baseline, row["run_id"]


def coupling_map_edges(backend) -> list:
    """Sorted directed edge list of the backend's coupling map ([] if dense/None)."""
    try:
        cm = backend.coupling_map
    except Exception:
        return []
    if cm is None:
        return []
    return sorted(tuple(e) for e in cm.get_edges())


def coupling_map_id(edges: list) -> str:
    """Short stable hash identifying a coupling map by its edge set."""
    return hashlib.sha1(repr(edges).encode()).hexdigest()[:12]


def target_description(backend) -> dict:
    """Serialise a backend's transpiler Target: basis, coupling map and per-
    qubit/per-gate calibration (T1/T2, gate and readout errors, durations)."""
    target = backend.target
    qprops = target.qubit_properties or []
    qubits = []
    for i in range(backend.num_qubits):
        qp = qprops[i] if i < len(qprops) else None
        qubits.append(
            {"qubit": i, "t1": getattr(qp, "t1", None), "t2": getattr(qp, "t2", None)}
        )
    instructions = {}
    for name in target.operation_names:
        entries = []
        for qargs, props in target[name].items():
            entries.append(
                {
                    "qubits": list(qargs) if qargs is not None else None,
                    "error": getattr(props, "error", None) if props else None,
                    "duration": getattr(props, "duration", None) if props else None,
                }
            )
        instructions[name] = entries
    edges = coupling_map_edges(backend)
    return {
        "name": backend.name,
        "num_qubits": backend.num_qubits,
        "basis_gates": sorted(target.operation_names),
        "coupling_map": [list(e) for e in edges],
        "coupling_pairs": len({frozenset(e) for e in edges}),
        "coupling_map_id": coupling_map_id(edges),
        "qubits": qubits,
        "instructions": instructions,
    }


def ci_run_url() -> str | None:
    """URL of the GitHub Actions run that produced this MLflow run, if in CI."""
    server = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if server and repo and run_id:
        return f"{server}/{repo}/actions/runs/{run_id}"
    return None


def _make_gate_plot(run_name: str, transpiled: QuantumCircuit, out_path: Path) -> bool:
    """Bar chart of the transpiled gate counts. Returns False if matplotlib
    is unavailable (the run is still logged, just without the plot)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    ops = transpiled.count_ops()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(list(ops), list(ops.values()), color="#4063D8")
    ax.set_title(run_name)
    ax.set_xlabel("gate")
    ax.set_ylabel("count")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def _log_run_artifacts(
    run_name: str,
    source_file: str,
    transpiled: QuantumCircuit,
    backend,
    metrics: dict,
    status: str,
    baseline_run_id: str | None,
) -> None:
    """Attach source + transpiled circuits, backend target, metrics JSON, plot."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        source = Path(source_file)
        if source.exists():
            mlflow.log_artifact(str(source), artifact_path="source")

        transpiled_path = tmp / f"{run_name}.qasm3"
        transpiled_path.write_text(qasm3.dumps(transpiled))
        mlflow.log_artifact(str(transpiled_path), artifact_path="transpiled")

        target_path = tmp / f"{backend.name}-target.json"
        target_path.write_text(json.dumps(target_description(backend), indent=2))
        mlflow.log_artifact(str(target_path), artifact_path="backend")

        metrics_path = tmp / f"{run_name}-metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "drift_status": status,
                    "baseline_run_id": baseline_run_id,
                    "metrics": metrics,
                },
                indent=2,
            )
        )
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        plot_path = tmp / f"{run_name}-gates.png"
        if _make_gate_plot(run_name, transpiled, plot_path):
            mlflow.log_artifact(str(plot_path), artifact_path="plots")


def compile_and_log(
    circuit: QuantumCircuit,
    circuit_name: str,
    source_file: str,
    backend_name: str,
    optimization_level: int,
) -> dict:
    """Transpile the circuit for one backend/level and record an MLflow run."""
    backend = get_backend(backend_name)

    seed = int(os.environ.get("QSET_TRANSPILER_SEED", DEFAULT_TRANSPILER_SEED))
    pass_manager = generate_preset_pass_manager(
        optimization_level=optimization_level, backend=backend, seed_transpiler=seed
    )
    start = time.perf_counter()
    transpiled = pass_manager.run(circuit)
    transpile_seconds = time.perf_counter() - start

    run_name = f"{circuit_name}-{backend_name}-O{optimization_level}"

    structural = {
        "depth": transpiled.depth(),
        "total_ops": transpiled.size(),
        "two_qubit_gates": two_qubit_gate_count(transpiled),
        "qubits_used": len(transpiled.layout.final_index_layout())
        if transpiled.layout
        else transpiled.num_qubits,
    }

    # Score drift against the previous run for this circuit/backend/level
    # before opening the new run (so it isn't its own baseline).
    baseline = fetch_baseline_metrics(circuit_name, backend_name, optimization_level)
    baseline_metrics, baseline_run_id = baseline if baseline else (None, None)
    drift = structural_drift(structural, baseline_metrics) if baseline_metrics else 0.0
    warn = float(os.environ.get("QSET_DRIFT_WARN", DRIFT_WARN_DEFAULT))
    fail = float(os.environ.get("QSET_DRIFT_FAIL", DRIFT_FAIL_DEFAULT))
    status = drift_status(drift, warn, fail)

    edges = coupling_map_edges(backend)

    metrics = {
        "structural_drift": round(drift, 4),
        **structural,
        "transpile_seconds": round(transpile_seconds, 4),
        # pre-transpile (logical) circuit, to gauge compilation overhead
        "source_depth": circuit.depth(),
        "source_ops": circuit.size(),
        "source_two_qubit_gates": two_qubit_gate_count(circuit),
    }
    gate_counts = {f"gate_count.{g}": c for g, c in transpiled.count_ops().items()}

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "circuit": circuit_name,
                "backend": backend_name,
                "git_sha": os.environ.get("GITHUB_SHA", "local"),
                "git_ref": os.environ.get("GITHUB_REF_NAME", "local"),
                "drift_status": status,
                "drift_baseline": "none" if baseline_metrics is None else "previous_run",
                "drift_baseline_run_id": baseline_run_id or "none",
                **({"ci_run_url": ci_run_url()} if ci_run_url() else {}),
            }
        )
        mlflow.log_params(
            {
                "circuit": circuit_name,
                "source_file": source_file,
                "backend": backend_name,
                "backend_num_qubits": backend.num_qubits,
                "basis_gates": ",".join(sorted(backend.operation_names)),
                "coupling_map_id": coupling_map_id(edges),
                "coupling_pairs": len({frozenset(e) for e in edges}),
                "optimization_level": optimization_level,
                "transpiler": "qiskit",
                "transpiler_version": qiskit.__version__,
                "transpiler_seed": seed,
                "qiskit_version": qiskit.__version__,
                "drift_warn_threshold": warn,
                "drift_fail_threshold": fail,
                "github_run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            }
        )

        mlflow.log_metrics({**metrics, **gate_counts})
        _log_run_artifacts(
            run_name, source_file, transpiled, backend,
            {**metrics, **gate_counts}, status, baseline_run_id,
        )

    return {
        "backend": backend_name,
        "opt_level": optimization_level,
        "drift_status": status,
        **metrics,
    }


def write_github_summary(circuit_name: str, results: list[dict]) -> None:
    """Append a results table to the GitHub Actions job summary, if available."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return
    lines = [
        f"### Compilation results: `{circuit_name}`",
        "",
        "| Backend | Opt level | Depth | Total ops | 2q gates | Drift | Status | Transpile (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['backend']} | O{r['opt_level']} | {r['depth']} "
            f"| {r['total_ops']} | {r['two_qubit_gates']} | {r['structural_drift']} "
            f"| {r['drift_status']} | {r['transpile_seconds']} |"
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
    # QASM 2 files carry no name, so Qiskit auto-assigns "circuit-N"; fall back
    # to the filename stem whenever the name is auto-generated.
    circuit_name = (
        circuit.name
        if circuit.name and not circuit.name.startswith("circuit")
        else args.circuit_file.stem
    )

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
                f"drift={result['structural_drift']} [{result['drift_status']}] "
                f"({result['transpile_seconds']}s)"
            )
            results.append(result)

    write_github_summary(circuit_name, results)
    print(f"Logged {len(results)} runs to experiment '{args.experiment}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
