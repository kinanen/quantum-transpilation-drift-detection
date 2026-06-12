# Q-SET Demo — Qiskit compilation tracking in CI

Showcase project: a GitHub Actions workflow that takes Qiskit circuit files,
compiles (transpiles) them against several quantum backends, and stores the
compilation data in [MLflow](https://mlflow.org).

## How it works

1. Each file in `circuits/` defines a circuit (a `.py` file with a
   `build_circuit()` function, or an OpenQASM 2 `.qasm` file).
2. `qset-compile` transpiles the circuit against a set of fake IBM backends
   (`FakeManilaV2`, `FakeBrisbane`, `FakeTorino` by default — no IBM Quantum
   credentials needed) at optimization levels 0–3.
3. Every (backend, optimization level) pair becomes one MLflow run with:
   - **params**: backend, basis gates, optimization level, Qiskit version, …
   - **metrics**: circuit depth, total ops, two-qubit gate count, per-gate
     counts, transpile time
   - **artifacts**: the transpiled circuit as OpenQASM 3
4. The CI workflow (`.github/workflows/compile.yml`) runs this on every push
   and PR, prints a results table to the job summary, and uploads the MLflow
   file store as a workflow artifact.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install .

qset-compile circuits/bell.py
qset-compile circuits/ghz.py --backends FakeTorino --opt-levels 1 3

# Browse the results
mlflow ui
```

## Use a remote MLflow server in CI

Set these repository secrets and the workflow logs there instead of the
local file store:

- `MLFLOW_TRACKING_URI` (e.g. `https://mlflow.example.com`)
- `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` (if required)

## Inspect CI results without a server

Download the `mlruns` artifact from a workflow run, unzip it, and run
`mlflow ui` in the directory containing `mlruns/`.
