# Q-SET Demo — Qiskit compilation tracking in CI

Showcase project: a GitHub Actions workflow that takes Qiskit circuit files,
compiles (transpiles) them against several quantum backends, and stores the
compilation data in [MLflow](https://mlflow.org).

## How it works

1. Each file in `circuits/` defines a circuit (a `.py` file with a
   `build_circuit()` function, or an OpenQASM 2 `.qasm` file).
2. `qset-compile` transpiles the circuit against backends built from real
   device calibration data in `hardware_specs/` (Braket-style JSON):
   - **Cepheus** — 107-qubit superconducting device (CZ-based)
   - **Garnet** — 20-qubit IQM superconducting device (CZ, square lattice)
   - **Forte** — 36-qubit IonQ trapped-ion device (all-to-all connectivity)

   The calibration data (T1/T2, gate and readout errors) is loaded into each
   backend's transpiler target, so noise-aware layout picks the
   best-calibrated qubits. Levels 0–3 are swept by default.
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
qset-compile circuits/ghz.py --backends Garnet Forte --opt-levels 1 3

# Browse the results
mlflow ui
```

## Local MLflow server

`scripts/mlflow-server.sh` starts a tracking server at
<http://127.0.0.1:5001> (port 5001 because macOS AirPlay occupies 5000),
backed by `mlflow.db` with artifacts under `mlartifacts/`. Point runs at it
with:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
qset-compile circuits/bell.py
```

## Send local runs to the hosted MLflow server

To make locally-run compiles land in the same hosted server that CI uses,
copy the template and fill in the real connection settings:

```bash
cp .env.example .env
# edit .env — set MLFLOW_TRACKING_URI (and username/password if required)
```

`qset-compile` auto-loads `.env` on startup, so runs go straight to the
hosted server:

```bash
qset-compile circuits/ghz.py
```

`.env` is gitignored — never commit real credentials. Any variable already
set in your shell or in CI takes precedence over `.env`, so this never
interferes with the GitHub-secret values used in Actions.

## Use a remote MLflow server in CI

Set these repository secrets and the workflow logs there instead of the
local file store:

- `MLFLOW_TRACKING_URI` (e.g. `https://mlflow.example.com`)
- `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` (if required)

## Inspect CI results without a server

Download the `mlruns` artifact from a workflow run, unzip it, and run
`mlflow ui` in the directory containing `mlruns/`.
