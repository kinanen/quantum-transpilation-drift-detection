# Quantum Transpilation Drift Detection — Qiskit compilation tracking in CI

Showcase project: a GitHub Actions workflow that takes Qiskit circuit files,
compiles (transpiles) them against several quantum backends, and stores the
compilation data in [MLflow](https://mlflow.org).

## How it works

1. Each file in `circuits/` defines a circuit (a `.py` file with a
   `build_circuit()` function, or an OpenQASM 2 `.qasm` file).
2. `qset-compile` transpiles the circuit against backends built from real
   device calibration data in `hardware_specs/` (Braket-style JSON):
   - **Cepheus** — 107-qubit superconducting device (CZ-based)
   - **Emerald** — 54-qubit IQM superconducting device (CZ, square lattice)
   - **Forte** — 36-qubit IonQ trapped-ion device (all-to-all connectivity)

   The calibration data (T1/T2, gate and readout errors) is loaded into each
   backend's transpiler target, so noise-aware layout picks the
   best-calibrated qubits. Levels 0–3 are swept by default.
3. Every (backend, optimization level) pair becomes one MLflow run with:
   - **params**: backend, basis gates, optimization level, Qiskit version, …
   - **metrics**: circuit depth, total ops, two-qubit gate count, per-gate
     counts, transpile time, structural drift (see below)
   - **artifacts**: the transpiled circuit as OpenQASM 3
4. The CI workflow (`.github/workflows/compile.yml`) runs this on every push
   and PR (and can be triggered manually). It prints a results table to the
   job summary and logs every run to the hosted MLflow server configured via
   repository secrets.

## Structural drift

Each run is scored against a **baseline** — the most recent prior run for the
same circuit, backend, and optimization level — to flag how much the compiled
realization changed. Over the count-valued metrics
`<depth, total_ops, two_qubit_gates, qubits_used>`:

```
D = (1/K) * sum_k |m_k(run) - m_k(base)| / max(m_k(base), epsilon)   (epsilon = 1)
```

It is a CI signal, not a formal circuit distance. Each run is tagged with a
`drift_status` derived from configurable thresholds (defaults shown):

| Status | Condition          | Env var            |
|--------|--------------------|--------------------|
| pass   | `D < 0.15`         | `QSET_DRIFT_WARN`  |
| warn   | `0.15 <= D < 0.40` | —                  |
| fail   | `D >= 0.40`        | `QSET_DRIFT_FAIL`  |

The first run of a new circuit/backend/level has no baseline, so its drift is
`0` (status `pass`). The score is recorded as the `structural_drift` metric and
shown in the CI job-summary table; it does not currently fail the build.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install .

qset-compile circuits/bv12.qasm
qset-compile circuits/ghz6.qasm --backends Emerald Forte --opt-levels 1 3
```

Runs are logged to the hosted MLflow server (see below); browse them in its
web UI.

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
qset-compile circuits/ghz6.qasm
```

`.env` is gitignored — never commit real credentials. Any variable already
set in your shell or in CI takes precedence over `.env`, so this never
interferes with the GitHub-secret values used in Actions.

## Configure CI to log to the hosted MLflow server

The workflow reads its MLflow connection from **repository secrets**. Set
them once (Settings → Secrets and variables → Actions, or with `gh`):

```bash
gh secret set MLFLOW_TRACKING_URI       # e.g. https://mlflow.example.com
gh secret set MLFLOW_TRACKING_USERNAME  # if the server needs basic auth
gh secret set MLFLOW_TRACKING_PASSWORD  # if the server needs basic auth
```

With these set, every run logs to the hosted server.

> The secrets must live on **this repository's** Actions secrets — not a
> GitHub Project board, repository *Variables*, or another repo — or the
> workflow fails because it has no MLflow server to reach.

## Run the workflow

```bash
gh workflow run "Quantum compilation CI"   # manual trigger (workflow_dispatch)
gh run watch                               # follow the latest run
```

It also runs automatically on every push to `main` and on pull requests.
