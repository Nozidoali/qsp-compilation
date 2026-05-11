# State Preparation Compilation

Quantum state preparation algorithms in Python/Qiskit: rotation-based (sparse & dense) and QROM-based (alias sampling with Select/SelectSwap).

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Dependencies are pinned in `requirements.txt`. Key version: `pygridsynth==2.0.0` (Clifford+T decomposition via gridsynth).

## Quick Start

```bash
# 1. Run the T-friendly comparison experiment
python -m experiments.compare_t_friendly -v

# 2. Run the general T-cost benchmark
python -m experiments.compare_t_cost --n-max 5 -v

# 3. Plot infidelity vs T-cost
python -m experiments.plot_infidelity_vs_tcost
```

The T-friendly experiment builds circuits using the qubit-reduction template
(Ry, CRY, CCRY, ...) with each demux Ry angle sampled from exact T-friendly
rotations. It compares the T-cost against sparse rotation, QROM, and
select-swap. You should see T-friendly dense rotation winning at every qubit
count.

## Experiment Arguments

### `compare_t_friendly`

| Flag      | Default                  | Description                                |
| --------- | ------------------------ | ------------------------------------------ |
| `--k-max` | 2                        | Max T gates per angle                      |
| `--n-min` | 3                        | Min qubits                                 |
| `--n-max` | 8                        | Max qubits                                 |
| `-s`      | 5                        | Random circuit samples per n               |
| `-b`      | 10                       | Precision bits for QROM / sparse gridsynth |
| `--seed`  | 42                       | Random seed                                |
| `-o`      | `t_friendly_results.csv` | Output CSV path                            |
| `-v`      | off                      | Verbose output                             |

### `compare_t_cost`

| Flag                | Default                  | Description                                         |
| ------------------- | ------------------------ | --------------------------------------------------- |
| `--n-min`           | 3                        | Min qubits                                          |
| `--n-max`           | 8                        | Max qubits                                          |
| `-f`                | `all`                    | State family (`w`, `dicke_2`, `sparse_random`, ...) |
| `-b`                | `1,2,4,8`                | Comma-separated precision bits                      |
| `-s`                | 3                        | Seeds for random families                           |
| `-t`                | 120                      | Timeout per synthesis (seconds)                     |
| `-o`                | `qsp_t_cost_results.csv` | Output path                                         |
| `--json`            | off                      | Output JSON instead of CSV                          |
| `--benchmarks-dir`  | —                        | Load states from JSON files                         |
| `--dump-benchmarks` | —                        | Generate benchmark states and exit                  |
| `-v`                | off                      | Verbose output                                      |

## Tests

```bash
pytest tests/ -v
```

## Structure

- `src/qsp/state.py` — Sparse state representation and generators
- `src/qsp/rotation/` — Cardinality reduction (sparse) and qubit reduction (dense)
- `src/qsp/qrom/` — QROM synthesis (Select, SelectSwap) and alias sampling
- `experiments/` — T-cost comparison benchmarks and plotting

## T-cost Accounting

Toffoli-via-Gidney-AND counts assume **measurement-based uncompute** (Gidney
[arxiv:1709.06648](https://arxiv.org/abs/1709.06648)): 4 T per AND pair (4 T
compute + 0 T uncompute). This applies uniformly to alias sampling's
comparator, controlled-swap, and the unary-iteration QROM in both the CCX-mode
(`clifford_t=False`) and AND-mode (`clifford_t=True`) synthesis paths.

For CCX-mode unary iteration the literal circuit emits both compute and
uncompute Toffolis as separate `ccx` gates, but they share one Gidney AND
ancilla pair in the asymptotic cost model, so each conjugate pair is counted
once (4 T per pair, not 4 T per `ccx`). For AND-mode the circuit already has
explicit `and_compute`/`and_uncompute` pairs.

The emitted circuit, however, always uses the **unitary adjoint** for the
AND uncompute (4 T compute + 4 T uncompute = 8 T per AND pair). Reason:
OpenQASM 2 can't serialize mid-circuit measurement + classically-controlled
gates, and `experiments/run.py` exports to QASM 2. Reported T-counts reflect
the asymptotically achievable cost rather than the literally emitted gate
count. Falling back to `clean_ancilla=False` switches to the standard 7-T
Toffoli decomposition and counts each emitted `ccx` independently (no AND
sharing).
