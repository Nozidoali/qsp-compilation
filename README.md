# State Preparation Compilation

Quantum state preparation algorithms in Python/Qiskit: rotation-based (sparse &
dense) and QROM-based (alias sampling with Select / SelectSwap). Given a target
state as JSON, `main.py` compiles it to an OpenQASM 2 circuit.

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key version: `pygridsynth==2.0.0` (Clifford+T decomposition via gridsynth).

## Usage

```bash
# Sparse rotation (exact), QASM to stdout
python main.py benchmarks/basic/w_n3.json -m sparse

# SelectSwap alias sampling, precision 8, to a file
python main.py benchmarks/basic/w_n3.json -m selectswap -b 8 -o w_n3.qasm

# Clifford+T lowering of the sparse-alias circuit
python main.py benchmarks/basic/w_n3.json -m sparse_alias -b 6 --clifford-t
```

### Arguments

| Flag           | Default  | Description                                                   |
| -------------- | -------- | ------------------------------------------------------------- |
| `state`        | —        | Path to the state JSON file                                   |
| `-m`           | `sparse` | Method: `sparse`, `dense`, `qrom`, `selectswap`, `sparse_alias` |
| `-b`           | `8`      | Precision bits (alias-table precision; Clifford+T angle precision) |
| `--clifford-t` | off      | Emit the Clifford+T circuit instead of the raw one            |
| `-o`           | stdout   | Output QASM path                                              |
| `-t`           | `120`    | Per-synthesis timeout in seconds (rotation methods)           |

### Methods

- `sparse` — rotation, cardinality reduction (exact); best for sparse states.
- `dense` — rotation, qubit reduction (exact).
- `qrom` — alias sampling with basic Select (approximate).
- `selectswap` — alias sampling with SelectSwap (Babbush et al. [arXiv:1812.00954](https://arxiv.org/abs/1812.00954)).
- `sparse_alias` — alias sampling over the compressed support plus an
  index-mapping QROM; addresses `2^ceil(log2(card))` bins instead of `2^n`.

## State JSON format

```json
{ "n_bits": 3, "amplitudes": { "001": 0.577, "010": 0.577, "100": 0.577 } }
```

Keys are zero-padded binary strings of length `n_bits`; values are real
amplitudes. See `benchmarks/` for examples.

## Structure

```
main.py            # entry point: state JSON + method -> QASM
src/
  state.py         # sparse state representation, JSON (de)serialization
  common.py        # rotation helpers (X/CNOT reduction, angle optimization)
  sparse.py        # cardinality-reduction rotation synthesis
  dense.py         # qubit-reduction rotation synthesis
  synthesis.py     # QROM: basic Select and SelectSwap (CCX and AND variants)
  alias_sampling.py# Walker alias sampling (dense and sparse/compressed)
  circuit_utils.py # comparator, controlled-swap, Gidney AND, Clifford+T lowering
benchmarks/        # example target states (basic / t_friendly / thc)
tests/             # pytest suite for the src modules
```

## Tests

```bash
pytest
```

## T-cost accounting

Toffoli-via-Gidney-AND assumes measurement-based uncompute (Gidney
[arXiv:1709.06648](https://arxiv.org/abs/1709.06648)): 4 T per AND pair. The
emitted QASM uses the unitary AND adjoint (8 T per pair) because OpenQASM 2
cannot serialize mid-circuit measurement + classically-controlled gates.
