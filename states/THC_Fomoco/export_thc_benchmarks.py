from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np

from thc_state_prep import THCFactors, build_nonorthogonal_prepare_terms


ROOT = Path(__file__).resolve().parent
FEMOCO_DIR = ROOT / "Lee_code" / "femoco"
DEFAULT_OUT_DIR = ROOT / "data"
FAMILIES = ("li", "reiher")
SOURCES = ("approximated", "exact")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Lee-code THC Hamiltonians as benchmark-style sparse JSON states."
    )
    parser.add_argument(
        "--family",
        nargs="+",
        choices=FAMILIES,
        default=list(FAMILIES),
        help="Which FeMoco Hamiltonian family to export.",
    )
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default="approximated",
        help="Use the approximated THC factors from the paper, or the raw THC tensors.",
    )
    parser.add_argument(
        "--M",
        dest="ranks",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of THC ranks M to export. Defaults to all available ranks for the chosen source.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where benchmark JSON files are written.",
    )
    return parser.parse_args()


def iter_tensor_files(family: str, source: str) -> List[Tuple[int, Path]]:
    if source == "approximated":
        directory = FEMOCO_DIR / "THC_tensors_approximated" / family
        paths = sorted(directory.glob("M_*_beta_*_eta_*.h5"))
    else:
        directory = FEMOCO_DIR / "THC_tensors" / family
        paths = sorted(directory.glob("M_*.h5"))

    ranked: List[Tuple[int, Path]] = []
    for path in paths:
        match = re.search(r"M_(\d+)", path.name)
        if not match:
            continue
        ranked.append((int(match.group(1)), path))
    return ranked


def choose_tensor_files(
    family: str, source: str, ranks: Iterable[int] | None
) -> List[Tuple[int, Path]]:
    available = dict(iter_tensor_files(family, source))
    if ranks is None:
        return sorted(available.items())

    chosen: List[Tuple[int, Path]] = []
    missing = [rank for rank in ranks if rank not in available]
    if missing:
        raise FileNotFoundError(
            f"Missing {source} THC files for family={family} at M={missing}. "
            f"Available ranks: {sorted(available)}"
        )
    for rank in ranks:
        chosen.append((rank, available[rank]))
    return chosen


def load_integrals(family: str) -> Tuple[np.ndarray, np.ndarray]:
    path = FEMOCO_DIR / "integrals" / f"eri_{family}.h5"
    with h5py.File(path, "r") as handle:
        eri = handle["eri"][()]
        h0 = handle["h0"][()]
    return eri, h0


def load_thc_factors(path: Path) -> THCFactors:
    with h5py.File(path, "r") as handle:
        mpq = handle["MPQ"][()]
        eta = handle["etaPp"][()]
    return THCFactors(chi=eta.T, zeta=mpq)


def compute_one_body_eigenvalues(
    eri_exact: np.ndarray,
    h0: np.ndarray,
    factors: THCFactors,
) -> np.ndarray:
    eta = factors.chi.T
    mpq = factors.zeta

    # Exact exchange contribution, plus the THC Coulomb contribution without
    # materializing the full 4-index THC tensor.
    exchange = np.einsum("illj->ij", eri_exact, optimize=True)
    row_norm_sq = np.einsum("Pp,Pp->P", eta, eta, optimize=True)
    coulomb_weights = np.einsum("P,PQ->Q", row_norm_sq, mpq, optimize=True)
    coulomb = np.einsum("Q,Qi,Qj->ij", coulomb_weights, eta, eta, optimize=True)

    t_matrix = h0 - 0.5 * exchange + coulomb
    eigenvalues, _ = np.linalg.eigh(t_matrix)
    return eigenvalues


def flatten_pair(a: int, b: int, side: int) -> int:
    return a * side + b


def prepare_state_to_sparse_dict(
    prep: Dict[str, object],
    side: int,
) -> Dict[int, float]:
    state: Dict[int, float] = {}

    for label, amplitude, sign in prep["one_body_terms"]:
        idx = flatten_pair(label[0], label[1], side)
        state[idx] = float(sign * amplitude)

    for label, amplitude, sign in prep["two_body_terms"]:
        idx = flatten_pair(label[0], label[1], side)
        state[idx] = float(sign * amplitude)

    norm = math.sqrt(sum(value * value for value in state.values()))
    if not np.isclose(norm, 1.0, atol=1e-8):
        raise ValueError(f"State is not normalized: norm={norm}")
    return state


def state_to_json_dict(state: Dict[int, float], n_bits: int, metadata: Dict[str, object]) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "n_bits": n_bits,
        "amplitudes": {str(index): amplitude for index, amplitude in sorted(state.items())},
    }
    payload.update(metadata)
    return payload


def export_family(
    family: str,
    source: str,
    ranks: Iterable[int] | None,
    out_dir: Path,
) -> List[Path]:
    eri_exact, h0 = load_integrals(family)
    written: List[Path] = []

    for rank, tensor_path in choose_tensor_files(family, source, ranks):
        factors_raw = load_thc_factors(tensor_path)
        factors_norm = factors_raw.normalize_columns_paper_convention()
        t_eigs = compute_one_body_eigenvalues(eri_exact, h0, factors_raw)
        prep = build_nonorthogonal_prepare_terms(t=t_eigs, zeta=factors_norm.zeta)

        side = rank + 1
        n_bits = math.ceil(math.log2(side * side))
        state = prepare_state_to_sparse_dict(prep, side=side)

        out_path = out_dir / f"thc_{family}_{source}_M{rank}_nonorthogonal_prepare.json"
        metadata = {
            "family": family,
            "source": source,
            "M": rank,
            "n_orb": factors_raw.n_orb,
            "n_terms": len(state),
            "lambda_total": float(prep["lambda_total"]),
            "lambda_zeta": float(factors_norm.lambda_zeta_nonorthogonal(assume_normalized=True)),
            "lambda_T": float(np.sum(np.abs(t_eigs))),
            "tensor_file": str(tensor_path.relative_to(ROOT)),
            "basis_encoding": {
                "kind": "pair_index",
                "side": side,
                "index_formula": "index = a * side + b; one-body labels use (ell, M), two-body labels use (mu, nu)",
            },
        }
        out_path.write_text(json.dumps(state_to_json_dict(state, n_bits, metadata), indent=2))
        written.append(out_path)

    return written


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for family in args.family:
        written.extend(export_family(family, args.source, args.ranks, args.out_dir))

    for path in written:
        print(path)
    print(f"Wrote {len(written)} benchmark files to {args.out_dir}")


if __name__ == "__main__":
    main()
