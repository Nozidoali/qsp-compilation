from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


Array = np.ndarray


@dataclass
class THCFactors:
    """
    Tensor-hypercontraction factors for

        G[p,q,r,s] = sum_{mu,nu} chi[p,mu] chi[q,mu] zeta[mu,nu] chi[r,nu] chi[s,nu].

    Conventions used here:
    - chi has shape (n_orb, M), where n_orb = N/2 in the paper's spatial-orbital notation.
    - zeta has shape (M, M) and is usually symmetric.
    """

    chi: Array
    zeta: Array

    def __post_init__(self) -> None:
        self.chi = np.asarray(self.chi, dtype=float)
        self.zeta = np.asarray(self.zeta, dtype=float)

        if self.chi.ndim != 2:
            raise ValueError("chi must have shape (n_orb, M)")
        if self.zeta.ndim != 2:
            raise ValueError("zeta must have shape (M, M)")
        if self.chi.shape[1] != self.zeta.shape[0] or self.zeta.shape[0] != self.zeta.shape[1]:
            raise ValueError(
                f"Incompatible shapes: chi.shape={self.chi.shape}, zeta.shape={self.zeta.shape}"
            )

    @property
    def n_orb(self) -> int:
        return self.chi.shape[0]

    @property
    def rank(self) -> int:
        return self.chi.shape[1]

    def normalize_columns_paper_convention(self, eps: float = 1e-14) -> "THCFactors":
        """
        Normalize each chi[:, mu] to unit 2-norm and absorb the scale into zeta.

        If chi[:,mu] = a_mu * chi_norm[:,mu], then because chi[:,mu] appears twice in G
        for a given mu, the correct rescaling is

            zeta_new[mu,nu] = zeta[mu,nu] * a_mu^2 * a_nu^2.

        This matches the paper's convention that one can take each chi^(mu) normalized
        without loss of generality.
        """
        norms = np.linalg.norm(self.chi, axis=0)
        if np.any(norms < eps):
            raise ValueError("At least one THC column has near-zero norm; cannot normalize safely.")

        chi_new = self.chi / norms[None, :]
        scale = norms**2
        zeta_new = self.zeta * scale[:, None] * scale[None, :]
        return THCFactors(chi=chi_new, zeta=zeta_new)

    def compute_G(self) -> Array:
        """
        Compute the THC-approximated two-electron tensor G[p,q,r,s].

        Returns
        -------
        G : ndarray, shape (n_orb, n_orb, n_orb, n_orb)
        """
        # G[p,q,r,s] = sum_{mu,nu} chi[p,mu] chi[q,mu] zeta[mu,nu] chi[r,nu] chi[s,nu]
        return np.einsum("pm,qm,mn,rn,sn->pqrs", self.chi, self.chi, self.zeta, self.chi, self.chi, optimize=True)

    def compute_G_element(self, p: int, q: int, r: int, s: int) -> float:
        """Compute a single tensor element G[p,q,r,s]."""
        return float(np.einsum("m,m,mn,n,n->", self.chi[p], self.chi[q], self.zeta, self.chi[r], self.chi[s], optimize=True))

    def lambda_thc_direct(self) -> float:
        """
        Compute the direct-THC lambda used in Appendix E / Eq. (9):

            lambda_THC = sum_{p,q,r,s} sum_{mu,nu}
                         | chi[p,mu] chi[q,mu] zeta[mu,nu] chi[r,nu] chi[s,nu] |.

        Because the absolute value acts termwise, this can be factorized efficiently as

            sum_{mu,nu} |zeta[mu,nu]| (sum_p |chi[p,mu]|)^2 (sum_r |chi[r,nu]|)^2.
        """
        l1_cols = np.sum(np.abs(self.chi), axis=0)
        weights = l1_cols**2
        return float(np.sum(np.abs(self.zeta) * weights[:, None] * weights[None, :]))

    def lambda_zeta_nonorthogonal(self, assume_normalized: bool = False) -> float:
        """
        Compute lambda_zeta = (1/2) sum_{mu,nu} |zeta[mu,nu]| from Eq. (13).

        Important:
        - Eq. (13) assumes the paper's convention that each chi^(mu) is normalized.
        - If your chi columns are not normalized, set assume_normalized=False and the
          function will first absorb norms into zeta before evaluating lambda_zeta.
        """
        factors = self if assume_normalized else self.normalize_columns_paper_convention()
        return 0.5 * float(np.sum(np.abs(factors.zeta)))


def thc_loss(V: Array, factors: THCFactors) -> float:
    """L2 objective ||V - G||_F^2 from Eq. (6)."""
    V = np.asarray(V, dtype=float)
    G = factors.compute_G()
    if V.shape != G.shape:
        raise ValueError(f"Shape mismatch: V.shape={V.shape}, G.shape={G.shape}")
    diff = V - G
    return float(np.vdot(diff, diff).real)


def fit_zeta_least_squares(V: Array, chi: Array) -> Array:
    """
    Given V[p,q,r,s] and a fixed chi[p,mu], solve for zeta[mu,nu] by least squares.

    This uses the fact that the THC model is linear in zeta once chi is fixed.
    It is often the cleanest subroutine inside an alternating/minimize-over-chi scheme.
    """
    V = np.asarray(V, dtype=float)
    chi = np.asarray(chi, dtype=float)

    if V.ndim != 4:
        raise ValueError("V must have shape (n_orb, n_orb, n_orb, n_orb)")
    n_orb, n_orb2, n_orb3, n_orb4 = V.shape
    if not (n_orb == n_orb2 == n_orb3 == n_orb4):
        raise ValueError("V must be a square 4-index tensor with equal dimensions")
    if chi.shape[0] != n_orb:
        raise ValueError("chi.shape[0] must match V.shape[0]")

    M = chi.shape[1]
    X = np.einsum("pm,qm->pqm", chi, chi, optimize=True).reshape(n_orb * n_orb, M)
    Vmat = V.reshape(n_orb * n_orb, n_orb * n_orb)

    # Solve Vmat ≈ X zeta X^T in Frobenius norm.
    # If X has full column rank, zeta = pinv(X) V pinv(X)^T.
    X_pinv = np.linalg.pinv(X)
    zeta = X_pinv @ Vmat @ X_pinv.T
    # Symmetrize because zeta is assumed symmetric in the paper.
    zeta = 0.5 * (zeta + zeta.T)
    return zeta


def initial_random_thc(V: Array, rank_M: int, seed: Optional[int] = None) -> THCFactors:
    """
    Produce a random initial guess for THC factors.

    This is only an initializer. The paper emphasizes that finding good THC factors is a
    difficult nonlinear optimization problem and used L-BFGS-B plus AdaGrad for robust
    convergence.
    """
    V = np.asarray(V, dtype=float)
    if V.ndim != 4 or len(set(V.shape)) != 1:
        raise ValueError("V must have shape (n_orb, n_orb, n_orb, n_orb)")

    rng = np.random.default_rng(seed)
    n_orb = V.shape[0]
    chi = rng.normal(size=(n_orb, rank_M))
    zeta = fit_zeta_least_squares(V, chi)
    return THCFactors(chi=chi, zeta=zeta)


def refine_zeta_only(V: Array, factors: THCFactors) -> THCFactors:
    """Given current chi, recompute the optimal least-squares zeta."""
    zeta = fit_zeta_least_squares(V, factors.chi)
    return THCFactors(chi=factors.chi.copy(), zeta=zeta)


def build_nonorthogonal_prepare_terms(
    t: Sequence[float],
    zeta: Array,
) -> Dict[str, object]:
    """
    Build the coefficient data for the main PREPARE state in Eq. (27):

        |psi> = 1/sqrt(lambda) |+>|+> [ sum_l sqrt(|t_l|) |l, M+1>
                                      + (1/sqrt(2)) sum_{mu,nu} sqrt(|zeta_{mu,nu}|) |mu,nu> ]

    Returns a compact classical description instead of an exponentially large statevector.

    Parameters
    ----------
    t : array_like, shape (n_orb,)
        One-body coefficients t_l appearing in Eq. (26).
    zeta : ndarray, shape (M, M)
        Two-body THC matrix for the nonorthogonal representation.

    Returns
    -------
    dict with keys:
      - 'lambda_total'
      - 'one_body_terms': list of (label, amplitude, sign)
      - 'two_body_terms': list of (label, amplitude, sign)
    """
    t = np.asarray(t, dtype=float)
    zeta = np.asarray(zeta, dtype=float)
    if zeta.ndim != 2 or zeta.shape[0] != zeta.shape[1]:
        raise ValueError("zeta must have shape (M, M)")

    lambda_total = float(np.sum(np.abs(t)) + 0.5 * np.sum(np.abs(zeta)))
    if lambda_total <= 0:
        raise ValueError("lambda_total must be positive")

    M = zeta.shape[0]
    one_body_terms: List[Tuple[Tuple[int, int], float, int]] = []
    two_body_terms: List[Tuple[Tuple[int, int], float, int]] = []

    # Labels use the paper's convention: (ell, M+1) flags a one-body term.
    for ell, coeff in enumerate(t):
        if coeff == 0.0:
            continue
        amp = np.sqrt(abs(coeff) / lambda_total)
        sign = 1 if coeff >= 0 else -1
        one_body_terms.append(((ell, M), amp, sign))  # Python index M corresponds to paper label M+1.

    for mu in range(M):
        for nu in range(M):
            coeff = zeta[mu, nu]
            if coeff == 0.0:
                continue
            amp = np.sqrt(abs(coeff) / (2.0 * lambda_total))
            sign = 1 if coeff >= 0 else -1
            two_body_terms.append(((mu, nu), amp, sign))

    return {
        "lambda_total": lambda_total,
        "one_body_terms": one_body_terms,
        "two_body_terms": two_body_terms,
    }


def build_direct_thc_two_body_distribution(factors: THCFactors) -> Dict[str, object]:
    """
    Build a compact classical description of the factorized two-body state in Appendix E,
    Eqs. (E4)-(E5).

    The full state is huge, so this routine returns the factored probability data:
      1) a distribution over (mu, nu) proportional to |zeta_{mu,nu}|,
      2) conditional distributions over p for each mu proportional to |chi[p,mu]|,
      3) conditional distributions over r for each nu proportional to |chi[r,nu]|,
    together with the direct-THC lambda from Eq. (9).
    """
    chi = factors.chi
    zeta = factors.zeta
    M = factors.rank
    n_orb = factors.n_orb

    lambda_direct = factors.lambda_thc_direct()
    if lambda_direct <= 0:
        raise ValueError("Direct THC lambda must be positive")

    # Pair weights for the outer (mu,nu) registers in the factorized state.
    mu_nu_weights = np.abs(zeta)
    mu_nu_probs = mu_nu_weights / np.sum(mu_nu_weights)

    # Conditional distributions over orbital labels.
    chi_abs = np.abs(chi)
    chi_probs = np.zeros_like(chi_abs)
    for mu in range(M):
        denom = np.sum(chi_abs[:, mu])
        if denom > 0:
            chi_probs[:, mu] = chi_abs[:, mu] / denom

    return {
        "lambda_direct": lambda_direct,
        "mu_nu_probs": mu_nu_probs,
        "chi_probs_given_mu": chi_probs,
        "sign_zeta": np.sign(zeta).astype(int),
        "sign_chi": np.sign(chi).astype(int),
        "n_orb": n_orb,
        "rank_M": M,
    }


def dense_direct_thc_two_body_amplitudes(
    factors: THCFactors,
    max_dimension: int = 12,
) -> Array:
    """
    Construct the dense amplitude tensor

        A[mu,nu,p,q,r,s] = sqrt(|chi[p,mu] chi[q,mu] zeta[mu,nu] chi[r,nu] chi[s,nu]| / (2 lambda_direct))

    corresponding to the magnitude-only two-body part of Eq. (E4).

    This object scales as O(M^2 n_orb^4), so the function refuses large instances by default.
    Use only for toy examples / sanity checks.
    """
    if factors.n_orb > max_dimension or factors.rank > max_dimension:
        raise ValueError(
            f"Refusing dense tensor with n_orb={factors.n_orb}, M={factors.rank}. "
            f"Increase max_dimension only for toy problems."
        )

    lambda_direct = factors.lambda_thc_direct()
    A = np.einsum(
        "pm,qm,mn,rn,sn->mnpqrs",
        np.sqrt(np.abs(factors.chi)),
        np.sqrt(np.abs(factors.chi)),
        np.sqrt(np.abs(factors.zeta) / (2.0 * lambda_direct)),
        np.sqrt(np.abs(factors.chi)),
        np.sqrt(np.abs(factors.chi)),
        optimize=True,
    )
    return A


if __name__ == "__main__":
    # Minimal self-test / demo with a tiny random example.
    rng = np.random.default_rng(7)
    n_orb = 4
    M = 3

    chi = rng.normal(size=(n_orb, M))
    zeta = rng.normal(size=(M, M))
    zeta = 0.5 * (zeta + zeta.T)
    factors = THCFactors(chi=chi, zeta=zeta)

    G = factors.compute_G()
    print("G shape:", G.shape)
    print("G[0,1,2,3] =", factors.compute_G_element(0, 1, 2, 3))
    print("lambda_THC (direct Eq. 9) =", factors.lambda_thc_direct())
    print("lambda_zeta (Eq. 13, with normalized chi convention) =", factors.lambda_zeta_nonorthogonal())

    t = rng.normal(size=n_orb)
    prep = build_nonorthogonal_prepare_terms(t=t, zeta=factors.normalize_columns_paper_convention().zeta)
    print("lambda_total (Eq. 27) =", prep["lambda_total"])
    print("number of one-body terms =", len(prep["one_body_terms"]))
    print("number of two-body terms =", len(prep["two_body_terms"]))
