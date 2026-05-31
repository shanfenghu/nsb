from __future__ import annotations

"""
Experiment C: Stress tests for model-mismatch and observation noise.

This script produces two sweeps on synthetic offspring distributions:
- Piecewise mean shift: compare a piecewise law vs. an averaged (stationary) law.
- Binomial thinning: demonstrate the effect of underreporting (observation rate q_obs).

Outputs:
- A sweep CSV (long format).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo-root imports work regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from experiments.rebuttal.rebuttal_utils import (  # noqa: E402
    ensure_parent_dir,
    kl_divergence,
    piecewise_likelihood_surface,
    posterior_z_given_c,
    set_global_seed,
    tv_distance,
)


def nb_pmf(*, k_max: int, r: float, p: float) -> np.ndarray:
    """
    Negative Binomial PMF (n=r, p=p) on k=0..k_max, normalized.
    This is used only for the stress test (controlled mean shift).
    """
    from scipy.special import gammaln  # local import

    k = np.arange(k_max + 1, dtype=float)
    log_pmf = gammaln(k + r) - gammaln(k + 1) - gammaln(r) + r * np.log(1 - p) + k * np.log(p)
    pmf = np.exp(log_pmf)
    pmf_sum = pmf.sum()
    return pmf / pmf_sum


def find_nb_p_for_target_mean(*, r: float, target_mean: float) -> float:
    """
    Solve for NB parameter p (in this parameterization) given target mean.
    """
    # Solve target_mean = r*(1-p)/p  =>  p = r / (r + target_mean)
    if target_mean <= 0:
        raise ValueError("target_mean must be positive.")
    return float(r / (r + target_mean))


def thin_offspring_pmf_binomial(p: np.ndarray, q_obs: float) -> np.ndarray:
    """
    Binomial thinning of offspring counts: each child independently observed with prob q_obs.
    p_thin(j) = sum_{k>=j} p(k) * Binom(j | k, q_obs)
    """
    if not (0.0 < q_obs <= 1.0):
        raise ValueError("q_obs must be in (0, 1].")
    p = np.asarray(p, dtype=float)
    K = len(p) - 1
    from scipy.stats import binom  # local import

    p_thin = np.zeros_like(p)
    js = np.arange(K + 1)
    for k in range(K + 1):
        if p[k] <= 0:
            continue
        pmf_j = binom.pmf(js[: k + 1], n=k, p=q_obs)
        p_thin[: k + 1] += p[k] * pmf_j
    s = p_thin.sum()
    if s <= 0:
        raise RuntimeError("Thinned PMF has non-positive mass.")
    return p_thin / s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n1", type=int, default=25)
    ap.add_argument("--K", type=int, default=400)
    ap.add_argument("--r", type=float, default=0.8)
    ap.add_argument("--R0_1", type=float, default=2.5)
    ap.add_argument("--R0_drop", type=float, default=0.2)  # used if no sweep list provided
    ap.add_argument("--prior", type=str, default="flat", choices=["flat", "community", "clustered"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--q_obs", type=float, default=0.8)  # used if no sweep list provided
    ap.add_argument("--demo_underreport_n", type=int, default=41)
    ap.add_argument(
        "--drops",
        type=str,
        default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4",
        help="Comma-separated list of relative drops (e.g., 0.2 means 20%%).",
    )
    ap.add_argument(
        "--q_obs_list",
        type=str,
        default="0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6",
        help="Comma-separated list of observation rates for thinning sweep.",
    )
    ap.add_argument(
        "--summary_csv",
        type=str,
        default="results/rebuttal/shift_noise/rebuttal_shift_noise_sweep_summary.csv",
        help="Where to write the sweep CSV summary.",
    )
    args = ap.parse_args()

    if args.n <= 0:
        raise ValueError("n must be positive.")
    if not (0 < args.n1 < args.n):
        raise ValueError("require 0 < n1 < n")
    n2 = args.n - args.n1
    if args.K <= 1:
        raise ValueError("K must be > 1.")

    set_global_seed(args.seed)

    def _parse_float_list(s: str) -> list[float]:
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    drops = _parse_float_list(args.drops)
    q_list = _parse_float_list(args.q_obs_list)
    for d in drops:
        if not (0.0 < d < 1.0):
            raise ValueError(f"drop must be in (0,1), got {d}")
    for q in q_list:
        if not (0.0 < q <= 1.0):
            raise ValueError(f"q_obs must be in (0,1], got {q}")

    summary_csv = (REPO_ROOT / args.summary_csv).resolve() if args.summary_csv else None
    if summary_csv is not None:
        ensure_parent_dir(summary_csv)

    # --- Experiment C sweep: piecewise shift severity ---
    rows = []
    for drop in drops:
        R0_1 = args.R0_1
        R0_2 = (1.0 - drop) * R0_1
        p1 = nb_pmf(k_max=args.K, r=args.r, p=find_nb_p_for_target_mean(r=args.r, target_mean=R0_1))
        p2 = nb_pmf(k_max=args.K, r=args.r, p=find_nb_p_for_target_mean(r=args.r, target_mean=R0_2))

        lik_piece = piecewise_likelihood_surface(p1, p2, n1=args.n1, n2=n2, n=args.n)
        prior = np.ones(args.n) / args.n
        post_piece = lik_piece * prior
        post_piece = post_piece / post_piece.sum()

        p_avg = (args.n1 / args.n) * p1 + (n2 / args.n) * p2
        post_avg = posterior_z_given_c(p_avg, n=args.n, prior_type="flat")

        tv_c = tv_distance(post_piece, post_avg)
        kl_c = kl_divergence(post_piece, post_avg)
        map_piece = int(np.argmax(post_piece) + 1)
        map_avg = int(np.argmax(post_avg) + 1)

        rows.append(
            {
                "sweep": "R0_drop",
                "n": args.n,
                "n1": args.n1,
                "n2": n2,
                "K": args.K,
                "r": args.r,
                "R0_1": R0_1,
                "R0_2": R0_2,
                "drop": drop,
                "TV_piece_vs_avg": tv_c,
                "KL_piece_vs_avg": kl_c,
                "MAP_piece": map_piece,
                "MAP_avg": map_avg,
                "MAP_changed": map_piece != map_avg,
            }
        )

    # --- Experiment C' sweep: underreporting rates ---
    # Use drop = args.R0_drop for a single representative averaged law.
    drop0 = args.R0_drop
    if not (0.0 < drop0 < 1.0):
        drop0 = 0.2
    R0_1 = args.R0_1
    R0_2 = (1.0 - drop0) * R0_1
    p1 = nb_pmf(k_max=args.K, r=args.r, p=find_nb_p_for_target_mean(r=args.r, target_mean=R0_1))
    p2 = nb_pmf(k_max=args.K, r=args.r, p=find_nb_p_for_target_mean(r=args.r, target_mean=R0_2))
    p_avg0 = (args.n1 / args.n) * p1 + (n2 / args.n) * p2
    post_full0 = posterior_z_given_c(p_avg0, n=args.demo_underreport_n, prior_type="flat")

    for q_obs in q_list:
        p_thin = thin_offspring_pmf_binomial(p_avg0, q_obs=q_obs)
        post_thin = posterior_z_given_c(p_thin, n=args.demo_underreport_n, prior_type="flat")
        tv_u = tv_distance(post_full0, post_thin)
        kl_u = kl_divergence(post_full0, post_thin)
        rows.append(
            {
                "sweep": "q_obs",
                "n": args.demo_underreport_n,
                "n1": args.n1,
                "n2": n2,
                "K": args.K,
                "r": args.r,
                "R0_1": R0_1,
                "R0_2": R0_2,
                "drop": drop0,
                "q_obs": q_obs,
                "TV_full_vs_thin": tv_u,
                "KL_full_vs_thin": kl_u,
            }
        )

    df = pd.DataFrame(rows)
    if summary_csv is not None:
        df.to_csv(summary_csv, index=False)
    # Metadata JSON output is disabled by default to support anonymous review.

    print("\n=== Rebuttal Experiment C sweep: piecewise shift severity ===")
    print(f"n={args.n}  n1={args.n1}  n2={n2}  K={args.K}  r={args.r}  R0_1={args.R0_1}  drops={drops}")
    print("drop\tTV\tKL\tMAP_changed")
    for r in df[df["sweep"] == "R0_drop"].to_dict(orient="records"):
        print(f"{r['drop']:.3g}\t{r['TV_piece_vs_avg']:.6g}\t{r['KL_piece_vs_avg']:.6g}\t{r['MAP_changed']}")

    print("\n=== Rebuttal Experiment C' sweep: underreporting rates ===")
    print(f"n={args.demo_underreport_n}  K={args.K}  q_obs_list={q_list}  (using drop={drop0})")
    print("q_obs\tTV\tKL")
    for r in df[df["sweep"] == "q_obs"].to_dict(orient="records"):
        print(f"{r['q_obs']:.3g}\t{r['TV_full_vs_thin']:.6g}\t{r['KL_full_vs_thin']:.6g}")
    if summary_csv is not None:
        print(f"wrote_csv={summary_csv}")


if __name__ == "__main__":
    main()

