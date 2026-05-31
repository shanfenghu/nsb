from __future__ import annotations

"""
Experiment B: Truncation-depth (K) sensitivity / ablation study.

This script measures how posterior summaries change as the offspring PMF is truncated
to a smaller K. Two truncation modes are reported:
- "defective": truncate without renormalizing (captures mass leakage)
- "renormalized": truncate and renormalize (isolates shape changes)

Outputs:
- An aggregated CSV with mean ± std across multiple training seeds for learned laws.
- Optional deterministic summaries for synthetic heavy-tail distributions.
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
    maybe_load_cached_pmf,
    posterior_z_given_c,
    save_cached_pmf,
    set_global_seed,
    tv_distance,
    train_nsb_on_outbreaktrees,
)


def parse_int_list(csv: str) -> list[int]:
    """
    Parse a comma-separated list of ints.
    """
    return [int(x.strip()) for x in csv.split(",") if x.strip()]


def heavy_tail_nb_pmf(*, k_max: int, r: float, p: float) -> np.ndarray:
    """
    A quick heavy-tail synthetic PMF: Negative Binomial (n=r, p=p) in scipy parameterization.
    Implemented directly to avoid extra dependencies.
    """
    from scipy.special import gammaln  # local import

    k = np.arange(k_max + 1, dtype=float)
    log_pmf = gammaln(k + r) - gammaln(k + 1) - gammaln(r) + r * np.log(1 - p) + k * np.log(p)
    pmf = np.exp(log_pmf)
    pmf_sum = pmf.sum()
    return pmf / pmf_sum


def power_law_pmf(*, k_max: int, alpha: float) -> np.ndarray:
    """
    Discrete power-law over k=0..k_max:
      p(k) ∝ (k+1)^(-alpha)
    """
    if alpha <= 1.0:
        raise ValueError("alpha should be > 1 for a normalizable discrete power-law tail.")
    k = np.arange(k_max + 1, dtype=float)
    w = np.power(k + 1.0, -alpha)
    w_sum = float(w.sum())
    return w / w_sum


def summarize_k_grid(
    *,
    p_ref: np.ndarray,
    n: int,
    Ks: list[int],
    prior: str,
) -> list[dict]:
    """
    For each K, compute posterior stability relative to largest K for:
      (a) defective truncation (no renorm)
      (b) renormalized truncation
    """
    K_ref = max(Ks)
    p_ref_slice = p_ref[:K_ref].copy()

    post_ref_def = posterior_z_given_c(p_ref_slice, n=n, prior_type=prior)
    post_ref_norm = posterior_z_given_c(p_ref_slice / p_ref_slice.sum(), n=n, prior_type=prior)

    rows: list[dict] = []
    for K in Ks:
        pK = p_ref[:K].copy()
        gap = float(1.0 - pK.sum())

        post_def = posterior_z_given_c(pK, n=n, prior_type=prior)
        post_norm = posterior_z_given_c(pK / pK.sum(), n=n, prior_type=prior)

        z = np.arange(1, n + 1, dtype=float)
        ez_def = float(np.sum(z * post_def))
        ez_norm = float(np.sum(z * post_norm))
        # Entropy in nats (log base e); stable for small probabilities.
        eps = 1e-12
        h_def = float(-np.sum(post_def * np.log(post_def + eps)))
        h_norm = float(-np.sum(post_norm * np.log(post_norm + eps)))

        rows.append(
            {
                "K": K,
                "gap": gap,
                "TV_def_to_ref": tv_distance(post_def, post_ref_def),
                "MAP_def": int(np.argmax(post_def) + 1),
                "EZ_def": ez_def,
                "H_def": h_def,
                "TV_norm_to_ref": tv_distance(post_norm, post_ref_norm),
                "MAP_norm": int(np.argmax(post_norm) + 1),
                "EZ_norm": ez_norm,
                "H_norm": h_norm,
            }
        )
    return rows


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n=== {title} ===")
    print("K\tgap\tTV_def\tMAP_def\tEZ_def\tH_def\tTV_norm\tMAP_norm\tEZ_norm\tH_norm")
    for r in rows:
        print(
            f"{r['K']}\t{r['gap']:.6g}\t{r['TV_def_to_ref']:.6g}\t{r['MAP_def']}\t"
            f"{r['EZ_def']:.6g}\t{r['H_def']:.6g}\t"
            f"{r['TV_norm_to_ref']:.6g}\t{r['MAP_norm']}\t{r['EZ_norm']:.6g}\t{r['H_norm']:.6g}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=41)
    ap.add_argument("--Ks", type=str, default="25,50,100,150,300,500")
    ap.add_argument("--K_ref_train", type=int, default=1000)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=10, help="Number of independent NSB training seeds for B1.")
    ap.add_argument("--prior", type=str, default="flat", choices=["flat", "community", "clustered"])
    ap.add_argument("--data_path", type=str, default="data/outbreaktrees_sars_mers_counts.csv")
    ap.add_argument("--pmf_cache", type=str, default="results/rebuttal/cache/outbreaktrees_pmf.npy")
    ap.add_argument(
        "--summary_csv",
        type=str,
        default="results/rebuttal/k_ablation/rebuttal_k_ablation_summary.csv",
        help="Where to write aggregated CSV summary.",
    )
    ap.add_argument("--include_synthetic_heavy_tail", action="store_true")
    ap.add_argument("--synthetic_k_max", type=int, default=2000)
    ap.add_argument("--synthetic_r", type=float, default=0.3)
    ap.add_argument("--synthetic_p", type=float, default=0.85)
    ap.add_argument("--include_powerlaw", action="store_true")
    ap.add_argument("--powerlaw_alpha", type=float, default=1.5)
    ap.add_argument("--powerlaw_k_max", type=int, default=5000)
    args = ap.parse_args()

    if args.n <= 0:
        raise ValueError("n must be positive.")
    if args.runs <= 0:
        raise ValueError("runs must be positive.")

    Ks = parse_int_list(args.Ks)
    if not Ks:
        raise ValueError("Ks must contain at least one K.")
    if min(Ks) <= 2:
        raise ValueError("Ks should be >= 3.")

    set_global_seed(args.seed)
    data_path = (REPO_ROOT / args.data_path).resolve()
    pmf_cache = (REPO_ROOT / args.pmf_cache).resolve() if args.pmf_cache else None
    summary_csv = (REPO_ROOT / args.summary_csv).resolve() if args.summary_csv else None
    if summary_csv is not None:
        ensure_parent_dir(summary_csv)

    # --- B1: learned outbreaktrees law across multiple training seeds ---
    per_run_rows = []
    for run_idx in range(args.runs):
        train_seed = args.seed + run_idx

        # Use a seed-specific cache file to avoid retraining if re-run.
        if pmf_cache is not None:
            seed_cache = pmf_cache.with_name(f"{pmf_cache.stem}_seed{train_seed}{pmf_cache.suffix}")
        else:
            seed_cache = None

        p_cached = maybe_load_cached_pmf(seed_cache)
        if p_cached is not None and len(p_cached) >= args.K_ref_train + 1:
            p_ref = p_cached[: args.K_ref_train + 1].copy()
            meta = {"pmf_loaded_from_cache": True, "pmf_cache": str(seed_cache), "pmf_seed": train_seed}
        else:
            p_ref, meta = train_nsb_on_outbreaktrees(
                data_path=data_path,
                k_max=args.K_ref_train,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                seed=train_seed,
            )
            if seed_cache is not None:
                save_cached_pmf(seed_cache, p_ref)
            meta["pmf_loaded_from_cache"] = False
            meta["pmf_cache"] = str(seed_cache) if seed_cache is not None else None
            meta["pmf_seed"] = train_seed

        rows = summarize_k_grid(p_ref=p_ref, n=args.n, Ks=Ks, prior=args.prior)
        for r in rows:
            per_run_rows.append(
                {
                    "dataset": "outbreaktrees",
                    "run_idx": run_idx,
                    "train_seed": train_seed,
                    "n": args.n,
                    "K": r["K"],
                    "gap": r["gap"],
                    "TV_def_to_ref": r["TV_def_to_ref"],
                    "MAP_def": r["MAP_def"],
                    "EZ_def": r["EZ_def"],
                    "H_def": r["H_def"],
                    "TV_norm_to_ref": r["TV_norm_to_ref"],
                    "MAP_norm": r["MAP_norm"],
                    "EZ_norm": r["EZ_norm"],
                    "H_norm": r["H_norm"],
                }
            )

    df_runs = pd.DataFrame(per_run_rows)
    # Aggregate mean/std + MAP stability rate across training seeds.
    agg_rows = []
    for K in Ks:
        dK = df_runs[df_runs["K"] == K]
        agg_rows.append(
            {
                "dataset": "outbreaktrees",
                "n": args.n,
                "K": K,
                "runs": args.runs,
                "gap_mean": dK["gap"].mean(),
                "gap_std": dK["gap"].std(ddof=1) if len(dK) > 1 else 0.0,
                "TV_def_mean": dK["TV_def_to_ref"].mean(),
                "TV_def_std": dK["TV_def_to_ref"].std(ddof=1) if len(dK) > 1 else 0.0,
                "EZ_def_mean": dK["EZ_def"].mean(),
                "EZ_def_std": dK["EZ_def"].std(ddof=1) if len(dK) > 1 else 0.0,
                "H_def_mean": dK["H_def"].mean(),
                "H_def_std": dK["H_def"].std(ddof=1) if len(dK) > 1 else 0.0,
                "TV_norm_mean": dK["TV_norm_to_ref"].mean(),
                "TV_norm_std": dK["TV_norm_to_ref"].std(ddof=1) if len(dK) > 1 else 0.0,
                "EZ_norm_mean": dK["EZ_norm"].mean(),
                "EZ_norm_std": dK["EZ_norm"].std(ddof=1) if len(dK) > 1 else 0.0,
                "H_norm_mean": dK["H_norm"].mean(),
                "H_norm_std": dK["H_norm"].std(ddof=1) if len(dK) > 1 else 0.0,
                "MAP_def_mode": int(dK["MAP_def"].mode().iloc[0]),
                "MAP_def_stability_rate": float((dK["MAP_def"] == int(dK["MAP_def"].mode().iloc[0])).mean()),
                "MAP_norm_mode": int(dK["MAP_norm"].mode().iloc[0]),
                "MAP_norm_stability_rate": float((dK["MAP_norm"] == int(dK["MAP_norm"].mode().iloc[0])).mean()),
            }
        )
    df_agg = pd.DataFrame(agg_rows)
    print("\n=== Rebuttal Experiment B1 (multi-run): K-ablation on learned outbreaktrees law ===")
    print(f"n={args.n}  Ks={Ks}  K_ref_train={args.K_ref_train}  runs={args.runs}  seed_start={args.seed}")
    # Print a compact aggregate view for quick copy/paste.
    print("K\tgap_mean\tgap_std\tTV_def_mean\tTV_def_std\tEZ_def_mean\tEZ_def_std\tH_def_mean\tH_def_std\tTV_norm_mean\tTV_norm_std\tEZ_norm_mean\tEZ_norm_std\tH_norm_mean\tH_norm_std\tMAP_def_mode\tMAP_def_stab\tMAP_norm_mode\tMAP_norm_stab")
    for _, row in df_agg[df_agg["dataset"] == "outbreaktrees"].iterrows():
        print(
            f"{int(row['K'])}\t{row['gap_mean']:.6g}\t{row['gap_std']:.6g}\t"
            f"{row['TV_def_mean']:.6g}\t{row['TV_def_std']:.6g}\t"
            f"{row['EZ_def_mean']:.6g}\t{row['EZ_def_std']:.6g}\t"
            f"{row['H_def_mean']:.6g}\t{row['H_def_std']:.6g}\t"
            f"{row['TV_norm_mean']:.6g}\t{row['TV_norm_std']:.6g}\t"
            f"{row['EZ_norm_mean']:.6g}\t{row['EZ_norm_std']:.6g}\t"
            f"{row['H_norm_mean']:.6g}\t{row['H_norm_std']:.6g}\t"
            f"{int(row['MAP_def_mode'])}\t{row['MAP_def_stability_rate']:.3f}\t"
            f"{int(row['MAP_norm_mode'])}\t{row['MAP_norm_stability_rate']:.3f}"
        )
    if summary_csv is not None:
        print(f"wrote_csv={summary_csv}")

    if args.include_synthetic_heavy_tail:
        p_syn = heavy_tail_nb_pmf(k_max=args.synthetic_k_max, r=args.synthetic_r, p=args.synthetic_p)
        rows_syn = summarize_k_grid(p_ref=p_syn, n=args.n, Ks=Ks, prior=args.prior)
        print_table(
            f"Rebuttal Experiment B2: K-ablation on synthetic heavy-tail NB "
            f"(r={args.synthetic_r}, p={args.synthetic_p}, n={args.n})",
            rows_syn,
        )

        # Add deterministic synthetic NB summary rows (std=0).
        for r in rows_syn:
            df_agg = pd.concat(
                [
                    df_agg,
                    pd.DataFrame(
                        [
                            {
                                "dataset": f"synthetic_nb_r{args.synthetic_r}_p{args.synthetic_p}",
                                "n": args.n,
                                "K": r["K"],
                                "runs": 1,
                                "gap_mean": r["gap"],
                                "gap_std": 0.0,
                                "TV_def_mean": r["TV_def_to_ref"],
                                "TV_def_std": 0.0,
                                "EZ_def_mean": r["EZ_def"],
                                "EZ_def_std": 0.0,
                                "H_def_mean": r["H_def"],
                                "H_def_std": 0.0,
                                "TV_norm_mean": r["TV_norm_to_ref"],
                                "TV_norm_std": 0.0,
                                "EZ_norm_mean": r["EZ_norm"],
                                "EZ_norm_std": 0.0,
                                "H_norm_mean": r["H_norm"],
                                "H_norm_std": 0.0,
                                "MAP_def_mode": r["MAP_def"],
                                "MAP_def_stability_rate": 1.0,
                                "MAP_norm_mode": r["MAP_norm"],
                                "MAP_norm_stability_rate": 1.0,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    if args.include_powerlaw:
        p_pl = power_law_pmf(k_max=args.powerlaw_k_max, alpha=args.powerlaw_alpha)
        rows_pl = summarize_k_grid(p_ref=p_pl, n=args.n, Ks=Ks, prior=args.prior)
        print_table(
            f"Rebuttal Experiment B3: K-ablation on synthetic power-law "
            f"(alpha={args.powerlaw_alpha}, n={args.n})",
            rows_pl,
        )
        for r in rows_pl:
            df_agg = pd.concat(
                [
                    df_agg,
                    pd.DataFrame(
                        [
                            {
                                "dataset": f"synthetic_powerlaw_alpha{args.powerlaw_alpha}",
                                "n": args.n,
                                "K": r["K"],
                                "runs": 1,
                                "gap_mean": r["gap"],
                                "gap_std": 0.0,
                                "TV_def_mean": r["TV_def_to_ref"],
                                "TV_def_std": 0.0,
                                "EZ_def_mean": r["EZ_def"],
                                "EZ_def_std": 0.0,
                                "H_def_mean": r["H_def"],
                                "H_def_std": 0.0,
                                "TV_norm_mean": r["TV_norm_to_ref"],
                                "TV_norm_std": 0.0,
                                "EZ_norm_mean": r["EZ_norm"],
                                "EZ_norm_std": 0.0,
                                "H_norm_mean": r["H_norm"],
                                "H_norm_std": 0.0,
                                "MAP_def_mode": r["MAP_def"],
                                "MAP_def_stability_rate": 1.0,
                                "MAP_norm_mode": r["MAP_norm"],
                                "MAP_norm_stability_rate": 1.0,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    if summary_csv is not None:
        df_agg.to_csv(summary_csv, index=False)
    # Metadata JSON output is disabled by default to support anonymous review.


if __name__ == "__main__":
    main()

