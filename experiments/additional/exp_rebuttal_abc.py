from __future__ import annotations

"""
Experiment A: Compare an exact posterior P(Z | C=n) computed via the spectral engine
against an accept/reject Galton–Watson simulator (ABC-style baseline) on small n.

Outputs:
- A per-run CSV with Monte Carlo variability across independent sampler seeds.
- An aggregate row with mean ± std for key discrepancies.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo-root imports work regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from experiments.rebuttal.rebuttal_utils import (  # noqa: E402
    ensure_parent_dir,
    kl_divergence,
    maybe_load_cached_pmf,
    posterior_z_given_c,
    save_cached_pmf,
    set_global_seed,
    time_call,
    tv_distance,
    train_nsb_on_outbreaktrees,
)


def _ez_and_entropy(post: np.ndarray) -> tuple[float, float]:
    """
    Returns (E[Z], H) where H is entropy in nats.
    """
    post = np.asarray(post, dtype=float)
    n = len(post)
    z = np.arange(1, n + 1, dtype=float)
    ez = float(np.sum(z * post))
    eps = 1e-12
    h = float(-np.sum(post * np.log(post + eps)))
    return ez, h


def simulate_total_progeny_accept_reject(
    *,
    rng: np.random.Generator,
    p: np.ndarray,
    n: int,
    z: int,
) -> bool:
    """
    Simulate a Galton–Watson process with Z=z founders until extinction or total>n.
    Accept iff total progeny C equals exactly n.

    Early-stops when total>n to avoid superspreading blow-ups.
    """
    total = int(z)
    active = int(z)
    k_vals = np.arange(len(p), dtype=int)

    while active > 0:
        if total > n:
            return False

        offspring = rng.choice(k_vals, size=active, replace=True, p=p)
        new_active = int(offspring.sum())
        total += new_active

        if total > n:
            return False

        active = new_active

    return total == n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--K_ref", type=int, default=400)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prior", type=str, default="flat", choices=["flat", "community", "clustered"])
    ap.add_argument("--N_accept", type=int, default=5000)
    ap.add_argument("--max_trials", type=int, default=5_000_000)
    ap.add_argument("--data_path", type=str, default="data/outbreaktrees_sars_mers_counts.csv")
    ap.add_argument("--pmf_cache", type=str, default="results/rebuttal/cache/outbreaktrees_pmf.npy")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument(
        "--train_once",
        action="store_true",
        help="If set, train/load one PMF (seed=--seed) and only vary the sampler seed across runs.",
    )
    ap.add_argument(
        "--summary_csv",
        type=str,
        default="results/rebuttal/abc/rebuttal_abc_summary.csv",
        help="Where to write the per-run + aggregate CSV summary.",
    )
    args = ap.parse_args()

    if args.n <= 0:
        raise ValueError("n must be positive.")
    if args.K_ref <= 1:
        raise ValueError("K_ref must be > 1.")
    if args.N_accept <= 0:
        raise ValueError("N_accept must be positive.")
    if args.max_trials <= 0:
        raise ValueError("max_trials must be positive.")
    if args.runs <= 0:
        raise ValueError("runs must be positive.")

    set_global_seed(args.seed)

    data_path = (REPO_ROOT / args.data_path).resolve()
    pmf_cache = (REPO_ROOT / args.pmf_cache).resolve() if args.pmf_cache else None
    summary_csv = (REPO_ROOT / args.summary_csv).resolve() if args.summary_csv else None

    if summary_csv is not None:
        ensure_parent_dir(summary_csv)

    def _get_or_train_pmf(*, pmf_seed: int) -> tuple[np.ndarray, dict]:
        if pmf_cache is None:
            p_ref, pmf_meta = train_nsb_on_outbreaktrees(
                data_path=data_path,
                k_max=args.K_ref,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                seed=pmf_seed,
            )
            pmf_meta["pmf_loaded_from_cache"] = False
            pmf_meta["pmf_cache"] = None
            pmf_meta["pmf_seed"] = pmf_seed
            return p_ref, pmf_meta

        # If training per-run, use a seed-specific cache; otherwise use the base path.
        cache_path = pmf_cache
        if not args.train_once:
            cache_path = pmf_cache.with_name(f"{pmf_cache.stem}_seed{pmf_seed}{pmf_cache.suffix}")

        p_cached = maybe_load_cached_pmf(cache_path)
        if p_cached is not None and len(p_cached) >= args.K_ref + 1:
            p_ref = p_cached[: args.K_ref + 1].copy()
            pmf_meta = {"pmf_loaded_from_cache": True, "pmf_cache": str(cache_path), "pmf_seed": pmf_seed}
            return p_ref, pmf_meta

        p_ref, pmf_meta = train_nsb_on_outbreaktrees(
            data_path=data_path,
            k_max=args.K_ref,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=pmf_seed,
        )
        save_cached_pmf(cache_path, p_ref)
        pmf_meta["pmf_loaded_from_cache"] = False
        pmf_meta["pmf_cache"] = str(cache_path)
        pmf_meta["pmf_seed"] = pmf_seed
        return p_ref, pmf_meta

    rows = []
    pmf_meta_first: dict | None = None
    t_exact_ms_first: float | None = None
    ez_exact_first: float | None = None
    h_exact_first: float | None = None
    for run_idx in range(args.runs):
        pmf_seed = args.seed if args.train_once else (args.seed + run_idx)
        sampler_seed = args.seed + run_idx
        rng = np.random.default_rng(sampler_seed)

        p_ref, pmf_meta = _get_or_train_pmf(pmf_seed=pmf_seed)
        p = p_ref / p_ref.sum()
        post_exact, t_exact = time_call(posterior_z_given_c, p, n=args.n, prior_type=args.prior)
        ez_exact, h_exact = _ez_and_entropy(post_exact)
        if pmf_meta_first is None:
            pmf_meta_first = dict(pmf_meta)
            t_exact_ms_first = t_exact.ms
            ez_exact_first = ez_exact
            h_exact_first = h_exact

        z_counts = np.zeros(args.n, dtype=int)
        accepted = 0
        trials = 0
        t0 = time.perf_counter()

        while accepted < args.N_accept and trials < args.max_trials:
            trials += 1
            z = int(rng.integers(1, args.n + 1))
            ok = simulate_total_progeny_accept_reject(rng=rng, p=p, n=args.n, z=z)
            if ok:
                z_counts[z - 1] += 1
                accepted += 1

        t1 = time.perf_counter()
        t_sampler_s = t1 - t0

        if accepted == 0:
            raise RuntimeError(
                f"ABC sampler accepted 0 samples on run_idx={run_idx}, sampler_seed={sampler_seed}. "
                "Increase max_trials or decrease N_accept."
            )

        post_abc = z_counts / z_counts.sum()
        tv = tv_distance(post_exact, post_abc)
        kl = kl_divergence(post_exact, post_abc)
        ez_abc, h_abc = _ez_and_entropy(post_abc)

        rows.append(
            {
                "run_idx": run_idx,
                "sampler_seed": sampler_seed,
                "pmf_seed": pmf_seed,
                "n": args.n,
                "K_ref": args.K_ref,
                "prior": args.prior,
                "N_accept": args.N_accept,
                "max_trials": args.max_trials,
                "accepted": accepted,
                "trials": trials,
                "accept_rate": accepted / trials,
                "TV": tv,
                "KL": kl,
                "EZ_exact": ez_exact,
                "H_exact": h_exact,
                "EZ_abc": ez_abc,
                "H_abc": h_abc,
                "EZ_abs_diff": abs(ez_exact - ez_abc),
                "H_abs_diff": abs(h_exact - h_abc),
                "t_exact_ms": t_exact.ms,
                "t_sampler_s": t_sampler_s,
            }
        )

    df = pd.DataFrame(rows)
    if pmf_meta_first is None:
        raise RuntimeError("Internal error: expected at least one run.")

    agg = {
        "run_idx": "AGG",
        "sampler_seed": "",
        "pmf_seed": "",
        "n": args.n,
        "K_ref": args.K_ref,
        "prior": args.prior,
        "N_accept": args.N_accept,
        "max_trials": args.max_trials,
        "accepted": df["accepted"].mean(),
        "trials": df["trials"].mean(),
        "accept_rate": df["accept_rate"].mean(),
        "TV": df["TV"].mean(),
        "KL": df["KL"].mean(),
        "EZ_exact": float(ez_exact_first),
        "H_exact": float(h_exact_first),
        "EZ_abc": df["EZ_abc"].mean(),
        "H_abc": df["H_abc"].mean(),
        "EZ_abs_diff": df["EZ_abs_diff"].mean(),
        "H_abs_diff": df["H_abs_diff"].mean(),
        "t_exact_ms": float(t_exact_ms_first),
        "t_sampler_s": df["t_sampler_s"].mean(),
        "TV_std": df["TV"].std(ddof=1) if len(df) > 1 else 0.0,
        "KL_std": df["KL"].std(ddof=1) if len(df) > 1 else 0.0,
        "EZ_abc_std": df["EZ_abc"].std(ddof=1) if len(df) > 1 else 0.0,
        "H_abc_std": df["H_abc"].std(ddof=1) if len(df) > 1 else 0.0,
        "EZ_abs_diff_std": df["EZ_abs_diff"].std(ddof=1) if len(df) > 1 else 0.0,
        "H_abs_diff_std": df["H_abs_diff"].std(ddof=1) if len(df) > 1 else 0.0,
        "t_sampler_s_std": df["t_sampler_s"].std(ddof=1) if len(df) > 1 else 0.0,
    }
    df_out = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)

    if summary_csv is not None:
        df_out.to_csv(summary_csv, index=False)
    # Metadata JSON output is disabled by default to support anonymous review.

    print("\n=== Rebuttal Experiment A (multi-run): Exact vs ABC accept/reject ===")
    print(f"n={args.n}  K_ref={args.K_ref}  prior={args.prior}  pmf_seed={args.seed}  runs={args.runs}")
    print(
        f"TV={agg['TV']:.6g} ± {agg['TV_std']:.6g}  "
        f"KL={agg['KL']:.6g} ± {agg['KL_std']:.6g}  "
        f"EZ_abs_diff={agg['EZ_abs_diff']:.6g} ± {agg['EZ_abs_diff_std']:.6g}  "
        f"H_abs_diff={agg['H_abs_diff']:.6g} ± {agg['H_abs_diff_std']:.6g}  "
        f"t_exact={agg['t_exact_ms']:.3f} ms  "
        f"t_sampler={agg['t_sampler_s']:.3f} ± {agg['t_sampler_s_std']:.3f} s"
    )
    if summary_csv is not None:
        print(f"wrote_csv={summary_csv}")


if __name__ == "__main__":
    main()

