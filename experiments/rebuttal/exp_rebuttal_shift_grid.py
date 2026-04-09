from __future__ import annotations

"""
Non-stationarity grid sweep for piecewise vs averaged offspring laws.

Purpose:
- Produce at least one clearly non-negligible regime where the posterior under a
  piecewise offspring law differs from the posterior under a stationary averaged law.
- Provide a "worst-case over a small grid" number that is hard to argue against.

Output:
- A long-format CSV with TV/KL/MAP statistics for each grid point.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo-root imports work regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from experiments.rebuttal.exp_rebuttal_shift_noise import (  # noqa: E402
    find_nb_p_for_target_mean,
    nb_pmf,
)
from experiments.rebuttal.rebuttal_utils import (  # noqa: E402
    ensure_parent_dir,
    kl_divergence,
    piecewise_likelihood_surface,
    posterior_z_given_c,
    set_global_seed,
    tv_distance,
)


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=1200)
    ap.add_argument("--R0_1", type=float, default=3.0)
    ap.add_argument("--ns", type=str, default="50,100,150")
    ap.add_argument("--n1_fracs", type=str, default="0.1,0.25,0.5,0.75,0.9")
    ap.add_argument("--rs", type=str, default="0.1,0.2,0.5,0.8")
    ap.add_argument("--drops", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7")
    ap.add_argument(
        "--summary_csv",
        type=str,
        default="results/rebuttal/shift_noise/rebuttal_shift_grid_summary.csv",
        help="Where to write the grid-sweep CSV.",
    )
    args = ap.parse_args()

    if args.K <= 1:
        raise ValueError("K must be > 1.")
    if args.R0_1 <= 0:
        raise ValueError("R0_1 must be positive.")

    set_global_seed(args.seed)

    ns = _parse_int_list(args.ns)
    n1_fracs = _parse_float_list(args.n1_fracs)
    rs = _parse_float_list(args.rs)
    drops = _parse_float_list(args.drops)
    if not ns:
        raise ValueError("ns must be non-empty.")
    for f in n1_fracs:
        if not (0.0 < f < 1.0):
            raise ValueError(f"n1_fracs entries must be in (0,1), got {f}")
    for r in rs:
        if r <= 0:
            raise ValueError(f"r must be positive, got {r}")
    for d in drops:
        if not (0.0 < d < 1.0):
            raise ValueError(f"drop must be in (0,1), got {d}")

    summary_csv = (REPO_ROOT / args.summary_csv).resolve() if args.summary_csv else None
    if summary_csv is not None:
        ensure_parent_dir(summary_csv)

    rows: list[dict] = []
    for n in ns:
        for frac in n1_fracs:
            n1 = int(round(frac * n))
            n1 = min(max(n1, 1), n - 1)
            n2 = n - n1
            for r in rs:
                p1 = nb_pmf(k_max=args.K, r=r, p=find_nb_p_for_target_mean(r=r, target_mean=args.R0_1))
                for drop in drops:
                    R0_2 = (1.0 - drop) * args.R0_1
                    p2 = nb_pmf(k_max=args.K, r=r, p=find_nb_p_for_target_mean(r=r, target_mean=R0_2))

                    lik_piece = piecewise_likelihood_surface(p1, p2, n1=n1, n2=n2, n=n)
                    post_piece = lik_piece / max(lik_piece.sum(), 1e-30)

                    p_avg = (n1 / n) * p1 + (n2 / n) * p2
                    post_avg = posterior_z_given_c(p_avg, n=n, prior_type="flat")

                    tv = tv_distance(post_piece, post_avg)
                    kl = kl_divergence(post_piece, post_avg)
                    map_piece = int(np.argmax(post_piece) + 1)
                    map_avg = int(np.argmax(post_avg) + 1)

                    rows.append(
                        {
                            "n": n,
                            "n1": n1,
                            "n2": n2,
                            "n1_frac": n1 / n,
                            "K": args.K,
                            "r": r,
                            "R0_1": args.R0_1,
                            "R0_2": R0_2,
                            "drop": drop,
                            "TV_piece_vs_avg": tv,
                            "KL_piece_vs_avg": kl,
                            "MAP_piece": map_piece,
                            "MAP_avg": map_avg,
                            "MAP_changed": map_piece != map_avg,
                        }
                    )

    df = pd.DataFrame(rows)
    if summary_csv is not None:
        df.to_csv(summary_csv, index=False)
    # Metadata JSON output is disabled by default to support anonymous review.

    # Print the top regimes by TV for quick inspection.
    top = df.sort_values("TV_piece_vs_avg", ascending=False).head(10)
    print("\n=== Non-stationarity grid sweep (top by TV) ===")
    cols = ["n", "n1", "n1_frac", "K", "r", "drop", "TV_piece_vs_avg", "KL_piece_vs_avg", "MAP_changed"]
    print(top[cols].to_string(index=False))
    if summary_csv is not None:
        print(f"wrote_csv={summary_csv}")


if __name__ == "__main__":
    main()

