from __future__ import annotations

"""
Aggregate rebuttal experiment outputs into a single, compact summary file.

This is intentionally conservative and schema-tolerant:
- It reads the per-experiment CSVs.
- It extracts key quantitative results (typically from the AGG row).
- It writes a single long-format CSV that is easy to paste into a rebuttal draft.

The goal is not to perfectly preserve every column from every experiment; those remain
in the original CSVs. Instead, this provides a curated "index" of what to cite.
"""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def _add_kv(rows: list[dict[str, Any]], *, experiment: str, key: str, value: Any, std: Any = "", notes: str = "") -> None:
    rows.append(
        {
            "experiment": experiment,
            "key": key,
            "value": value,
            "std": std,
            "notes": notes,
        }
    )


def _summarize_abc(results_dir: Path, rows: list[dict[str, Any]]) -> None:
    abc_dir = results_dir / "abc"
    if not abc_dir.exists():
        return

    for csv_path in sorted(abc_dir.glob("rebuttal_abc*_summary.csv")):
        df = pd.read_csv(csv_path)
        agg = df[df["run_idx"].astype(str) == "AGG"]
        if len(agg) != 1:
            continue
        r = agg.iloc[0].to_dict()
        n = r.get("n", "")
        exp = f"A_abc_exact_vs_sim(n={n})"

        _add_kv(rows, experiment=exp, key="TV_mean", value=r.get("TV"), std=r.get("TV_std"))
        _add_kv(rows, experiment=exp, key="KL_mean", value=r.get("KL"), std=r.get("KL_std"))
        _add_kv(rows, experiment=exp, key="EZ_abs_diff_mean", value=r.get("EZ_abs_diff"), std=r.get("EZ_abs_diff_std"))
        _add_kv(rows, experiment=exp, key="H_abs_diff_mean_nats", value=r.get("H_abs_diff"), std=r.get("H_abs_diff_std"))
        _add_kv(rows, experiment=exp, key="t_exact_ms", value=r.get("t_exact_ms"))
        _add_kv(rows, experiment=exp, key="t_sampler_s_mean", value=r.get("t_sampler_s"), std=r.get("t_sampler_s_std"))


def _summarize_k_ablation(results_dir: Path, rows: list[dict[str, Any]]) -> None:
    exp = "B_k_ablation"
    csv_path = results_dir / "k_ablation" / "rebuttal_k_ablation_summary.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    # Focus on learned law summary (dataset == outbreaktrees).
    d = df[df.get("dataset", "") == "outbreaktrees"].copy()
    if d.empty:
        return

    # Provide a compact subset: smallest K and largest K in the table.
    ks = sorted(d["K"].astype(int).unique().tolist())
    k_small, k_large = ks[0], ks[-1]
    for k in [k_small, k_large]:
        r = d[d["K"].astype(int) == k].iloc[0].to_dict()
        prefix = f"K={k}"
        _add_kv(rows, experiment=exp, key=f"{prefix}/gap_mean", value=r.get("gap_mean"), std=r.get("gap_std"))
        _add_kv(rows, experiment=exp, key=f"{prefix}/TV_def_mean", value=r.get("TV_def_mean"), std=r.get("TV_def_std"))
        _add_kv(rows, experiment=exp, key=f"{prefix}/TV_norm_mean", value=r.get("TV_norm_mean"), std=r.get("TV_norm_std"))
        _add_kv(rows, experiment=exp, key=f"{prefix}/EZ_def_mean", value=r.get("EZ_def_mean"), std=r.get("EZ_def_std"))
        _add_kv(rows, experiment=exp, key=f"{prefix}/H_def_mean_nats", value=r.get("H_def_mean"), std=r.get("H_def_std"))
        _add_kv(rows, experiment=exp, key=f"{prefix}/MAP_def_stability_rate", value=r.get("MAP_def_stability_rate"))
        _add_kv(rows, experiment=exp, key=f"{prefix}/MAP_norm_stability_rate", value=r.get("MAP_norm_stability_rate"))


def _summarize_shift_noise(results_dir: Path, rows: list[dict[str, Any]]) -> None:
    shift_dir = results_dir / "shift_noise"
    if not shift_dir.exists():
        return

    for csv_path in sorted(shift_dir.glob("rebuttal_shift_noise*_sweep_summary.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        # Try to label by key parameters.
        label_bits = []
        try:
            first_shift = df[df["sweep"] == "R0_drop"].iloc[0]
            label_bits.append(f"n={int(first_shift['n'])}")
            label_bits.append(f"n1={int(first_shift['n1'])}")
            label_bits.append(f"K={int(first_shift['K'])}")
            label_bits.append(f"r={float(first_shift['r']):g}")
        except Exception:
            pass
        exp = "C_shift_and_thinning" + (f"({', '.join(label_bits)})" if label_bits else "")

        # Shift sweep summary.
        d_shift = df[df["sweep"] == "R0_drop"].copy()
        if not d_shift.empty:
            tv_max = float(d_shift["TV_piece_vs_avg"].max())
            kl_max = float(d_shift["KL_piece_vs_avg"].max())
            changed_rate = float(d_shift["MAP_changed"].mean())
            _add_kv(rows, experiment=exp, key="shift/TV_max", value=tv_max)
            _add_kv(rows, experiment=exp, key="shift/KL_max", value=kl_max)
            _add_kv(rows, experiment=exp, key="shift/MAP_changed_rate", value=changed_rate)

        # Underreporting sweep summary.
        d_q = df[df["sweep"] == "q_obs"].copy()
        if not d_q.empty:
            tv_max = float(d_q["TV_full_vs_thin"].max())
            kl_max = float(d_q["KL_full_vs_thin"].max())
            q_mid = d_q.iloc[(d_q["q_obs"] - 0.8).abs().argsort()[:1]].iloc[0]
            _add_kv(rows, experiment=exp, key="thinning/TV_max", value=tv_max)
            _add_kv(rows, experiment=exp, key="thinning/KL_max", value=kl_max)
            _add_kv(
                rows,
                experiment=exp,
                key="thinning/TV_at_q≈0.8",
                value=float(q_mid["TV_full_vs_thin"]),
                notes=f"q_obs={float(q_mid['q_obs']):.3g}",
            )
            _add_kv(
                rows,
                experiment=exp,
                key="thinning/KL_at_q≈0.8",
                value=float(q_mid["KL_full_vs_thin"]),
                notes=f"q_obs={float(q_mid['q_obs']):.3g}",
            )


def _summarize_shift_grid(results_dir: Path, rows: list[dict[str, Any]]) -> None:
    exp = "C_shift_grid_search"
    csv_path = results_dir / "shift_noise" / "rebuttal_shift_grid_summary.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    best = df.sort_values("TV_piece_vs_avg", ascending=False).iloc[0].to_dict()
    _add_kv(rows, experiment=exp, key="best/TV", value=best.get("TV_piece_vs_avg"))
    _add_kv(rows, experiment=exp, key="best/KL", value=best.get("KL_piece_vs_avg"))
    _add_kv(rows, experiment=exp, key="best/MAP_changed", value=best.get("MAP_changed"))
    _add_kv(rows, experiment=exp, key="best/n", value=best.get("n"))
    _add_kv(rows, experiment=exp, key="best/n1_frac", value=best.get("n1_frac"))
    _add_kv(rows, experiment=exp, key="best/r", value=best.get("r"))
    _add_kv(rows, experiment=exp, key="best/drop", value=best.get("drop"))

    # Also provide a robustness summary: 95th percentile TV across the grid.
    _add_kv(rows, experiment=exp, key="TV_p95", value=float(df["TV_piece_vs_avg"].quantile(0.95)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results/rebuttal")
    ap.add_argument(
        "--out_csv",
        type=str,
        default="results/rebuttal/rebuttal_aggregate_summary.csv",
        help="Where to write the consolidated summary CSV.",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    results_dir = (repo_root / args.results_dir).resolve()
    out_csv = (repo_root / args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    _summarize_abc(results_dir, rows)
    _summarize_k_ablation(results_dir, rows)
    _summarize_shift_noise(results_dir, rows)
    _summarize_shift_grid(results_dir, rows)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    print(f"wrote_csv={out_csv}")


if __name__ == "__main__":
    main()

