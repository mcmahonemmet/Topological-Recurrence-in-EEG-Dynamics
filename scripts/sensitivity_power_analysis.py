#!/usr/bin/env python3
"""
sensitivity_power_analysis.py
=============================

Standalone sensitivity power analysis for the headline REM-versus-Wake K0
contrast.

For each of four cohort specifications, the script computes:

  - Observed paired-t effect size d_z on subject-level (or person-level)
    REM - Wake K0 means.
  - Minimum detectable effect (MDE) at 80% power and alpha = 0.05 (two-sided)
    using both:
      (i)  the closed-form normal approximation
           MDE_dz = (z_{1-alpha/2} + z_{1-beta}) / sqrt(n)
      (ii) the exact non-central-t calculation via
           statsmodels.stats.power.TTestPower.solve_power()
  - The ratio observed_dz / MDE.
  - For a small grid of effect sizes, the achieved power at each n.

Cohort specifications (all REM and W epoch averages from
``outputs/tda_epoch_features_all.csv``, joined with
``outputs/demographics_per_night.csv``):

  per_night            n = 197 paired records (one row per PSG)
  unique_subject       n = 100 paired records (one row per unique person;
                              Telemetry contributes its placebo night)
  cassette_only        Sleep-Cassette only, per-night (n = 153)
  cassette_subjects    Sleep-Cassette unique subjects (n = 78)
  telemetry_only       Sleep-Telemetry, per-night (n = 44)
  telemetry_subjects   Sleep-Telemetry unique subjects (n = 22)

Outputs:
  - outputs/sensitivity_power_analysis.csv   per-specification numbers
  - outputs/sensitivity_power_curves.csv     power-vs-effect-size grid
  - outputs/figures/sensitivity_power_curves.png
                                              power curves at each cohort
                                              size with observed d_z marked

Usage:
  python scripts/sensitivity_power_analysis.py
  python scripts/sensitivity_power_analysis.py --no-figure
  python scripts/sensitivity_power_analysis.py --force
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import math
import numpy as np
import pandas as pd
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
import pipeline as P  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUT_DIR = P.OUT_DIR
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_TABLE  = OUT_DIR / "sensitivity_power_analysis.csv"
OUT_GRID   = OUT_DIR / "sensitivity_power_curves.csv"
OUT_FIG    = FIG_DIR / "sensitivity_power_curves.png"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def banner(s: str):
    bar = "═" * max(len(s) + 4, 60)
    print(f"\n{bar}\n  {s}\n{bar}", flush=True)


def log(s: str):
    print(s, flush=True)


def closed_form_mde(n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Closed-form normal approximation for paired-t MDE."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return (z_a + z_b) / math.sqrt(n)


def nct_mde(n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Exact non-central-t MDE via statsmodels TTestPower."""
    try:
        from statsmodels.stats.power import TTestPower
    except Exception:
        return float("nan")
    try:
        return float(TTestPower().solve_power(
            effect_size=None, nobs=n, alpha=alpha, power=power,
            alternative="two-sided",
        ))
    except Exception:
        return float("nan")


def nct_power(d_z: float, n: int, alpha: float = 0.05) -> float:
    """Achieved power for a given d_z, n at α (two-sided paired-t)."""
    try:
        from statsmodels.stats.power import TTestPower
    except Exception:
        return float("nan")
    try:
        return float(TTestPower().solve_power(
            effect_size=d_z, nobs=n, alpha=alpha, power=None,
            alternative="two-sided",
        ))
    except Exception:
        return float("nan")


def paired_dz(diff: np.ndarray) -> Dict:
    """Compute paired-t statistics on a vector of subject-level differences."""
    d = np.asarray(diff, dtype=float)
    d = d[~np.isnan(d)]
    n = int(d.size)
    if n < 2:
        return {"n": n, "mean_diff": float("nan"), "sd_diff": float("nan"),
                "d_z": float("nan"), "t": float("nan"), "df": 0,
                "p_two_sided": float("nan")}
    m = float(d.mean())
    sd = float(d.std(ddof=1))
    dz = m / sd if sd > 0 else float("nan")
    t = m / (sd / math.sqrt(n)) if sd > 0 else float("nan")
    p = 2 * stats.t.sf(abs(t), df=n - 1) if not math.isnan(t) else float("nan")
    return {"n": n, "mean_diff": m, "sd_diff": sd, "d_z": dz,
            "t": t, "df": n - 1, "p_two_sided": p}


# ──────────────────────────────────────────────────────────────────────────────
# Build cohort-specification frames
# ──────────────────────────────────────────────────────────────────────────────

def build_subject_diffs() -> Dict[str, pd.DataFrame]:
    """Return per-cohort DataFrames of subject_key -> REM_K0 - W_K0 means."""
    tda_path = OUT_DIR / "tda_epoch_features_all.csv"
    demo_path = OUT_DIR / "demographics_per_night.csv"
    if not tda_path.exists():
        raise SystemExit(f"missing {tda_path}; run pipeline.py first")
    if not demo_path.exists():
        raise SystemExit(f"missing {demo_path}; run scripts/demographics_breakdown.py first")
    tda = pd.read_csv(tda_path, comment="#")
    demo = pd.read_csv(demo_path, comment="#")

    # Within-recording z-score so K0 is the within-night recurrence measure
    # (matches the headline contrast specification in pipeline.run_mixedlm_analysis).
    tda = tda[tda["stage"].isin(P.STAGES_MAIN)].copy()
    tda["K0_tot"] = tda.groupby("subject")["H1_totpers"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0
    )

    # Per-recording subject-stage means (one row per recording × stage)
    keys = ["subject", "psg_file", "hyp_file"]
    rec_means = (tda.groupby(keys + ["stage"], as_index=False)["K0_tot"].mean())

    # Pivot to (recording, stage) → K0 mean
    rec_wide = rec_means.pivot(index=keys, columns="stage", values="K0_tot").reset_index()
    rec_wide = rec_wide.dropna(subset=["REM", "W"])
    rec_wide["diff_REM_W"] = rec_wide["REM"] - rec_wide["W"]

    # Attach demographics for cohort splits and unique-person rollup
    demo_keys = ["psg_file", "hyp_file"]
    rec_wide = rec_wide.merge(
        demo[demo_keys + ["cohort", "person_id", "drug"]],
        on=demo_keys, how="left",
    )

    # ───── Build the six specifications ─────
    out: Dict[str, pd.DataFrame] = {}

    # 1. per_night — every recording
    out["per_night"] = rec_wide[["subject", "psg_file", "diff_REM_W"]].copy()

    # 2. unique_subject — one row per (cohort, person_id):
    #    Cassette person → mean across their two nights
    #    Telemetry person → placebo night only (to avoid the drug confound)
    rw = rec_wide.copy()
    cas = rw[rw["cohort"] == "Cassette"].copy()
    tel = rw[(rw["cohort"] == "Telemetry") & (rw["drug"] == "placebo")].copy()
    cas_subj = (cas.groupby(["cohort", "person_id"], as_index=False)["diff_REM_W"].mean())
    tel_subj = tel[["cohort", "person_id", "diff_REM_W"]].copy()
    uniq = pd.concat([cas_subj, tel_subj], ignore_index=True)
    uniq = uniq.rename(columns={"person_id": "subject"})
    out["unique_subject"] = uniq[["cohort", "subject", "diff_REM_W"]]

    # 3. cassette_only — every Cassette night
    out["cassette_only"] = rec_wide.loc[
        rec_wide["cohort"] == "Cassette",
        ["subject", "psg_file", "diff_REM_W"]
    ].copy()

    # 4. cassette_subjects — Cassette unique subjects (mean across nights)
    out["cassette_subjects"] = (
        rec_wide.loc[rec_wide["cohort"] == "Cassette"]
                .groupby(["cohort", "person_id"], as_index=False)["diff_REM_W"].mean()
                .rename(columns={"person_id": "subject"})
    )

    # 5. telemetry_only — every Telemetry night (placebo + Temazepam)
    out["telemetry_only"] = rec_wide.loc[
        rec_wide["cohort"] == "Telemetry",
        ["subject", "psg_file", "drug", "diff_REM_W"]
    ].copy()

    # 6. telemetry_subjects — Telemetry placebo nights (one per person)
    out["telemetry_subjects"] = rec_wide.loc[
        (rec_wide["cohort"] == "Telemetry") & (rec_wide["drug"] == "placebo"),
        ["cohort", "person_id", "diff_REM_W"]
    ].rename(columns={"person_id": "subject"})

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────────────

ALPHA = 0.05
POWER = 0.80


def compute_table(specs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in specs.items():
        st = paired_dz(df["diff_REM_W"].to_numpy())
        n = st["n"]
        mde_cf = closed_form_mde(n, ALPHA, POWER) if n >= 2 else float("nan")
        mde_nct = nct_mde(n, ALPHA, POWER) if n >= 2 else float("nan")
        ratio_cf = (st["d_z"] / mde_cf) if (mde_cf and not math.isnan(mde_cf)) else float("nan")
        ratio_nct = (st["d_z"] / mde_nct) if (mde_nct and not math.isnan(mde_nct)) else float("nan")
        rows.append({
            "cohort_spec": name,
            "n": n,
            "observed_mean_diff": st["mean_diff"],
            "observed_sd_diff":   st["sd_diff"],
            "observed_d_z":       st["d_z"],
            "observed_t":         st["t"],
            "observed_df":        st["df"],
            "observed_p_two_sided": st["p_two_sided"],
            "mde_dz_closed_form_80pct": mde_cf,
            "mde_dz_nct_exact_80pct":   mde_nct,
            "observed_over_mde_closed_form": ratio_cf,
            "observed_over_mde_nct":         ratio_nct,
            "alpha": ALPHA, "power_target": POWER,
        })
    return pd.DataFrame(rows)


def compute_power_grid(specs: Dict[str, pd.DataFrame],
                       d_z_grid: np.ndarray = None) -> pd.DataFrame:
    if d_z_grid is None:
        d_z_grid = np.round(np.arange(0.05, 2.01, 0.05), 3)
    rows = []
    for name, df in specs.items():
        n = int(df["diff_REM_W"].dropna().shape[0])
        if n < 2:
            continue
        for dz in d_z_grid:
            rows.append({
                "cohort_spec": name, "n": n, "d_z": float(dz),
                "power_nct": nct_power(float(dz), n, ALPHA),
                "alpha": ALPHA,
            })
    return pd.DataFrame(rows)


def make_figure(table: pd.DataFrame, grid: pd.DataFrame, out_path: Path):
    if grid.empty or table.empty:
        log("  ! cannot draw figure (empty input)")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    palette = {
        "per_night":         "#1f77b4",
        "unique_subject":    "#d62728",
        "cassette_only":     "#2ca02c",
        "cassette_subjects": "#9467bd",
        "telemetry_only":    "#ff7f0e",
        "telemetry_subjects":"#8c564b",
    }
    pretty = {
        "per_night":         "Per-night (n = 197)",
        "unique_subject":    "Unique subject (n = 100)",
        "cassette_only":     "Cassette per-night (n = 153)",
        "cassette_subjects": "Cassette subjects (n = 78)",
        "telemetry_only":    "Telemetry per-night (n = 44)",
        "telemetry_subjects":"Telemetry subjects (n = 22)",
    }
    # Draw a power-vs-dz curve for each cohort spec
    for name, sub in grid.groupby("cohort_spec"):
        sub = sub.sort_values("d_z")
        n = int(sub["n"].iloc[0])
        ax.plot(sub["d_z"], sub["power_nct"],
                color=palette.get(name, "#444"),
                lw=1.6, label=pretty.get(name, name))
    # Reference lines
    ax.axhline(POWER, color="grey", lw=0.8, ls="--", label=f"target {POWER:.0%} power")
    # Mark observed d_z (use the unique_subject row as the headline)
    headline = table[table["cohort_spec"] == "unique_subject"]
    if not headline.empty:
        obs_dz = float(headline.iloc[0]["observed_d_z"])
        ax.axvline(obs_dz, color="black", lw=0.8, ls=":")
        ax.annotate(
            f"observed d_z = {obs_dz:.2f}",
            xy=(obs_dz, 0.05), xytext=(obs_dz - 0.6, 0.10),
            fontsize=9, color="black",
            arrowprops=dict(arrowstyle="-", color="black", lw=0.6),
        )
    ax.set_xlabel("Effect size d_z (paired)")
    ax.set_ylabel("Power (two-sided, α = 0.05)")
    ax.set_title("Sensitivity power curves: paired-t REM−Wake K0 contrast")
    ax.set_xlim(0, 2.0); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  · wrote {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Sensitivity power analysis for the headline REM-Wake K0 contrast"
    )
    p.add_argument("--force", action="store_true",
                   help="Recompute and overwrite outputs even if they exist.")
    p.add_argument("--no-figure", action="store_true",
                   help="Skip the power-curves figure.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    banner("Sensitivity power analysis (paired-t MDE on headline REM−Wake K0)")

    if (OUT_TABLE.exists() and OUT_GRID.exists()
            and (args.no_figure or OUT_FIG.exists()) and not args.force):
        log("  ✓ outputs exist; use --force to recompute")
        return

    log("  · building per-cohort subject-level diff arrays")
    specs = build_subject_diffs()
    for name, df in specs.items():
        log(f"    {name:>20s}: n = {df['diff_REM_W'].dropna().shape[0]}")

    log("\n  · computing observed d_z + MDE per cohort specification")
    table = compute_table(specs)
    table.to_csv(OUT_TABLE, index=False)
    log(f"  · wrote {OUT_TABLE.name} ({len(table)} rows)")

    log("\n  · computing achieved-power grid (d_z = 0.05 … 2.00)")
    grid = compute_power_grid(specs)
    grid.to_csv(OUT_GRID, index=False)
    log(f"  · wrote {OUT_GRID.name} ({len(grid)} rows)")

    if not args.no_figure:
        log("\n  · drawing power-curves figure")
        make_figure(table, grid, OUT_FIG)

    # Console summary
    log("")
    log("  Headline summary:")
    cols = ["cohort_spec", "n", "observed_d_z",
            "mde_dz_nct_exact_80pct", "observed_over_mde_nct"]
    width_map = [20, 5, 14, 22, 22]
    log(f"    {'cohort_spec':<20s} {'n':>5s}  {'obs d_z':>10s}  {'MDE d_z (NCT 80%)':>22s}  {'obs/MDE':>9s}")
    for _, r in table.iterrows():
        log(f"    {r['cohort_spec']:<20s} {int(r['n']):>5d}  "
            f"{r['observed_d_z']:>10.3f}  "
            f"{r['mde_dz_nct_exact_80pct']:>22.3f}  "
            f"{r['observed_over_mde_nct']:>9.1f}x")
    log("")


if __name__ == "__main__":
    main()
