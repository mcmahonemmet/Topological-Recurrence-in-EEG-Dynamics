#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats


# -----------------------------
# Multiple-comparison correction
# -----------------------------
def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, returned in original order."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for i, idx in enumerate(order):
        adj_i = (m - i) * pvals[idx]
        adj_i = max(adj_i, prev)  # monotone
        prev = adj_i
        adj[idx] = min(adj_i, 1.0)
    return adj


def pretty_table(df: pd.DataFrame, fmt: str = "markdown") -> str:
    """Render a DataFrame as markdown (github) or plain text."""
    df2 = df.copy()
    try:
        from tabulate import tabulate
        if fmt == "markdown":
            return tabulate(df2, headers="keys", tablefmt="github", showindex=False)
        elif fmt == "plain":
            return tabulate(df2, headers="keys", tablefmt="simple", showindex=False)
        else:
            return tabulate(df2, headers="keys", tablefmt="github", showindex=False)
    except Exception:
        return df2.to_string(index=False)


def save_table(text: str, outdir: str, filename: str) -> str:
    """Save a rendered table string to a file in outdir. Returns the filepath."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def within_subject_z(df: pd.DataFrame, subject_col: str, value_col: str) -> pd.Series:
    """Within-subject z-score."""
    def z(x):
        s = x.std(ddof=1)
        if s == 0 or np.isnan(s):
            return (x - x.mean()) * 0.0
        return (x - x.mean()) / s
    return df.groupby(subject_col)[value_col].transform(z)


def paired_contrast(wide: pd.DataFrame, a: str, b: str) -> dict:
    """
    Paired contrast a-b using subject-level stage means.

    Returns:
      estimate (mean difference),
      95% CI (t-based on mean diff),
      Cohen's dz,
      p (one-sample t-test on paired differences)
    """
    d = (wide[a] - wide[b]).dropna()
    n = len(d)
    if n < 3:
        return {
            "n": n, "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
            "dz": np.nan, "p": np.nan
        }

    mean = d.mean()
    sd = d.std(ddof=1)
    dz = mean / sd if sd > 0 else np.nan

    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean - tcrit * se
    ci_high = mean + tcrit * se

    tstat, p = stats.ttest_1samp(d, popmean=0.0, nan_policy="omit")

    return {"n": n, "estimate": mean, "ci_low": ci_low, "ci_high": ci_high, "dz": dz, "p": p}


# -----------------------------
# Main analysis
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outputs_dir",
        default=r"C:\Users\mcmah\Documents\sleep_tda_project\outputs",
        help="Path to outputs directory containing the CSVs."
    )
    ap.add_argument(
        "--format",
        default="markdown",
        choices=["markdown", "plain"],
        help="Table output format."
    )
    ap.add_argument(
        "--no_robustness",
        action="store_true",
        help="Skip robustness check even if robustness file exists."
    )
    args = ap.parse_args()

    outdir = args.outputs_dir

    # Expected files
    tda_epoch = os.path.join(outdir, "tda_epoch_features_all.csv")
    baseline_epoch = os.path.join(outdir, "baseline_epoch_features_all.csv")
    robustness_contrasts = os.path.join(outdir, "tda_robustness_mixedlm_planned_contrasts.csv")

    missing = [p for p in [tda_epoch, baseline_epoch] if not os.path.exists(p)]
    if missing:
        print("ERROR: Missing required file(s):")
        for p in missing:
            print("  -", p)
        sys.exit(1)

    # -----------------------------
    # Load
    # -----------------------------
    tda = pd.read_csv(tda_epoch)
    base = pd.read_csv(baseline_epoch)

    # Basic column checks
    for col in ["subject", "stage", "H1_totpers"]:
        if col not in tda.columns:
            raise ValueError(f"{tda_epoch} missing required column: {col}")

    for col in ["subject", "stage", "spec_entropy", "perm_entropy", "lz_complexity",
                "log_delta", "log_theta", "log_alpha", "log_beta"]:
        if col not in base.columns:
            raise ValueError(f"{baseline_epoch} missing required column: {col}")

    # -----------------------------
    # Compute K0 and stage descriptives
    # -----------------------------
    tda = tda.copy()
    tda["K0"] = within_subject_z(tda, "subject", "H1_totpers")

    stage_desc = (
        tda.groupby("stage")["K0"]
        .agg(
            Mean="mean",
            SD=lambda x: x.std(ddof=1),
            Median="median",
            IQR=lambda x: x.quantile(0.75) - x.quantile(0.25),
        )
        .reset_index()
    )

    # Canonical stage order
    stage_order = ["W", "N1", "N2", "N3", "REM"]
    stage_desc["stage"] = pd.Categorical(stage_desc["stage"], categories=stage_order, ordered=True)
    stage_desc = stage_desc.sort_values("stage").reset_index(drop=True)
    stage_desc.rename(columns={"stage": "Sleep stage"}, inplace=True)

    # Round for display
    stage_desc_disp = stage_desc.copy()
    for c in ["Mean", "SD", "Median", "IQR"]:
        stage_desc_disp[c] = stage_desc_disp[c].astype(float).round(2)

    # -----------------------------
    # Planned contrasts for K0 (paired, subject-level means)
    # -----------------------------
    wide_k0 = (
        tda.groupby(["subject", "stage"])["K0"]
        .mean()
        .unstack()
    )

    contrast_specs = [
        ("REM–Wake", "REM", "W"),
        ("REM–N3", "REM", "N3"),
        ("N1–N2", "N1", "N2"),
    ]

    rows = []
    for label, a, b in contrast_specs:
        res = paired_contrast(wide_k0, a, b)
        rows.append({
            "Contrast": label,
            "Estimate (SD units)": res["estimate"],
            "95% CI": f"[{res['ci_low']:.2f}, {res['ci_high']:.2f}]",
            "Cohen’s dz": res["dz"],
            "p (raw)": res["p"],
            "n": res["n"]
        })

    contrasts_df = pd.DataFrame(rows)
    contrasts_df["Holm-adjusted p"] = holm_adjust(contrasts_df["p (raw)"].values)

    # Display rounding
    contrasts_disp = contrasts_df.copy()
    contrasts_disp["Estimate (SD units)"] = contrasts_disp["Estimate (SD units)"].astype(float).round(2)
    contrasts_disp["Cohen’s dz"] = contrasts_disp["Cohen’s dz"].astype(float).round(2)
    contrasts_disp["Holm-adjusted p"] = contrasts_disp["Holm-adjusted p"].astype(float).map(
        lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}"
    )
    contrasts_disp["p (raw)"] = contrasts_disp["p (raw)"].astype(float).map(
        lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}"
    )

    # -----------------------------
    # EEG metric comparison (paired REM–Wake) with Holm correction across metrics
    # -----------------------------
    metric_map = [
        ("Spectral entropy", "spec_entropy", "Shannon entropy of power spectrum"),
        ("Permutation entropy", "perm_entropy", "Ordinal pattern entropy"),
        ("Lempel–Ziv complexity", "lz_complexity", "Algorithmic compressibility"),
        ("Delta power", "log_delta", "Log band power (0.5–4 Hz)"),
        ("Theta power", "log_theta", "Log band power (4–8 Hz)"),
        ("Alpha power", "log_alpha", "Log band power (8–12 Hz)"),
        ("Beta power", "log_beta", "Log band power (13–30 Hz)"),
    ]

    base_subj = (
        base.groupby(["subject", "stage"])[[m[1] for m in metric_map]]
        .mean()
        .reset_index()
    )

    comp_rows = []
    raw_ps = []
    tmp_store = []

    for name, col, qty in metric_map:
        wide = base_subj.pivot(index="subject", columns="stage", values=col)
        res = paired_contrast(wide, "REM", "W")
        tmp_store.append((name, qty, res))
        raw_ps.append(res["p"])

    raw_ps = np.array(raw_ps, dtype=float)
    holm_ps = holm_adjust(raw_ps)

    for (name, qty, res), p_holm in zip(tmp_store, holm_ps):
        comp_rows.append({
            "Metric": name,
            "Primary quantity": qty,
            "REM–Wake estimate (SD)": res["estimate"],
            "95% CI": f"[{res['ci_low']:.2f}, {res['ci_high']:.2f}]",
            "Cohen’s dz": res["dz"],
            "Holm-adj. p": p_holm,
            "n": res["n"]
        })

    comp_df = pd.DataFrame(comp_rows)

    # Add K0 row at top
    k0_rw = contrasts_df.loc[contrasts_df["Contrast"] == "REM–Wake"].iloc[0]
    k0_row = pd.DataFrame([{
        "Metric": "Recurrence (K0)",
        "Primary quantity": "H1 total persistence (within-subject z-score)",
        "REM–Wake estimate (SD)": float(k0_rw["Estimate (SD units)"]),
        "95% CI": contrasts_df.loc[contrasts_df["Contrast"] == "REM–Wake", "95% CI"].iloc[0],
        "Cohen’s dz": float(k0_rw["Cohen’s dz"]),
        "Holm-adj. p": float(k0_rw["Holm-adjusted p"]),
        "n": int(k0_rw["n"])
    }])

    comp_df = pd.concat([k0_row, comp_df], ignore_index=True)

    # Robustness flag (optional)
    robust_flag = "Unknown"
    if not args.no_robustness and os.path.exists(robustness_contrasts):
        rob = pd.read_csv(robustness_contrasts)
        ccol = "contrast" if "contrast" in rob.columns else ("Contrast" if "Contrast" in rob.columns else None)
        pcol = "p_holm" if "p_holm" in rob.columns else ("p_holm_corrected" if "p_holm_corrected" in rob.columns else None)

        if ccol and pcol:
            rw = rob[
                rob[ccol].astype(str).str.replace(" ", "").str.contains("REM", case=False) &
                rob[ccol].astype(str).str.replace(" ", "").str.contains("W", case=False)
            ]
            if len(rw) > 0:
                robust_flag = "Yes" if (rw[pcol] < 0.05).all() else "No"
        else:
            robust_flag = "Could not parse robustness file columns"

    comp_df.insert(len(comp_df.columns) - 1, "Robust (m, τ)", ["Yes" if i == 0 else "N/A" for i in range(len(comp_df))])
    if comp_df.loc[0, "Metric"] == "Recurrence (K0)":
        comp_df.loc[0, "Robust (m, τ)"] = robust_flag

    # Display rounding
    comp_disp = comp_df.copy()
    comp_disp["REM–Wake estimate (SD)"] = comp_disp["REM–Wake estimate (SD)"].astype(float).round(2)
    comp_disp["Cohen’s dz"] = comp_disp["Cohen’s dz"].astype(float).round(2)
    comp_disp["Holm-adj. p"] = comp_disp["Holm-adj. p"].astype(float).map(
        lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}"
    )

    # -----------------------------
    # Render + Print + Save the three tables
    # -----------------------------
    ext = "md" if args.format == "markdown" else "txt"

    # Table 1
    table1 = pretty_table(stage_desc_disp, fmt=args.format)
    print("\n=== Sleep stage descriptives for Recurrence (K0) ===")
    print(table1)
    save_table(table1, outdir, f"table_k0_stage_descriptives.{ext}")

    # Table 2
    table2 = pretty_table(
        contrasts_disp[["Contrast", "Estimate (SD units)", "95% CI", "Cohen’s dz", "Holm-adjusted p", "n"]],
        fmt=args.format
    )
    print("\n=== Planned contrasts for Recurrence (K0) ===")
    print(table2)
    save_table(table2, outdir, f"table_k0_planned_contrasts.{ext}")

    # Table 3
    table3 = pretty_table(
        comp_disp[["Metric", "Primary quantity", "REM–Wake estimate (SD)", "95% CI",
                   "Cohen’s dz", "Holm-adj. p", "Robust (m, τ)", "n"]],
        fmt=args.format
    )
    print("\n=== Comparison with EEG metrics (paired REM–Wake) ===")
    print(table3)
    save_table(table3, outdir, f"table_metric_comparison.{ext}")

    print(f"\nSaved tables to: {outdir}")
    print(f" - table_k0_stage_descriptives.{ext}")
    print(f" - table_k0_planned_contrasts.{ext}")
    print(f" - table_metric_comparison.{ext}")


if __name__ == "__main__":
    main()
