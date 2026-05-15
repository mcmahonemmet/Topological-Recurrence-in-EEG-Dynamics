#!/usr/bin/env python3
"""
stratified_effects.py
=====================

Standalone script that breaks the headline REM-vs-Wake K0 contrast down by
demographic stratification:

  Section 1 — Lifespan x Sex
       Refit the REM-Wake mixed-LM planned contrast within each (age-group,
       sex) cell, and within the marginal age-group and sex strata.

  Section 2 — Drug study (Sleep-Telemetry, within-subject Temazepam crossover)
       Refit the REM-Wake contrast separately within placebo and Temazepam
       nights, fit a stage × drug interaction model on the full Telemetry
       cohort, and report a per-subject paired comparison of the REM-Wake
       contrast under placebo vs Temazepam.

  Section 3 — Figures
       Forest plot of REM-Wake K0 estimates by age-group, faceted by sex,
       plus a placebo-vs-Temazepam panel for the drug study.

Inputs:
  - outputs/tda_epoch_features_all.csv     (per-epoch K0/H0/H1 features)
  - outputs/demographics_per_night.csv     (produced by demographics_breakdown.py)

Outputs (written to outputs/ and outputs/figures/):
  - strat_lifespan_age_sex_remw.csv        REM-Wake mixed-LM contrast within
                                           every (age_group, sex) cell, plus
                                           marginal and pooled rows.
  - strat_drug_remw_by_condition.csv       REM-Wake contrast under placebo
                                           and under Temazepam separately
                                           (Telemetry only).
  - strat_drug_stage_x_drug.csv            stage × drug interaction model on
                                           Telemetry: omnibus interaction LR
                                           and the four planned contrast rows
                                           (REM-W under placebo, REM-W under
                                           temazepam, stage main effect,
                                           drug × REM-vs-W interaction).
  - strat_drug_subject_paired_remw.csv     per-subject within-person REM-Wake
                                           on placebo vs Temazepam, with
                                           paired t-test summary.
  - figures/strat_forest_age_sex_remw.png  forest plot.
  - figures/strat_drug_remw.png            placebo-vs-Temazepam panel.

Cohort handling for the lifespan analysis:
  The Sleep-Telemetry sub-cohort is a within-subject drug crossover, so
  including BOTH Telemetry nights per subject would weight Telemetry people
  twice and confound any age effect with the drug protocol. By default this
  script restricts the lifespan analysis to one night per Telemetry person —
  the placebo night — so each of the 100 subjects contributes one night.
  Sensitivity options:
    --tel-policy placebo     (default) Telemetry contributes its placebo night
    --tel-policy first       Telemetry contributes its night-1 record
    --tel-policy all         use both Telemetry nights (caveat above)
    --tel-policy exclude     Cassette only

Age groups default to {18-29, 30-44, 45-59, 60-74, 75+}. Override with
``--age-bins 18,30,45,60,75,120`` (CSV of bin edges; right-open).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy import stats

# Make pipeline.py importable.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import pipeline as P  # noqa: E402

import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUT_DIR = P.OUT_DIR
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_LIFESPAN  = OUT_DIR / "strat_lifespan_age_sex_remw.csv"
OUT_DRUG_BYC  = OUT_DIR / "strat_drug_remw_by_condition.csv"
OUT_DRUG_INT  = OUT_DIR / "strat_drug_stage_x_drug.csv"
OUT_DRUG_PAIR = OUT_DIR / "strat_drug_subject_paired_remw.csv"
FIG_FOREST    = FIG_DIR / "strat_forest_age_sex_remw.png"
FIG_DRUG      = FIG_DIR / "strat_drug_remw.png"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def banner(s: str):
    bar = "═" * max(len(s) + 4, 60)
    print(f"\n{bar}\n  {s}\n{bar}", flush=True)

def log(s: str):
    print(s, flush=True)

def _within_subject_z(df: pd.DataFrame, src: str = "H1_totpers",
                      key: str = "subject", dst: str = "K0_tot") -> pd.DataFrame:
    df = df.copy()
    df[dst] = df.groupby(key)[src].transform(
        lambda x: (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0
    )
    return df

def _bin_age(age_series: pd.Series, edges: Sequence[int]) -> pd.Series:
    edges = list(edges)
    labels = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i+1]
        labels.append(f"{a}-{b-1}" if b < 200 else f"{a}+")
    out = pd.cut(age_series, bins=edges, right=False, labels=labels,
                 include_lowest=True)
    return out


def _fit_remw_contrast(df: pd.DataFrame, label: str,
                       group_col: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Fit a stage-level mixed-LM on K0_tot with subject random intercepts and
    return the planned-contrast row for REM - W. Returns None if the cell is
    too small or the model fails to converge."""
    n_subj = df["subject"].nunique()
    stages_present = sorted(df["stage"].unique().tolist())
    if "REM" not in stages_present or "W" not in stages_present:
        return None
    if n_subj < 5:
        return None
    ss = df.groupby(["subject", "stage"], as_index=False)["K0_tot"].mean()
    try:
        res, lr, df_diff, p_lr = P.fit_mixedlm_stage(
            ss, "K0_tot", P.STAGES_MAIN, method="powell")
    except Exception:
        try:
            res, lr, df_diff, p_lr = P.fit_mixedlm_stage(
                ss, "K0_tot", P.STAGES_MAIN, method="lbfgs")
        except Exception:
            return None
    contr = P.planned_contrasts(res, "K0_tot", [("REM", "W")], P.STAGES_MAIN)
    contr["stratum"] = label
    contr["n_subjects"] = n_subj
    contr["n_nights"] = int(df.groupby(["subject", "psg_file"]).ngroups)
    contr["LR_omnibus"] = lr
    contr["df_omnibus"] = df_diff
    contr["p_omnibus"] = p_lr
    if group_col is not None:
        contr["group_col"] = group_col
    return contr


# ──────────────────────────────────────────────────────────────────────────────
# Section 1 — Lifespan × Sex stratification
# ──────────────────────────────────────────────────────────────────────────────

def section_1_lifespan(merged: pd.DataFrame, age_edges: Sequence[int],
                       tel_policy: str) -> pd.DataFrame:
    banner(f"Section 1 — Lifespan × Sex (age bins {list(age_edges)}, "
           f"Telemetry policy: {tel_policy})")
    df = merged.copy()

    # Telemetry policy
    if tel_policy == "placebo":
        keep = (df["cohort"] == "Cassette") | (
            (df["cohort"] == "Telemetry") & (df["drug"] == "placebo"))
        df = df[keep]
    elif tel_policy == "first":
        keep = (df["cohort"] == "Cassette") | (
            (df["cohort"] == "Telemetry") & (df["night"] == 1))
        df = df[keep]
    elif tel_policy == "all":
        pass
    elif tel_policy == "exclude":
        df = df[df["cohort"] == "Cassette"]
    else:
        raise SystemExit(f"unknown --tel-policy: {tel_policy}")

    # Within-subject K0 (using a per-person key so the same person's two
    # Cassette nights both contribute to one z-score baseline).
    df["person_key"] = df["cohort"].astype(str) + "_" + df["person_id"].astype(str)
    df = _within_subject_z(df, src="H1_totpers", key="person_key")
    # Use person_key as the random-effect "subject" so two-night individuals
    # don't get double-weighted in the mixed-LM.
    df = df.rename(columns={"subject": "_subject_filename"})
    df["subject"] = df["person_key"]
    df["age_group"] = _bin_age(df["age"], age_edges).astype(str)

    age_groups_order = []
    for i in range(len(age_edges) - 1):
        a, b = age_edges[i], age_edges[i+1]
        age_groups_order.append(f"{a}-{b-1}" if b < 200 else f"{a}+")

    rows = []

    # Pooled (all subjects)
    r = _fit_remw_contrast(df, "Pooled (all subjects)")
    if r is not None:
        r["age_group"] = "All"; r["sex"] = "All"
        rows.append(r)

    # Marginal sex
    for sex in ["F", "M"]:
        sub = df[df["sex"] == sex]
        r = _fit_remw_contrast(sub, f"All-age, {sex}")
        if r is not None:
            r["age_group"] = "All"; r["sex"] = sex
            rows.append(r)

    # Marginal age-group
    for ag in age_groups_order:
        sub = df[df["age_group"] == ag]
        r = _fit_remw_contrast(sub, f"{ag}, both sexes")
        if r is not None:
            r["age_group"] = ag; r["sex"] = "All"
            rows.append(r)

    # Age-group × sex
    for ag in age_groups_order:
        for sex in ["F", "M"]:
            sub = df[(df["age_group"] == ag) & (df["sex"] == sex)]
            r = _fit_remw_contrast(sub, f"{ag}, {sex}")
            if r is not None:
                r["age_group"] = ag; r["sex"] = sex
                rows.append(r)

    if not rows:
        log("  ! no fits succeeded")
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out[["age_group", "sex", "stratum", "n_subjects", "n_nights",
               "metric", "contrast", "estimate", "SE", "z", "p", "p_holm",
               "CI95_low", "CI95_high",
               "LR_omnibus", "df_omnibus", "p_omnibus"]]
    out.to_csv(OUT_LIFESPAN, index=False)
    log(f"  · wrote {OUT_LIFESPAN.name} ({len(out)} rows)")

    # Console digest
    log("")
    log(f"  {'age_group':<10s} {'sex':<4s} {'n':>4s}  {'REM-W (SD)':>14s}  {'CI95':>22s}  p")
    for _, r in out.iterrows():
        log(f"  {str(r['age_group']):<10s} {r['sex']:<4s} {int(r['n_subjects']):>4d}  "
            f"{r['estimate']:+.3f}        "
            f"[{r['CI95_low']:+.2f}, {r['CI95_high']:+.2f}]   "
            f"{r['p']:.2e}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Section 2 — Drug study (Telemetry)
# ──────────────────────────────────────────────────────────────────────────────

def section_2_drug(merged: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    banner("Section 2 — Drug study (Sleep-Telemetry: placebo vs Temazepam)")
    df = merged[merged["cohort"] == "Telemetry"].copy()
    if df.empty:
        log("  ! no Telemetry rows")
        return {}
    df["person_key"] = "Telemetry_" + df["person_id"].astype(str)

    # ─── (a) REM-Wake contrast within each drug condition ──────────────────
    by_cond_rows = []
    for drug in ["placebo", "temazepam"]:
        sub = df[df["drug"] == drug].copy()
        # Within-subject z within this drug subset (so K0 reflects the
        # subject's own baseline ON that drug).
        sub = _within_subject_z(sub, src="H1_totpers", key="person_key")
        sub = sub.rename(columns={"subject": "_subject_filename"})
        sub["subject"] = sub["person_key"]
        r = _fit_remw_contrast(sub, f"Telemetry-only, {drug}")
        if r is not None:
            r["drug"] = drug
            by_cond_rows.append(r)
    by_cond = pd.concat(by_cond_rows, ignore_index=True) if by_cond_rows else pd.DataFrame()
    if not by_cond.empty:
        by_cond.to_csv(OUT_DRUG_BYC, index=False)
        log(f"  · wrote {OUT_DRUG_BYC.name}")
        for _, r in by_cond.iterrows():
            log(f"    {r['drug']:<10s} REM-W = {r['estimate']:+.3f} SD "
                f"[{r['CI95_low']:+.2f}, {r['CI95_high']:+.2f}], "
                f"n_subjects = {int(r['n_subjects'])}, p = {r['p']:.2e}")

    # ─── (b) stage × drug interaction model (full Telemetry data) ─────────
    full = _within_subject_z(df, src="H1_totpers", key="person_key")
    full = full.rename(columns={"subject": "_subject_filename"})
    full["subject"] = full["person_key"]
    full = full[full["stage"].isin(P.STAGES_MAIN)].copy()
    # Subject-level means per (subject, stage, drug) so the interaction term
    # operates on the same aggregated unit as the headline analysis.
    ss = (full.groupby(["subject", "stage", "drug"], as_index=False)["K0_tot"]
              .mean())
    ss["stage"] = pd.Categorical(ss["stage"], categories=P.STAGES_MAIN, ordered=True)
    ss["drug"] = pd.Categorical(ss["drug"], categories=["placebo", "temazepam"])
    int_rows = []
    try:
        full_m = smf.mixedlm("K0_tot ~ C(stage) * C(drug)", ss,
                             groups=ss["subject"]).fit(
            reml=False, method="powell", maxiter=2000, disp=False)
        # Likelihood-ratio interaction test vs additive model
        add_m = smf.mixedlm("K0_tot ~ C(stage) + C(drug)", ss,
                            groups=ss["subject"]).fit(
            reml=False, method="powell", maxiter=2000, disp=False)
        lr_int = 2 * (full_m.llf - add_m.llf)
        df_int = full_m.df_modelwc - add_m.df_modelwc
        p_int = stats.chi2.sf(lr_int, df_int) if df_int > 0 else np.nan
        int_rows.append({"contrast": "stage × drug (omnibus interaction)",
                         "LR": float(lr_int), "df": int(df_int),
                         "p": float(p_int)})

        # Pull the specific interaction term: how much does drug=temazepam
        # change the REM-vs-W contrast? Statsmodels reference coding gives us:
        #   beta_REM       (effect of REM under placebo, vs W under placebo)
        #   beta_drugTemaz (effect of temazepam at W stage)
        #   beta_REM:drugTemaz (additional REM effect under temazepam)
        params = full_m.params; cov = full_m.cov_params()
        idx = params.index.tolist()
        def _get(term):
            return idx.index(term) if term in idx else None
        i_remw_pla   = _get("C(stage)[T.REM]")
        i_remw_tema  = _get("C(stage)[T.REM]:C(drug)[T.temazepam]")
        if i_remw_pla is not None:
            est = float(params.iloc[i_remw_pla])
            se  = float(np.sqrt(cov.iloc[i_remw_pla, i_remw_pla]))
            int_rows.append({
                "contrast": "REM - W under placebo (model coef)",
                "estimate": est, "SE": se,
                "CI95_low": est - 1.96*se, "CI95_high": est + 1.96*se,
                "z": est/se if se > 0 else np.nan,
                "p": 2 * stats.norm.sf(abs(est/se)) if se > 0 else np.nan,
            })
        if (i_remw_pla is not None) and (i_remw_tema is not None):
            # REM-W contrast under Temazepam = beta_REM + beta_REM:Temaz
            L = np.zeros(len(idx))
            L[i_remw_pla] = 1.0; L[i_remw_tema] = 1.0
            est = float(np.dot(L, params))
            se  = float(np.sqrt(np.dot(L, np.dot(cov, L))))
            int_rows.append({
                "contrast": "REM - W under temazepam (model coef)",
                "estimate": est, "SE": se,
                "CI95_low": est - 1.96*se, "CI95_high": est + 1.96*se,
                "z": est/se if se > 0 else np.nan,
                "p": 2 * stats.norm.sf(abs(est/se)) if se > 0 else np.nan,
            })
            # The drug-modulation of the REM-W contrast (interaction beta)
            est = float(params.iloc[i_remw_tema])
            se  = float(np.sqrt(cov.iloc[i_remw_tema, i_remw_tema]))
            int_rows.append({
                "contrast": "Δ(REM-W): temazepam - placebo (interaction term)",
                "estimate": est, "SE": se,
                "CI95_low": est - 1.96*se, "CI95_high": est + 1.96*se,
                "z": est/se if se > 0 else np.nan,
                "p": 2 * stats.norm.sf(abs(est/se)) if se > 0 else np.nan,
            })
    except Exception as ex:
        log(f"  ! interaction model failed: {ex}")
    interactions = pd.DataFrame(int_rows)
    if not interactions.empty:
        interactions.to_csv(OUT_DRUG_INT, index=False)
        log(f"  · wrote {OUT_DRUG_INT.name}")
        for _, r in interactions.iterrows():
            est = r.get("estimate", np.nan); ci_lo = r.get("CI95_low", np.nan)
            ci_hi = r.get("CI95_high", np.nan); p = r.get("p", np.nan)
            extras = ""
            if "LR" in r and not pd.isna(r["LR"]):
                extras = f"LR = {r['LR']:.2f}, df = {int(r['df'])}, p = {r['p']:.3e}"
            else:
                extras = f"est = {est:+.3f} [{ci_lo:+.2f}, {ci_hi:+.2f}], p = {p:.3e}"
            log(f"    {r['contrast']}: {extras}")

    # ─── (c) per-subject paired REM-W contrasts under placebo vs temazepam ─
    paired_rows = []
    for pid in sorted(df["person_id"].unique()):
        sub = df[df["person_id"] == pid].copy()
        sub = _within_subject_z(sub, src="H1_totpers", key="person_id")
        for drug in ["placebo", "temazepam"]:
            d = sub[sub["drug"] == drug]
            if d.empty:
                continue
            stage_means = d.groupby("stage")["K0_tot"].mean()
            if "REM" in stage_means and "W" in stage_means:
                paired_rows.append({
                    "person_id": int(pid), "drug": drug,
                    "K0_REM": float(stage_means["REM"]),
                    "K0_W":   float(stage_means["W"]),
                    "REM_minus_W": float(stage_means["REM"] - stage_means["W"]),
                })
    paired_long = pd.DataFrame(paired_rows)
    paired_summary = pd.DataFrame()
    if not paired_long.empty:
        # Wide form
        wide = paired_long.pivot_table(index="person_id", columns="drug",
                                       values="REM_minus_W")
        wide = wide.dropna(subset=["placebo", "temazepam"])
        if len(wide):
            diff = wide["temazepam"] - wide["placebo"]
            t, p = stats.ttest_rel(wide["temazepam"], wide["placebo"])
            wsr = stats.wilcoxon(wide["temazepam"], wide["placebo"])
            paired_summary = pd.DataFrame([{
                "n_paired_subjects": int(len(wide)),
                "remw_placebo_mean":   float(wide["placebo"].mean()),
                "remw_placebo_sd":     float(wide["placebo"].std(ddof=1)),
                "remw_temazepam_mean": float(wide["temazepam"].mean()),
                "remw_temazepam_sd":   float(wide["temazepam"].std(ddof=1)),
                "delta_mean":  float(diff.mean()),
                "delta_sd":    float(diff.std(ddof=1)),
                "delta_ci95_low":  float(diff.mean() - 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))),
                "delta_ci95_high": float(diff.mean() + 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))),
                "paired_t": float(t), "paired_t_p": float(p),
                "wilcoxon_W": float(wsr.statistic), "wilcoxon_p": float(wsr.pvalue),
            }])
            log(f"    Per-subject paired Δ(REM-W) = "
                f"{diff.mean():+.3f} SD, paired-t p = {p:.3f}, "
                f"Wilcoxon p = {wsr.pvalue:.3f}, n = {len(wide)}")
    # Save the long-form per-person rows; append summary row(s) at the bottom.
    if not paired_long.empty:
        out = paired_long.copy()
        if not paired_summary.empty:
            for col in paired_summary.columns:
                out.loc["_summary_" + col, col] = paired_summary.iloc[0][col]
        out.to_csv(OUT_DRUG_PAIR, index=False)
        log(f"  · wrote {OUT_DRUG_PAIR.name}")

    return {"by_condition": by_cond, "interactions": interactions,
            "paired_summary": paired_summary, "paired_long": paired_long}


# ──────────────────────────────────────────────────────────────────────────────
# Section 3 — Figures
# ──────────────────────────────────────────────────────────────────────────────

def make_forest(lifespan: pd.DataFrame, out_path: Path):
    if lifespan.empty:
        log("  ! lifespan table empty; skipping forest plot")
        return
    sub = lifespan[lifespan["age_group"] != "All"].copy()
    age_groups = list(dict.fromkeys(sub["age_group"].tolist()))   # ordered unique
    sexes = ["F", "M", "All"]
    palette = {"F": "#bf5b8c", "M": "#5b8cbf", "All": "#444444"}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.0), sharey=True,
                             gridspec_kw={"wspace": 0.18})
    for ax, sex in zip(axes, sexes):
        rows = sub[sub["sex"] == sex]
        rows = rows.set_index("age_group").reindex(age_groups)
        y = np.arange(len(age_groups))
        est = rows["estimate"].values.astype(float)
        lo = rows["CI95_low"].values.astype(float)
        hi = rows["CI95_high"].values.astype(float)
        ns = rows["n_subjects"].values
        ax.errorbar(est, y, xerr=[est - lo, hi - est],
                    fmt="o", color=palette[sex], lw=1.4, capsize=3,
                    markersize=5)
        for yi, n in enumerate(ns):
            if pd.isna(n): continue
            ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 1.5 else 1.55,
                    yi, f"n={int(n)}", va="center", fontsize=8, color="grey")
        ax.axvline(0, color="grey", ls=":", lw=0.8)
        ax.set_xlabel("REM−Wake K0 (within-subject SD)")
        ax.set_yticks(y); ax.set_yticklabels(age_groups)
        ax.set_title(f"{'Female' if sex=='F' else 'Male' if sex=='M' else 'Both sexes'}")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("REM−Wake K0 contrast by age and sex", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  · wrote {out_path.name}")


def make_drug_panel(by_cond: pd.DataFrame, paired_long: pd.DataFrame,
                    out_path: Path):
    if by_cond.empty and paired_long.empty:
        log("  ! drug-study tables empty; skipping drug figure")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    # Left — group estimates
    ax = axes[0]
    if not by_cond.empty:
        x = np.arange(len(by_cond))
        colours = ["#888888" if d == "placebo" else "#7B4B94" for d in by_cond["drug"]]
        ax.bar(x, by_cond["estimate"], yerr=[by_cond["estimate"] - by_cond["CI95_low"],
                                              by_cond["CI95_high"] - by_cond["estimate"]],
               color=colours, capsize=4, edgecolor="white", width=0.55)
        ax.set_xticks(x); ax.set_xticklabels(
            [d.capitalize() for d in by_cond["drug"]])
        ax.set_ylabel("REM−Wake K0 (within-subject SD)")
        ax.set_title("Group REM−Wake by drug condition (Telemetry)")
        ax.axhline(0, color="grey", ls=":", lw=0.8)
        ax.grid(axis="y", alpha=0.25)

    # Right — per-subject paired lines
    ax = axes[1]
    if not paired_long.empty:
        wide = paired_long.pivot_table(index="person_id", columns="drug",
                                       values="REM_minus_W").dropna()
        for pid, row in wide.iterrows():
            ax.plot([0, 1], [row["placebo"], row["temazepam"]],
                    color="grey", alpha=0.45, lw=0.9, marker="o", markersize=3)
        ax.scatter([0]*len(wide), wide["placebo"], color="#888888", zorder=3)
        ax.scatter([1]*len(wide), wide["temazepam"], color="#7B4B94", zorder=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Placebo", "Temazepam"])
        ax.set_ylabel("Per-subject REM−Wake K0 (SD)")
        ax.set_title(f"Per-subject paired Δ (n = {len(wide)})")
        ax.axhline(0, color="grey", ls=":", lw=0.8)
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  · wrote {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Stratified analysis: lifespan × sex and drug-study"
    )
    p.add_argument("--age-bins", type=str, default="18,30,45,60,75,120",
                   help="CSV of age bin edges (right-open). "
                        "Default: 18,30,45,60,75,120 → groups 18-29, 30-44, "
                        "45-59, 60-74, 75+.")
    p.add_argument("--tel-policy", type=str, default="placebo",
                   choices=["placebo", "first", "all", "exclude"],
                   help="How to include Telemetry nights in Section 1 "
                        "(default: placebo = one night per Telemetry person).")
    p.add_argument("--no-figure", action="store_true",
                   help="Skip the figures.")
    p.add_argument("--force", action="store_true",
                   help="Recompute outputs even if they exist.")
    return p.parse_args(argv)


def _load_inputs() -> pd.DataFrame:
    tda = OUT_DIR / "tda_epoch_features_all.csv"
    demo = OUT_DIR / "demographics_per_night.csv"
    if not tda.exists():
        sys.exit(f"missing {tda}; run pipeline.py first")
    if not demo.exists():
        sys.exit(f"missing {demo}; run scripts/demographics_breakdown.py first")
    t = pd.read_csv(tda, comment="#")
    d = pd.read_csv(demo, comment="#")
    keys = ["psg_file", "hyp_file"]
    keep_cols = ["psg_file", "hyp_file", "cohort", "person_id", "night",
                 "age", "sex", "drug"]
    merged = t.merge(d[keep_cols], on=["psg_file", "hyp_file"], how="inner")
    merged = merged[merged["stage"].isin(P.STAGES_MAIN)].copy()
    return merged


def main(argv=None):
    args = parse_args(argv)
    edges = [int(x) for x in args.age_bins.split(",") if x.strip()]
    if len(edges) < 2:
        sys.exit("--age-bins needs at least two edges")

    have_all = all(p.exists() for p in [
        OUT_LIFESPAN, OUT_DRUG_BYC, OUT_DRUG_INT, OUT_DRUG_PAIR
    ])
    if have_all and (args.no_figure or all(p.exists() for p in [FIG_FOREST, FIG_DRUG])) \
            and not args.force:
        log("  ✓ outputs exist; use --force to recompute")
        return

    merged = _load_inputs()
    log(f"Loaded {len(merged):,} epoch rows from "
        f"{merged.groupby(['cohort','person_id']).ngroups} unique persons "
        f"({merged.groupby(['psg_file']).ngroups} nights).")

    lifespan = section_1_lifespan(merged, edges, args.tel_policy)
    drug = section_2_drug(merged)

    if not args.no_figure:
        banner("Section 3 — Figures")
        make_forest(lifespan, FIG_FOREST)
        make_drug_panel(
            drug.get("by_condition", pd.DataFrame()),
            drug.get("paired_long", pd.DataFrame()),
            FIG_DRUG,
        )

    log("\nDone.")


if __name__ == "__main__":
    main()
