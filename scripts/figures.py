#!/usr/bin/env python3
"""
figures.py
==========

Single figure-generation script. By default it creates every manuscript /
supplementary figure into ``outputs/figures/``. Use ``--only`` to make a
subset, e.g. ``--only heatmap,k0_subject_lines``.

Available figures (id -> file):

    heatmap                 robustness_heatmap_K0_tot_REM-W.png       Figure 1
    fig2_example            fig2_panels.png + four sub-panels         Figure 2
    wake_counts             wake_subclass_counts.png
    qc_distributions        qc_feature_distributions.png
    boxplot_raw             raw_tda_boxplots_h1tot.png
    boxplot_corrected       corrected_tda_boxplots_h1tot.png
    boxplot_eog             eog_tda_boxplots_h1tot.png
    contrast_estimates      contrast_estimates_h1tot.png
    k0_mean_sem             fig_k0tot_by_stage_mean_sem.png
    k0_subject_lines        fig_k0tot_by_stage_subject_lines.png
    k0_rem_contrasts        fig_k0tot_rem_contrasts_subject_lines.png

Most figures only need CSVs from ``outputs/`` and run instantly. Figure 2
(``fig2_example``) re-processes a single PSG/Hypnogram EDF pair to draw
the embedded trajectory and persistence diagrams, so it requires the
dataset to be present.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Sequence

# Windows: force UTF-8 stdout so unicode characters in our prints don't trip
# up the default cp1252 codec.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

STAGES_WAKE = ["W_quiet", "W_active_ocular", "REM"]
STAGE_LABELS = {"W_quiet": "Quiet wake", "W_active_ocular": "Active wake", "REM": "REM"}
QC_FEATURES = ["eog_rms", "eog_peak_count", "eeg_eog_corr", "emg_rms"]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_csv(name: str) -> pd.DataFrame | None:
    p = OUT_DIR / name
    if not p.exists():
        print(f"  ! missing {name}")
        return None
    return pd.read_csv(p)

def _save(fig, fname: str, dpi: int = 200):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / fname
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.relative_to(PROJECT_ROOT)}")

def _within_subject_z(df: pd.DataFrame, src: str = "H1_totpers", dst: str = "K0_tot") -> pd.DataFrame:
    df = df.copy()
    df[dst] = df.groupby("subject")[src].transform(
        lambda x: (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — K0 robustness heatmap
# ──────────────────────────────────────────────────────────────────────────────

def fig_heatmap(metric: str = "K0_tot", contrast: str = "REM - W"):
    df = _load_csv("tda_robustness_mixedlm_planned_contrasts.csv")
    if df is None:
        return
    d = df[(df["metric"] == metric) & (df["contrast"] == contrast)]
    if d.empty:
        print(f"  ! no rows for metric={metric} contrast={contrast}")
        return
    channels = sorted(d["channel"].unique()) if "channel" in d.columns else [None]
    for ch in channels:
        sub = d if ch is None else d[d["channel"] == ch]
        table = sub.pivot_table(index="m", columns="tau", values="estimate", aggfunc="mean")
        table = table.reindex(sorted(table.index)).reindex(columns=sorted(table.columns), copy=False)
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(table.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(table.columns))); ax.set_xticklabels([str(x) for x in table.columns])
        ax.set_yticks(range(len(table.index)));    ax.set_yticklabels([str(x) for x in table.index])
        ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$m$")
        title = f"{metric}: {contrast} estimate across (m, τ)"
        if ch is not None:
            title += f"\n{ch}"
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="estimate")
        suffix = f"_{ch.replace(' ', '_')}" if ch else ""
        fig.tight_layout()
        _save(fig, f"robustness_heatmap_{metric}_{contrast.replace(' ', '')}{suffix}.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — example trajectory + persistence diagrams
# ──────────────────────────────────────────────────────────────────────────────

def fig2_example():
    """Re-process one PSG to draw embedded trajectories and PDs for Wake vs REM."""
    try:
        import mne
        from ripser import ripser
        from persim import plot_diagrams
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as ex:
        print(f"  ! skipping fig2_example ({type(ex).__name__}: {ex})")
        return
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        from pipeline import (
            resolve_data_root, discover_pairs, load_intervals, stage_at,
            time_delay_embedding, EEG_PRIMARY, LOWCUT, HIGHCUT, TARGET_SFREQ,
            EPOCH_SEC, EMBED_M, EMBED_TAU, MIN_EMBED_POINTS_MAIN, MAXDIM,
        )
    except Exception as ex:
        print(f"  ! could not import pipeline helpers ({ex})")
        return

    data_root = resolve_data_root()
    pairs = discover_pairs(data_root)
    if not pairs:
        print("  ! no PSG/Hypnogram pairs found")
        return

    def _epochs_for_stage(p, target_stage: str, n_pick: int = 1):
        raw = mne.io.read_raw_edf(str(p.psg_path), preload=False, verbose="ERROR")
        ch = EEG_PRIMARY if EEG_PRIMARY in raw.ch_names else next(
            (c for c in raw.ch_names if "EEG" in c.upper()), None)
        if ch is None:
            return []
        raw.pick([ch]); raw.load_data()
        raw.filter(LOWCUT, HIGHCUT, verbose="ERROR")
        raw.resample(TARGET_SFREQ, verbose="ERROR")
        x = raw.get_data()[0]; sf = float(raw.info["sfreq"])
        L = int(EPOCH_SEC * sf); n = len(x) // L
        intervals = load_intervals(p.hyp_path)
        chosen = []
        for e in range(n):
            mid = (e + 0.5) * L / sf
            if stage_at(intervals, mid) == target_stage:
                chosen.append(x[e * L:(e + 1) * L])
                if len(chosen) >= n_pick:
                    break
        return chosen

    wake_seg = rem_seg = None
    for p in pairs:
        if wake_seg is None:
            ws = _epochs_for_stage(p, "W", 1)
            if ws: wake_seg = ws[0]
        if rem_seg is None:
            rs = _epochs_for_stage(p, "REM", 1)
            if rs: rem_seg = rs[0]
        if wake_seg is not None and rem_seg is not None:
            break
    if wake_seg is None or rem_seg is None:
        print("  ! could not find both Wake and REM example epochs")
        return

    def _embed(seg):
        seg = seg[::2]
        X = time_delay_embedding(seg, EMBED_M, EMBED_TAU, MIN_EMBED_POINTS_MAIN)
        if X is None:
            return None
        return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)

    Xw = _embed(wake_seg); Xr = _embed(rem_seg)
    Dw = ripser(Xw, maxdim=MAXDIM)["dgms"]
    Dr = ripser(Xr, maxdim=MAXDIM)["dgms"]

    # Panel A — wake trajectory (3-D projection of m=10 embedding)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(Xw[:, 0], Xw[:, 1], Xw[:, 2], lw=0.5)
    ax.set_title(f"Wake (delay-embedded), m={EMBED_M}, τ={EMBED_TAU}")
    ax.set_xlabel("x(t)"); ax.set_ylabel("x(t+τ)"); ax.set_zlabel("x(t+2τ)")
    _save(fig, "fig2A_wake_trajectory.png", dpi=300)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(Xr[:, 0], Xr[:, 1], Xr[:, 2], lw=0.5, color="tab:orange")
    ax.set_title(f"REM (delay-embedded), m={EMBED_M}, τ={EMBED_TAU}")
    ax.set_xlabel("x(t)"); ax.set_ylabel("x(t+τ)"); ax.set_zlabel("x(t+2τ)")
    _save(fig, "fig2B_rem_trajectory.png", dpi=300)

    fig = plt.figure(figsize=(5, 4))
    plot_diagrams([Dw[1]], labels=["$H_1$"], show=False)
    plt.title("Wake $H_1$ persistence")
    plt.xlabel("Birth"); plt.ylabel("Death")
    _save(fig, "fig2C_wake_h1_pd.png", dpi=300)

    fig = plt.figure(figsize=(5, 4))
    plot_diagrams([Dr[1]], labels=["$H_1$"], show=False)
    plt.title("REM $H_1$ persistence")
    plt.xlabel("Birth"); plt.ylabel("Death")
    _save(fig, "fig2D_rem_h1_pd.png", dpi=300)

    # Combined 2x2
    import matplotlib.image as mpimg
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    pieces = [("fig2A_wake_trajectory.png", "A: Wake trajectory"),
              ("fig2B_rem_trajectory.png", "B: REM trajectory"),
              ("fig2C_wake_h1_pd.png",     "C: Wake $H_1$"),
              ("fig2D_rem_h1_pd.png",      "D: REM $H_1$")]
    for ax, (fn, t) in zip(axes.flat, pieces):
        img = mpimg.imread(FIG_DIR / fn)
        ax.imshow(img); ax.axis("off"); ax.set_title(t)
    fig.tight_layout()
    _save(fig, "fig2_panels.png", dpi=300)


# ──────────────────────────────────────────────────────────────────────────────
# Wake subclass counts + QC distributions
# ──────────────────────────────────────────────────────────────────────────────

def fig_wake_counts():
    qc = _load_csv("wake_qc_epoch_table.csv")
    if qc is None:
        return
    wcol = "stage_grouped" if "stage_grouped" in qc.columns else "stage_original"
    w = qc[qc[wcol] == "W"]
    counts = (w["wake_subclass"].value_counts()
              .reindex(["W_quiet", "W_active_ocular", "W_bad"]).fillna(0))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(counts.index.tolist(), counts.values.tolist())
    ax.set_ylabel("Epoch count")
    ax.set_title("Wake subclass counts")
    fig.tight_layout()
    _save(fig, "wake_subclass_counts.png")


def fig_qc_distributions():
    qc = _load_csv("wake_qc_epoch_table.csv")
    if qc is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, feat in zip(axes.flat, QC_FEATURES):
        if feat not in qc.columns:
            ax.set_visible(False); continue
        data = []; labels = []
        for s in ["W_quiet", "W_active_ocular", "W_bad"]:
            v = qc.loc[qc["wake_subclass"] == s, feat].dropna()
            if len(v):
                data.append(v); labels.append(s)
        if not data:
            ax.set_visible(False); continue
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(feat)
        for label in ax.get_xticklabels():
            label.set_rotation(20)
    fig.suptitle("QC feature distributions across wake subclasses")
    fig.tight_layout()
    _save(fig, "qc_feature_distributions.png")


# ──────────────────────────────────────────────────────────────────────────────
# Boxplots: H1_totpers across wake subclasses + REM, per signal source
# ──────────────────────────────────────────────────────────────────────────────

def _boxplot_h1tot(csv_name: str, title: str, fname: str):
    df = _load_csv(csv_name)
    if df is None:
        return
    df = df[df["stage"].isin(STAGES_WAKE)]
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [df.loc[df["stage"] == s, "H1_totpers"].dropna().to_numpy() for s in STAGES_WAKE]
    ax.boxplot(data, labels=STAGES_WAKE, showfliers=False)
    ax.set_ylabel("H1_totpers"); ax.set_title(title)
    fig.tight_layout()
    _save(fig, fname)

def fig_boxplot_raw():
    _boxplot_h1tot("tda_epoch_features_wake_raw.csv",
                   "Raw EEG: H1_totpers by stage", "raw_tda_boxplots_h1tot.png")

def fig_boxplot_corrected():
    _boxplot_h1tot("tda_epoch_features_wake_corrected.csv",
                   "EOG-corrected EEG: H1_totpers by stage", "corrected_tda_boxplots_h1tot.png")

def fig_boxplot_eog():
    _boxplot_h1tot("tda_epoch_features_eog.csv",
                   "EOG channel: H1_totpers by stage", "eog_tda_boxplots_h1tot.png")


# ──────────────────────────────────────────────────────────────────────────────
# Contrast estimates: H1_totpers across raw / corrected / EOG
# ──────────────────────────────────────────────────────────────────────────────

def fig_contrast_estimates():
    sources = [("raw",       "raw_wake_mixedlm_planned_contrasts.csv"),
               ("corrected", "corrected_wake_mixedlm_planned_contrasts.csv"),
               ("eog",       "eog_wake_mixedlm_planned_contrasts.csv")]
    frames = []
    for label, name in sources:
        df = _load_csv(name)
        if df is None:
            continue
        if "analysis" not in df.columns:
            df["analysis"] = label
        frames.append(df[df["metric"] == "H1_totpers"].copy())
    if not frames:
        return
    big = pd.concat(frames, ignore_index=True)
    contrast_order = ["REM - W_quiet", "REM - W_active_ocular", "W_quiet - W_active_ocular"]
    analysis_order = ["raw", "corrected", "eog"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x_positions, x_labels = [], []
    pos = 0
    palette = {"raw": "tab:blue", "corrected": "tab:orange", "eog": "tab:green"}
    for c in contrast_order:
        for a in analysis_order:
            row = big[(big["contrast"] == c) & (big["analysis"] == a)]
            if not len(row):
                pos += 1; continue
            r = row.iloc[0]
            est = float(r["estimate"])
            lo = float(r.get("CI95_low", est))
            hi = float(r.get("CI95_high", est))
            ax.errorbar(pos, est, yerr=[[est - lo], [hi - est]], fmt="o",
                        capsize=4, color=palette.get(a, "k"), label=a if pos < 3 else None)
            x_positions.append(pos); x_labels.append(f"{c}\n[{a}]")
            pos += 1
        pos += 1  # gap
    ax.axhline(0.0, linewidth=1, color="grey")
    ax.set_xticks(x_positions); ax.set_xticklabels(x_labels, rotation=25, ha="right")
    ax.set_ylabel("Estimate (95% CI)")
    ax.set_title("Planned contrast estimates for H1_totpers")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # de-duplicate
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        ax.legend(seen.values(), seen.keys(), loc="best")
    fig.tight_layout()
    _save(fig, "contrast_estimates_h1tot.png")


# ──────────────────────────────────────────────────────────────────────────────
# K0_tot bar chart, subject-line plots, REM contrasts
# ──────────────────────────────────────────────────────────────────────────────

def _k0_subject_stage_pivot() -> pd.DataFrame | None:
    df = _load_csv("tda_epoch_features_wake_subclasses.csv")
    if df is None:
        return None
    df = df[df["stage"].isin(STAGES_WAKE)]
    if df.empty:
        return None
    df = _within_subject_z(df, "H1_totpers", "K0_tot")
    return df.groupby(["subject", "stage"])["K0_tot"].mean().unstack("stage")

def fig_k0_mean_sem():
    pivot = _k0_subject_stage_pivot()
    if pivot is None:
        return
    present = [s for s in STAGES_WAKE if s in pivot.columns]
    pivot = pivot[present]
    means = pivot.mean(0).values
    sems = pivot.std(0, ddof=1).values / np.sqrt(pivot.notna().sum(0).values)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(np.arange(len(present)), means, yerr=sems, capsize=5)
    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels([STAGE_LABELS.get(s, s) for s in present])
    ax.set_ylabel("K0_tot"); ax.set_title("K0_tot by stage (mean ± SEM)")
    ax.axhline(0.0, linewidth=1, color="grey")
    fig.tight_layout()
    _save(fig, "fig_k0tot_by_stage_mean_sem.png")

def fig_k0_subject_lines():
    pivot = _k0_subject_stage_pivot()
    if pivot is None:
        return
    present = [s for s in STAGES_WAKE if s in pivot.columns]
    pivot = pivot[present].dropna(axis=0, how="any")
    n = len(pivot)
    if n == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    xpos = np.arange(len(present))
    for _, row in pivot.iterrows():
        ax.plot(xpos, row.values.astype(float), alpha=0.18, linewidth=1)
    means = pivot.mean(0).values
    sems = pivot.std(0, ddof=1).values / np.sqrt(n)
    ax.plot(xpos, means, linewidth=3, marker="o")
    ax.errorbar(xpos, means, yerr=sems, fmt="none", capsize=5)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{STAGE_LABELS.get(s,s)}\n(n={n})" for s in present])
    ax.set_ylabel("K0_tot")
    ax.set_title("K0_tot by stage (paired subject means)")
    ax.axhline(0.0, linewidth=1, color="grey")
    fig.tight_layout()
    _save(fig, "fig_k0tot_by_stage_subject_lines.png")

def fig_k0_rem_contrasts():
    pivot = _k0_subject_stage_pivot()
    if pivot is None:
        return
    fig, axes = plt.subplots(figsize=(10, 5), ncols=2)
    for ax, (a, b) in zip(axes, [("W_quiet", "REM"), ("W_active_ocular", "REM")]):
        if a not in pivot.columns or b not in pivot.columns:
            ax.set_visible(False); continue
        d = pivot[[a, b]].dropna()
        n = len(d); xpos = np.arange(2)
        for _, row in d.iterrows():
            ax.plot(xpos, row.values.astype(float), alpha=0.18, linewidth=1)
        means = d.mean(0).values
        sems = d.std(0, ddof=1).values / np.sqrt(n)
        ax.plot(xpos, means, linewidth=3, marker="o")
        ax.errorbar(xpos, means, yerr=sems, fmt="none", capsize=5)
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"{STAGE_LABELS[a]}\n(n={n})", f"{STAGE_LABELS[b]}\n(n={n})"])
        ax.set_ylabel("K0_tot"); ax.axhline(0.0, linewidth=1, color="grey")
        ax.set_title(f"{STAGE_LABELS[a]} → {STAGE_LABELS[b]}")
    fig.suptitle("K0_tot REM contrasts (paired subject means)")
    fig.tight_layout()
    _save(fig, "fig_k0tot_rem_contrasts_subject_lines.png")


# ──────────────────────────────────────────────────────────────────────────────
# Main

# ──────────────────────────────────────────────────────────────────────────────
# Revision-round figures
# ──────────────────────────────────────────────────────────────────────────────

def fig_all_pairwise():
    """Heatmap of all 10 pairwise stage contrasts on K0."""
    df = _load_csv("stage_all_pairwise_contrasts.csv")
    if df is None or df.empty:
        return
    stages = ["W", "N1", "N2", "N3", "REM"]
    M = np.full((len(stages), len(stages)), np.nan)
    P = np.full_like(M, np.nan)
    for _, r in df.iterrows():
        a, b = str(r["contrast"]).split(" - ")
        if a in stages and b in stages:
            i, j = stages.index(a), stages.index(b)
            M[i, j] = float(r["estimate"])
            M[j, i] = -float(r["estimate"])
            P[i, j] = float(r.get("p_holm", np.nan))
            P[j, i] = float(r.get("p_holm", np.nan))
    fig, ax = plt.subplots(figsize=(7, 6))
    vlim = np.nanmax(np.abs(M)) if np.isfinite(np.nanmax(np.abs(M))) else 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages)
    ax.set_yticks(range(len(stages))); ax.set_yticklabels(stages)
    ax.set_xlabel("subtracted stage"); ax.set_ylabel("reference stage")
    for i in range(len(stages)):
        for j in range(len(stages)):
            if i == j or not np.isfinite(M[i, j]):
                continue
            star = ""
            if np.isfinite(P[i, j]):
                if P[i, j] < 0.001:   star = "***"
                elif P[i, j] < 0.01:  star = "**"
                elif P[i, j] < 0.05:  star = "*"
            ax.text(j, i, f"{M[i,j]:.2f}\n{star}",
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(M[i,j]) > 0.5 * vlim else "black")
    fig.colorbar(im, ax=ax, label="K0_tot estimate")
    ax.set_title("All-pairwise stage contrasts (K0_tot)\nHolm-corrected: * p<.05, ** p<.01, *** p<.001")
    fig.tight_layout()
    _save(fig, "all_pairwise_contrasts_heatmap.png")


def fig_subsampling_stability():
    """REM−W estimate vs cap, error bars across replicates."""
    df = _load_csv("subsampling_stability_contrasts.csv")
    if df is None or df.empty:
        return
    df = df[df["contrast"] == "REM - W"].copy()
    if df.empty:
        return
    g = df.groupby("cap")["estimate"]
    means, sds, caps = g.mean(), g.std(ddof=1), sorted(df["cap"].unique())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(caps, [means[c] for c in caps],
                yerr=[sds.get(c, 0) for c in caps], fmt="o-", capsize=4)
    ax.axhline(0.0, linewidth=1, color="grey")
    ax.set_xlabel("Epochs sampled per stage per night (cap)")
    ax.set_ylabel("REM − W estimate (K0_tot units)")
    ax.set_title("Subsampling stability of the REM − W contrast\n(mean ± SD across 10 replicates per cap)")
    fig.tight_layout()
    _save(fig, "subsampling_stability.png")


def fig_bootstrap_distribution():
    """For each headline contrast, draw the bootstrap CI as a horizontal bar."""
    df = _load_csv("bootstrap_contrasts.csv")
    if df is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(df))
    ax.errorbar(df["boot_mean"], y,
                xerr=[df["boot_mean"] - df["boot_ci95_low"],
                      df["boot_ci95_high"] - df["boot_mean"]],
                fmt="o", capsize=5, color="tab:blue", label="bootstrap mean (95% CI)")
    ax.scatter(df["point_estimate"], y, marker="x", color="tab:red", label="point estimate")
    ax.set_yticks(y); ax.set_yticklabels(df["contrast"])
    ax.axvline(0.0, linewidth=1, color="grey")
    ax.set_xlabel("Estimate (K0_tot units)")
    ax.set_title("Bootstrap 95% CIs (1000 subject-level resamples)")
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, "bootstrap_contrast_cis.png")


def fig_embedding_diagnostics():
    """AMI(τ) and FNN(m) curves with the chosen value highlighted."""
    ami = _load_csv("embedding_ami.csv")
    fnn = _load_csv("embedding_fnn.csv")
    if ami is None and fnn is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if ami is not None and not ami.empty:
        m_ami = ami.groupby("tau")["ami"].mean()
        s_ami = ami.groupby("tau")["ami"].std(ddof=1)
        axes[0].errorbar(m_ami.index, m_ami.values, yerr=s_ami.values,
                         fmt="o-", capsize=3)
        axes[0].axvline(2, color="tab:red", linestyle="--", label="chosen τ=2")
        axes[0].set_xlabel("τ (samples at 50 Hz)")
        axes[0].set_ylabel("Average mutual information")
        axes[0].set_title("AMI(τ): first local minimum suggests optimal τ")
        axes[0].legend()
    if fnn is not None and not fnn.empty:
        m_fnn = fnn.groupby("m")["fnn_fraction"].mean()
        s_fnn = fnn.groupby("m")["fnn_fraction"].std(ddof=1)
        axes[1].errorbar(m_fnn.index, m_fnn.values, yerr=s_fnn.values,
                         fmt="o-", capsize=3, color="tab:green")
        axes[1].axhline(0.05, linewidth=1, color="grey", linestyle=":")
        axes[1].axvline(10, color="tab:red", linestyle="--", label="chosen m=10")
        axes[1].set_xlabel("m (embedding dimension)")
        axes[1].set_ylabel("FNN fraction")
        axes[1].set_title("FNN(m): first m where FNN<5% suggests optimal m")
        axes[1].legend()
    fig.tight_layout()
    _save(fig, "embedding_diagnostics.png")


def fig_pz_oz_contrasts():
    """Side-by-side: Fpz-Cz vs Pz-Oz planned-contrast estimates."""
    fpz = _load_csv("tda_robustness_mixedlm_planned_contrasts.csv")
    pz = _load_csv("tda_pz_oz_mixedlm_planned_contrasts.csv")
    if pz is None or pz.empty:
        return
    pz = pz.copy(); pz["channel"] = "EEG Pz-Oz"
    if fpz is not None and not fpz.empty:
        fpz = fpz[(fpz.get("channel", "") == "EEG Fpz-Cz") &
                  (fpz["metric"] == "K0_tot") &
                  (fpz["m"] == 10) & (fpz["tau"] == 2)].copy() \
              if "channel" in fpz.columns else fpz.copy()
        big = pd.concat([fpz, pz], ignore_index=True)
    else:
        big = pz
    if big.empty:
        return
    contrasts = ["REM - W", "REM - N3", "N1 - N3"]
    big = big[big["contrast"].isin(contrasts)]
    if big.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"EEG Fpz-Cz": "tab:blue", "EEG Pz-Oz": "tab:orange"}
    pos = 0; xs, labels = [], []
    for c in contrasts:
        for ch in ["EEG Fpz-Cz", "EEG Pz-Oz"]:
            row = big[(big["contrast"] == c) & (big["channel"] == ch)]
            if not len(row):
                pos += 1; continue
            r = row.iloc[0]
            est = float(r["estimate"])
            lo = float(r.get("CI95_low", est)); hi = float(r.get("CI95_high", est))
            ax.errorbar(pos, est, yerr=[[est - lo], [hi - est]], fmt="o",
                        capsize=4, color=palette[ch], label=ch if pos < 2 else None)
            xs.append(pos); labels.append(f"{c}\n[{ch.split()[1]}]")
            pos += 1
        pos += 1
    ax.axhline(0.0, linewidth=1, color="grey")
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Estimate (95% CI)")
    ax.set_title("K0_tot planned contrasts: Fpz-Cz vs Pz-Oz")
    handles, lbls = ax.get_legend_handles_labels()
    if handles:
        seen = {}
        for h, l in zip(handles, lbls):
            if l not in seen: seen[l] = h
        ax.legend(seen.values(), seen.keys(), loc="best")
    fig.tight_layout()
    _save(fig, "pz_oz_contrasts.png")


def fig_preproc_sensitivity():
    """Heatmap of REM−W estimate across (bandpass, sfreq)."""
    df = _load_csv("preprocessing_sensitivity_contrasts.csv")
    if df is None or df.empty:
        return
    df = df[df["contrast"] == "REM - W"].copy()
    if df.empty:
        return
    df["bp"] = df.apply(lambda r: f"{r['bandpass_low']}–{r['bandpass_high']} Hz", axis=1)
    table = df.pivot_table(index="bp", columns="sfreq_target",
                           values="estimate", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(table.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([f"{c:g} Hz" for c in table.columns])
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    ax.set_xlabel("Resample target"); ax.set_ylabel("Bandpass")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="REM − W estimate (K0_tot)")
    ax.set_title("Preprocessing sensitivity of the REM − W contrast")
    fig.tight_layout()
    _save(fig, "preprocessing_sensitivity.png")


def fig_classification_summary():
    """Bar chart of mean LOSO AUC across (target × feature_set), one panel per model."""
    df = _load_csv("classification_summary.csv")
    if df is None or df.empty:
        return
    targets = ["REM_vs_W", "REM_vs_NREM", "REM_vs_other"]
    fsets = ["K0_only", "bandpower_only", "combined"]
    df = df[df["target"].isin(targets) & df["feature_set"].isin(fsets)]
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
    palette = {"K0_only": "tab:red", "bandpower_only": "tab:blue", "combined": "tab:purple"}
    for ax, model in zip(axes[0], models):
        sub = df[df["model"] == model]
        n_t = len(targets); n_fs = len(fsets); width = 0.25
        x = np.arange(n_t)
        for i, fs in enumerate(fsets):
            heights, errs = [], []
            for t in targets:
                row = sub[(sub["target"] == t) & (sub["feature_set"] == fs)]
                if len(row):
                    r = row.iloc[0]
                    heights.append(float(r["auc_mean"]))
                    errs.append(float(r["auc_sd"]))
                else:
                    heights.append(np.nan); errs.append(0.0)
            ax.bar(x + (i - 1) * width, heights, width=width, yerr=errs,
                   capsize=3, label=fs, color=palette.get(fs, None))
        ax.axhline(0.5, color="grey", linewidth=1, linestyle=":")
        ax.set_xticks(x); ax.set_xticklabels(targets, rotation=15)
        ax.set_ylabel("LOSO AUC (mean ± SD across folds)")
        ax.set_title(model)
        ax.set_ylim(0.4, 1.0)
        ax.legend(loc="lower right")
    fig.suptitle("LOSO classification: K0 vs band power vs combined")
    fig.tight_layout()
    _save(fig, "classification_summary.png")


def fig_cohort_replication():
    """REM−W estimate per cohort (Cassette vs Telemetry), K0 vs each baseline metric."""
    df = _load_csv("cohort_replication_contrasts.csv")
    if df is None or df.empty:
        return
    df = df[df["contrast"] == "REM - W"].copy()
    if df.empty:
        return
    # Order rows by metric
    metric_order = ["K0_tot", "spec_entropy", "perm_entropy", "lz_complexity",
                    "log_delta", "log_theta", "log_alpha", "log_beta", "log_sigma"]
    df["order"] = df["metric"].apply(
        lambda m: metric_order.index(m) if m in metric_order else len(metric_order))
    df = df.sort_values(["order", "cohort"])
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = {"Cassette": "tab:blue", "Telemetry": "tab:orange"}
    pos = 0; ys, lbls = [], []
    for met, g in df.groupby("metric", sort=False):
        for cohort in ["Cassette", "Telemetry"]:
            row = g[g["cohort"] == cohort]
            if not len(row):
                pos += 1; continue
            r = row.iloc[0]
            est = float(r["estimate"])
            lo = float(r.get("CI95_low", est)); hi = float(r.get("CI95_high", est))
            ax.errorbar(est, pos, xerr=[[est - lo], [hi - est]], fmt="o",
                        capsize=3, color=palette[cohort],
                        label=cohort if pos < 2 else None)
            ys.append(pos); lbls.append(f"{met} [{cohort}]"); pos += 1
        pos += 0.5
    ax.axvline(0.0, linewidth=1, color="grey")
    ax.set_yticks(ys); ax.set_yticklabels(lbls, fontsize=8)
    ax.set_xlabel("REM − W estimate (per-metric units)")
    ax.set_title("Cohort replication: Cassette vs Telemetry")
    handles, hl = ax.get_legend_handles_labels()
    if handles:
        seen = {}
        for h, l in zip(handles, hl):
            if l not in seen: seen[l] = h
        ax.legend(seen.values(), seen.keys(), loc="best")
    fig.tight_layout()
    _save(fig, "cohort_replication.png")


# ──────────────────────────────────────────────────────────────────────────────

ALL_FIGURES = {
    "heatmap":             fig_heatmap,
    "fig2_example":        fig2_example,
    "wake_counts":         fig_wake_counts,
    "qc_distributions":    fig_qc_distributions,
    "boxplot_raw":         fig_boxplot_raw,
    "boxplot_corrected":   fig_boxplot_corrected,
    "boxplot_eog":         fig_boxplot_eog,
    "contrast_estimates":  fig_contrast_estimates,
    "k0_mean_sem":         fig_k0_mean_sem,
    "k0_subject_lines":    fig_k0_subject_lines,
    "k0_rem_contrasts":    fig_k0_rem_contrasts,
    # Revision-round additions
    "all_pairwise":        fig_all_pairwise,
    "subsampling":         fig_subsampling_stability,
    "bootstrap":           fig_bootstrap_distribution,
    "embedding_diag":      fig_embedding_diagnostics,
    "pz_oz":               fig_pz_oz_contrasts,
    "preproc_sensitivity": fig_preproc_sensitivity,
    "classification":      fig_classification_summary,
    "cohort_replication":  fig_cohort_replication,
}


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate every manuscript / supplementary figure.")
    p.add_argument("--only", type=str, default="",
                   help=f"Comma-separated subset of figure ids ({', '.join(ALL_FIGURES)}).")
    p.add_argument("--skip-fig2", action="store_true",
                   help="Skip the Figure 2 example panels (which require raw EDF access).")
    args = p.parse_args(argv)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    selected = list(ALL_FIGURES.keys())
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        selected = [s for s in wanted if s in ALL_FIGURES]
        if not selected:
            print("No valid figure ids in --only.")
            return 1
    if args.skip_fig2 and "fig2_example" in selected:
        selected.remove("fig2_example")

    print(f"Writing figures to {FIG_DIR.relative_to(PROJECT_ROOT)}/")
    for fid in selected:
        print(f"\n• {fid}")
        try:
            ALL_FIGURES[fid]()
        except Exception as ex:
            print(f"  ✗ {fid} failed: {type(ex).__name__}: {ex}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
