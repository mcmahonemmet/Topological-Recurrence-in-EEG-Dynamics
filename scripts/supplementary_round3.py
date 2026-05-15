#!/usr/bin/env python3
"""
supplementary_round3.py
=======================

Standalone supplementary analyses for the round-3 manuscript revision.
Runs four independent analyses and writes CSVs / figures into ``outputs/``.

Sections
--------
  A. Diagnostic-favoured embedding cell.
        Re-run the headline TDA on Fpz-Cz at the AMI-suggested tau (=11) and
        FNN-favoured m (default {6, 10, 12}), then refit the REM-Wake planned
        contrast at each (m, tau) cell. Confirms the result survives at the
        embedding diagnostically motivated by AMI/FNN.

  B. LOSO classification refinements.
        Re-run leave-one-subject-out logistic regression for REM-vs-Wake at
        K0_only / bandpower_only / combined feature sets, saving per-epoch
        held-out scores so we can compute (i) bootstrap 95% CIs on AUCs, (ii)
        a paired subject-level bootstrap test for whether combined > bandpower,
        and (iii) pooled ROC curves. Also produces a K0 distribution figure
        across W_quiet / W_active_ocular / REM.

  C. ICA-based ocular-artefact sensitivity (Pz-Oz).
        On a subset of subjects, run ICA on the two EEG channels with the
        EOG horizontal channel as a reference, drop EOG-correlated components,
        recompute K0 on the cleaned Pz-Oz signal, and refit the REM-Wake
        contrast. Provides an artefact-rejection check beyond the
        linear-regression EOG correction reported in the main pipeline.

  D. Permutation-entropy order sensitivity.
        Recompute permutation entropy at orders 3, 4, 5, 6 on Fpz-Cz and refit
        the REM-Wake mixed-LM planned contrast at each order, to show the
        comparator result is not driven by the original order=5 choice.

Usage
-----
  python scripts/supplementary_round3.py                 # run all four sections
  python scripts/supplementary_round3.py --only A,B      # subset
  python scripts/supplementary_round3.py --n-jobs 8
  python scripts/supplementary_round3.py --ica-subset 20
  python scripts/supplementary_round3.py --force         # rerun, ignore cached

Reproducibility
---------------
RNG_SEED = 0 (inherited from pipeline.py); per-night sub-seeds are spawned
via ``np.random.SeedSequence`` so output is bit-for-bit identical regardless
of ``--n-jobs``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Force UTF-8 stdio so banner characters render on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd

# Make pipeline.py importable.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import pipeline as P  # noqa: E402  -- imports must follow the path tweak

import mne
from ripser import ripser
from scipy import stats
import statsmodels.formula.api as smf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Output paths
# ──────────────────────────────────────────────────────────────────────────────

OUT_DIR = P.OUT_DIR
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Section A
A_EPOCHS  = OUT_DIR / "supp_a_diagnostic_embedding_epoch_features.csv"
A_CONTR   = OUT_DIR / "supp_a_diagnostic_embedding_planned_contrasts.csv"

# Section B
B_LOSO    = OUT_DIR / "supp_b_loso_logistic_predictions.csv"
B_AUC_CI  = OUT_DIR / "supp_b_loso_auc_bootstrap_ci.csv"
B_PAIRED  = OUT_DIR / "supp_b_loso_paired_bootstrap_combined_vs_bandpower.csv"
B_ROC_FIG = FIG_DIR / "supp_b_roc_curves_REM_vs_W.png"
B_K0_FIG  = FIG_DIR / "supp_b_k0_distribution_wsubclass.png"

# Section C
C_DROPPED = OUT_DIR / "supp_c_ica_components_dropped.csv"
C_EPOCHS  = OUT_DIR / "supp_c_ica_pz_oz_epoch_features.csv"
C_CONTR   = OUT_DIR / "supp_c_ica_pz_oz_planned_contrasts.csv"

# Section D
D_EPOCHS  = OUT_DIR / "supp_d_pe_order_epoch_features.csv"
D_CONTR   = OUT_DIR / "supp_d_pe_order_planned_contrasts.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Tiny logger
# ──────────────────────────────────────────────────────────────────────────────

def banner(s: str):
    bar = "═" * max(len(s) + 4, 60)
    print(f"\n{bar}\n  {s}\n{bar}", flush=True)

def log(s: str):
    print(s, flush=True)

def _save_csv(df: pd.DataFrame, path: Path):
    if df is None or df.empty:
        path.write_text("# empty\n", encoding="utf-8")
        log(f"  · wrote {path.name} (empty placeholder)")
        return
    df.to_csv(path, index=False)
    log(f"  · wrote {path.name} ({len(df)} rows)")


# ──────────────────────────────────────────────────────────────────────────────
# Section A — Diagnostic-favoured embedding cell
# ──────────────────────────────────────────────────────────────────────────────

def _worker_tda_at_cell(
    pair: P.Pair,
    sub_seed: np.random.SeedSequence,
    cells: Sequence[Tuple[int, int]],
) -> List[dict]:
    """Per-night TDA on Fpz-Cz across W/N1/N2/N3/REM at the given (m, tau) cells.

    Mirrors pipeline.worker_main_tda (same epoch sampling, same downsampling)
    but parametrises the embedding so the diagnostic-favoured cells can be
    tested without disturbing the main pipeline's outputs.
    """
    rng = np.random.default_rng(sub_seed)
    try:
        x, sf, ch = P._load_eeg(pair.psg_path)
        intervals = P.load_intervals(pair.hyp_path)
        epoch_len = int(P.EPOCH_SEC * sf)
        n_epochs = len(x) // epoch_len
        by_stage: Dict[str, List[int]] = {s: [] for s in P.STAGES_MAIN}
        for e in range(n_epochs):
            s = P.stage_at(intervals, (e + 0.5) * epoch_len / sf)
            if s in by_stage:
                by_stage[s].append(e)
        # Per-stage epoch sampling (identical RNG draw to worker_main_tda for
        # the first cell; subsequent cells reuse the same indices).
        sampled: Dict[str, List[int]] = {}
        for s, idxs in by_stage.items():
            if not idxs:
                continue
            if len(idxs) > P.MAX_EPOCHS_MAIN:
                sampled[s] = list(rng.choice(idxs, size=P.MAX_EPOCHS_MAIN, replace=False))
            else:
                sampled[s] = idxs
        out = []
        psg_tag = P._tag(P._prefix(pair.psg_path))
        for stage, idxs in sampled.items():
            for e in idxs:
                seg = x[e * epoch_len:(e + 1) * epoch_len]
                # Apply the same downsample-by-2 as the pipeline.
                seg_ds = seg[::2]
                for (m, tau) in cells:
                    # min_pts = MIN_EMBED_POINTS_MAIN matches main pipeline
                    X = P.time_delay_embedding(seg_ds, m, tau, P.MIN_EMBED_POINTS_MAIN)
                    if X is None:
                        continue
                    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
                    dgms = ripser(X, maxdim=P.MAXDIM)["dgms"]
                    h0 = P.dgm_summaries(dgms[0]) if len(dgms) > 0 else {"count":0,"tot_pers":0.0,"max_pers":0.0}
                    h1 = P.dgm_summaries(dgms[1]) if len(dgms) > 1 else {"count":0,"tot_pers":0.0,"max_pers":0.0}
                    out.append({
                        "subject": pair.subject, "psg_tag": psg_tag,
                        "psg_file": pair.psg_path.name, "hyp_file": pair.hyp_path.name,
                        "channel": ch, "stage": stage, "epoch_index": int(e),
                        "m": int(m), "tau": int(tau),
                        "H0_count": h0["count"], "H0_totpers": h0["tot_pers"], "H0_maxpers": h0["max_pers"],
                        "H1_count": h1["count"], "H1_totpers": h1["tot_pers"], "H1_maxpers": h1["max_pers"],
                    })
        return out
    except Exception as ex:
        return [{"_ERROR_": f"{pair.psg_path.name} :: {type(ex).__name__}: {ex}"}]


def section_A_diagnostic_embedding(pairs: List[P.Pair], n_jobs: int, force: bool,
                                   m_list: Sequence[int], tau_value: int,
                                   limit: Optional[int]):
    banner("Section A — Diagnostic-favoured embedding (Fpz-Cz, AMI/FNN cells)")
    if A_EPOCHS.exists() and A_CONTR.exists() and not force:
        log("  ✓ skipping (outputs exist; use --force to rerun)")
        return

    cells = [(m, tau_value) for m in m_list]
    log(f"  · cells: {cells}")
    log(f"  · {len(pairs)} PSG pairs, n_jobs={n_jobs}")

    if limit is not None:
        pairs = pairs[:limit]
        log(f"  · DEBUG --limit {limit} → using first {len(pairs)} pairs only")

    # Spawn deterministic per-night seeds.
    seeds = np.random.SeedSequence(P.RNG_SEED).spawn(len(pairs))
    items = list(zip(pairs, seeds))

    def _wrap(pair_seed):
        pair, seed = pair_seed
        return _worker_tda_at_cell(pair, seed, cells)

    rows = P._parallel_bar(_wrap, items, n_jobs, "TDA @ diagnostic cells")
    flat = []
    errs = []
    for r in rows:
        for d in r:
            if "_ERROR_" in d:
                errs.append(d["_ERROR_"])
            else:
                flat.append(d)
    if errs:
        log(f"  ! {len(errs)} per-night errors; first: {errs[0]}")
    df = pd.DataFrame(flat)
    _save_csv(df, A_EPOCHS)

    if df.empty:
        log("  ! no epoch features computed; skipping contrasts")
        _save_csv(pd.DataFrame(), A_CONTR)
        return

    # K0 within-subject z per (m, tau)
    log("  · fitting mixed-LM REM-W contrast at each (m, tau) cell")
    contrast_rows = []
    for (m, tau) in cells:
        sub = df[(df["m"] == m) & (df["tau"] == tau)].copy()
        if sub.empty:
            continue
        sub["K0_tot"] = sub.groupby("subject")["H1_totpers"].transform(
            lambda x: (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0
        )
        ss = sub.groupby(["subject", "stage"], as_index=False)["K0_tot"].mean()
        try:
            res, lr, df_diff, p_lr = P.fit_mixedlm_stage(
                ss, "K0_tot", P.STAGES_MAIN, method="powell")
        except Exception:
            res, lr, df_diff, p_lr = P.fit_mixedlm_stage(
                ss, "K0_tot", P.STAGES_MAIN, method="lbfgs")
        contr = P.planned_contrasts(res, "K0_tot", P.PLANNED_MAIN, P.STAGES_MAIN)
        contr["m"] = int(m); contr["tau"] = int(tau)
        contr["channel"] = "EEG Fpz-Cz"
        contr["LR_omnibus"] = lr; contr["df_omnibus"] = df_diff; contr["p_omnibus"] = p_lr
        contr["n_subjects"] = ss["subject"].nunique()
        contrast_rows.append(contr)
        rw = contr[contr["contrast"] == "REM - W"].iloc[0]
        log(f"    m={m:>2} tau={tau:>2}: REM-W = {rw['estimate']:+.3f} SD "
            f"[{rw['CI95_low']:+.3f}, {rw['CI95_high']:+.3f}], "
            f"Holm p = {rw['p_holm']:.3e}")
    if contrast_rows:
        out = pd.concat(contrast_rows, ignore_index=True)
        _save_csv(out, A_CONTR)
    else:
        _save_csv(pd.DataFrame(), A_CONTR)


# ──────────────────────────────────────────────────────────────────────────────
# Section B — Classification refinements (CIs, paired bootstrap, ROC, K0 dist)
# ──────────────────────────────────────────────────────────────────────────────

def _refit_loso_with_scores(rng_seed: int = P.RNG_SEED) -> pd.DataFrame:
    """Re-run LOSO logistic for REM-vs-W, saving per-test-epoch held-out
    scores for K0_only / bandpower_only / combined feature sets. Mirrors
    pipeline.step_classification but stores predictions instead of summary
    statistics so we can build pooled ROC curves and bootstrap CIs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    tda  = P._safe_read_csv(OUT_DIR / "tda_epoch_features_all.csv")
    base = P._safe_read_csv(OUT_DIR / "baseline_epoch_features_all.csv")
    if tda.empty or base.empty:
        raise RuntimeError("prerequisite epoch CSVs missing or empty")

    keys = ["subject", "psg_file", "hyp_file", "epoch_index"]
    for d in (tda, base):
        d["epoch_index"] = d["epoch_index"].astype(int)
    t_small = tda[keys + ["stage", "H1_totpers"]]
    b_small = base[keys + list(P.BANDPOWER_COLS)]
    df = t_small.merge(b_small, on=keys, how="inner")
    df = df[df["stage"].isin(["W", "REM"])].copy()
    df["K0_tot"] = df.groupby("subject")["H1_totpers"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0
    )
    df["y"] = (df["stage"] == "REM").astype(int)

    feature_sets = {
        "K0_only":        ["K0_tot"],
        "bandpower_only": list(P.BANDPOWER_COLS),
        "combined":       ["K0_tot"] + list(P.BANDPOWER_COLS),
    }
    subjects = sorted(df["subject"].unique())
    log(f"  · LOSO logistic over {len(subjects)} subjects × 3 feature sets")
    rows = []
    pbar = P._inner_bar(len(subjects) * 3, "LOSO with stored scores")
    for held in subjects:
        tr_mask = df["subject"] != held
        te_mask = df["subject"] == held
        if te_mask.sum() < 5 or df.loc[te_mask, "y"].nunique() < 2:
            if pbar is not None:
                pbar.update(3)
            continue
        for fs_name, fs_cols in feature_sets.items():
            X_tr = df.loc[tr_mask, fs_cols].values
            y_tr = df.loc[tr_mask, "y"].values
            X_te = df.loc[te_mask, fs_cols].values
            y_te = df.loc[te_mask, "y"].values
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                                        random_state=rng_seed))])
            pipe.fit(X_tr, y_tr)
            scores = pipe.predict_proba(X_te)[:, 1]
            for j, idx in enumerate(df.index[te_mask]):
                rows.append({
                    "feature_set": fs_name,
                    "held_out_subject": held,
                    "row_index": int(idx),
                    "y_true": int(y_te[j]),
                    "y_score": float(scores[j]),
                })
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()
    return pd.DataFrame(rows)


def _bootstrap_auc_ci(preds: pd.DataFrame, n_boot: int = 1000,
                      seed: int = P.RNG_SEED) -> pd.DataFrame:
    """Subject-level bootstrap on AUCs. Resample held-out subjects with
    replacement, pool their predictions, recompute AUC, and report 95%
    percentile CI plus mean and SD of the bootstrap distribution.

    Optimisation: per-subject (y_true, y_score) numpy arrays are precomputed
    once per feature set so each bootstrap iteration is a single
    np.concatenate over indices rather than ~200 pandas DataFrame slices.
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    rows = []
    for fs_name, sub in preds.groupby("feature_set"):
        # Pre-bucket per subject as numpy arrays for fast resampling
        per_subj_yt: Dict[str, np.ndarray] = {}
        per_subj_ys: Dict[str, np.ndarray] = {}
        for s, g in sub.groupby("held_out_subject"):
            per_subj_yt[s] = g["y_true"].to_numpy(dtype=np.int8)
            per_subj_ys[s] = g["y_score"].to_numpy(dtype=np.float64)
        subjects = np.array(list(per_subj_yt.keys()))
        # Point estimate: pooled AUC across all held-out epochs
        all_yt = np.concatenate([per_subj_yt[s] for s in subjects])
        all_ys = np.concatenate([per_subj_ys[s] for s in subjects])
        try:
            point = float(roc_auc_score(all_yt, all_ys))
        except ValueError:
            point = np.nan
        boot = np.empty(n_boot, dtype=np.float64)
        n_ok = 0
        for i in range(n_boot):
            picks = rng.choice(subjects, size=len(subjects), replace=True)
            yt = np.concatenate([per_subj_yt[s] for s in picks])
            ys = np.concatenate([per_subj_ys[s] for s in picks])
            if np.unique(yt).size < 2:
                continue
            try:
                boot[n_ok] = float(roc_auc_score(yt, ys)); n_ok += 1
            except ValueError:
                continue
        boot = boot[:n_ok]
        rows.append({
            "feature_set": fs_name,
            "auc_point": point,
            "auc_boot_mean": float(np.mean(boot)) if boot.size else np.nan,
            "auc_boot_sd":   float(np.std(boot, ddof=1)) if boot.size > 1 else np.nan,
            "auc_boot_ci95_low":  float(np.quantile(boot, 0.025)) if boot.size else np.nan,
            "auc_boot_ci95_high": float(np.quantile(boot, 0.975)) if boot.size else np.nan,
            "n_boot_successful": int(boot.size),
            "n_subjects": int(len(subjects)),
        })
    return pd.DataFrame(rows)


def _paired_subject_bootstrap_combined_vs_bandpower(
    preds: pd.DataFrame, n_boot: int = 1000, seed: int = P.RNG_SEED
) -> pd.DataFrame:
    """Subject-level paired bootstrap test for combined > bandpower.

    For each bootstrap iteration: resample subjects with replacement; pool the
    resampled subjects' predictions for both feature sets; compute AUC for each
    feature set on the same resampled subject set; record the difference
    (combined − bandpower). Return point estimate, 95% percentile CI, and a
    one-sided bootstrap p-value (= fraction of resamples with diff <= 0)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    fs_a = "combined"; fs_b = "bandpower_only"
    a = preds[preds["feature_set"] == fs_a]
    b = preds[preds["feature_set"] == fs_b]
    common = sorted(set(a["held_out_subject"]) & set(b["held_out_subject"]))
    if not common:
        return pd.DataFrame()
    # Precompute per-subject numpy arrays for both feature sets
    a_yt = {s: g["y_true"].to_numpy(dtype=np.int8)   for s, g in a.groupby("held_out_subject")}
    a_ys = {s: g["y_score"].to_numpy(dtype=np.float64) for s, g in a.groupby("held_out_subject")}
    b_yt = {s: g["y_true"].to_numpy(dtype=np.int8)   for s, g in b.groupby("held_out_subject")}
    b_ys = {s: g["y_score"].to_numpy(dtype=np.float64) for s, g in b.groupby("held_out_subject")}
    subjects = np.array(common)
    diffs = np.empty(n_boot, dtype=np.float64); n_ok = 0
    for i in range(n_boot):
        picks = rng.choice(subjects, size=len(subjects), replace=True)
        ayt = np.concatenate([a_yt[s] for s in picks])
        ays = np.concatenate([a_ys[s] for s in picks])
        byt = np.concatenate([b_yt[s] for s in picks])
        bys = np.concatenate([b_ys[s] for s in picks])
        if np.unique(ayt).size < 2 or np.unique(byt).size < 2:
            continue
        try:
            auc_a = float(roc_auc_score(ayt, ays))
            auc_b = float(roc_auc_score(byt, bys))
            diffs[n_ok] = auc_a - auc_b; n_ok += 1
        except ValueError:
            continue
    diffs = diffs[:n_ok]
    # Point estimate from the pooled (un-resampled) data
    try:
        all_a_yt = np.concatenate([a_yt[s] for s in subjects])
        all_a_ys = np.concatenate([a_ys[s] for s in subjects])
        all_b_yt = np.concatenate([b_yt[s] for s in subjects])
        all_b_ys = np.concatenate([b_ys[s] for s in subjects])
        point_a = float(roc_auc_score(all_a_yt, all_a_ys))
        point_b = float(roc_auc_score(all_b_yt, all_b_ys))
        point_diff = point_a - point_b
    except ValueError:
        point_a = point_b = point_diff = np.nan
    # Two-sided bootstrap p-value: 2 × min(P(diff ≤ 0), P(diff ≥ 0)),
    # plus a one-sided p for the hypothesis combined > bandpower.
    p_one_sided = float(np.mean(diffs <= 0)) if diffs.size else np.nan
    p_two_sided = (2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
                   if diffs.size else np.nan)
    return pd.DataFrame([{
        "comparison": "combined - bandpower_only",
        "auc_combined_point": point_a,
        "auc_bandpower_point": point_b,
        "auc_diff_point": point_diff,
        "auc_diff_boot_mean": float(np.mean(diffs)) if diffs.size else np.nan,
        "auc_diff_boot_ci95_low":  float(np.quantile(diffs, 0.025)) if diffs.size else np.nan,
        "auc_diff_boot_ci95_high": float(np.quantile(diffs, 0.975)) if diffs.size else np.nan,
        "p_boot_one_sided_combined_gt_bandpower": p_one_sided,
        "p_boot_two_sided": p_two_sided,
        "n_boot_successful": int(diffs.size),
        "n_subjects": int(len(subjects)),
    }])


def _figure_roc_curves(preds: pd.DataFrame, out_path: Path):
    """Pooled ROC curves for K0_only / bandpower_only / combined."""
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    colours = {"K0_only": "#1f77b4", "bandpower_only": "#d62728", "combined": "#2ca02c"}
    pretty = {"K0_only": "K0 only", "bandpower_only": "Band power only", "combined": "Combined"}
    for fs in ["K0_only", "bandpower_only", "combined"]:
        sub = preds[preds["feature_set"] == fs]
        if sub.empty:
            continue
        fpr, tpr, _ = roc_curve(sub["y_true"], sub["y_score"])
        try:
            auc = float(roc_auc_score(sub["y_true"], sub["y_score"]))
        except ValueError:
            auc = float("nan")
        ax.plot(fpr, tpr, color=colours[fs], lw=1.8,
                label=f"{pretty[fs]} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("LOSO ROC curves: REM vs Wake (Fpz-Cz)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  · wrote {out_path.name}")


def _figure_k0_distribution_wsubclass(out_path: Path):
    """Violin plot of K0_tot across W_quiet / W_active_ocular / REM."""
    df = P._safe_read_csv(OUT_DIR / "tda_epoch_features_wake_subclasses.csv")
    if df.empty or "stage" not in df.columns:
        log("  ! tda_epoch_features_wake_subclasses.csv missing/empty; skipping K0-dist figure")
        return
    df = df[df["stage"].isin(["W_quiet", "W_active_ocular", "REM"])].copy()
    if df.empty:
        log("  ! no W_quiet/W_active_ocular/REM rows; skipping K0-dist figure")
        return
    df["K0_tot"] = df.groupby("subject")["H1_totpers"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0
    )
    order = ["W_quiet", "W_active_ocular", "REM"]
    pretty = {"W_quiet": "Quiet wake", "W_active_ocular": "Active-ocular wake", "REM": "REM"}
    data = [df.loc[df["stage"] == s, "K0_tot"].dropna().values for s in order]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    parts = ax.violinplot(data, showextrema=False, widths=0.85)
    palette = ["#7fc97f", "#fdc086", "#beaed4"]
    for pc, c in zip(parts["bodies"], palette):
        pc.set_facecolor(c); pc.set_edgecolor("#444"); pc.set_alpha(0.85)
    bp = ax.boxplot(data, widths=0.18, patch_artist=False, showcaps=False,
                    showfliers=False,
                    medianprops=dict(color="black", lw=1.2),
                    whiskerprops=dict(color="black", lw=0.8),
                    boxprops=dict(color="black", lw=0.8))
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([pretty[s] for s in order])
    ax.set_ylabel("K0 (within-subject z of H1 total persistence)")
    ax.set_title("K0 distributions across wake subclasses and REM")
    ax.axhline(0, color="grey", lw=0.6, ls=":")
    # Annotate subject-level means
    sm = df.groupby(["subject", "stage"], as_index=False)["K0_tot"].mean()
    for j, s in enumerate(order):
        vals = sm.loc[sm["stage"] == s, "K0_tot"].dropna().values
        if vals.size:
            ax.scatter([j + 1] * len(vals), vals, s=8, color="black", alpha=0.35,
                       zorder=3, label="subject mean" if j == 0 else None)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  · wrote {out_path.name}")


def section_B_classification_refinements(force: bool, n_boot: int):
    banner("Section B — LOSO classification refinements (CIs, ROC, K0 dist)")
    have_all = all(p.exists() for p in [B_LOSO, B_AUC_CI, B_PAIRED, B_ROC_FIG, B_K0_FIG])
    if have_all and not force:
        log("  ✓ skipping (outputs exist; use --force to rerun)")
        return

    # 1. Refit LOSO with stored scores
    if B_LOSO.exists() and not force:
        log("  ✓ reusing cached LOSO predictions")
        preds = pd.read_csv(B_LOSO, comment="#")
    else:
        try:
            preds = _refit_loso_with_scores()
        except Exception as ex:
            log(f"  ! LOSO refit failed: {ex}")
            preds = pd.DataFrame()
        _save_csv(preds, B_LOSO)
    if preds.empty:
        log("  ! no predictions; skipping CIs/ROC")
    else:
        # 2. Bootstrap AUC CIs
        log(f"  · bootstrapping AUC CIs ({n_boot} iterations, subject-level resample)")
        ci = _bootstrap_auc_ci(preds, n_boot=n_boot)
        for _, r in ci.iterrows():
            log(f"    {r['feature_set']:>16s}: AUC = {r['auc_point']:.3f} "
                f"[{r['auc_boot_ci95_low']:.3f}, {r['auc_boot_ci95_high']:.3f}]")
        _save_csv(ci, B_AUC_CI)

        # 3. Paired bootstrap test combined vs bandpower
        log("  · paired-subject bootstrap test: combined > bandpower")
        paired = _paired_subject_bootstrap_combined_vs_bandpower(preds, n_boot=n_boot)
        if not paired.empty:
            r = paired.iloc[0]
            log(f"    Δ AUC = {r['auc_diff_point']:+.4f} "
                f"[{r['auc_diff_boot_ci95_low']:+.4f}, {r['auc_diff_boot_ci95_high']:+.4f}], "
                f"one-sided p = {r['p_boot_one_sided_combined_gt_bandpower']:.4f}")
        _save_csv(paired, B_PAIRED)

        # 4. ROC curves figure
        log("  · ROC curves figure")
        _figure_roc_curves(preds, B_ROC_FIG)

    # 5. K0 distribution figure (independent of LOSO predictions)
    log("  · K0 distribution figure (W_quiet / W_active_ocular / REM)")
    _figure_k0_distribution_wsubclass(B_K0_FIG)


# ──────────────────────────────────────────────────────────────────────────────
# Section C — ICA-based ocular-artefact sensitivity for Pz-Oz
# ──────────────────────────────────────────────────────────────────────────────

def _worker_ica_pz_oz(pair: P.Pair, sub_seed: np.random.SeedSequence) -> dict:
    """Per-night ICA + Pz-Oz cleaning + TDA.

    Loads Fpz-Cz, Pz-Oz, and EOG horizontal at full resolution; band-pass
    filters EEG channels to [0.5, 40] Hz and EOG to [0.3, 15] Hz; resamples to
    50 Hz; fits MNE ICA with n_components = 2 (== n EEG channels) and uses the
    EOG channel as a reference for find_bads_eog. Drops any flagged components
    and returns: (a) per-epoch cleaned-Pz-Oz K0 features, (b) the indices of
    components dropped per night.
    """
    rng = np.random.default_rng(sub_seed)
    epochs_out: List[dict] = []
    drop_info: dict = {
        "subject": pair.subject, "psg_file": pair.psg_path.name,
        "n_components": 0, "components_dropped": "",
        "n_components_dropped": 0, "max_eog_corr": np.nan, "error": "",
    }
    try:
        raw = mne.io.read_raw_edf(str(pair.psg_path), preload=True, verbose="ERROR")
        # Find available channels
        wanted_eeg  = [c for c in [P.EEG_PRIMARY, P.EEG_SECONDARY] if c in raw.ch_names]
        wanted_eog  = P.EOG_CH if P.EOG_CH in raw.ch_names else None
        if P.EEG_SECONDARY not in wanted_eeg:
            drop_info["error"] = "Pz-Oz channel missing"
            return {"epochs": [], "drop_info": drop_info}
        if wanted_eog is None:
            drop_info["error"] = "EOG channel missing"
            return {"epochs": [], "drop_info": drop_info}
        keep = wanted_eeg + [wanted_eog]
        raw.pick(keep)
        raw.set_channel_types({wanted_eog: "eog"})
        # Band-pass filter EEG channels and EOG separately by picking before filtering
        raw.filter(P.LOWCUT, P.HIGHCUT, picks=wanted_eeg, verbose="ERROR")
        raw.filter(P.EOG_LOW, P.EOG_HIGH, picks=[wanted_eog], verbose="ERROR")
        raw.resample(P.TARGET_SFREQ, verbose="ERROR")

        # Fit ICA on EEG channels, with EOG passed as a reference signal to
        # find_bads_eog. n_components must be ≤ number of input EEG channels.
        n_eeg = len(wanted_eeg)
        ica = mne.preprocessing.ICA(
            n_components=n_eeg, method="fastica",
            random_state=int(rng.integers(0, 2**31 - 1)), max_iter="auto",
        )
        # Restrict ICA to EEG picks; MNE handles EOG separately for detection.
        ica.fit(raw, picks=wanted_eeg)
        drop_info["n_components"] = int(ica.n_components_)
        # Identify EOG-correlated components
        eog_inds, eog_scores = ica.find_bads_eog(raw, ch_name=wanted_eog,
                                                 verbose="ERROR")
        drop_info["components_dropped"] = ",".join(str(i) for i in eog_inds)
        drop_info["n_components_dropped"] = int(len(eog_inds))
        if len(eog_scores):
            drop_info["max_eog_corr"] = float(np.max(np.abs(eog_scores)))
        ica.exclude = list(map(int, eog_inds))
        cleaned = raw.copy()
        ica.apply(cleaned, verbose="ERROR")

        # Pull cleaned Pz-Oz signal and run TDA
        ch_idx = cleaned.ch_names.index(P.EEG_SECONDARY)
        x = cleaned.get_data(picks=[ch_idx])[0].astype(np.float64)
        sf = float(cleaned.info["sfreq"])
        epoch_len = int(P.EPOCH_SEC * sf)
        intervals = P.load_intervals(pair.hyp_path)
        n_epochs = len(x) // epoch_len
        by_stage: Dict[str, List[int]] = {s: [] for s in P.STAGES_MAIN}
        for e in range(n_epochs):
            s = P.stage_at(intervals, (e + 0.5) * epoch_len / sf)
            if s in by_stage:
                by_stage[s].append(e)
        rng_epochs = np.random.default_rng(sub_seed)
        psg_tag = P._tag(P._prefix(pair.psg_path))
        for stage, idxs in by_stage.items():
            if not idxs:
                continue
            if len(idxs) > P.MAX_EPOCHS_MAIN:
                idxs = list(rng_epochs.choice(idxs, size=P.MAX_EPOCHS_MAIN, replace=False))
            for e in idxs:
                seg = x[e * epoch_len:(e + 1) * epoch_len]
                feats = P.persistence_features(seg, P.EMBED_M, P.EMBED_TAU,
                                               P.MAXDIM, P.MIN_EMBED_POINTS_MAIN,
                                               downsample=True)
                if feats is None:
                    continue
                epochs_out.append({
                    "subject": pair.subject, "psg_tag": psg_tag,
                    "psg_file": pair.psg_path.name, "hyp_file": pair.hyp_path.name,
                    "channel": P.EEG_SECONDARY + " (ICA-cleaned)",
                    "stage": stage, "epoch_index": int(e), **feats,
                })
        return {"epochs": epochs_out, "drop_info": drop_info}
    except Exception as ex:
        drop_info["error"] = f"{type(ex).__name__}: {ex}"
        return {"epochs": [], "drop_info": drop_info}


def section_C_ica_pz_oz(pairs: List[P.Pair], n_jobs: int, force: bool,
                        subset_size: int):
    banner(f"Section C — ICA-based ocular-artefact sensitivity (Pz-Oz, n={subset_size})")
    if C_EPOCHS.exists() and C_CONTR.exists() and C_DROPPED.exists() and not force:
        log("  ✓ skipping (outputs exist; use --force to rerun)")
        return

    # Deterministic subset selection: sort pairs by subject ID, take first
    # subset_size unique subjects (keep one PSG per subject).
    seen = set(); chosen = []
    for pair in sorted(pairs, key=lambda p: p.subject):
        if pair.subject not in seen:
            chosen.append(pair); seen.add(pair.subject)
        if len(chosen) >= subset_size:
            break
    log(f"  · running ICA on {len(chosen)} subjects (one PSG per subject)")

    seeds = np.random.SeedSequence(P.RNG_SEED).spawn(len(chosen))
    items = list(zip(chosen, seeds))
    def _wrap(pair_seed):
        return _worker_ica_pz_oz(*pair_seed)
    results = P._parallel_bar(_wrap, items, n_jobs, "ICA + TDA on Pz-Oz")

    epoch_rows = []
    drop_rows = []
    for r in results:
        epoch_rows.extend(r.get("epochs", []))
        drop_rows.append(r.get("drop_info", {}))
    df_epochs = pd.DataFrame(epoch_rows)
    df_drop = pd.DataFrame(drop_rows)
    _save_csv(df_drop, C_DROPPED)
    _save_csv(df_epochs, C_EPOCHS)

    n_failed = int(df_drop["error"].astype(bool).sum()) if "error" in df_drop.columns else 0
    if n_failed:
        log(f"  · {n_failed} subject(s) failed (channel missing or ICA error)")

    if df_epochs.empty:
        _save_csv(pd.DataFrame(), C_CONTR)
        return

    # Mixed-LM REM-W contrast on cleaned Pz-Oz
    log("  · fitting mixed-LM REM-W contrast on ICA-cleaned Pz-Oz")
    df_epochs["K0_tot"] = df_epochs.groupby("subject")["H1_totpers"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0
    )
    ss = df_epochs.groupby(["subject", "stage"], as_index=False)["K0_tot"].mean()
    try:
        res, lr, df_diff, p_lr = P.fit_mixedlm_stage(
            ss, "K0_tot", P.STAGES_MAIN, method="powell")
    except Exception:
        res, lr, df_diff, p_lr = P.fit_mixedlm_stage(
            ss, "K0_tot", P.STAGES_MAIN, method="lbfgs")
    contr = P.planned_contrasts(res, "K0_tot", P.PLANNED_MAIN, P.STAGES_MAIN)
    contr["channel"] = "EEG Pz-Oz (ICA-cleaned)"
    contr["LR_omnibus"] = lr
    contr["df_omnibus"] = df_diff
    contr["p_omnibus"] = p_lr
    contr["n_subjects"] = ss["subject"].nunique()
    rw = contr[contr["contrast"] == "REM - W"].iloc[0]
    log(f"    REM-W (cleaned Pz-Oz) = {rw['estimate']:+.3f} SD "
        f"[{rw['CI95_low']:+.3f}, {rw['CI95_high']:+.3f}], Holm p = {rw['p_holm']:.3e}")
    _save_csv(contr, C_CONTR)


# ──────────────────────────────────────────────────────────────────────────────
# Section D — Permutation-entropy order sensitivity
# ──────────────────────────────────────────────────────────────────────────────

def _worker_pe_orders(pair: P.Pair, orders: Sequence[int]) -> List[dict]:
    try:
        x, sf, ch = P._load_eeg(pair.psg_path)
        intervals = P.load_intervals(pair.hyp_path)
        epoch_len = int(P.EPOCH_SEC * sf)
        n_epochs = len(x) // epoch_len
        out = []
        for e in range(n_epochs):
            s = P.stage_at(intervals, (e + 0.5) * epoch_len / sf)
            if s not in P.STAGES_MAIN:
                continue
            seg = x[e * epoch_len:(e + 1) * epoch_len]
            row = {
                "subject": pair.subject,
                "psg_file": pair.psg_path.name, "hyp_file": pair.hyp_path.name,
                "channel": ch, "stage": s, "epoch_index": int(e),
            }
            for order in orders:
                row[f"perm_entropy_order_{order}"] = P.permutation_entropy(
                    seg, order=order, delay=1)
            out.append(row)
        return out
    except Exception as ex:
        return [{"_ERROR_": f"{pair.psg_path.name} :: {type(ex).__name__}: {ex}"}]


def section_D_pe_order_sensitivity(pairs: List[P.Pair], n_jobs: int, force: bool,
                                   orders: Sequence[int],
                                   limit: Optional[int]):
    banner(f"Section D — Permutation-entropy order sensitivity (orders={list(orders)})")
    if D_EPOCHS.exists() and D_CONTR.exists() and not force:
        log("  ✓ skipping (outputs exist; use --force to rerun)")
        return

    if limit is not None:
        pairs = pairs[:limit]
        log(f"  · DEBUG --limit {limit} → first {len(pairs)} pairs only")

    log(f"  · {len(pairs)} PSG pairs, n_jobs={n_jobs}")
    def _wrap(pair):
        return _worker_pe_orders(pair, orders)
    rows = P._parallel_bar(_wrap, pairs, n_jobs, "PE @ orders 3-6")
    flat = []; errs = []
    for r in rows:
        for d in r:
            if "_ERROR_" in d:
                errs.append(d["_ERROR_"])
            else:
                flat.append(d)
    if errs:
        log(f"  ! {len(errs)} per-night errors; first: {errs[0]}")
    df = pd.DataFrame(flat)
    _save_csv(df, D_EPOCHS)

    if df.empty:
        _save_csv(pd.DataFrame(), D_CONTR)
        return

    # Per-order: within-subject z-score and refit REM-W contrast
    log("  · fitting mixed-LM REM-W contrast at each PE order")
    contrast_rows = []
    for order in orders:
        col_raw = f"perm_entropy_order_{order}"
        col_z = f"PE_z_order_{order}"
        # Within-subject z-score of the PE values (for parity with K0)
        df[col_z] = df.groupby("subject")[col_raw].transform(
            lambda x: (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0
        )
        ss = df.groupby(["subject", "stage"], as_index=False)[col_z].mean()
        try:
            res, lr, df_diff, p_lr = P.fit_mixedlm_stage(
                ss, col_z, P.STAGES_MAIN, method="powell")
        except Exception:
            res, lr, df_diff, p_lr = P.fit_mixedlm_stage(
                ss, col_z, P.STAGES_MAIN, method="lbfgs")
        contr = P.planned_contrasts(res, col_z, P.PLANNED_MAIN, P.STAGES_MAIN)
        contr["pe_order"] = int(order)
        contr["LR_omnibus"] = lr
        contr["df_omnibus"] = df_diff
        contr["p_omnibus"] = p_lr
        contr["n_subjects"] = ss["subject"].nunique()
        contrast_rows.append(contr)
        rw = contr[contr["contrast"] == "REM - W"].iloc[0]
        log(f"    order={order}: REM-W = {rw['estimate']:+.3f} SD "
            f"[{rw['CI95_low']:+.3f}, {rw['CI95_high']:+.3f}], "
            f"Holm p = {rw['p_holm']:.3e}")
    out = pd.concat(contrast_rows, ignore_index=True)
    _save_csv(out, D_CONTR)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

ALL_SECTIONS = ["A", "B", "C", "D"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Round-3 supplementary analyses (diagnostic embedding, "
                    "LOSO refinements, ICA Pz-Oz, PE order sensitivity)"
    )
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Parallel workers for per-night TDA / ICA / PE.")
    p.add_argument("--only", type=str, default=",".join(ALL_SECTIONS),
                   help=f"Comma-separated list of sections to run "
                        f"(any of {ALL_SECTIONS}); default = all.")
    p.add_argument("--force", action="store_true",
                   help="Rerun a section even if its outputs already exist.")
    p.add_argument("--limit", type=int, default=None,
                   help="(Debug) cap the number of PSG pairs processed in "
                        "Section A and Section D.")
    p.add_argument("--ica-subset", type=int, default=20,
                   help="Number of subjects to run ICA on in Section C.")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="Bootstrap iterations for Section B AUC CIs and the "
                        "paired combined-vs-bandpower test.")
    p.add_argument("--m-list", type=str, default="6,10,12",
                   help="Comma-separated m values for Section A "
                        "(default: 6,10,12 — FNN-min, original, conservative).")
    p.add_argument("--tau", type=int, default=11,
                   help="tau for Section A (default: 11 = AMI first local min).")
    p.add_argument("--pe-orders", type=str, default="3,4,5,6",
                   help="Comma-separated permutation-entropy orders "
                        "for Section D.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sections = [s.strip().upper() for s in args.only.split(",") if s.strip()]
    invalid = [s for s in sections if s not in ALL_SECTIONS]
    if invalid:
        sys.exit(f"Unknown section(s): {invalid}; allowed: {ALL_SECTIONS}")

    # Discover dataset for sections that need raw EDFs (A, C, D).
    needs_data = bool(set(sections) & {"A", "C", "D"})
    pairs: List[P.Pair] = []
    if needs_data:
        data_root = P.resolve_data_root()
        log(f"Dataset root: {data_root}")
        pairs = P.discover_pairs(data_root)
        log(f"Discovered {len(pairs)} PSG/Hypnogram pairs.")

    t0 = time.time()
    if "A" in sections:
        m_list = [int(x) for x in args.m_list.split(",") if x.strip()]
        section_A_diagnostic_embedding(
            pairs, args.n_jobs, args.force, m_list, args.tau, args.limit)
    if "B" in sections:
        section_B_classification_refinements(args.force, args.n_boot)
    if "C" in sections:
        section_C_ica_pz_oz(pairs, args.n_jobs, args.force, args.ica_subset)
    if "D" in sections:
        orders = [int(x) for x in args.pe_orders.split(",") if x.strip()]
        section_D_pe_order_sensitivity(
            pairs, args.n_jobs, args.force, orders, args.limit)

    elapsed = time.time() - t0
    log(f"\nDone. Sections run: {sections} | total elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
