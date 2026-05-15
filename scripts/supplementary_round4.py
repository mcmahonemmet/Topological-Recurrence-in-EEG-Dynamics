#!/usr/bin/env python3
"""
supplementary_round4.py
=======================

Two regime-specific supplementary analyses that close the loop on the
"if band power can already classify REM at AUC ≈ 0.91, what's left for K0
to do?" question.

Sections
--------
  A. Regime-specific LOSO classification (REM vs W_quiet, REM vs W_active_ocular).
        The pooled REM-vs-Wake AUC of 0.91 hides large heterogeneity across
        the wake subclasses. This section refits LOSO logistic regression
        for each (target ∈ {W_quiet, W_active_ocular}) × (feature_set ∈
        {K0_only, bandpower_only, combined}) cell and reports pooled AUCs
        with 95% subject-level bootstrap CIs plus a paired-bootstrap test
        for combined > bandpower in each regime. Two ROC-curve panels are
        also produced.

  B. EOG-regressed band-power AUC.
        The existing artefact-control sensitivity uses EOG-regressed K0;
        the band-power analysis was never re-run on the corrected EEG. This
        section computes Welch band power on the EOG-regressed Fpz-Cz signal
        (cached in outputs/corrected_epochs/*.npz from the main pipeline)
        and refits the LOSO logistic for REM vs W_active_ocular on raw vs
        corrected band power, so the manuscript can quantify how much of
        the band-power AUC is residual ocular signal rather than cortical.

Outputs (written to outputs/ and outputs/figures/):
  - supp_d_regime_loso_predictions.csv         per-test-epoch held-out scores
                                               for every (target, feature_set)
                                               cell.
  - supp_d_regime_auc_ci.csv                   pooled AUC + 95% bootstrap CI
                                               per (target, feature_set).
  - supp_d_regime_paired_bootstrap.csv         combined - bandpower paired
                                               bootstrap test per target.
  - supp_d_corrected_bandpower_epoch_features.csv
                                               per-epoch band-power on EOG-
                                               regressed Fpz-Cz across REM
                                               and W_active_ocular epochs.
  - supp_d_corrected_bandpower_loso_auc.csv    raw vs corrected band-power
                                               (and corrected K0) AUCs for
                                               REM vs W_active_ocular.
  - figures/supp_d_roc_regime_REM_vs_Wquiet.png
  - figures/supp_d_roc_regime_REM_vs_Wactive.png
  - figures/supp_d_corrected_bandpower_auc_bars.png

Reproducibility
---------------
RNG_SEED = 0 (inherited from pipeline.py); per-subject bootstrap draws use
``np.random.default_rng(RNG_SEED)``. LOSO classifier (logistic regression
with StandardScaler) matches pipeline.step_classification.

Usage
-----
  python scripts/supplementary_round4.py                # all sections
  python scripts/supplementary_round4.py --only A       # regime LOSO only
  python scripts/supplementary_round4.py --only B       # EOG-corrected BP only
  python scripts/supplementary_round4.py --n-jobs 8     # parallel band-power for B
  python scripts/supplementary_round4.py --n-boot 1000  # bootstrap iterations
  python scripts/supplementary_round4.py --force        # overwrite cached outputs
"""

from __future__ import annotations

import argparse
import sys
import time
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
import mne

# Make pipeline.py importable.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import pipeline as P  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

OUT_DIR = P.OUT_DIR
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Section A
A_LOSO    = OUT_DIR / "supp_d_regime_loso_predictions.csv"
A_AUC_CI  = OUT_DIR / "supp_d_regime_auc_ci.csv"
A_PAIRED  = OUT_DIR / "supp_d_regime_paired_bootstrap.csv"
A_FIG_QUI = FIG_DIR / "supp_d_roc_regime_REM_vs_Wquiet.png"
A_FIG_ACT = FIG_DIR / "supp_d_roc_regime_REM_vs_Wactive.png"

# Section B
B_BP      = OUT_DIR / "supp_d_corrected_bandpower_epoch_features.csv"
B_AUC     = OUT_DIR / "supp_d_corrected_bandpower_loso_auc.csv"
B_FIG     = FIG_DIR / "supp_d_corrected_bandpower_auc_bars.png"


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
# Shared LOSO + bootstrap helpers (numpy-fast)
# ──────────────────────────────────────────────────────────────────────────────

def _loso_logistic_with_scores(df: pd.DataFrame, feature_sets: Dict[str, List[str]],
                               y_col: str = "y", subj_col: str = "subject",
                               desc: str = "LOSO logistic") -> pd.DataFrame:
    """Run LOSO logistic regression on each feature set in ``feature_sets``,
    storing per-test-epoch held-out scores. Returns a long DataFrame with
    columns: feature_set, held_out_subject, row_index, y_true, y_score."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    subjects = sorted(df[subj_col].unique())
    rows = []
    pbar = P._inner_bar(len(subjects) * len(feature_sets), desc)
    for held in subjects:
        tr = df[subj_col] != held
        te = df[subj_col] == held
        if te.sum() < 5 or df.loc[te, y_col].nunique() < 2:
            if pbar is not None:
                pbar.update(len(feature_sets))
            continue
        for fs_name, fs_cols in feature_sets.items():
            X_tr = df.loc[tr, fs_cols].values
            y_tr = df.loc[tr, y_col].values
            X_te = df.loc[te, fs_cols].values
            y_te = df.loc[te, y_col].values
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                                        random_state=P.RNG_SEED))])
            pipe.fit(X_tr, y_tr)
            scores = pipe.predict_proba(X_te)[:, 1]
            for j, idx in enumerate(df.index[te]):
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
                      seed: int = P.RNG_SEED, target_col: str = "_target") -> pd.DataFrame:
    """Subject-level bootstrap on AUCs. Per-subject (y_true, y_score) numpy
    arrays are precomputed once per (target, feature_set) cell so each
    bootstrap iteration is a single ``np.concatenate`` over indices."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    rows = []
    group_cols = ([target_col, "feature_set"] if target_col in preds.columns
                  else ["feature_set"])
    for keys, sub in preds.groupby(group_cols):
        per_subj_yt: Dict[str, np.ndarray] = {}
        per_subj_ys: Dict[str, np.ndarray] = {}
        for s, g in sub.groupby("held_out_subject"):
            per_subj_yt[s] = g["y_true"].to_numpy(dtype=np.int8)
            per_subj_ys[s] = g["y_score"].to_numpy(dtype=np.float64)
        subjects = np.array(list(per_subj_yt.keys()))
        all_yt = np.concatenate([per_subj_yt[s] for s in subjects])
        all_ys = np.concatenate([per_subj_ys[s] for s in subjects])
        try:
            point = float(roc_auc_score(all_yt, all_ys))
        except ValueError:
            point = np.nan
        boot = np.empty(n_boot, dtype=np.float64); n_ok = 0
        for _ in range(n_boot):
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
        rec = {} if isinstance(keys, str) else {}
        if isinstance(keys, tuple):
            for col, val in zip(group_cols, keys):
                rec[col] = val
        else:
            rec[group_cols[0]] = keys
        rec.update({
            "auc_point": point,
            "auc_boot_mean": float(np.mean(boot)) if boot.size else np.nan,
            "auc_boot_sd":   float(np.std(boot, ddof=1)) if boot.size > 1 else np.nan,
            "auc_boot_ci95_low":  float(np.quantile(boot, 0.025)) if boot.size else np.nan,
            "auc_boot_ci95_high": float(np.quantile(boot, 0.975)) if boot.size else np.nan,
            "n_boot_successful": int(boot.size),
            "n_subjects": int(len(subjects)),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def _paired_bootstrap_combined_vs_bandpower(preds: pd.DataFrame,
                                            n_boot: int = 1000,
                                            seed: int = P.RNG_SEED,
                                            target_col: str = "_target") -> pd.DataFrame:
    """Per-target paired subject-level bootstrap test for combined > bandpower."""
    from sklearn.metrics import roc_auc_score
    out_rows = []
    rng = np.random.default_rng(seed)
    targets = preds[target_col].unique() if target_col in preds.columns else [None]
    for t in targets:
        sub = preds if t is None else preds[preds[target_col] == t]
        a = sub[sub["feature_set"] == "combined"]
        b = sub[sub["feature_set"] == "bandpower_only"]
        common = sorted(set(a["held_out_subject"]) & set(b["held_out_subject"]))
        if not common:
            continue
        a_yt = {s: g["y_true"].to_numpy(dtype=np.int8)   for s, g in a.groupby("held_out_subject")}
        a_ys = {s: g["y_score"].to_numpy(dtype=np.float64) for s, g in a.groupby("held_out_subject")}
        b_yt = {s: g["y_true"].to_numpy(dtype=np.int8)   for s, g in b.groupby("held_out_subject")}
        b_ys = {s: g["y_score"].to_numpy(dtype=np.float64) for s, g in b.groupby("held_out_subject")}
        subjects = np.array(common)
        diffs = np.empty(n_boot, dtype=np.float64); n_ok = 0
        for _ in range(n_boot):
            picks = rng.choice(subjects, size=len(subjects), replace=True)
            ayt = np.concatenate([a_yt[s] for s in picks])
            ays = np.concatenate([a_ys[s] for s in picks])
            byt = np.concatenate([b_yt[s] for s in picks])
            bys = np.concatenate([b_ys[s] for s in picks])
            if np.unique(ayt).size < 2 or np.unique(byt).size < 2:
                continue
            try:
                diffs[n_ok] = roc_auc_score(ayt, ays) - roc_auc_score(byt, bys)
                n_ok += 1
            except ValueError:
                continue
        diffs = diffs[:n_ok]
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
        p_one = float(np.mean(diffs <= 0)) if diffs.size else np.nan
        p_two = (2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
                 if diffs.size else np.nan)
        out_rows.append({
            "target": t,
            "auc_combined_point": point_a,
            "auc_bandpower_point": point_b,
            "auc_diff_point": point_diff,
            "auc_diff_boot_mean": float(np.mean(diffs)) if diffs.size else np.nan,
            "auc_diff_boot_ci95_low":  float(np.quantile(diffs, 0.025)) if diffs.size else np.nan,
            "auc_diff_boot_ci95_high": float(np.quantile(diffs, 0.975)) if diffs.size else np.nan,
            "p_boot_one_sided_combined_gt_bandpower": p_one,
            "p_boot_two_sided": p_two,
            "n_boot_successful": int(diffs.size),
            "n_subjects": int(len(subjects)),
        })
    return pd.DataFrame(out_rows)


# ──────────────────────────────────────────────────────────────────────────────
# Section A — Regime-specific LOSO (REM vs W_quiet, REM vs W_active_ocular)
# ──────────────────────────────────────────────────────────────────────────────

def _build_regime_dataset() -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Merge K0 + band-power features with wake-subclass labels so each row
    is one epoch tagged with stage ∈ {REM, W_quiet, W_active_ocular}."""
    tda  = P._safe_read_csv(OUT_DIR / "tda_epoch_features_all.csv")
    base = P._safe_read_csv(OUT_DIR / "baseline_epoch_features_all.csv")
    wsub = P._safe_read_csv(OUT_DIR / "wake_epoch_subclasses.csv")
    if tda.empty or base.empty or wsub.empty:
        raise RuntimeError("prerequisite epoch CSVs missing or empty")
    keys = ["subject", "psg_file", "hyp_file", "epoch_index"]
    for d in (tda, base, wsub):
        d["epoch_index"] = d["epoch_index"].astype(int)

    # Wake subclass label keyed by epoch
    wsub_small = wsub[keys + ["wake_subclass"]].copy()
    # Merge TDA + bandpower
    t_small = tda[keys + ["stage", "H1_totpers"]]
    b_small = base[keys + list(P.BANDPOWER_COLS)]
    df = t_small.merge(b_small, on=keys, how="inner")
    # Attach wake subclass; will be NaN for non-wake epochs (we re-derive
    # the final stage from the wake_subclass column for wake epochs and
    # keep "REM" from the original stage column).
    df = df.merge(wsub_small, on=keys, how="left")
    # Construct the final classification stage:
    #   REM         -> REM
    #   W and wake_subclass = W_quiet         -> W_quiet
    #   W and wake_subclass = W_active_ocular -> W_active_ocular
    #   anything else -> drop
    keep_mask = pd.Series(False, index=df.index)
    df["stage_final"] = ""
    is_rem = df["stage"] == "REM"
    df.loc[is_rem, "stage_final"] = "REM"; keep_mask |= is_rem
    is_w = (df["stage"] == "W")
    is_wq = is_w & (df["wake_subclass"] == "W_quiet")
    is_wa = is_w & (df["wake_subclass"] == "W_active_ocular")
    df.loc[is_wq, "stage_final"] = "W_quiet";        keep_mask |= is_wq
    df.loc[is_wa, "stage_final"] = "W_active_ocular"; keep_mask |= is_wa
    df = df[keep_mask].copy()
    # K0_tot: within-subject z of H1_totpers across all kept epochs
    df["K0_tot"] = df.groupby("subject")["H1_totpers"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0
    )
    feature_sets = {
        "K0_only":        ["K0_tot"],
        "bandpower_only": list(P.BANDPOWER_COLS),
        "combined":       ["K0_tot"] + list(P.BANDPOWER_COLS),
    }
    return df, feature_sets


def section_A_regime_loso(force: bool, n_boot: int):
    banner("Section A — Regime-specific LOSO (REM vs W_quiet, REM vs W_active_ocular)")
    have_all = all(p.exists() for p in [A_LOSO, A_AUC_CI, A_PAIRED, A_FIG_QUI, A_FIG_ACT])
    if have_all and not force:
        log("  ✓ skipping (outputs exist; use --force to rerun)")
        return

    df, feature_sets = _build_regime_dataset()
    log(f"  · {len(df):,} epochs across "
        f"{df['stage_final'].value_counts().to_dict()}")

    targets = {
        "REM_vs_W_quiet":         ("REM", "W_quiet"),
        "REM_vs_W_active_ocular": ("REM", "W_active_ocular"),
    }
    all_preds = []
    for tname, (pos, neg) in targets.items():
        sub = df[df["stage_final"].isin([pos, neg])].copy()
        sub["y"] = (sub["stage_final"] == pos).astype(int)
        log(f"  · {tname}: {len(sub):,} epochs from "
            f"{sub['subject'].nunique()} subjects")
        preds = _loso_logistic_with_scores(
            sub, feature_sets, desc=f"LOSO {tname}",
        )
        preds["_target"] = tname
        all_preds.append(preds)
    preds_all = pd.concat(all_preds, ignore_index=True)
    _save_csv(preds_all, A_LOSO)

    log(f"  · bootstrapping AUC CIs ({n_boot} iterations)")
    ci = _bootstrap_auc_ci(preds_all, n_boot=n_boot)
    log(f"\n    {'target':<25s} {'feature_set':<18s}  AUC   95% CI")
    for _, r in ci.iterrows():
        log(f"    {r['_target']:<25s} {r['feature_set']:<18s}  "
            f"{r['auc_point']:.3f}  "
            f"[{r['auc_boot_ci95_low']:.3f}, {r['auc_boot_ci95_high']:.3f}]")
    _save_csv(ci, A_AUC_CI)

    log(f"\n  · paired bootstrap: combined - bandpower per regime")
    paired = _paired_bootstrap_combined_vs_bandpower(preds_all, n_boot=n_boot)
    for _, r in paired.iterrows():
        log(f"    {r['target']:<25s}  Δ AUC = {r['auc_diff_point']:+.4f} "
            f"[{r['auc_diff_boot_ci95_low']:+.4f}, {r['auc_diff_boot_ci95_high']:+.4f}], "
            f"one-sided p = {r['p_boot_one_sided_combined_gt_bandpower']:.4f}")
    _save_csv(paired, A_PAIRED)

    # ROC figures
    _figure_roc_regime(preds_all, "REM_vs_W_quiet",
                       "REM vs Quiet Wake", A_FIG_QUI)
    _figure_roc_regime(preds_all, "REM_vs_W_active_ocular",
                       "REM vs Active-Ocular Wake", A_FIG_ACT)


def _figure_roc_regime(preds_all: pd.DataFrame, target: str, title: str,
                       out_path: Path):
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    colours = {"K0_only": "#1f77b4", "bandpower_only": "#d62728",
               "combined": "#2ca02c"}
    pretty = {"K0_only": "K0 only", "bandpower_only": "Band power only",
              "combined": "Combined"}
    sub = preds_all[preds_all["_target"] == target]
    for fs in ["K0_only", "bandpower_only", "combined"]:
        s = sub[sub["feature_set"] == fs]
        if s.empty:
            continue
        fpr, tpr, _ = roc_curve(s["y_true"], s["y_score"])
        try:
            auc = float(roc_auc_score(s["y_true"], s["y_score"]))
        except ValueError:
            auc = float("nan")
        ax.plot(fpr, tpr, color=colours[fs], lw=1.8,
                label=f"{pretty[fs]} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"LOSO ROC: {title}")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  · wrote {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Section B — EOG-regressed band-power AUC for REM vs W_active_ocular
# ──────────────────────────────────────────────────────────────────────────────

def _bandpower_one_epoch(seg: np.ndarray, sf: float) -> Dict[str, float]:
    """Welch PSD + log band powers, matching pipeline.worker_baseline."""
    psd, freqs = mne.time_frequency.psd_array_welch(
        seg, sfreq=sf, fmin=0.5, fmax=40.0,
        n_fft=min(2048, len(seg)), verbose="ERROR",
    )
    return {
        "log_delta": float(np.log(P.bandpower(psd, freqs, 0.5, 4.0)  + 1e-12)),
        "log_theta": float(np.log(P.bandpower(psd, freqs, 4.0, 8.0)  + 1e-12)),
        "log_alpha": float(np.log(P.bandpower(psd, freqs, 8.0, 12.0) + 1e-12)),
        "log_sigma": float(np.log(P.bandpower(psd, freqs, 12.0, 15.0)+ 1e-12)),
        "log_beta":  float(np.log(P.bandpower(psd, freqs, 15.0, 30.0)+ 1e-12)),
    }


def _coerce_str(v) -> str:
    """Numpy 0-d arrays in npz files come back wrapped; unwrap to plain str."""
    arr = np.asarray(v)
    if arr.ndim == 0:
        return str(arr.item())
    if arr.size == 1:
        return str(arr.flat[0])
    return str(arr.tolist())


def _worker_corrected_bandpower(args) -> List[dict]:
    """Per-night band-power on EOG-regressed Fpz-Cz. Reads the cached
    corrected-epoch arrays produced by the main pipeline's
    ``step_corrected_eeg`` rather than re-computing the regression.

    args = (npz_path, target_epoch_idx_set_for_this_psg). Only the target
    epochs (REM + W_active_ocular for this PSG file) are Welch-computed."""
    npz_path, target_idx_set = args
    try:
        z = np.load(str(npz_path), allow_pickle=True)
        eeg_corr = z["eeg_corrected_epochs"]
        epoch_idx = z["epoch_index"]
        sf = float(np.asarray(z["sfreq"]).flat[0])
        subj = _coerce_str(z["subject"])
        psg  = _coerce_str(z["psg_file"])
        hyp  = _coerce_str(z["hyp_file"])
        rows = []
        for k in range(eeg_corr.shape[0]):
            ei = int(epoch_idx[k])
            if target_idx_set is not None and ei not in target_idx_set:
                continue
            seg = eeg_corr[k].astype(np.float64)
            bp = _bandpower_one_epoch(seg, sf)
            rows.append({
                "subject": subj, "psg_file": psg, "hyp_file": hyp,
                "epoch_index": ei,
                **{f"corr_{k_}": v_ for k_, v_ in bp.items()},
            })
        return rows
    except Exception as ex:
        return [{"_ERROR_": f"{npz_path.name} :: {type(ex).__name__}: {ex}"}]


def _compute_corrected_bandpower(n_jobs: int, force: bool) -> pd.DataFrame:
    if B_BP.exists() and not force:
        log(f"  ✓ reusing {B_BP.name}")
        return pd.read_csv(B_BP, comment="#")
    manifest_path = OUT_DIR / "corrected_epochs_manifest.csv"
    if not manifest_path.exists():
        raise RuntimeError("corrected_epochs_manifest.csv missing — "
                           "run pipeline.py first (step_corrected_eeg).")
    manifest = pd.read_csv(manifest_path, comment="#")
    npz_dir = OUT_DIR / "corrected_epochs"
    log(f"  · computing band-power on {len(manifest)} corrected-epoch NPZs "
        f"(n_jobs={n_jobs})")

    # Per-PSG target epoch sets (REM + W_active_ocular only) so each worker
    # Welch-computes ~30–80 epochs per night instead of ~2,500.
    wsub = P._safe_read_csv(OUT_DIR / "wake_epoch_subclasses.csv")
    base = P._safe_read_csv(OUT_DIR / "baseline_epoch_features_all.csv")
    for d in (wsub, base):
        d["epoch_index"] = d["epoch_index"].astype(int)
    rem_keys  = base.loc[base["stage"] == "REM",  ["psg_file", "epoch_index"]]
    wact_keys = wsub.loc[wsub["wake_subclass"] == "W_active_ocular",
                         ["psg_file", "epoch_index"]]
    target_keys = pd.concat([rem_keys, wact_keys], ignore_index=True).drop_duplicates()
    by_psg: Dict[str, set] = {}
    for psg, g in target_keys.groupby("psg_file"):
        by_psg[psg] = set(int(x) for x in g["epoch_index"].tolist())
    log(f"  · target epoch set size: {sum(len(v) for v in by_psg.values()):,} "
        f"(REM={len(rem_keys):,} + W_active_ocular={len(wact_keys):,})")

    items = []
    for psg_file in manifest["npz_file"]:
        # Reconstruct the original PSG filename from the npz filename
        # (corrected_epochs naming: "<PSG-stem>_corrected_epochs.npz").
        base_name = psg_file.replace("_corrected_epochs.npz", ".edf")
        items.append((npz_dir / psg_file, by_psg.get(base_name, set())))

    rows = P._parallel_bar(_worker_corrected_bandpower, items, n_jobs,
                           "corrected band-power")
    flat = []
    errs = []
    for r in rows:
        for d in (r if isinstance(r, list) else [r]):
            if "_ERROR_" in d:
                errs.append(d["_ERROR_"])
            else:
                flat.append(d)
    if errs:
        log(f"  ! {len(errs)} per-night error(s); first: {errs[0]}")
    df = pd.DataFrame(flat)
    _save_csv(df, B_BP)
    return df


def section_B_corrected_bandpower(n_jobs: int, force: bool, n_boot: int):
    banner("Section B — EOG-regressed band-power AUC for REM vs W_active_ocular")
    if B_AUC.exists() and B_FIG.exists() and B_BP.exists() and not force:
        log("  ✓ skipping (outputs exist; use --force to rerun)")
        return

    bp_corr = _compute_corrected_bandpower(n_jobs, force)
    if bp_corr.empty:
        log("  ! no corrected band-power features computed")
        return

    # Build the comparison dataset:
    #   raw_bandpower      from baseline_epoch_features_all.csv
    #   corr_bandpower     from supp_d_corrected_bandpower_epoch_features.csv
    #   K0 raw/corrected   from tda_epoch_features_all.csv and
    #                      tda_epoch_features_wake_corrected.csv (raw-track
    #                      H1 stays the headline; corrected-track is the
    #                      EOG-regressed equivalent)
    base = P._safe_read_csv(OUT_DIR / "baseline_epoch_features_all.csv")
    tda_raw  = P._safe_read_csv(OUT_DIR / "tda_epoch_features_all.csv")
    wsub     = P._safe_read_csv(OUT_DIR / "wake_epoch_subclasses.csv")
    keys = ["subject", "psg_file", "hyp_file", "epoch_index"]
    for d in (base, tda_raw, wsub, bp_corr):
        d["epoch_index"] = d["epoch_index"].astype(int)
    base_small = base[keys + ["stage"] + list(P.BANDPOWER_COLS)]
    tda_small  = tda_raw[keys + ["H1_totpers"]].rename(columns={"H1_totpers": "H1_raw"})
    wsub_small = wsub[keys + ["wake_subclass"]]
    df = base_small.merge(tda_small, on=keys, how="inner")
    df = df.merge(wsub_small, on=keys, how="left")
    df = df.merge(bp_corr, on=keys, how="inner")
    # Pick the REM and W_active_ocular epochs only.
    keep = (df["stage"] == "REM") | ((df["stage"] == "W") & (df["wake_subclass"] == "W_active_ocular"))
    df = df[keep].copy()
    df["y"] = (df["stage"] == "REM").astype(int)
    df["K0_raw"] = df.groupby("subject")["H1_raw"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0
    )
    log(f"  · merged dataset: {len(df):,} epochs from "
        f"{df['subject'].nunique()} subjects")

    feature_sets = {
        "bandpower_raw":       list(P.BANDPOWER_COLS),
        "bandpower_corrected": [f"corr_{c}" for c in P.BANDPOWER_COLS],
        "K0_raw":              ["K0_raw"],
        "combined_raw":        ["K0_raw"] + list(P.BANDPOWER_COLS),
        "combined_corrected":  ["K0_raw"] + [f"corr_{c}" for c in P.BANDPOWER_COLS],
    }
    log(f"  · feature sets: {list(feature_sets.keys())}")
    preds = _loso_logistic_with_scores(df, feature_sets,
                                        desc="LOSO BP raw vs corrected")
    log(f"  · bootstrapping AUC CIs ({n_boot} iterations)")
    ci = _bootstrap_auc_ci(preds, n_boot=n_boot, target_col="_no_target")
    log(f"\n    {'feature_set':<22s}  AUC   95% CI")
    for _, r in ci.iterrows():
        log(f"    {r['feature_set']:<22s}  {r['auc_point']:.3f}  "
            f"[{r['auc_boot_ci95_low']:.3f}, {r['auc_boot_ci95_high']:.3f}]")
    _save_csv(ci, B_AUC)

    # Bar figure
    _figure_corrected_bandpower(ci, B_FIG)


def _figure_corrected_bandpower(ci: pd.DataFrame, out_path: Path):
    order = ["bandpower_raw", "bandpower_corrected",
             "K0_raw", "combined_raw", "combined_corrected"]
    pretty = {
        "bandpower_raw":       "Band power\n(raw EEG)",
        "bandpower_corrected": "Band power\n(EOG-corrected)",
        "K0_raw":              "K0 only",
        "combined_raw":        "Combined\n(K0 + raw BP)",
        "combined_corrected":  "Combined\n(K0 + corr. BP)",
    }
    palette = {
        "bandpower_raw":       "#d62728",
        "bandpower_corrected": "#7B4B94",
        "K0_raw":              "#1f77b4",
        "combined_raw":        "#2ca02c",
        "combined_corrected":  "#bcbd22",
    }
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(order))
    sub = ci.set_index("feature_set").reindex(order)
    aucs = sub["auc_point"].values.astype(float)
    los  = sub["auc_boot_ci95_low"].values.astype(float)
    his  = sub["auc_boot_ci95_high"].values.astype(float)
    yerr = np.vstack([aucs - los, his - aucs])
    colours = [palette[k] for k in order]
    ax.bar(x, aucs, yerr=yerr, capsize=4, color=colours, edgecolor="white",
           width=0.65)
    for xi, a in zip(x, aucs):
        ax.text(xi, a + 0.005, f"{a:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([pretty[k] for k in order])
    ax.set_ylabel("Pooled LOSO AUC")
    ax.set_title("REM vs W_active_ocular: AUC across feature sets and EOG-correction")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  · wrote {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

ALL_SECTIONS = ["A", "B"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Round-4 supplementary: regime-specific AUCs + "
                    "EOG-regressed band-power sensitivity"
    )
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Parallel workers for Section B band-power computation.")
    p.add_argument("--only", type=str, default=",".join(ALL_SECTIONS),
                   help=f"Sections to run (any of {ALL_SECTIONS}).")
    p.add_argument("--force", action="store_true",
                   help="Recompute and overwrite existing outputs.")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="Bootstrap iterations for AUC CIs / paired tests.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sections = [s.strip().upper() for s in args.only.split(",") if s.strip()]
    invalid = [s for s in sections if s not in ALL_SECTIONS]
    if invalid:
        sys.exit(f"Unknown section(s): {invalid}; allowed: {ALL_SECTIONS}")

    t0 = time.time()
    if "A" in sections:
        section_A_regime_loso(args.force, args.n_boot)
    if "B" in sections:
        section_B_corrected_bandpower(args.n_jobs, args.force, args.n_boot)
    log(f"\nDone. Sections run: {sections} | total elapsed: "
        f"{(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
