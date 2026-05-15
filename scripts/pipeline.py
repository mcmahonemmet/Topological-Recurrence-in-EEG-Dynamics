#!/usr/bin/env python3
"""
pipeline.py
===========

Consolidated end-to-end analysis pipeline for the manuscript:

    Topological Recurrence in EEG Dynamics:
    Distinguishing REM Sleep from Wakefulness via Persistent Homology

This single script reproduces the full quantitative analysis previously spread
across ~30 individual scripts. It is intentionally one large, well-sectioned
file so that the entire computation can be read top-to-bottom and audited.

Stages performed (each one writes its output CSV/NPZ as soon as it finishes;
re-running the script will skip stages whose output already exists unless
``--force`` is passed):

  1.  Pair PSG and hypnogram EDFs.
  2.  Main TDA: per-epoch persistent homology on EEG Fpz-Cz.
  3.  Robustness grid: TDA across embedding parameters m x tau.
  4.  Baselines: Welch band power, spectral / permutation entropy, LZ.
  5.  Wake subclass labelling (track 2: simple recording-percentile rules).
  6.  Wake QC table (track 3: richer per-subject MAD-based thresholding).
  7.  TDA on wake subclasses (track 2 stage labels).
  8.  EOG-corrected EEG epoch bundles (full-recording linear regression).
  9.  TDA on raw / corrected / EOG channel for wake subclasses.
 10.  Mixed-effects models with planned contrasts for every metric / track.
 11.  Wake-subclass robustness grid.
 12.  Incremental K0 vs band power binomial GLM.
 13.  Review / comparison / supplementary tables.

Reproducibility
---------------
RNG_SEED = 0. Per-night sub-seeds are spawned via ``np.random.SeedSequence``
so output is bit-for-bit identical regardless of the ``--n-jobs`` setting.

Filtering, resampling, epoching, embedding, and statistical model parameters
are unchanged from the original published analysis. See the constants block
below.

Usage
-----
    python scripts/pipeline.py                # all steps, single core
    python scripts/pipeline.py --n-jobs 8     # parallel per-night TDA
    python scripts/pipeline.py --force        # rerun every step
    python scripts/pipeline.py --only main,baselines

Use ``run.py`` from the project root for a guided interactive run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

# Force UTF-8 on stdout/stderr so Unicode banner characters render on Windows
# consoles (default cp1252 codec) without UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import mne
from ripser import ripser
from scipy import stats
from scipy.signal import butter, filtfilt, find_peaks, welch
import statsmodels.api as sm
import statsmodels.formula.api as smf

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# NumPy 2.x removed np.trapz in favour of np.trapezoid; keep both versions
# working with a single shim.
_trapz = getattr(np, "trapezoid", None) or np.trapz



# ──────────────────────────────────────────────────────────────────────────────
# Constants (preserved verbatim from the original published pipeline)
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"
CORRECTED_DIR = OUT_DIR / "corrected_epochs"
LOG_DIR = OUT_DIR / "logs"

# Signal pre-processing
TARGET_SFREQ = 50.0
LOWCUT, HIGHCUT = 0.5, 40.0
EOG_LOW, EOG_HIGH = 0.3, 15.0
EMG_LOW, EMG_HIGH = 10.0, 40.0
EPOCH_SEC = 30.0

# Channels
EEG_PRIMARY = "EEG Fpz-Cz"
EEG_SECONDARY = "EEG Pz-Oz"
EOG_CH = "EOG horizontal"
EMG_CH = "EMG submental"
EVENT_CH = "Event marker"

# TDA
EMBED_M, EMBED_TAU, MAXDIM = 10, 2, 1
RNG_SEED = 0
MAX_EPOCHS_MAIN = 30
MAX_EPOCHS_ROBUSTNESS = 25
MIN_EMBED_POINTS_MAIN = 20
MIN_EMBED_POINTS_ROBUSTNESS = 30

# Robustness grids
M_GRID_MAIN = [6, 8, 10, 12]
TAU_GRID_MAIN = [1, 2, 4]
M_GRID_WAKE = [8, 10, 12]
TAU_GRID_WAKE = [1, 2, 3]

# Stage labels
STAGES_MAIN = ["W", "N1", "N2", "N3", "REM"]
STAGES_WAKE = ["W_quiet", "W_active_ocular", "REM"]
PLANNED_MAIN = [("REM", "W"), ("REM", "N3"), ("N1", "N3")]
PLANNED_WAKE = [
    ("REM", "W_quiet"),
    ("REM", "W_active_ocular"),
    ("W_quiet", "W_active_ocular"),
]

# Baseline metrics
BASELINE_METRICS = [
    "log_delta", "log_theta", "log_alpha", "log_sigma", "log_beta",
    "spec_entropy", "perm_entropy", "lz_complexity",
]
BANDPOWER_COLS = ["log_delta", "log_theta", "log_alpha", "log_beta"]

STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Sleep stage ?": None,
    "Movement time": None,
}


# ──────────────────────────────────────────────────────────────────────────────
# Logging / narration
# ──────────────────────────────────────────────────────────────────────────────

class Narrator:
    """Tiny logger that prints to stdout AND mirrors to a master log file."""
    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n[run started] {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n")

    def log(self, msg: str = ""):
        print(msg, flush=True)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    def banner(self, title: str):
        bar = "─" * 78
        self.log("")
        self.log(bar)
        self.log(f"▶ {title}")
        self.log(bar)

    def step(self, n: int, total: int, name: str, explainer: str = ""):
        self.banner(f"STEP {n}/{total}  {name}")
        if explainer:
            self.log(explainer)


# ──────────────────────────────────────────────────────────────────────────────
# Pairing
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Pair:
    subject: str
    psg_path: Path
    hyp_path: Path

def read_env_var(path: Path, key: str) -> Optional[str]:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip().strip("'").strip('"')
    return None

def _prefix(p: Path) -> str:    return p.name.split("-", 1)[0]
def _subj(prefix: str) -> str:  return prefix[:6]
def _tag(prefix: str) -> str:   return prefix[6:]
def _lead(t: str) -> Optional[str]: return t[0] if t else None

def discover_pairs(data_root: Path) -> List[Pair]:
    """Pair each PSG with its hypnogram by matching subject + tag-lead-letter,
    selecting the alphabetically-first match if multiple exist."""
    psg_files = sorted(data_root.rglob("*-PSG.edf"))
    hyp_files = sorted(data_root.rglob("*-Hypnogram.edf"))
    by_subj: Dict[str, List[Path]] = defaultdict(list)
    for h in hyp_files:
        by_subj[_subj(_prefix(h))].append(h)
    pairs: List[Pair] = []
    for p in psg_files:
        pp = _prefix(p)
        subj = _subj(pp)
        want = _lead(_tag(pp))
        if want is None:
            continue
        matches = [h for h in by_subj.get(subj, [])
                   if _lead(_tag(_prefix(h))) == want]
        if not matches:
            continue
        pairs.append(Pair(subj, p, sorted(matches, key=lambda x: x.name)[0]))
    return pairs

def resolve_data_root() -> Path:
    rel = read_env_var(PROJECT_ROOT / "config.env", "SLEEP_EDF_ROOT")
    candidates = []
    if rel:
        candidates.append((PROJECT_ROOT / rel).resolve())
    candidates += [
        PROJECT_ROOT / "data" / "sleep-edfx" / "sleep-edf-database-expanded-1.0.0",
        PROJECT_ROOT / "data" / "sleep-edfx" / "sleep-edf-database-expanded-1.0.0" / "sleep-cassette",
        PROJECT_ROOT / "data" / "sleep-edfx" / "sleep-edf-database-expanded-1.0.0" / "sleep-telemetry",
    ]
    for cand in candidates:
        cand = cand.resolve()
        if cand.exists() and any(cand.rglob("*.edf")):
            return cand
    raise SystemExit(
        "Could not locate Sleep-EDF dataset.\n"
        "Set SLEEP_EDF_ROOT in config.env (relative to project root) or place\n"
        "the dataset under data/sleep-edfx/sleep-edf-database-expanded-1.0.0/."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Stage handling
# ──────────────────────────────────────────────────────────────────────────────

def load_intervals(hyp_path: Path) -> List[Tuple[float, float, Optional[str]]]:
    ann = mne.read_annotations(str(hyp_path))
    return [
        (float(o), float(o + d), STAGE_MAP.get(str(s).strip(), None))
        for o, d, s in zip(ann.onset, ann.duration, ann.description)
    ]

def stage_at(intervals, t_sec: float) -> Optional[str]:
    for a, b, s in intervals:
        if a <= t_sec < b:
            return s
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Signal helpers
# ──────────────────────────────────────────────────────────────────────────────

def butter_bp(x, fs, lo, hi, order=4):
    nyq = fs / 2.0
    return filtfilt(*butter(order, [max(lo/nyq, 1e-6), min(hi/nyq, 0.999999)],
                            btype="band"), x)

def rms(x):    return float(np.sqrt(np.mean(np.square(x))))
def ptp_uv(x): return float(1e6 * (np.max(x) - np.min(x)))

def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = np.median(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return 0.6745 * (x - med) / mad

def robust_hi(vals: pd.Series, k: float = 4.0) -> float:
    med = float(vals.median())
    mad = float(np.median(np.abs(vals - med)))
    return med + k * mad if mad > 0 else med


# ──────────────────────────────────────────────────────────────────────────────
# TDA helpers
# ──────────────────────────────────────────────────────────────────────────────

def time_delay_embedding(x: np.ndarray, m: int, tau: int, min_pts: int) -> Optional[np.ndarray]:
    n = x.shape[0] - (m - 1) * tau
    if n <= min_pts:
        return None
    return np.stack([x[i:i + n] for i in range(0, m * tau, tau)], axis=1)

def dgm_summaries(dgm: np.ndarray) -> Dict[str, float]:
    if dgm.size == 0:
        return {"count": 0, "tot_pers": 0.0, "max_pers": 0.0}
    finite = np.isfinite(dgm[:, 1])
    pers = (dgm[:, 1] - dgm[:, 0])[finite]
    return {
        "count": int(dgm.shape[0]),
        "tot_pers": float(pers.sum()) if pers.size else 0.0,
        "max_pers": float(pers.max()) if pers.size else 0.0,
    }

def persistence_features(seg: np.ndarray, m: int, tau: int, maxdim: int,
                         min_pts: int, downsample: bool = True) -> Optional[Dict[str, float]]:
    if downsample:
        seg = seg[::2]
    X = time_delay_embedding(seg, m, tau, min_pts)
    if X is None:
        return None
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    dgms = ripser(X, maxdim=maxdim)["dgms"]
    h0 = dgm_summaries(dgms[0]) if len(dgms) > 0 else {"count":0,"tot_pers":0.0,"max_pers":0.0}
    h1 = dgm_summaries(dgms[1]) if len(dgms) > 1 else {"count":0,"tot_pers":0.0,"max_pers":0.0}
    return {
        "H0_count": h0["count"], "H0_totpers": h0["tot_pers"], "H0_maxpers": h0["max_pers"],
        "H1_count": h1["count"], "H1_totpers": h1["tot_pers"], "H1_maxpers": h1["max_pers"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Baseline EEG features
# ──────────────────────────────────────────────────────────────────────────────

def bandpower(psd, freqs, fmin, fmax) -> float:
    mask = (freqs >= fmin) & (freqs < fmax)
    return float(_trapz(psd[mask], freqs[mask])) if np.any(mask) else float("nan")

def spectral_entropy(psd) -> float:
    p = psd / (psd.sum() + 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))

def permutation_entropy(x, order: int = 5, delay: int = 1) -> float:
    n = len(x) - (order - 1) * delay
    if n <= 10:
        return float("nan")
    patterns = np.empty(n, dtype=np.int64)
    weights = (order ** np.arange(order)).astype(np.int64)
    for i in range(n):
        patterns[i] = np.argsort(x[i:i + order * delay:delay]).dot(weights)
    _, counts = np.unique(patterns, return_counts=True)
    p = counts / counts.sum()
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))

def lz_complexity_binary(x) -> float:
    """LZ76 on median-thresholded binary string, normalised by n / log2(n)."""
    b = (x > np.median(x)).astype(np.uint8)
    s = "".join("1" if v else "0" for v in b.tolist())
    n = len(s)
    if n < 20:
        return float("nan")
    i, k, l = 0, 1, 1
    c = 1
    while True:
        if i + k > n or l + k > n:
            c += 1
            break
        if s[i:i + k] == s[l:l + k]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > 1:
                i += 1
                k -= 1
            else:
                c += 1
                l += 1
                i = 0
                k = 1
                if l >= n:
                    break
    return float(c * np.log2(n) / n)


# ──────────────────────────────────────────────────────────────────────────────
# Per-night workers (parallel-safe)
# ──────────────────────────────────────────────────────────────────────────────

def _load_eeg(psg_path: Path, channel: str = EEG_PRIMARY,
              fallback_to_first_eeg: bool = True) -> Tuple[np.ndarray, float, str]:
    raw = mne.io.read_raw_edf(str(psg_path), preload=False, verbose="ERROR")
    ch = channel
    if ch not in raw.ch_names:
        if not fallback_to_first_eeg:
            raise RuntimeError(f"{psg_path.name}: missing channel '{channel}'")
        eegs = [c for c in raw.ch_names if "EEG" in c.upper()]
        if not eegs:
            raise RuntimeError(f"{psg_path.name}: no EEG channels found")
        ch = eegs[0]
    raw.pick([ch])
    raw.load_data()
    raw.filter(LOWCUT, HIGHCUT, verbose="ERROR")
    raw.resample(TARGET_SFREQ, verbose="ERROR")
    return raw.get_data()[0].astype(np.float64), float(raw.info["sfreq"]), ch


def worker_main_tda(pair: Pair, sub_seed: np.random.SeedSequence) -> List[dict]:
    """Per-night TDA on EEG Fpz-Cz across W/N1/N2/N3/REM, sampled to 30/stage."""
    rng = np.random.default_rng(sub_seed)
    try:
        x, sf, ch = _load_eeg(pair.psg_path)
        intervals = load_intervals(pair.hyp_path)
        epoch_len = int(EPOCH_SEC * sf)
        n_epochs = len(x) // epoch_len
        by_stage: Dict[str, List[int]] = {s: [] for s in STAGES_MAIN}
        for e in range(n_epochs):
            s = stage_at(intervals, (e + 0.5) * epoch_len / sf)
            if s in by_stage:
                by_stage[s].append(e)
        out = []
        psg_tag = _tag(_prefix(pair.psg_path))
        for stage, idxs in by_stage.items():
            if not idxs:
                continue
            if len(idxs) > MAX_EPOCHS_MAIN:
                idxs = list(rng.choice(idxs, size=MAX_EPOCHS_MAIN, replace=False))
            for e in idxs:
                seg = x[e * epoch_len:(e + 1) * epoch_len]
                feats = persistence_features(seg, EMBED_M, EMBED_TAU, MAXDIM,
                                             MIN_EMBED_POINTS_MAIN, downsample=True)
                if feats is None:
                    continue
                out.append({
                    "subject": pair.subject, "psg_tag": psg_tag,
                    "psg_file": pair.psg_path.name, "hyp_file": pair.hyp_path.name,
                    "channel": ch, "stage": stage, "epoch_index": int(e), **feats,
                })
        return out
    except Exception as ex:
        return [{"_ERROR_": f"{pair.psg_path.name} :: {type(ex).__name__}: {ex}"}]


def worker_robustness(pair: Pair, sub_seed: np.random.SeedSequence) -> List[dict]:
    """Per-night TDA across the m x tau grid, max 25 epochs/stage, both EEGs if present."""
    rng = np.random.default_rng(sub_seed)
    try:
        raw = mne.io.read_raw_edf(str(pair.psg_path), preload=True, verbose="ERROR")
        ch = next((c for c in [EEG_PRIMARY, EEG_SECONDARY] if c in raw.ch_names), None)
        if ch is None:
            eegs = [c for c in raw.ch_names if "EEG" in c.upper()]
            if not eegs:
                return []
            ch = eegs[0]
        raw.pick([ch]); raw.filter(LOWCUT, HIGHCUT, verbose="ERROR")
        raw.resample(TARGET_SFREQ, verbose="ERROR")
        x = raw.get_data()[0]
        sf = float(raw.info["sfreq"])
        epoch_len = int(EPOCH_SEC * sf)
        intervals = load_intervals(pair.hyp_path)
        n_epochs = len(x) // epoch_len

        by_stage: Dict[str, List[int]] = {s: [] for s in STAGES_MAIN}
        for e in range(n_epochs):
            s = stage_at(intervals, (e + 0.5) * epoch_len / sf)
            if s in by_stage:
                by_stage[s].append(e)

        sampled: Dict[str, List[int]] = {}
        for s, idxs in by_stage.items():
            if not idxs:
                continue
            sampled[s] = (list(rng.choice(idxs, size=MAX_EPOCHS_ROBUSTNESS, replace=False))
                          if len(idxs) > MAX_EPOCHS_ROBUSTNESS else idxs)

        out = []
        for s, idxs in sampled.items():
            for e in idxs:
                seg = x[e * epoch_len:(e + 1) * epoch_len][::2]
                for m in M_GRID_MAIN:
                    for tau in TAU_GRID_MAIN:
                        X = time_delay_embedding(seg, m, tau, MIN_EMBED_POINTS_ROBUSTNESS)
                        if X is None:
                            continue
                        X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
                        dgms = ripser(X, maxdim=MAXDIM)["dgms"]
                        h1 = dgm_summaries(dgms[1]) if len(dgms) > 1 else {"count":0,"tot_pers":0.0,"max_pers":0.0}
                        out.append({
                            "subject": pair.subject,
                            "psg_file": pair.psg_path.name, "hyp_file": pair.hyp_path.name,
                            "channel": ch, "stage": s, "epoch_index": int(e),
                            "m": int(m), "tau": int(tau),
                            "H1_count": h1["count"], "H1_totpers": h1["tot_pers"],
                            "H1_maxpers": h1["max_pers"],
                        })
        return out
    except Exception as ex:
        return [{"_ERROR_": f"{pair.psg_path.name} :: {type(ex).__name__}: {ex}"}]


def worker_baseline(pair: Pair) -> List[dict]:
    try:
        x, sf, ch = _load_eeg(pair.psg_path)
        intervals = load_intervals(pair.hyp_path)
        epoch_len = int(EPOCH_SEC * sf)
        n_epochs = len(x) // epoch_len
        out = []
        for e in range(n_epochs):
            s = stage_at(intervals, (e + 0.5) * epoch_len / sf)
            if s not in STAGES_MAIN:
                continue
            seg = x[e * epoch_len:(e + 1) * epoch_len]
            psd, freqs = mne.time_frequency.psd_array_welch(
                seg, sfreq=sf, fmin=0.5, fmax=40.0,
                n_fft=min(2048, len(seg)), verbose="ERROR",
            )
            out.append({
                "subject": pair.subject,
                "psg_file": pair.psg_path.name, "hyp_file": pair.hyp_path.name,
                "channel": ch, "epoch_index": int(e), "stage": s,
                "log_delta": float(np.log(bandpower(psd, freqs, 0.5, 4.0)  + 1e-12)),
                "log_theta": float(np.log(bandpower(psd, freqs, 4.0, 8.0)  + 1e-12)),
                "log_alpha": float(np.log(bandpower(psd, freqs, 8.0, 12.0) + 1e-12)),
                "log_sigma": float(np.log(bandpower(psd, freqs, 12.0, 15.0)+ 1e-12)),
                "log_beta":  float(np.log(bandpower(psd, freqs, 15.0, 30.0)+ 1e-12)),
                "spec_entropy": spectral_entropy(psd),
                "perm_entropy": permutation_entropy(seg, order=5, delay=1),
                "lz_complexity": lz_complexity_binary(seg),
            })
        return out
    except Exception as ex:
        return [{"_ERROR_": f"{pair.psg_path.name} :: {type(ex).__name__}: {ex}"}]


def worker_wake_subclass(pair: Pair) -> List[dict]:
    """Track 2 wake subclass features. Recording-percentile + 6-vote rule."""
    try:
        raw = mne.io.read_raw_edf(str(pair.psg_path), preload=True, verbose="ERROR")
        need = [EEG_PRIMARY, EEG_SECONDARY, EOG_CH, EMG_CH, EVENT_CH]
        picks = [c for c in need if c in raw.ch_names]
        if EEG_PRIMARY not in picks or EOG_CH not in picks:
            raise RuntimeError("missing required EEG/EOG channel")
        raw.pick(picks); raw.filter(LOWCUT, HIGHCUT, verbose="ERROR")
        raw.resample(TARGET_SFREQ, verbose="ERROR")
        sigs = {c: raw.get_data(picks=[c])[0].astype(np.float64) for c in raw.ch_names}
        sf = float(raw.info["sfreq"])
        intervals = load_intervals(pair.hyp_path)
        epoch_len = int(EPOCH_SEC * sf)
        n_epochs = len(sigs[EEG_PRIMARY]) // epoch_len

        eeg = sigs[EEG_PRIMARY]
        eog = sigs[EOG_CH]
        eeg_pz = sigs.get(EEG_SECONDARY)
        emg = sigs.get(EMG_CH)
        evt = sigs.get(EVENT_CH)

        rows = []
        for e in range(n_epochs):
            mid = (e + 0.5) * epoch_len / sf
            if stage_at(intervals, mid) != "W":
                continue
            sl = slice(e * epoch_len, (e + 1) * epoch_len)
            eeg_seg = eeg[sl]; eog_seg = eog[sl]
            if len(eeg_seg) != epoch_len:
                continue
            eog_lf = butter_bp(eog_seg, sf, 0.3, 5.0)
            eeg_d = butter_bp(eeg_seg, sf, 0.5, 4.0)
            alpha_seg = (eeg_pz[sl] if eeg_pz is not None else eeg_seg)
            # alpha ratio
            nperseg = min(len(alpha_seg), int(4 * sf))
            ar = np.nan
            if nperseg >= 16:
                f, ps = welch(alpha_seg, fs=sf, nperseg=nperseg)
                num = _trapz(ps[(f>=8)&(f<=12)], f[(f>=8)&(f<=12)])
                den = _trapz(ps[(f>=1)&(f<=20)], f[(f>=1)&(f<=20)])
                ar = float(num/den) if den > 0 else np.nan
            # eog peaks
            sigma = np.std(eog_lf)
            n_peaks = 0
            if sigma > 0:
                p, _ = find_peaks(np.abs(eog_lf), height=2.5*sigma,
                                  distance=max(1, int(0.20 * sf)))
                n_peaks = int(len(p))
            corr = float(np.corrcoef(eeg_seg, eog_seg)[0,1]) if (np.std(eeg_seg)>0 and np.std(eog_seg)>0) else 0.0

            rows.append({
                "subject": pair.subject,
                "psg_file": pair.psg_path.name, "hyp_file": pair.hyp_path.name,
                "epoch_index": int(e), "stage_original": "W",
                "channel_eeg": EEG_PRIMARY, "channel_eog": EOG_CH,
                "channel_emg": EMG_CH if emg is not None else "",
                "channel_event": EVENT_CH if evt is not None else "",
                "eog_rms_lf": rms(eog_lf),
                "eog_peak_count": n_peaks,
                "eeg_eog_corr": corr,
                "alpha_ratio": ar,
                "fpz_delta_var": float(np.var(eeg_d)),
                "eeg_ptp_uv": ptp_uv(eeg_seg),
                "eog_ptp_uv": ptp_uv(eog_seg),
                "emg_rms": rms(emg[sl]) if emg is not None else np.nan,
                "emg_ptp_uv": ptp_uv(emg[sl]) if emg is not None else np.nan,
                "event_rms": rms(evt[sl]) if evt is not None else np.nan,
                "event_ptp_uv": ptp_uv(evt[sl]) if evt is not None else np.nan,
            })
        if not rows:
            return []

        df = pd.DataFrame(rows)
        df["z_eeg_ptp"]   = robust_z(df["eeg_ptp_uv"])
        df["z_eog_ptp"]   = robust_z(df["eog_ptp_uv"])
        df["z_emg_rms"]   = robust_z(df["emg_rms"].fillna(df["emg_rms"].median()))
        df["z_event_ptp"] = robust_z(df["event_ptp_uv"].fillna(df["event_ptp_uv"].median()))

        gross = (
            (df["eeg_ptp_uv"] > 500.0) | (df["eog_ptp_uv"] > 1000.0) |
            (df["z_eeg_ptp"] > 5.0) | (df["z_eog_ptp"] > 5.0) |
            (df["z_emg_rms"] > 5.0) | (df["z_event_ptp"] > 5.0)
        )
        clean = df.loc[~gross]
        if clean.empty:
            df["wake_subclass"] = "W_bad"
            df["gross_bad"] = True
            df["active_score"] = 0
            return df.to_dict("records")

        eog_q75   = clean["eog_rms_lf"].quantile(0.75)
        alpha_q25 = clean["alpha_ratio"].quantile(0.25)
        delta_q75 = clean["fpz_delta_var"].quantile(0.75)
        emg_q75   = clean["emg_rms"].quantile(0.75) if clean["emg_rms"].notna().any() else np.inf

        c1 = df["eog_rms_lf"]   > eog_q75
        c2 = df["eog_peak_count"] >= 6
        c3 = df["eeg_eog_corr"].abs() > 0.35
        c4 = df["alpha_ratio"]   < alpha_q25
        c5 = df["fpz_delta_var"] > delta_q75
        c6 = df["emg_rms"] > emg_q75 if np.isfinite(emg_q75) else pd.Series(False, index=df.index)

        score = c1.astype(int)+c2.astype(int)+c3.astype(int)+c4.astype(int)+c5.astype(int)+c6.astype(int)
        sub = np.full(len(df), "W_quiet", dtype=object)
        sub[gross.values] = "W_bad"
        sub[(~gross & (score >= 2)).values] = "W_active_ocular"
        df["gross_bad"] = gross
        df["active_score"] = score
        df["wake_subclass"] = sub
        df["crit_high_eog_rms"] = c1
        df["crit_many_peaks"]   = c2
        df["crit_high_corr"]    = c3
        df["crit_low_alpha"]    = c4
        df["crit_high_delta"]   = c5
        df["crit_high_emg"]     = c6
        return df.to_dict("records")
    except Exception as ex:
        return [{"_ERROR_": f"{pair.psg_path.name} :: {type(ex).__name__}: {ex}"}]


def worker_wake_qc(pair: Pair) -> List[dict]:
    """Track 3 wake-QC table. Per-subject MAD-based thresholds, 3-vote ocular rule."""
    try:
        raw = mne.io.read_raw_edf(str(pair.psg_path), preload=True, verbose="ERROR")
        for need in [EEG_PRIMARY, EOG_CH]:
            if need not in raw.ch_names:
                raise RuntimeError(f"missing {need}")
        picks = [c for c in [EEG_PRIMARY, EOG_CH, EMG_CH] if c in raw.ch_names]
        raw.pick(picks); raw.load_data()
        sf_in = float(raw.info["sfreq"])
        eeg = butter_bp(raw.get_data(picks=[EEG_PRIMARY])[0], sf_in, LOWCUT, HIGHCUT)
        eog = butter_bp(raw.get_data(picks=[EOG_CH])[0],     sf_in, EOG_LOW, EOG_HIGH)
        emg = (butter_bp(raw.get_data(picks=[EMG_CH])[0], sf_in, EMG_LOW, EMG_HIGH)
               if EMG_CH in picks else None)
        # resample
        from scipy.signal import resample_poly
        from math import gcd
        def _resample(x, sf_in, sf_out):
            g = gcd(int(round(sf_in * 1000)), int(round(sf_out * 1000)))
            up = int(round(sf_out * 1000)) // g
            down = int(round(sf_in * 1000)) // g
            return resample_poly(x, up, down)
        eeg = _resample(eeg, sf_in, TARGET_SFREQ)
        eog = _resample(eog, sf_in, TARGET_SFREQ)
        if emg is not None:
            emg = _resample(emg, sf_in, TARGET_SFREQ)
        sf = TARGET_SFREQ
        epoch_len = int(EPOCH_SEC * sf)
        n_epochs = len(eeg) // epoch_len
        intervals = load_intervals(pair.hyp_path)
        psg_tag = _tag(_prefix(pair.psg_path))

        rows = []
        for e in range(n_epochs):
            mid = (e + 0.5) * epoch_len / sf
            stage_orig = stage_at(intervals, mid)
            if stage_orig is None:
                continue
            sl = slice(e * epoch_len, (e + 1) * epoch_len)
            eeg_seg = eeg[sl]; eog_seg = eog[sl]
            emg_seg = emg[sl] if emg is not None else None
            if len(eeg_seg) != epoch_len:
                continue
            eeg_drift = butter_bp(eeg_seg, sf, 0.3, 2.0)
            eog_f = butter_bp(eog_seg, sf, 0.3, 5.0)
            sigma = np.std(eog_f)
            n_peaks = 0
            if sigma > 0:
                p, _ = find_peaks(np.abs(eog_f), height=2.5 * sigma,
                                  distance=max(1, int(round(0.20 * sf))))
                n_peaks = int(len(p))
            corr = float(np.corrcoef(eeg_seg, eog_seg)[0,1]) if (np.std(eeg_seg)>0 and np.std(eog_seg)>0) else 0.0
            if not np.isfinite(corr):
                corr = 0.0
            ptp_eeg = ptp_uv(eeg_seg)
            flat = int(float(np.var(eeg_seg)) < 1e-12 or ptp_eeg < 1.0)
            rows.append({
                "subject": pair.subject, "psg_tag": psg_tag,
                "psg_file": pair.psg_path.name, "hyp_file": pair.hyp_path.name,
                "epoch_index": int(e),
                "stage_original": stage_orig,
                "stage_grouped": stage_orig,
                "eeg_ptp_uv": ptp_eeg,
                "eeg_drift_uv": ptp_uv(eeg_drift),
                "eeg_flat_flag": flat,
                "eog_rms": rms(eog_f),
                "eog_ptp_uv": ptp_uv(eog_seg),
                "eog_peak_count": n_peaks,
                "eeg_eog_corr": corr,
                "emg_rms": rms(emg_seg) if emg_seg is not None else np.nan,
            })
        if not rows:
            return []
        df = pd.DataFrame(rows)
        # subject-wise thresholds from W epochs only
        w = df[df["stage_grouped"] == "W"]
        if w.empty:
            return df.to_dict("records")
        eeg_ptp_thr = robust_hi(w["eeg_ptp_uv"], k=4.0)
        drift_thr   = robust_hi(w["eeg_drift_uv"], k=4.0)
        emg_thr     = robust_hi(w["emg_rms"].dropna(), k=4.0) if w["emg_rms"].notna().any() else np.inf
        eog_rms_thr = float(w["eog_rms"].quantile(0.75))

        gross = (
            (df["eeg_ptp_uv"]   > eeg_ptp_thr) |
            (df["eeg_drift_uv"] > drift_thr) |
            ((df["emg_rms"]     > emg_thr) if np.isfinite(emg_thr) else False) |
            (df["eeg_flat_flag"] == 1)
        )
        ocular_votes = (
            (df["eog_rms"] > eog_rms_thr).astype(int) +
            (df["eog_peak_count"] >= 6).astype(int) +
            (df["eeg_eog_corr"].abs() > 0.35).astype(int)
        )
        ocular = ocular_votes >= 2

        df["gross_bad_flag"] = gross.astype(int)
        df["ocular_heavy_flag"] = ocular.astype(int)
        # subclass only for stage_grouped == W
        sub = np.full(len(df), "", dtype=object)
        is_w = (df["stage_grouped"] == "W").values
        sub[is_w] = "W_quiet"
        sub[is_w & ocular.values & ~gross.values] = "W_active_ocular"
        sub[is_w & gross.values] = "W_bad"
        df["wake_subclass"] = sub

        reasons = []
        for _, r in df.iterrows():
            rr = []
            if r["gross_bad_flag"]:
                if r["eeg_flat_flag"] == 1: rr.append("flat")
                if r["eeg_ptp_uv"]    > eeg_ptp_thr: rr.append("eeg_ptp")
                if r["eeg_drift_uv"]  > drift_thr:   rr.append("drift")
                if np.isfinite(emg_thr) and r["emg_rms"] > emg_thr: rr.append("emg")
            elif r["ocular_heavy_flag"]:
                rr.append("ocular")
            reasons.append(",".join(rr))
        df["qc_reason"] = reasons
        return df.to_dict("records")
    except Exception as ex:
        return [{"_ERROR_": f"{pair.psg_path.name} :: {type(ex).__name__}: {ex}"}]


def worker_corrected_eeg(pair: Pair) -> Optional[dict]:
    """Generate EOG-corrected EEG epoch bundle (NPZ). Returns manifest row."""
    try:
        raw = mne.io.read_raw_edf(str(pair.psg_path), preload=True, verbose="ERROR")
        if EEG_PRIMARY not in raw.ch_names or EOG_CH not in raw.ch_names:
            return None
        raw.pick([EEG_PRIMARY, EOG_CH])
        sf_in = float(raw.info["sfreq"])
        eeg = butter_bp(raw.get_data(picks=[EEG_PRIMARY])[0], sf_in, LOWCUT, HIGHCUT)
        eog = butter_bp(raw.get_data(picks=[EOG_CH])[0],     sf_in, EOG_LOW, EOG_HIGH)
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(round(sf_in*1000)), int(round(TARGET_SFREQ*1000)))
        up = int(round(TARGET_SFREQ*1000))//g; down = int(round(sf_in*1000))//g
        eeg = resample_poly(eeg, up, down); eog = resample_poly(eog, up, down)
        sf = TARGET_SFREQ
        eeg0 = eeg - eeg.mean(); eog0 = eog - eog.mean()
        denom = float(np.dot(eog0, eog0))
        beta = float(np.dot(eeg0, eog0) / denom) if denom > 0 else 0.0
        eeg_corr = eeg - beta * eog

        epoch_len = int(EPOCH_SEC * sf)
        n_epochs = len(eeg) // epoch_len
        idx = np.arange(n_epochs, dtype=np.int32)
        def _ep(arr):
            return np.stack([arr[i*epoch_len:(i+1)*epoch_len] for i in idx]).astype(np.float32)
        eeg_raw_e = _ep(eeg); eeg_cor_e = _ep(eeg_corr); eog_e = _ep(eog)

        psg_stem = pair.psg_path.stem
        out = CORRECTED_DIR / f"{psg_stem}_corrected_epochs.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            eeg_raw_epochs=eeg_raw_e,
            eeg_corrected_epochs=eeg_cor_e,
            eog_epochs=eog_e,
            epoch_index=idx,
            beta=np.array([beta], dtype=np.float64),
            sfreq=np.array([sf], dtype=np.float64),
            subject=np.array([pair.subject], dtype=object),
            psg_tag=np.array([_tag(_prefix(pair.psg_path))], dtype=object),
            psg_file=np.array([pair.psg_path.name], dtype=object),
            hyp_file=np.array([pair.hyp_path.name], dtype=object),
        )
        return {
            "subject": pair.subject, "psg_tag": _tag(_prefix(pair.psg_path)),
            "psg_file": pair.psg_path.name, "hyp_file": pair.hyp_path.name,
            "npz_file": out.name, "n_epochs_saved": int(n_epochs),
            "beta": beta, "sfreq": sf,
        }
    except Exception:
        return None


def _tda_on_signal_array(arr: np.ndarray, sf: float, intervals,
                         qc_lookup: Dict[Tuple, Tuple[str, int]],
                         subject: str, psg_file: str, hyp_file: str, psg_tag: str,
                         channel_label: str, sub_seed: np.random.SeedSequence,
                         min_pts: int = MIN_EMBED_POINTS_MAIN,
                         m: int = EMBED_M, tau: int = EMBED_TAU,
                         downsample: bool = True,
                         max_per_stage: int = MAX_EPOCHS_MAIN) -> List[dict]:
    """Compute TDA over W_quiet/W_active_ocular/REM epochs. arr is one channel
    already filtered/resampled. qc_lookup maps (subject,psg_file,hyp_file,epoch)
    -> (wake_subclass_or_'', gross_bad_flag)."""
    rng = np.random.default_rng(sub_seed)
    epoch_len = int(EPOCH_SEC * sf)
    n_epochs = len(arr) // epoch_len
    cands: Dict[str, List[Tuple[int, np.ndarray]]] = {s: [] for s in STAGES_WAKE}
    for e in range(n_epochs):
        mid = (e + 0.5) * epoch_len / sf
        s_orig = stage_at(intervals, mid)
        if s_orig not in {"W", "REM"}:
            continue
        if s_orig == "W":
            key = (subject, psg_file, hyp_file, int(e))
            sub, gross = qc_lookup.get(key, ("", 0))
            if gross or sub not in {"W_quiet", "W_active_ocular"}:
                continue
            stage = sub
        else:
            stage = "REM"
        seg = arr[e * epoch_len:(e + 1) * epoch_len]
        if len(seg) != epoch_len:
            continue
        cands[stage].append((int(e), seg))
    out = []
    for stage, items in cands.items():
        if not items:
            continue
        if len(items) > max_per_stage:
            sel = rng.choice(len(items), size=max_per_stage, replace=False)
            items = [items[i] for i in sorted(sel)]
        for e, seg in items:
            feats = persistence_features(seg, m, tau, MAXDIM, min_pts, downsample=downsample)
            if feats is None:
                continue
            out.append({
                "subject": subject, "psg_tag": psg_tag,
                "psg_file": psg_file, "hyp_file": hyp_file,
                "channel": channel_label, "stage": stage, "epoch_index": int(e),
                **feats,
            })
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ──────────────────────────────────────────────────────────────────────────────

def holm(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    adj = np.minimum(1.0, (m - np.arange(m)) * sorted_p)
    adj = np.maximum.accumulate(adj)
    out = np.empty(m); out[order] = adj
    return out

def within_subject_z(df: pd.DataFrame, col: str) -> pd.Series:
    mu = df.groupby("subject")[col].transform("mean")
    sd = df.groupby("subject")[col].transform("std").replace(0, np.nan)
    return ((df[col] - mu) / sd).fillna(0.0)

def fit_mixedlm_stage(ss: pd.DataFrame, y: str, stages: Sequence[str],
                      method: str = "powell"):
    ss = ss.copy()
    ss["stage"] = pd.Categorical(ss["stage"], categories=list(stages), ordered=True)
    if method == "powell":
        full = smf.mixedlm(f"{y} ~ C(stage)", ss, groups=ss["subject"]).fit(
            reml=False, method="powell", maxiter=2000, disp=False)
        null = smf.mixedlm(f"{y} ~ 1", ss, groups=ss["subject"]).fit(
            reml=False, method="powell", maxiter=2000, disp=False)
    else:
        full = smf.mixedlm(f"{y} ~ C(stage)", ss, groups=ss["subject"]).fit(
            reml=False, method="lbfgs", disp=False)
        null = smf.mixedlm(f"{y} ~ 1", ss, groups=ss["subject"]).fit(
            reml=False, method="lbfgs", disp=False)
    lr = 2 * (full.llf - null.llf)
    df_diff = full.df_modelwc - null.df_modelwc
    p_lr = stats.chi2.sf(lr, df_diff)
    return full, float(lr), int(df_diff), float(p_lr)

def planned_contrasts(res, yname: str, planned: Sequence[Tuple[str, str]],
                      stages: Sequence[str]) -> pd.DataFrame:
    """Compute planned contrasts using statsmodels' default reference coding,
    where the reference category is stages[0]."""
    params = res.params; cov = res.cov_params(); idx = params.index.tolist()
    ref = stages[0]

    def L_for(a: str, b: str) -> np.ndarray:
        L = np.zeros(len(idx))
        if a != ref:
            L[idx.index(f"C(stage)[T.{a}]")] += 1.0
        if b != ref:
            L[idx.index(f"C(stage)[T.{b}]")] -= 1.0
        return L

    rows = []
    for a, b in planned:
        L = L_for(a, b)
        est = float(np.dot(L, params))
        se = float(np.sqrt(np.dot(L, np.dot(cov, L))))
        z = est / se if se > 0 else np.nan
        p = 2 * stats.norm.sf(abs(z)) if se > 0 else np.nan
        rows.append({
            "metric": yname, "contrast": f"{a} - {b}",
            "estimate": est, "SE": se, "z": z, "p": p,
            "CI95_low": est - 1.96 * se, "CI95_high": est + 1.96 * se,
        })
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p"].values)
    return df


def run_mixedlm_analysis(df: pd.DataFrame, metrics: Sequence[str],
                         stages: Sequence[str], planned: Sequence[Tuple[str, str]],
                         method: str = "powell",
                         derive_k0: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run omnibus + planned contrasts for each metric, with optional within-subject z."""
    df = df[df["stage"].isin(stages)].copy()
    omni_rows, contrast_rows = [], []
    for y in metrics:
        if derive_k0:
            ycol = "K0_" + y.split("_")[-1]
            df[ycol] = within_subject_z(df, y)
            yname = ycol
        else:
            yname = y
        ss = df.groupby(["subject", "stage"], as_index=False)[yname].mean()
        n_rows = len(ss); n_subj = ss["subject"].nunique()
        try:
            res, lr, df_diff, p_lr = fit_mixedlm_stage(ss, yname, stages, method=method)
        except Exception as ex:
            omni_rows.append({"metric": yname, "LR": np.nan, "df": 0, "p": np.nan,
                              "n_rows": n_rows, "n_subjects": n_subj,
                              "error": f"{type(ex).__name__}: {ex}"})
            continue
        omni_rows.append({"metric": yname, "LR": lr, "df": df_diff, "p": p_lr,
                          "n_rows": n_rows, "n_subjects": n_subj})
        contrast_rows.append(planned_contrasts(res, yname, planned, stages))
    omni = pd.DataFrame(omni_rows)
    contr = pd.concat(contrast_rows, ignore_index=True) if contrast_rows else pd.DataFrame()
    return omni, contr


# ──────────────────────────────────────────────────────────────────────────────
# Step orchestration helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save(df: pd.DataFrame, path: Path, n: Narrator):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if df.empty:
        # Make sure the file is not literally zero-bytes; pandas needs at least
        # a header line to read it back without raising EmptyDataError.
        if path.stat().st_size == 0:
            path.write_text("# empty\n", encoding="utf-8")
        n.log(f"  ! wrote {path.relative_to(PROJECT_ROOT)}  (EMPTY — 0 rows)")
    else:
        n.log(f"  ✓ wrote {path.relative_to(PROJECT_ROOT)}  ({len(df):,} rows)")


def _safe_read_csv(path: Path) -> pd.DataFrame:
    """Read a prerequisite CSV; return an empty DataFrame for empty / header-less files
    instead of raising EmptyDataError. Handles the case where a producer step ran but
    yielded zero rows (e.g. ``# empty`` placeholder)."""
    try:
        if (not path.exists()) or path.stat().st_size == 0:
            return pd.DataFrame()
        # Skip lines starting with '#' so our placeholder doesn't choke pandas.
        return pd.read_csv(path, comment="#")
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()

def _exists(*paths: Path) -> bool:
    """True only if all paths exist AND are non-trivially populated.
    For CSV files, requires at least one data row beyond the header.
    Zero-byte files, '# empty' placeholders, and header-only CSVs are all
    treated as missing so the producing step will re-run on next invocation."""
    for p in paths:
        if not p.exists():
            return False
        try:
            sz = p.stat().st_size
        except OSError:
            return False
        if sz == 0:
            return False
        suffix = p.suffix.lower()
        if suffix == ".csv":
            # A populated CSV needs at least 2 non-comment, non-blank lines
            # (header + ≥1 data row).
            try:
                non_comment = 0
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        s = line.strip()
                        if not s or s.startswith("#"):
                            continue
                        non_comment += 1
                        if non_comment >= 2:
                            break
                if non_comment < 2:
                    return False
            except Exception:
                pass
        else:
            # Non-CSV: small placeholder files are flagged as missing.
            if sz < 32:
                try:
                    head = p.read_text(encoding="utf-8", errors="replace").strip()
                    if head == "" or head == "# empty":
                        return False
                except Exception:
                    pass
    return True

import contextlib

@contextlib.contextmanager
def _tqdm_joblib(tqdm_obj):
    """Patch joblib so a tqdm bar updates as each task completes."""
    if not HAS_JOBLIB:
        yield tqdm_obj
        return
    import joblib.parallel as jp
    original = jp.BatchCompletionCallBack

    class _Cb(original):
        def __call__(self, *args, **kwargs):
            tqdm_obj.update(n=self.batch_size)
            return original.__call__(self, *args, **kwargs)

    jp.BatchCompletionCallBack = _Cb
    try:
        yield tqdm_obj
    finally:
        jp.BatchCompletionCallBack = original
        tqdm_obj.close()


def _parallel_map(fn: Callable, items: Sequence, n_jobs: int, desc: str, n: Narrator,
                  with_seed: bool = False):
    """Map fn over items with a tqdm progress bar (parallel via joblib if n_jobs>1).
    Optionally pass each worker a per-item SeedSequence derived from RNG_SEED so
    output is reproducible regardless of n_jobs."""
    seeds = None
    if with_seed:
        seeds = list(np.random.SeedSequence(RNG_SEED).spawn(len(items)))
    n.log(f"  · {desc}: {len(items)} items, n_jobs={n_jobs}")
    t0 = time.time()

    bar_kwargs = dict(total=len(items), desc=f"    {desc}", ncols=90,
                      mininterval=1.0, ascii=False, leave=True,
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    if HAS_JOBLIB and n_jobs and n_jobs > 1:
        if HAS_TQDM:
            with _tqdm_joblib(tqdm(**bar_kwargs)):
                if with_seed:
                    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
                        delayed(fn)(it, seeds[i]) for i, it in enumerate(items))
                else:
                    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
                        delayed(fn)(it) for it in items)
        else:
            if with_seed:
                results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
                    delayed(fn)(it, seeds[i]) for i, it in enumerate(items))
            else:
                results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
                    delayed(fn)(it) for it in items)
    else:
        if HAS_TQDM:
            iterator = tqdm(items, **{k: v for k, v in bar_kwargs.items() if k != "total"})
        else:
            iterator = items
        results = []
        for i, it in enumerate(iterator):
            if with_seed:
                results.append(fn(it, seeds[i]))
            else:
                results.append(fn(it))
    n.log(f"    done in {time.time()-t0:0.1f}s")
    return results


def _parallel_bar(worker: Callable, items: Sequence, n_jobs: int, desc: str):
    """Run worker(item) over items with a live tqdm progress bar.
    Returns the list of results in submission order."""
    bar_kwargs = dict(total=len(items), desc=f"    {desc}", ncols=90,
                      mininterval=1.0, leave=True,
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    if HAS_JOBLIB and n_jobs and n_jobs > 1:
        if HAS_TQDM:
            with _tqdm_joblib(tqdm(**bar_kwargs)):
                return Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
                    delayed(worker)(it) for it in items)
        return Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(worker)(it) for it in items)
    if HAS_TQDM:
        iterator = tqdm(items, **{k: v for k, v in bar_kwargs.items() if k != "total"})
    else:
        iterator = items
    return [worker(it) for it in iterator]

def _inner_bar(total: int, desc: str):
    """Return a tqdm bar for inner loops (mixed-LM / GLM refits) or None if tqdm absent."""
    if not HAS_TQDM:
        return None
    return tqdm(total=total, desc=f"    {desc}", ncols=90, mininterval=1.0,
                leave=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")



# ──────────────────────────────────────────────────────────────────────────────
# Steps
# ──────────────────────────────────────────────────────────────────────────────

def step_main_tda(pairs, n_jobs, force, n: Narrator):
    out_e = OUT_DIR / "tda_epoch_features_all.csv"
    out_s = OUT_DIR / "tda_stage_summary_all.csv"
    if _exists(out_e, out_s) and not force:
        n.log("  ✓ skipping (outputs exist)")
        return
    results = _parallel_map(worker_main_tda, pairs, n_jobs, "main TDA per night", n, with_seed=True)
    rows, errs, empty = [], [], 0
    for r in results:
        if not r:
            empty += 1; continue
        ok = [x for x in r if "_ERROR_" not in x]
        bad = [x["_ERROR_"] for x in r if "_ERROR_" in x]
        rows.extend(ok); errs.extend(bad)
    n.log(f"  · collected {len(rows)} rows, {len(errs)} workers errored, {empty} returned empty")
    if errs:
        n.log("  · first few errors:")
        for e in errs[:5]:
            n.log(f"    ! {e}")
    df = pd.DataFrame(rows)
    _save(df, out_e, n)
    metrics = ["H0_count","H0_totpers","H0_maxpers","H1_count","H1_totpers","H1_maxpers"]
    if not df.empty:
        summary = df.groupby("stage")[metrics].agg(["mean","std","count"])
        summary.to_csv(out_s)
        n.log(f"  ✓ wrote {out_s.relative_to(PROJECT_ROOT)}")
    if errs:
        (OUT_DIR / "tda_errors.log").write_text("\n".join(e["_ERROR_"] for e in errs), encoding="utf-8")
        n.log(f"  ! {len(errs)} nights errored (see outputs/tda_errors.log)")


def step_robustness_grid(pairs, n_jobs, force, n: Narrator):
    out = OUT_DIR / "tda_robustness_grid_epochs.csv"
    if _exists(out) and not force:
        n.log("  ✓ skipping (output exists)")
        return
    results = _parallel_map(worker_robustness, pairs, n_jobs, "robustness grid per night",
                            n, with_seed=True)
    rows = [x for r in results for x in r if "_ERROR_" not in x]
    _save(pd.DataFrame(rows), out, n)


def step_baselines(pairs, n_jobs, force, n: Narrator):
    out = OUT_DIR / "baseline_epoch_features_all.csv"
    if _exists(out) and not force:
        n.log("  ✓ skipping (output exists)")
        return
    results = _parallel_map(worker_baseline, pairs, n_jobs, "baseline metrics per night", n)
    rows = [x for r in results for x in r if "_ERROR_" not in x]
    _save(pd.DataFrame(rows), out, n)


def step_wake_subclasses(pairs, n_jobs, force, n: Narrator):
    out_w = OUT_DIR / "wake_epoch_subclasses.csv"
    out_s = OUT_DIR / "wake_subclass_summary.csv"
    if _exists(out_w, out_s) and not force:
        n.log("  ✓ skipping (outputs exist)")
        return
    results = _parallel_map(worker_wake_subclass, pairs, n_jobs, "wake subclasses per night", n)
    rows, errors, empty = [], [], 0
    for r in results:
        if not r:
            empty += 1; continue
        ok = [x for x in r if "_ERROR_" not in x]
        bad = [x["_ERROR_"] for x in r if "_ERROR_" in x]
        if ok:
            rows.extend(ok)
        else:
            errors.extend(bad)
    n.log(f"  · collected {len(rows)} rows, {len(errors)} workers errored, {empty} returned empty")
    if errors:
        n.log("  · first few errors:")
        for e in errors[:5]:
            n.log(f"    ! {e}")
        if len(errors) > 5:
            n.log(f"    ! ... and {len(errors) - 5} more")
        # Also save full error list
        (OUT_DIR / "wake_subclasses_errors.log").write_text(
            "\n".join(errors), encoding="utf-8")
        n.log(f"  · full error list at outputs/wake_subclasses_errors.log")
    df = pd.DataFrame(rows)
    _save(df, out_w, n)
    if df.empty:
        return
    summary = (df.groupby(["subject","psg_file","hyp_file"])["wake_subclass"]
                 .value_counts().unstack(fill_value=0).reset_index())
    for c in ["W_quiet","W_active_ocular","W_bad"]:
        if c not in summary.columns:
            summary[c] = 0
    summary["n_wake_total"] = summary[["W_quiet","W_active_ocular","W_bad"]].sum(axis=1)
    for c in ["W_quiet","W_active_ocular","W_bad"]:
        summary[f"pct_{c.replace('W_','')}"] = 100.0 * summary[c] / summary["n_wake_total"].replace(0, np.nan)
    _save(summary.sort_values(["subject","psg_file"]).reset_index(drop=True), out_s, n)


def step_wake_qc(pairs, n_jobs, force, n: Narrator):
    out = OUT_DIR / "wake_qc_epoch_table.csv"
    if _exists(out) and not force:
        n.log("  ✓ skipping (output exists)")
        return
    results = _parallel_map(worker_wake_qc, pairs, n_jobs, "wake QC per night", n)
    rows, errors, empty = [], [], 0
    for r in results:
        if not r:
            empty += 1; continue
        ok = [x for x in r if "_ERROR_" not in x]
        bad = [x["_ERROR_"] for x in r if "_ERROR_" in x]
        if ok: rows.extend(ok)
        errors.extend(bad)
    n.log(f"  · collected {len(rows)} rows, {len(errors)} workers errored, {empty} returned empty")
    if errors:
        for e in errors[:5]:
            n.log(f"    ! {e}")
    _save(pd.DataFrame(rows), out, n)


def step_corrected_eeg(pairs, n_jobs, force, n: Narrator):
    manifest_csv = OUT_DIR / "corrected_epochs_manifest.csv"
    if _exists(manifest_csv) and not force:
        n.log("  ✓ skipping (manifest exists)")
        return
    CORRECTED_DIR.mkdir(parents=True, exist_ok=True)
    results = _parallel_map(worker_corrected_eeg, pairs, n_jobs, "corrected EEG per night", n)
    rows = [r for r in results if r is not None]
    _save(pd.DataFrame(rows), manifest_csv, n)


def step_tda_wake_subclasses(pairs, n_jobs, force, n: Narrator):
    """Track 2: TDA on wake subclasses + REM, using wake_epoch_subclasses.csv."""
    out_e = OUT_DIR / "tda_epoch_features_wake_subclasses.csv"
    out_s = OUT_DIR / "tda_stage_summary_wake_subclasses.csv"
    if _exists(out_e, out_s) and not force:
        n.log("  ✓ skipping (outputs exist)")
        return
    wake_csv = OUT_DIR / "wake_epoch_subclasses.csv"
    if not _exists(wake_csv):
        n.log("  ! prerequisite missing or empty: wake_epoch_subclasses.csv")
        return
    wdf = _safe_read_csv(wake_csv)
    if wdf.empty:
        n.log("  ! prerequisite empty: wake_epoch_subclasses.csv "
              "(delete the file and re-run the wake_subclasses step)")
        return
    qc_lookup = {
        (r.subject, r.psg_file, r.hyp_file, int(r.epoch_index)): (r.wake_subclass, 0)
        for r in wdf.itertuples(index=False)
    }
    seeds = list(np.random.SeedSequence(RNG_SEED).spawn(len(pairs)))

    def _one(pair_idx):
        pair, sub_seed = pair_idx
        try:
            x, sf, ch = _load_eeg(pair.psg_path)
            intervals = load_intervals(pair.hyp_path)
            return _tda_on_signal_array(
                x, sf, intervals, qc_lookup,
                pair.subject, pair.psg_path.name, pair.hyp_path.name,
                _tag(_prefix(pair.psg_path)), ch, sub_seed,
            )
        except Exception:
            return []
    items = list(zip(pairs, seeds))
    n.log(f"  · TDA wake subclasses: {len(items)} nights, n_jobs={n_jobs}")
    t0 = time.time()
    results = _parallel_bar(_one, items, n_jobs, "TDA wake subclasses")
    n.log(f"    done in {time.time()-t0:0.1f}s")
    rows = [x for r in results for x in r]
    df = pd.DataFrame(rows)
    _save(df, out_e, n)
    if not df.empty:
        metrics = ["H0_count","H0_totpers","H0_maxpers","H1_count","H1_totpers","H1_maxpers"]
        df.groupby("stage")[metrics].agg(["mean","std","count"]).reset_index().to_csv(out_s, index=False)
        n.log(f"  ✓ wrote {out_s.relative_to(PROJECT_ROOT)}")


def _tda_on_track3(pairs, n_jobs, n: Narrator,
                   signal_loader: Callable[[Pair], Tuple[np.ndarray, float]],
                   channel_label: str,
                   out_epoch: Path, out_summary: Path,
                   force: bool):
    """Track 3 helper: TDA on a chosen signal across W_quiet/W_active_ocular/REM."""
    if _exists(out_epoch, out_summary) and not force:
        n.log(f"  ✓ skipping {out_epoch.name} (exists)")
        return
    qc_csv = OUT_DIR / "wake_qc_epoch_table.csv"
    if not _exists(qc_csv):
        n.log("  ! prerequisite missing or empty: wake_qc_epoch_table.csv")
        return
    qc = _safe_read_csv(qc_csv)
    qc_lookup = {
        (r.subject, r.psg_file, r.hyp_file, int(r.epoch_index)):
            (str(r.wake_subclass) if isinstance(r.wake_subclass, str) else "",
             int(r.gross_bad_flag) if pd.notna(r.gross_bad_flag) else 0)
        for r in qc.itertuples(index=False)
    }
    seeds = list(np.random.SeedSequence(RNG_SEED).spawn(len(pairs)))

    def _one(pair_seed):
        pair, sub_seed = pair_seed
        try:
            arr, sf = signal_loader(pair)
            intervals = load_intervals(pair.hyp_path)
            return _tda_on_signal_array(
                arr, sf, intervals, qc_lookup,
                pair.subject, pair.psg_path.name, pair.hyp_path.name,
                _tag(_prefix(pair.psg_path)), channel_label, sub_seed,
            )
        except Exception:
            return []
    items = list(zip(pairs, seeds))
    n.log(f"  · TDA on {channel_label}: {len(items)} nights, n_jobs={n_jobs}")
    t0 = time.time()
    results = _parallel_bar(_one, items, n_jobs, f"TDA {channel_label}")
    n.log(f"    done in {time.time()-t0:0.1f}s")
    rows = [x for r in results for x in r]
    df = pd.DataFrame(rows)
    _save(df, out_epoch, n)
    if not df.empty:
        metrics = ["H0_count","H0_totpers","H0_maxpers","H1_count","H1_totpers","H1_maxpers"]
        df.groupby("stage")[metrics].agg(["mean","std","count"]).reset_index().to_csv(out_summary, index=False)
        n.log(f"  ✓ wrote {out_summary.relative_to(PROJECT_ROOT)}")


def _load_raw_eeg_for_track3(pair: Pair) -> Tuple[np.ndarray, float]:
    """Same preprocessing as wake_qc: bandpass on raw, then resample with poly."""
    raw = mne.io.read_raw_edf(str(pair.psg_path), preload=True, verbose="ERROR")
    if EEG_PRIMARY not in raw.ch_names:
        raise RuntimeError("missing EEG Fpz-Cz")
    raw.pick([EEG_PRIMARY])
    sf_in = float(raw.info["sfreq"])
    x = butter_bp(raw.get_data()[0], sf_in, LOWCUT, HIGHCUT)
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(round(sf_in*1000)), int(round(TARGET_SFREQ*1000)))
    up = int(round(TARGET_SFREQ*1000))//g; down = int(round(sf_in*1000))//g
    return resample_poly(x, up, down), TARGET_SFREQ


def _load_corrected_eeg_for_track3(pair: Pair) -> Tuple[np.ndarray, float]:
    """Reconstruct continuous corrected EEG from saved NPZ epoch bundle."""
    npz_path = CORRECTED_DIR / f"{pair.psg_path.stem}_corrected_epochs.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"no NPZ for {pair.psg_path.name}")
    z = np.load(npz_path, allow_pickle=True)
    arr = z["eeg_corrected_epochs"].reshape(-1).astype(np.float64)
    return arr, float(z["sfreq"][0])


def _load_eog_for_track3(pair: Pair) -> Tuple[np.ndarray, float]:
    raw = mne.io.read_raw_edf(str(pair.psg_path), preload=True, verbose="ERROR")
    if EOG_CH not in raw.ch_names:
        raise RuntimeError("missing EOG horizontal")
    raw.pick([EOG_CH])
    sf_in = float(raw.info["sfreq"])
    x = butter_bp(raw.get_data()[0], sf_in, EOG_LOW, EOG_HIGH)
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(round(sf_in*1000)), int(round(TARGET_SFREQ*1000)))
    up = int(round(TARGET_SFREQ*1000))//g; down = int(round(sf_in*1000))//g
    return resample_poly(x, up, down), TARGET_SFREQ


def step_tda_track3(pairs, n_jobs, force, n: Narrator):
    _tda_on_track3(pairs, n_jobs, n, _load_raw_eeg_for_track3,       "EEG Fpz-Cz",
                   OUT_DIR/"tda_epoch_features_wake_raw.csv",       OUT_DIR/"tda_stage_summary_wake_raw.csv", force)
    _tda_on_track3(pairs, n_jobs, n, _load_corrected_eeg_for_track3, "EEG Fpz-Cz (EOG-corrected)",
                   OUT_DIR/"tda_epoch_features_wake_corrected.csv", OUT_DIR/"tda_stage_summary_wake_corrected.csv", force)
    _tda_on_track3(pairs, n_jobs, n, _load_eog_for_track3,           "EOG horizontal",
                   OUT_DIR/"tda_epoch_features_eog.csv",            OUT_DIR/"tda_stage_summary_eog.csv", force)


def step_wake_subclass_robustness(pairs, n_jobs, force, n: Narrator):
    """TDA grid sweep on raw EEG for wake subclasses (M_GRID_WAKE x TAU_GRID_WAKE)."""
    out_grid = OUT_DIR / "wake_subclass_robustness_grid.csv"
    out_omni = OUT_DIR / "wake_subclass_robustness_mixedlm_omnibus.csv"
    out_pc = OUT_DIR / "wake_subclass_robustness_planned_contrasts.csv"
    if _exists(out_grid, out_omni, out_pc) and not force:
        n.log("  ✓ skipping (outputs exist)")
        return
    qc_csv = OUT_DIR / "wake_qc_epoch_table.csv"
    if not _exists(qc_csv):
        n.log("  ! prerequisite missing or empty: wake_qc_epoch_table.csv")
        return
    qc = _safe_read_csv(qc_csv)
    qc_lookup = {
        (r.subject, r.psg_file, r.hyp_file, int(r.epoch_index)):
            (str(r.wake_subclass) if isinstance(r.wake_subclass, str) else "",
             int(r.gross_bad_flag) if pd.notna(r.gross_bad_flag) else 0)
        for r in qc.itertuples(index=False)
    }
    seeds = list(np.random.SeedSequence(RNG_SEED).spawn(len(pairs)))

    def _one(pair_seed):
        pair, sub_seed = pair_seed
        rng = np.random.default_rng(sub_seed)
        try:
            arr, sf = _load_raw_eeg_for_track3(pair)
            intervals = load_intervals(pair.hyp_path)
            epoch_len = int(EPOCH_SEC * sf)
            n_epochs = len(arr) // epoch_len
            cands: Dict[str, List[Tuple[int, np.ndarray]]] = {s: [] for s in STAGES_WAKE}
            for e in range(n_epochs):
                mid = (e + 0.5) * epoch_len / sf
                s_orig = stage_at(intervals, mid)
                if s_orig not in {"W", "REM"}:
                    continue
                if s_orig == "W":
                    sub, gross = qc_lookup.get((pair.subject, pair.psg_path.name, pair.hyp_path.name, int(e)), ("", 0))
                    if gross or sub not in {"W_quiet", "W_active_ocular"}:
                        continue
                    stage = sub
                else:
                    stage = "REM"
                seg = arr[e*epoch_len:(e+1)*epoch_len]
                if len(seg) != epoch_len:
                    continue
                cands[stage].append((int(e), seg))
            sampled: Dict[str, List[Tuple[int, np.ndarray]]] = {}
            for s, items in cands.items():
                if not items:
                    continue
                if len(items) > MAX_EPOCHS_ROBUSTNESS:
                    sel = rng.choice(len(items), size=MAX_EPOCHS_ROBUSTNESS, replace=False)
                    items = [items[i] for i in sorted(sel)]
                sampled[s] = items
            out = []
            for s, items in sampled.items():
                for e, seg in items:
                    seg2 = seg[::2]
                    for m in M_GRID_WAKE:
                        for tau in TAU_GRID_WAKE:
                            X = time_delay_embedding(seg2, m, tau, MIN_EMBED_POINTS_ROBUSTNESS)
                            if X is None:
                                continue
                            X = (X - X.mean(0, keepdims=True))/(X.std(0, keepdims=True)+1e-8)
                            dgms = ripser(X, maxdim=MAXDIM)["dgms"]
                            h1 = dgm_summaries(dgms[1]) if len(dgms)>1 else {"count":0,"tot_pers":0.0,"max_pers":0.0}
                            out.append({
                                "subject": pair.subject,
                                "psg_tag": _tag(_prefix(pair.psg_path)),
                                "psg_file": pair.psg_path.name, "hyp_file": pair.hyp_path.name,
                                "channel": "EEG Fpz-Cz", "stage": s, "epoch_index": int(e),
                                "m": int(m), "tau": int(tau),
                                "H1_count": h1["count"], "H1_totpers": h1["tot_pers"],
                                "H1_maxpers": h1["max_pers"],
                            })
            return out
        except Exception:
            return []
    items = list(zip(pairs, seeds))
    n.log(f"  · wake-subclass robustness: {len(items)} nights, n_jobs={n_jobs}")
    t0 = time.time()
    results = _parallel_bar(_one, items, n_jobs, "wake-subclass robustness")
    n.log(f"    done in {time.time()-t0:0.1f}s")
    rows = [x for r in results for x in r]
    df = pd.DataFrame(rows)
    _save(df, out_grid, n)
    if df.empty:
        return
    omni_rows, contrast_rows = [], []
    grid_groups = list(df.groupby(["channel","m","tau"]))
    pbar = _inner_bar(len(grid_groups) * 3, "wake-subclass mixed-LM (grid × metric)")
    _wsr_fail = [0]; _wsr_err = [""]
    for (channel, m, tau), d in grid_groups:
        d = d.copy()
        d["K0_tot"] = within_subject_z(d, "H1_totpers")
        d["K0_max"] = within_subject_z(d, "H1_maxpers")
        d["K0_cnt"] = within_subject_z(d, "H1_count")
        for y in ["K0_tot","K0_max","K0_cnt"]:
            ss = d.groupby(["subject","stage"], as_index=False)[y].mean()
            try:
                # Powell first (more robust); lbfgs as fallback. Original code used
                # lbfgs which hits singular-matrix errors on most cells at full N.
                try:
                    res, lr, df_diff, p_lr = fit_mixedlm_stage(ss, y, STAGES_WAKE, method="powell")
                except Exception:
                    res, lr, df_diff, p_lr = fit_mixedlm_stage(ss, y, STAGES_WAKE, method="lbfgs")
                omni_rows.append({"channel": channel, "m": int(m), "tau": int(tau),
                                  "metric": y, "LR": lr, "df": df_diff, "p": p_lr})
                pc = planned_contrasts(res, y, PLANNED_WAKE, STAGES_WAKE)
                pc["channel"] = channel; pc["m"] = int(m); pc["tau"] = int(tau)
                contrast_rows.append(pc)
            except Exception as ex:
                _wsr_fail[0] += 1
                _wsr_err[0] = f"{type(ex).__name__}: {ex}"
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()
    if _wsr_fail[0]:
        n.log(f"  ! {_wsr_fail[0]} mixed-LM fits failed silently (e.g. '{_wsr_err[0]}')")
        n.log("    (this is common with small N; full run on 197 nights should converge)")
    _save(pd.DataFrame(omni_rows), out_omni, n)
    _save(pd.concat(contrast_rows, ignore_index=True) if contrast_rows else pd.DataFrame(), out_pc, n)


def step_main_robustness_mixedlm(force, n: Narrator):
    in_path = OUT_DIR / "tda_robustness_grid_epochs.csv"
    if not _exists(in_path):
        n.log("  ! prerequisite missing or empty: tda_robustness_grid_epochs.csv")
        return
    out_omni = OUT_DIR / "tda_robustness_mixedlm_omnibus.csv"
    out_pc   = OUT_DIR / "tda_robustness_mixedlm_planned_contrasts.csv"
    if _exists(out_omni, out_pc) and not force:
        n.log("  ✓ skipping (outputs exist)")
        return
    df = _safe_read_csv(in_path)
    df = df[df["stage"].isin(STAGES_MAIN)].copy()
    omni_rows, contrast_rows = [], []
    grid_groups = list(df.groupby(["channel","m","tau"]))
    pbar = _inner_bar(len(grid_groups) * 3, "robustness mixed-LM (grid × metric)")
    _fail_count = [0]; _last_err = [""]
    for (channel, m, tau), d in grid_groups:
        d = d.copy()
        d["K0_tot"] = within_subject_z(d, "H1_totpers")
        d["K0_max"] = within_subject_z(d, "H1_maxpers")
        d["K0_cnt"] = within_subject_z(d, "H1_count")
        for y in ["K0_tot","K0_max","K0_cnt"]:
            ss = d.groupby(["subject","stage"], as_index=False)[y].mean()
            try:
                res, lr, df_diff, p_lr = fit_mixedlm_stage(ss, y, STAGES_MAIN, method="powell")
                omni_rows.append({"channel": channel, "m": int(m), "tau": int(tau),
                                  "metric": y, "LR": lr, "df": df_diff, "p": p_lr})
                pc = planned_contrasts(res, y, PLANNED_MAIN, STAGES_MAIN)
                pc["channel"] = channel; pc["m"] = int(m); pc["tau"] = int(tau)
                contrast_rows.append(pc)
            except Exception as ex:
                _fail_count[0] += 1
                _last_err[0] = f"{type(ex).__name__}: {ex}"
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()
    if _fail_count[0]:
        n.log(f"  ! {_fail_count[0]} mixed-LM fits failed silently (e.g. '{_last_err[0]}')")
    _save(pd.DataFrame(omni_rows), out_omni, n)
    _save(pd.concat(contrast_rows, ignore_index=True) if contrast_rows else pd.DataFrame(), out_pc, n)


def step_baseline_mixedlm(force, n: Narrator):
    in_path = OUT_DIR / "baseline_epoch_features_all.csv"
    if not _exists(in_path):
        n.log("  ! prerequisite missing or empty: baseline_epoch_features_all.csv")
        return
    out_omni = OUT_DIR / "baseline_mixedlm_omnibus.csv"
    out_pc   = OUT_DIR / "baseline_mixedlm_planned_contrasts.csv"
    if _exists(out_omni, out_pc) and not force:
        n.log("  ✓ skipping (outputs exist)")
        return
    df = _safe_read_csv(in_path)
    if df.empty:
        n.log(f"  ! prerequisite {in_path.name} is empty (delete it and re-run baselines)")
        return
    omni, contr = run_mixedlm_analysis(df, BASELINE_METRICS, STAGES_MAIN, PLANNED_MAIN, method="powell")
    _save(omni, out_omni, n); _save(contr, out_pc, n)


def step_wake_subclass_mixedlm(force, n: Narrator):
    """Track 2 mixed-LM on the wake-subclass TDA features."""
    in_path = OUT_DIR / "tda_epoch_features_wake_subclasses.csv"
    if not _exists(in_path):
        n.log("  ! prerequisite missing or empty: tda_epoch_features_wake_subclasses.csv")
        return
    out_omni = OUT_DIR / "tda_wake_subclasses_mixedlm_omnibus.csv"
    out_pc   = OUT_DIR / "tda_wake_subclasses_mixedlm_planned_contrasts.csv"
    if _exists(out_omni, out_pc) and not force:
        n.log("  ✓ skipping (outputs exist)")
        return
    df = _safe_read_csv(in_path)
    df = df[df["stage"].isin(STAGES_WAKE)].copy()
    if df.empty:
        return
    if "channel" in df.columns:
        omni_rows, contrast_rows = [], []
        for ch, d in df.groupby("channel"):
            for src, k0 in [("H1_totpers","K0_tot"),("H1_maxpers","K0_max"),("H1_count","K0_cnt")]:
                d2 = d.copy(); d2[k0] = within_subject_z(d2, src)
                ss = d2.groupby(["subject","stage"], as_index=False)[k0].mean()
                try:
                    res, lr, df_diff, p_lr = fit_mixedlm_stage(ss, k0, STAGES_WAKE, method="powell")
                except Exception:
                    continue
                omni_rows.append({"channel": ch, "metric": k0, "LR": lr, "df": df_diff, "p": p_lr})
                pc = planned_contrasts(res, k0, PLANNED_WAKE, STAGES_WAKE)
                pc["channel"] = ch
                contrast_rows.append(pc)
        _save(pd.DataFrame(omni_rows), out_omni, n)
        _save(pd.concat(contrast_rows, ignore_index=True) if contrast_rows else pd.DataFrame(), out_pc, n)


def step_baseline_wake_mixedlm(force, n: Narrator):
    out_omni = OUT_DIR / "baseline_wake_subclasses_mixedlm_omnibus.csv"
    out_pc   = OUT_DIR / "baseline_wake_subclasses_mixedlm_planned_contrasts.csv"
    if _exists(out_omni, out_pc) and not force:
        n.log("  ✓ skipping (outputs exist)")
        return
    base = OUT_DIR / "baseline_epoch_features_all.csv"
    wake = OUT_DIR / "wake_epoch_subclasses.csv"
    if not (base.exists() and wake.exists()):
        return
    bdf = _safe_read_csv(base); wdf = _safe_read_csv(wake)
    if bdf.empty or wdf.empty:
        n.log(f"  ! prerequisite empty: baseline={bdf.empty}, wake={wdf.empty}")
        return
    keys = ["subject","psg_file","hyp_file","epoch_index"]
    merged = bdf.merge(wdf[keys+["wake_subclass"]], on=keys, how="left")
    merged.loc[merged["stage"]=="W", "stage"] = merged.loc[merged["stage"]=="W", "wake_subclass"]
    merged = merged[merged["stage"].isin(STAGES_WAKE)].copy()
    omni, contr = run_mixedlm_analysis(merged, BASELINE_METRICS, STAGES_WAKE, PLANNED_WAKE, method="powell")
    _save(omni, out_omni, n); _save(contr, out_pc, n)


def _track3_mixedlm(in_csv: Path, prefix: str, force: bool, n: Narrator):
    out_omni = OUT_DIR / f"{prefix}_wake_mixedlm_omnibus.csv"
    out_pc   = OUT_DIR / f"{prefix}_wake_mixedlm_planned_contrasts.csv"
    if _exists(out_omni, out_pc) and not force:
        n.log(f"  ✓ skipping {prefix}_wake_mixedlm_*.csv (exist)")
        return
    if not _exists(in_csv):
        n.log(f"  ! prerequisite missing or empty: {in_csv.name}")
        return
    df = _safe_read_csv(in_csv)
    if df.empty:
        n.log(f"  ! prerequisite empty: {in_csv.name}")
        return
    df = df[df["stage"].isin(STAGES_WAKE)].copy()
    if df.empty:
        return
    metrics = ["H0_count","H0_totpers","H0_maxpers","H1_count","H1_totpers","H1_maxpers"]
    omni_rows, contrast_rows = [], []
    for y in metrics:
        ss = df.groupby(["subject","stage"], as_index=False)[y].mean()
        n_rows = len(ss); n_subj = ss["subject"].nunique()
        # Powell first (more robust); lbfgs as fallback. Original code used
        # lbfgs which fails on most metrics at full N.
        res = lr = df_diff = p_lr = None
        for method in ("powell", "lbfgs"):
            try:
                res, lr, df_diff, p_lr = fit_mixedlm_stage(ss, y, STAGES_WAKE, method=method)
                break
            except Exception:
                continue
        if res is None:
            continue
        omni_rows.append({"metric": y, "LR": lr, "df": df_diff, "p": p_lr,
                          "n_rows": n_rows, "n_subjects": n_subj})
        pc = planned_contrasts(res, y, PLANNED_WAKE, STAGES_WAKE)
        pc.insert(0, "analysis", prefix)
        contrast_rows.append(pc)
    _save(pd.DataFrame(omni_rows), out_omni, n)
    _save(pd.concat(contrast_rows, ignore_index=True) if contrast_rows else pd.DataFrame(), out_pc, n)


def step_track3_mixedlm(force, n: Narrator):
    _track3_mixedlm(OUT_DIR/"tda_epoch_features_wake_raw.csv",       "raw",       force, n)
    _track3_mixedlm(OUT_DIR/"tda_epoch_features_wake_corrected.csv", "corrected", force, n)
    _track3_mixedlm(OUT_DIR/"tda_epoch_features_eog.csv",            "eog",       force, n)


def step_incremental_glm(force, n: Narrator):
    out_fits = OUT_DIR / "incremental_k0_vs_bandpower_model_fits.csv"
    out_lr   = OUT_DIR / "incremental_k0_vs_bandpower_lr_tests.csv"
    out_co   = OUT_DIR / "incremental_k0_vs_bandpower_coefficients.csv"
    if _exists(out_fits, out_lr, out_co) and not force:
        n.log("  ✓ skipping (outputs exist)")
        return
    tda = OUT_DIR / "tda_epoch_features_wake_subclasses.csv"
    base = OUT_DIR / "baseline_epoch_features_all.csv"
    wake = OUT_DIR / "wake_epoch_subclasses.csv"
    if not (tda.exists() and base.exists() and wake.exists()):
        n.log("  ! missing prerequisites for incremental GLM")
        return
    t = _safe_read_csv(tda); b = _safe_read_csv(base)
    if t.empty or b.empty:
        n.log(f"  ! prerequisite empty: tda={t.empty}, baseline={b.empty}")
        return
    keys = ["subject","psg_file","hyp_file","epoch_index"]
    for d in (t, b):
        d["epoch_index"] = d["epoch_index"].astype(int)
    # tda_epoch_features_wake_subclasses.csv already has stage =
    # REM / W_quiet / W_active_ocular (no W_bad), so no wake-table merge is needed.
    t_small = t[keys+["stage","H1_totpers"]]
    b_small = b[keys+BANDPOWER_COLS]
    m = t_small.merge(b_small, on=keys, how="inner")
    if m.empty:
        n.log(f"  ! merge of TDA wake-subclasses + baselines produced 0 rows; "
              f"check that prerequisites cover overlapping epochs")
        return
    m["K0_tot"] = m.groupby("subject")["H1_totpers"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0)
    n.log(f"  · merged: {len(m):,} epochs, {m['subject'].nunique()} subjects, "
          f"stages: {sorted(m['stage'].unique())}")

    fits, lrs, coefs = [], [], []
    for cname, sa, sb in [("REM_vs_W_quiet","REM","W_quiet"),
                          ("REM_vs_W_active_ocular","REM","W_active_ocular")]:
        d = m[m["stage"].isin([sa, sb])].dropna(subset=["K0_tot"]+BANDPOWER_COLS).copy()
        d["is_rem"] = (d["stage"] == sa).astype(int)
        ok_subj = d.groupby("subject")["is_rem"].nunique()
        keep = ok_subj[ok_subj >= 2].index
        d_filt = d[d["subject"].isin(keep)]
        n.log(f"  · {cname}: {len(d):,} rows pre-filter, "
              f"{len(d_filt):,} rows post-filter, "
              f"{len(keep)} subjects with both classes "
              f"(needed both REM and {sb})")
        d = d_filt
        if d.empty:
            n.log(f"    ! skipping {cname}: no subjects with both classes")
            continue

        models = {
            "A_bandpower_plus_subject":          "is_rem ~ " + " + ".join(BANDPOWER_COLS) + " + C(subject)",
            "B_bandpower_plus_K0_plus_subject":  "is_rem ~ " + " + ".join(BANDPOWER_COLS) + " + K0_tot + C(subject)",
            "C_K0_plus_subject":                 "is_rem ~ K0_tot + C(subject)",
            "D_K0_plus_bandpower_plus_subject":  "is_rem ~ K0_tot + " + " + ".join(BANDPOWER_COLS) + " + C(subject)",
        }
        results = {}
        n_glm_fail = 0; last_glm_err = ""
        for name, formula in models.items():
            try:
                res = smf.glm(formula, d, family=sm.families.Binomial()).fit(disp=False)
                results[name] = res
                fits.append({
                    "contrast": cname, "model": name, "formula": formula,
                    "n_epochs": int(len(d)), "n_subjects": int(d["subject"].nunique()),
                    "logLik": float(res.llf), "AIC": float(res.aic), "BIC": float(res.bic),
                    "df_model": int(res.df_model),
                })
            except Exception as ex:
                n_glm_fail += 1
                last_glm_err = f"{type(ex).__name__}: {ex}"
        if n_glm_fail:
            n.log(f"    ! {n_glm_fail}/4 GLMs failed for {cname} (e.g. '{last_glm_err}'); "
                  f"often perfect separation with C(subject) on small N")
        if results:
            n.log(f"    ✓ {len(results)}/4 GLMs converged for {cname}")

        def _lr(small, large, comp_name):
            if small not in results or large not in results:
                return
            rs = results[small]; rl = results[large]
            lr = 2*(rl.llf - rs.llf); df_d = int(rl.df_model - rs.df_model)
            p = float(stats.chi2.sf(lr, df_d)) if df_d > 0 else np.nan
            lrs.append({
                "contrast": cname, "comparison": comp_name,
                "LR": float(lr), "df": df_d, "p": p,
                "AIC_small": float(rs.aic), "AIC_large": float(rl.aic),
                "BIC_small": float(rs.bic), "BIC_large": float(rl.bic),
                "delta_AIC_large_minus_small": float(rl.aic - rs.aic),
                "delta_BIC_large_minus_small": float(rl.bic - rs.bic),
            })
        _lr("A_bandpower_plus_subject", "B_bandpower_plus_K0_plus_subject", "A_vs_B_add_K0_to_bandpower")
        _lr("C_K0_plus_subject",        "D_K0_plus_bandpower_plus_subject", "C_vs_D_add_bandpower_to_K0")

        for name, res in results.items():
            for term in res.params.index:
                if term.startswith("C(subject)") or term == "Intercept":
                    continue
                est = float(res.params[term]); se = float(res.bse[term])
                z = est/se if se>0 else np.nan
                p = float(2*stats.norm.sf(abs(z))) if se>0 else np.nan
                lo, hi = est - 1.96*se, est + 1.96*se
                coefs.append({
                    "contrast": cname, "model": name, "term": term,
                    "estimate": est, "SE": se, "z": z, "p": p,
                    "CI95_low": lo, "CI95_high": hi,
                    "odds_ratio": float(np.exp(est)),
                    "OR_CI95_low": float(np.exp(lo)), "OR_CI95_high": float(np.exp(hi)),
                })

    _save(pd.DataFrame(fits), out_fits, n)
    _save(pd.DataFrame(lrs),  out_lr,  n)
    _save(pd.DataFrame(coefs), out_co, n)


# ──────────────────────────────────────────────────────────────────────────────
# Review / comparison / supplementary tables
# ──────────────────────────────────────────────────────────────────────────────

def _stage_means_table(df: pd.DataFrame, metrics: Sequence[str], stages: Sequence[str],
                       has_channel: bool) -> pd.DataFrame:
    rows = []
    grouper = ["channel"] if has_channel and "channel" in df.columns else []
    for keys, d in (df.groupby(grouper) if grouper else [((None,), df)]):
        ch = keys if grouper else None
        for met in metrics:
            if met not in d.columns:
                continue
            for s in stages:
                sub = d[d["stage"] == s]
                if sub.empty:
                    continue
                ss = sub.groupby("subject")[met].mean()
                rows.append({
                    **({"channel": ch} if ch is not None else {}),
                    "metric": met, "stage": s,
                    "mean": float(ss.mean()), "std": float(ss.std(ddof=1)),
                    "n_subjects": int(ss.shape[0]),
                })
    return pd.DataFrame(rows)

def _paired_dz_table(df: pd.DataFrame, metrics: Sequence[str],
                     contrasts: Sequence[Tuple[str,str,str]],
                     has_channel: bool) -> pd.DataFrame:
    """contrasts: (label, a, b) where label is 'A - B'."""
    rows = []
    grouper = ["channel"] if has_channel and "channel" in df.columns else []
    for keys, d in (df.groupby(grouper) if grouper else [((None,), df)]):
        ch = keys if grouper else None
        for met in metrics:
            if met not in d.columns:
                continue
            piv = d.groupby(["subject","stage"])[met].mean().unstack("stage")
            for label, a, b in contrasts:
                if a not in piv.columns or b not in piv.columns:
                    continue
                diff = (piv[a] - piv[b]).dropna()
                if diff.empty:
                    continue
                m_diff = float(diff.mean()); s_diff = float(diff.std(ddof=1))
                dz = m_diff / s_diff if s_diff > 0 else np.nan
                rows.append({
                    **({"channel": ch} if ch is not None else {}),
                    "metric": met, "contrast": label,
                    "n_subjects": int(diff.shape[0]),
                    "mean_diff": m_diff, "sd_diff": s_diff, "dz": dz,
                })
    return pd.DataFrame(rows)

def step_review_wake_subclass(force, n: Narrator):
    out_means = OUT_DIR / "review_wake_subclass_stage_means.csv"
    out_dz    = OUT_DIR / "review_wake_subclass_paired_effect_sizes.csv"
    out_sum   = OUT_DIR / "review_wake_subclass_summary_table.csv"
    if _exists(out_means, out_dz, out_sum) and not force:
        n.log("  ✓ skipping (outputs exist)")
        return
    epoch = OUT_DIR / "tda_epoch_features_wake_subclasses.csv"
    omni  = OUT_DIR / "tda_wake_subclasses_mixedlm_omnibus.csv"
    contr = OUT_DIR / "tda_wake_subclasses_mixedlm_planned_contrasts.csv"
    if not (epoch.exists() and omni.exists() and contr.exists()):
        return
    df = _safe_read_csv(epoch)
    df = df[df["stage"].isin(STAGES_WAKE)].copy()
    if "channel" in df.columns:
        for src, k0 in [("H1_totpers","K0_tot"),("H1_maxpers","K0_max"),("H1_count","K0_cnt")]:
            df[k0] = df.groupby(["channel","subject"])[src].transform(
                lambda x: (x - x.mean())/x.std(ddof=1) if x.std(ddof=1)>0 else 0.0)
    means = _stage_means_table(df, ["K0_tot","K0_max","K0_cnt"], STAGES_WAKE, True)
    dz = _paired_dz_table(df, ["K0_tot","K0_max","K0_cnt"],
                          [(f"{a} - {b}", a, b) for a,b in PLANNED_WAKE], True)
    _save(means, out_means, n); _save(dz, out_dz, n)

    o = _safe_read_csv(omni); c = _safe_read_csv(contr)
    join = ["channel","metric"] if "channel" in o.columns and "channel" in c.columns else ["metric"]
    merged = c.merge(o, on=join, how="left", suffixes=("","_omni"))
    merged = merged.rename(columns={"p_omni": "p_omni"})
    if "p" in merged.columns and "LR" in merged.columns:
        pass
    final = merged.merge(dz, on=join+["contrast"], how="left")
    _save(final, out_sum, n)

def step_review_baseline_wake_subclass(force, n: Narrator):
    out_means = OUT_DIR / "review_baseline_wake_subclass_stage_means.csv"
    out_dz    = OUT_DIR / "review_baseline_wake_subclass_paired_effect_sizes.csv"
    out_sum   = OUT_DIR / "review_baseline_wake_subclass_summary_table.csv"
    if _exists(out_means, out_dz, out_sum) and not force:
        n.log("  ✓ skipping (outputs exist)")
        return
    base = OUT_DIR / "baseline_epoch_features_all.csv"
    wake = OUT_DIR / "wake_epoch_subclasses.csv"
    omni = OUT_DIR / "baseline_wake_subclasses_mixedlm_omnibus.csv"
    contr = OUT_DIR / "baseline_wake_subclasses_mixedlm_planned_contrasts.csv"
    if not (base.exists() and wake.exists() and omni.exists() and contr.exists()):
        return
    bdf = _safe_read_csv(base); wdf = _safe_read_csv(wake)
    keys = ["subject","psg_file","hyp_file","epoch_index"]
    m = bdf.merge(wdf[keys+["wake_subclass"]], on=keys, how="left")
    m.loc[m["stage"]=="W","stage"] = m.loc[m["stage"]=="W","wake_subclass"]
    m = m[m["stage"].isin(STAGES_WAKE)].copy()
    metrics = [met for met in BASELINE_METRICS if met in m.columns]
    has_ch = "channel" in m.columns
    means = _stage_means_table(m, metrics, STAGES_WAKE, has_ch)
    dz = _paired_dz_table(m, metrics, [(f"{a} - {b}", a, b) for a,b in PLANNED_WAKE], has_ch)
    _save(means, out_means, n); _save(dz, out_dz, n)
    o = _safe_read_csv(omni); c = _safe_read_csv(contr)
    join = ["metric"]
    merged = c.merge(o, on=join, how="left", suffixes=("","_omni"))
    final = merged.merge(dz, on=["metric","contrast"], how="left")
    _save(final, out_sum, n)

def step_review_incremental(force, n: Narrator):
    out_sum = OUT_DIR / "review_incremental_k0_vs_bandpower_summary.csv"
    out_k0  = OUT_DIR / "review_incremental_k0_vs_bandpower_k0_only.csv"
    out_bp  = OUT_DIR / "review_incremental_k0_vs_bandpower_bandpower_terms.csv"
    if _exists(out_sum, out_k0, out_bp) and not force:
        n.log("  ✓ skipping (outputs exist)")
        return
    fits = OUT_DIR / "incremental_k0_vs_bandpower_model_fits.csv"
    lr   = OUT_DIR / "incremental_k0_vs_bandpower_lr_tests.csv"
    co   = OUT_DIR / "incremental_k0_vs_bandpower_coefficients.csv"
    if not (fits.exists() and lr.exists() and co.exists()):
        return
    fdf = _safe_read_csv(fits); ldf = _safe_read_csv(lr); cdf = _safe_read_csv(co)
    if fdf.empty or ldf.empty or cdf.empty:
        n.log(f"  ! prerequisite empty (incremental GLM produced no rows; "
              f"often the case with small N — try --limit > 20 or run on full cohort)")
        return

    rows_sum, rows_k0, rows_bp = [], [], []
    for cname, g in fdf.groupby("contrast"):
        get = lambda mn: g[g["model"]==mn].iloc[0] if (g["model"]==mn).any() else None
        A, B, C, D = (get(x) for x in [
            "A_bandpower_plus_subject","B_bandpower_plus_K0_plus_subject",
            "C_K0_plus_subject","D_K0_plus_bandpower_plus_subject"])
        lr_AB = ldf[(ldf["contrast"]==cname)&(ldf["comparison"]=="A_vs_B_add_K0_to_bandpower")]
        lr_CD = ldf[(ldf["contrast"]==cname)&(ldf["comparison"]=="C_vs_D_add_bandpower_to_K0")]
        lr_AB = lr_AB.iloc[0] if len(lr_AB) else None
        lr_CD = lr_CD.iloc[0] if len(lr_CD) else None
        k0_in_B = cdf[(cdf["contrast"]==cname)&(cdf["model"]=="B_bandpower_plus_K0_plus_subject")&(cdf["term"]=="K0_tot")]
        k0_in_C = cdf[(cdf["contrast"]==cname)&(cdf["model"]=="C_K0_plus_subject")&(cdf["term"]=="K0_tot")]
        k0_in_B = k0_in_B.iloc[0] if len(k0_in_B) else None
        k0_in_C = k0_in_C.iloc[0] if len(k0_in_C) else None

        n_e = int(A["n_epochs"]) if A is not None else np.nan
        n_s = int(A["n_subjects"]) if A is not None else np.nan

        rows_sum.append({
            "contrast": cname, "n_epochs": n_e, "n_subjects": n_s,
            "AIC_A_bandpower":         float(A["AIC"]) if A is not None else np.nan,
            "AIC_B_bandpower_plus_K0": float(B["AIC"]) if B is not None else np.nan,
            "Delta_AIC_B_minus_A":     (float(B["AIC"])-float(A["AIC"])) if (A is not None and B is not None) else np.nan,
            "LR_add_K0_to_bandpower":  float(lr_AB["LR"]) if lr_AB is not None else np.nan,
            "LR_p_add_K0_to_bandpower":float(lr_AB["p"])  if lr_AB is not None else np.nan,
            "K0_beta_in_B":            float(k0_in_B["estimate"]) if k0_in_B is not None else np.nan,
            "K0_p_in_B":               float(k0_in_B["p"])        if k0_in_B is not None else np.nan,
            "K0_OR_in_B":              float(k0_in_B["odds_ratio"]) if k0_in_B is not None else np.nan,
            "AIC_C_K0":                float(C["AIC"]) if C is not None else np.nan,
            "AIC_D_K0_plus_bandpower": float(D["AIC"]) if D is not None else np.nan,
            "Delta_AIC_D_minus_C":     (float(D["AIC"])-float(C["AIC"])) if (C is not None and D is not None) else np.nan,
            "LR_add_bandpower_to_K0":  float(lr_CD["LR"]) if lr_CD is not None else np.nan,
            "LR_p_add_bandpower_to_K0":float(lr_CD["p"])  if lr_CD is not None else np.nan,
            "K0_beta_in_C":            float(k0_in_C["estimate"]) if k0_in_C is not None else np.nan,
            "K0_p_in_C":               float(k0_in_C["p"])        if k0_in_C is not None else np.nan,
            "K0_OR_in_C":              float(k0_in_C["odds_ratio"]) if k0_in_C is not None else np.nan,
        })
        if A is not None and B is not None and lr_AB is not None and k0_in_B is not None:
            rows_k0.append({
                "contrast": cname, "n_epochs": n_e, "n_subjects": n_s,
                "Model A AIC": float(A["AIC"]), "Model B AIC": float(B["AIC"]),
                "Delta AIC (B-A)": float(B["AIC"]-A["AIC"]),
                "Model A BIC": float(A["BIC"]), "Model B BIC": float(B["BIC"]),
                "Delta BIC (B-A)": float(B["BIC"]-A["BIC"]),
                "LR add K0": float(lr_AB["LR"]), "LR df": int(lr_AB["df"]), "LR p": float(lr_AB["p"]),
                "K0 beta": float(k0_in_B["estimate"]),
                "K0 beta 95% CI": f'[{float(k0_in_B["CI95_low"]):.3f}, {float(k0_in_B["CI95_high"]):.3f}]',
                "K0 p": float(k0_in_B["p"]),
                "K0 OR": float(k0_in_B["odds_ratio"]),
                "K0 OR 95% CI": f'[{float(k0_in_B["OR_CI95_low"]):.3f}, {float(k0_in_B["OR_CI95_high"]):.3f}]',
            })
        for term in ["K0_tot"] + BANDPOWER_COLS:
            tr = cdf[(cdf["contrast"]==cname)&(cdf["model"]=="B_bandpower_plus_K0_plus_subject")&(cdf["term"]==term)]
            if len(tr):
                tr = tr.iloc[0]
                rows_bp.append({
                    "contrast": cname, "term": term,
                    "estimate": float(tr["estimate"]), "SE": float(tr["SE"]),
                    "z": float(tr["z"]), "p": float(tr["p"]),
                    "beta 95% CI": f'[{float(tr["CI95_low"]):.3f}, {float(tr["CI95_high"]):.3f}]',
                    "odds_ratio": float(tr["odds_ratio"]),
                    "OR 95% CI": f'[{float(tr["OR_CI95_low"]):.3f}, {float(tr["OR_CI95_high"]):.3f}]',
                })
    _save(pd.DataFrame(rows_sum), out_sum, n)
    _save(pd.DataFrame(rows_k0),  out_k0,  n)
    _save(pd.DataFrame(rows_bp),  out_bp,  n)


def step_comparison_table(force, n: Narrator):
    out_long = OUT_DIR / "comparison_table_rem_vs_wake_subclasses_long.csv"
    out_wide = OUT_DIR / "comparison_table_rem_vs_wake_subclasses.csv"
    if _exists(out_long, out_wide) and not force:
        n.log("  ✓ skipping (outputs exist)")
        return
    tda_sum = OUT_DIR / "review_wake_subclass_summary_table.csv"
    base_sum = OUT_DIR / "review_baseline_wake_subclass_summary_table.csv"
    if not (tda_sum.exists() and base_sum.exists()):
        return
    t = _safe_read_csv(tda_sum); b = _safe_read_csv(base_sum)
    if t.empty and b.empty:
        n.log("  ! prerequisites empty — run review_wake_subclass and review_baseline_wake_subclass first")
        return
    if t.empty: t = pd.DataFrame()
    if b.empty: b = pd.DataFrame()
    if not t.empty: t["source"] = "TDA"
    if not b.empty: b["source"] = "Baseline"
    keep = ["REM - W_quiet", "REM - W_active_ocular"]
    df = pd.concat([t, b], ignore_index=True)
    df = df[df["contrast"].isin(keep)].copy()

    labels = {
        "K0_tot":"Recurrence (K0_tot)", "K0_max":"Recurrence (K0_max)", "K0_cnt":"Recurrence (K0_cnt)",
        "log_delta":"Delta power (log)", "log_theta":"Theta power (log)",
        "log_alpha":"Alpha power (log)", "log_sigma":"Sigma power (log)", "log_beta":"Beta power (log)",
        "spec_entropy":"Spectral entropy", "perm_entropy":"Permutation entropy",
        "lz_complexity":"Lempel-Ziv complexity",
    }
    primary = {**{k:"Within-subject z(H1)" for k in ["K0_tot","K0_max","K0_cnt"]},
               **{k:"log power" for k in ["log_delta","log_theta","log_alpha","log_sigma","log_beta"]},
               "spec_entropy":"Shannon entropy of PSD",
               "perm_entropy":"Permutation entropy",
               "lz_complexity":"LZ76 (binary)"}
    order = ["K0_tot","spec_entropy","perm_entropy","lz_complexity",
             "log_delta","log_theta","log_alpha","log_beta","log_sigma","K0_max","K0_cnt"]
    df["row_order"] = df["metric"].apply(lambda m: order.index(m) if m in order else len(order))
    df["Metric"] = df["metric"].map(labels).fillna(df["metric"])
    df["Primary quantity"] = df["metric"].map(primary).fillna("")
    def fmt_p(p):
        try:
            p = float(p)
            return "<0.001" if p < 0.001 else f"{p:.3f}"
        except Exception:
            return ""
    df["Est. (95% CI)"] = df.apply(
        lambda r: f"{r['estimate']:.2f} [{r['CI95_low']:.2f}, {r['CI95_high']:.2f}]" if pd.notna(r.get("estimate")) else "", axis=1)
    df["dz_fmt"] = df["dz"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    df["Holm-p"] = df["p_holm"].apply(fmt_p)
    long_cols = ["row_order","source","metric","Metric","Primary quantity","contrast",
                 "Est. (95% CI)","dz_fmt","Holm-p","estimate","CI95_low","CI95_high","p_holm"]
    long_cols = [c for c in long_cols if c in df.columns]
    long_df = df[long_cols].sort_values(["row_order","source","metric","contrast"])
    _save(long_df, out_long, n)
    pivot_rows = []
    for met, g in df.groupby("metric"):
        row = {"Metric": labels.get(met, met),
               "Primary quantity": primary.get(met, ""),
               "row_order": order.index(met) if met in order else len(order)}
        for c in keep:
            sub = g[g["contrast"] == c]
            if len(sub):
                r = sub.iloc[0]
                row[f"{c} Est. (95% CI)"] = r["Est. (95% CI)"]
                row[f"{c} dz"] = r["dz_fmt"]
                row[f"{c} Holm-p"] = r["Holm-p"]
        pivot_rows.append(row)
    pivot = pd.DataFrame(pivot_rows).sort_values("row_order").drop(columns=["row_order"])
    _save(pivot, out_wide, n)


def step_supplementary_table(force, n: Narrator):
    out_main = OUT_DIR / "supplementary_table_wake_subclass_and_incremental_results.csv"
    out_long = OUT_DIR / "supplementary_table_wake_subclass_and_incremental_results_long.csv"
    if _exists(out_main, out_long) and not force:
        n.log("  ✓ skipping (outputs exist)")
        return
    wide = OUT_DIR / "comparison_table_rem_vs_wake_subclasses.csv"
    incr = OUT_DIR / "review_incremental_k0_vs_bandpower_k0_only.csv"
    if not (wide.exists() and incr.exists()):
        return
    w = _safe_read_csv(wide); i = _safe_read_csv(incr)
    if w.empty and i.empty:
        n.log("  ! both prerequisites empty — run comparison_table and review_incremental first")
        return
    if not w.empty: w.insert(0, "Section", "Wake-subclass contrasts")
    if not i.empty: i.insert(0, "Section", "Incremental K0 beyond band power")
    parts = [d for d in (w, i) if not d.empty]
    out = pd.concat(parts, ignore_index=True)
    _save(out, out_main, n)
    long_wide = OUT_DIR / "comparison_table_rem_vs_wake_subclasses_long.csv"
    if long_wide.exists():
        lw = _safe_read_csv(long_wide)
        lw.insert(0, "Section", "Wake-subclass contrasts")
        i_long = i.copy()
        i_long.insert(0, "row_order", 999)
        merged = pd.concat([lw, i_long], ignore_index=True)
        _save(merged, out_long, n)



# ──────────────────────────────────────────────────────────────────────────────
# Revision-round additions
# ──────────────────────────────────────────────────────────────────────────────
# The following steps were added to address reviewer comments at major-revision:
#   - all-pairwise stage contrasts + monotonicity (Editor #2)
#   - LOSO classification (Editor #3, Reviewer #3 #6)
#   - subsampling stability (Editor #5)
#   - cohort replication SC vs ST (Reviewer #1 #7)
#   - bootstrap CIs on contrasts (Reviewer #1 #1, #5)
#   - embedding diagnostics: AMI / FNN (Reviewer #3 #4)
#   - Pz-Oz multi-channel (Editor #4 / Reviewer #3 #2)
#   - preprocessing sensitivity (Reviewer #3 #3)
#   - statistical assumption diagnostics (Reviewer #1 #5)
# All steps reuse the same RNG_SEED, embedding params, mixed-LM spec, and Holm
# correction as the headline pipeline.

# ----- (a) all-pairwise contrasts + monotonicity ----------------------------

def step_all_pairwise(force, n: "Narrator"):
    """All 10 pairwise stage contrasts on K0 + per-stage descriptives + monotonicity."""
    out_descrip = OUT_DIR / "stage_descriptives_all.csv"
    out_pairs = OUT_DIR / "stage_all_pairwise_contrasts.csv"
    out_mono = OUT_DIR / "stage_monotonicity.csv"
    if _exists(out_descrip, out_pairs, out_mono) and not force:
        n.log("  ✓ skipping (outputs exist)")
        return
    in_path = OUT_DIR / "tda_epoch_features_all.csv"
    if not _exists(in_path):
        n.log("  ! prerequisite missing or empty: tda_epoch_features_all.csv")
        return
    df = _safe_read_csv(in_path)
    if df.empty:
        n.log(f"  ! prerequisite empty: tda_epoch_features_all.csv")
        return
    df = df[df["stage"].isin(STAGES_MAIN)].copy()
    if df.empty:
        return
    df["K0_tot"] = within_subject_z(df, "H1_totpers")

    rows = []
    for stage in STAGES_MAIN:
        sm = df.loc[df["stage"] == stage].groupby("subject")["K0_tot"].mean()
        if sm.empty:
            continue
        rows.append({
            "stage": stage, "n_subjects": int(len(sm)),
            "mean_K0": float(sm.mean()), "sd_K0": float(sm.std(ddof=1)),
            "median_K0": float(sm.median()),
            "iqr_low": float(sm.quantile(0.25)),
            "iqr_high": float(sm.quantile(0.75)),
        })
    _save(pd.DataFrame(rows), out_descrip, n)

    ss = df.groupby(["subject", "stage"], as_index=False)["K0_tot"].mean()
    try:
        res, lr, df_diff, p_lr = fit_mixedlm_stage(ss, "K0_tot", STAGES_MAIN, method="powell")
    except Exception as ex:
        n.log(f"  ! all-pairwise fit failed: {ex}")
        return
    pairs = [(a, b) for i, a in enumerate(STAGES_MAIN) for b in STAGES_MAIN[i + 1:]]
    pc = planned_contrasts(res, "K0_tot", pairs, STAGES_MAIN)
    _save(pc, out_pairs, n)

    # Monotonicity: per-subject Spearman of K0 across stage rank
    rank = {s: i for i, s in enumerate(STAGES_MAIN)}
    rhos = []
    for subj, sg in df.groupby("subject"):
        sm = sg.groupby("stage")["K0_tot"].mean().reindex(STAGES_MAIN).dropna()
        if len(sm) < 3:
            continue
        rho, _ = stats.spearmanr([rank[s] for s in sm.index], sm.values)
        if not np.isnan(rho):
            rhos.append(float(rho))
    if rhos:
        arr = np.array(rhos)
        t = stats.ttest_1samp(arr, 0.0) if len(arr) > 1 else None
        mono = pd.DataFrame([{
            "n_subjects": int(len(arr)),
            "mean_spearman_rho": float(arr.mean()),
            "sd_spearman_rho": float(arr.std(ddof=1)),
            "t_stat_vs_zero": float(t.statistic) if t is not None else np.nan,
            "p_t_test": float(t.pvalue) if t is not None else np.nan,
        }])
    else:
        mono = pd.DataFrame()
    _save(mono, out_mono, n)


# ----- (b) cohort replication (SC vs ST) ------------------------------------

def step_cohort_replication(force, n: "Narrator"):
    """Refit the headline contrasts on Sleep-Cassette vs Sleep-Telemetry subsets."""
    out = OUT_DIR / "cohort_replication_contrasts.csv"
    if _exists(out) and not force:
        n.log("  ✓ skipping (output exists)")
        return
    tda = OUT_DIR / "tda_epoch_features_all.csv"
    base = OUT_DIR / "baseline_epoch_features_all.csv"
    if not _exists(tda):
        n.log("  ! prerequisite missing or empty: tda_epoch_features_all.csv")
        return
    df = _safe_read_csv(tda)
    if df.empty:
        n.log("  ! prerequisite empty: tda_epoch_features_all.csv")
        return
    if df.empty:
        n.log(f"  ! prerequisite empty: tda_epoch_features_all.csv")
        return
    df = df[df["stage"].isin(STAGES_MAIN)].copy()
    df["cohort"] = df["psg_file"].str[:2].map({"SC": "Cassette", "ST": "Telemetry"})

    parts = []
    for cohort, sub in df.groupby("cohort"):
        if sub["subject"].nunique() < 5:
            continue
        sub = sub.copy()
        sub["K0_tot"] = within_subject_z(sub, "H1_totpers")
        ss = sub.groupby(["subject", "stage"], as_index=False)["K0_tot"].mean()
        try:
            res, lr, df_diff, p_lr = fit_mixedlm_stage(ss, "K0_tot", STAGES_MAIN, method="powell")
            pc = planned_contrasts(res, "K0_tot", PLANNED_MAIN, STAGES_MAIN)
            pc["cohort"] = cohort
            pc["source"] = "TDA_K0_tot"
            pc["n_subjects"] = int(ss["subject"].nunique())
            pc["lr_omnibus"] = float(lr); pc["p_omnibus"] = float(p_lr)
            parts.append(pc)
        except Exception:
            continue

    if base.exists():
        bdf = _safe_read_csv(base)
        bdf = bdf[bdf["stage"].isin(STAGES_MAIN)].copy()
        bdf["cohort"] = bdf["psg_file"].str[:2].map({"SC": "Cassette", "ST": "Telemetry"})
        for cohort, sub in bdf.groupby("cohort"):
            if sub["subject"].nunique() < 5:
                continue
            for met in BASELINE_METRICS:
                if met not in sub.columns:
                    continue
                ss = sub.groupby(["subject", "stage"], as_index=False)[met].mean()
                try:
                    res, lr, df_diff, p_lr = fit_mixedlm_stage(ss, met, STAGES_MAIN, method="powell")
                    pc = planned_contrasts(res, met, PLANNED_MAIN, STAGES_MAIN)
                    pc["cohort"] = cohort
                    pc["source"] = "baseline"
                    pc["n_subjects"] = int(ss["subject"].nunique())
                    pc["lr_omnibus"] = float(lr); pc["p_omnibus"] = float(p_lr)
                    parts.append(pc)
                except Exception:
                    continue

    out_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    _save(out_df, out, n)


# ----- (c) subsampling stability --------------------------------------------

def step_subsampling_stability(force, n: "Narrator", n_replicates: int = 10):
    """Post-hoc resampling of the existing TDA epochs at caps {5,10,15,20,25,30}."""
    out = OUT_DIR / "subsampling_stability_contrasts.csv"
    if _exists(out) and not force:
        n.log("  ✓ skipping (output exists)")
        return
    tda = OUT_DIR / "tda_epoch_features_all.csv"
    if not _exists(tda):
        n.log("  ! prerequisite missing or empty: tda_epoch_features_all.csv")
        return
    df = _safe_read_csv(tda)
    if df.empty:
        n.log(f"  ! prerequisite empty: tda_epoch_features_all.csv")
        return
    df = df[df["stage"].isin(STAGES_MAIN)].copy()
    caps = [5, 10, 15, 20, 25, 30]
    rng = np.random.default_rng(RNG_SEED)
    parts = []

    pbar = _inner_bar(len(caps) * n_replicates, "subsampling stability")
    for cap in caps:
        for r in range(n_replicates):
            seed = int(rng.integers(0, 2**31 - 1))
            # Per-(subject, psg, stage) capped subsample. Use shuffle + head()
            # rather than groupby.apply() because pandas 2.2's apply changed the
            # default include_groups behaviour and started dropping group keys.
            sub = (
                df.sample(frac=1.0, random_state=seed)
                  .groupby(["subject", "psg_file", "stage"], as_index=False, group_keys=False)
                  .head(cap)
                  .reset_index(drop=True)
                  .copy()
            )
            sub["K0_tot"] = within_subject_z(sub, "H1_totpers")
            ss = sub.groupby(["subject", "stage"], as_index=False)["K0_tot"].mean()
            try:
                res, _, _, _ = fit_mixedlm_stage(ss, "K0_tot", STAGES_MAIN, method="powell")
                pc = planned_contrasts(res, "K0_tot", PLANNED_MAIN, STAGES_MAIN)
                pc["cap"] = int(cap); pc["replicate"] = int(r)
                pc["n_subjects"] = int(ss["subject"].nunique())
                parts.append(pc)
            except Exception:
                pass
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()
    out_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    _save(out_df, out, n)


# ----- (d) bootstrap CIs ----------------------------------------------------

def step_bootstrap_contrasts(force, n: "Narrator", n_boot: int = 1000):
    """Subject-level non-parametric bootstrap for the headline planned contrasts."""
    out = OUT_DIR / "bootstrap_contrasts.csv"
    if _exists(out) and not force:
        n.log("  ✓ skipping (output exists)")
        return
    tda = OUT_DIR / "tda_epoch_features_all.csv"
    if not _exists(tda):
        n.log("  ! prerequisite missing or empty: tda_epoch_features_all.csv")
        return
    df = _safe_read_csv(tda)
    if df.empty:
        n.log(f"  ! prerequisite empty: tda_epoch_features_all.csv")
        return
    df = df[df["stage"].isin(STAGES_MAIN)].copy()
    df["K0_tot"] = within_subject_z(df, "H1_totpers")
    subjects = np.array(sorted(df["subject"].unique()))

    # Point estimates
    ss = df.groupby(["subject", "stage"], as_index=False)["K0_tot"].mean()
    res, _, _, _ = fit_mixedlm_stage(ss, "K0_tot", STAGES_MAIN, method="powell")
    point = planned_contrasts(res, "K0_tot", PLANNED_MAIN, STAGES_MAIN)
    point_dict = dict(zip(point["contrast"], point["estimate"]))

    rng = np.random.default_rng(RNG_SEED)
    boots: Dict[str, List[float]] = {f"{a} - {b}": [] for a, b in PLANNED_MAIN}
    n_failed = 0
    n.log(f"  · running {n_boot} bootstrap replicates...")
    pbar = _inner_bar(n_boot, "bootstrap (subject resamples)")
    for b in range(n_boot):
        sample = rng.choice(subjects, size=len(subjects), replace=True)
        rows = []
        for i, s in enumerate(sample):
            sub = df.loc[df["subject"] == s].copy()
            sub["subject"] = f"{s}_b{i}"
            rows.append(sub)
        bdf = pd.concat(rows, ignore_index=True)
        ss = bdf.groupby(["subject", "stage"], as_index=False)["K0_tot"].mean()
        try:
            res_b, _, _, _ = fit_mixedlm_stage(ss, "K0_tot", STAGES_MAIN, method="powell")
            pc_b = planned_contrasts(res_b, "K0_tot", PLANNED_MAIN, STAGES_MAIN)
            for _, row in pc_b.iterrows():
                boots[row["contrast"]].append(float(row["estimate"]))
        except Exception:
            n_failed += 1
        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()

    rows_out = []
    for contrast, vals in boots.items():
        arr = np.asarray(vals)
        if arr.size < 10:
            continue
        rows_out.append({
            "contrast": contrast,
            "point_estimate": float(point_dict.get(contrast, np.nan)),
            "boot_mean": float(arr.mean()),
            "boot_se": float(arr.std(ddof=1)),
            "boot_ci95_low": float(np.percentile(arr, 2.5)),
            "boot_ci95_high": float(np.percentile(arr, 97.5)),
            "n_boot_successful": int(arr.size),
            "n_boot_failed": int(n_failed),
        })
    _save(pd.DataFrame(rows_out), out, n)


# ----- (e) embedding diagnostics: AMI + FNN ---------------------------------

def avg_mutual_info(x: np.ndarray, max_tau: int = 20, n_bins: int = 16) -> np.ndarray:
    """Average mutual information AMI(τ) for τ = 1..max_tau."""
    n = len(x); x_min, x_max = float(x.min()), float(x.max())
    if x_max <= x_min:
        return np.full(max_tau, np.nan)
    bins = np.linspace(x_min, x_max + 1e-12, n_bins + 1)
    xb = np.digitize(x, bins) - 1
    xb = np.clip(xb, 0, n_bins - 1)
    out = np.empty(max_tau)
    for tau in range(1, max_tau + 1):
        a = xb[:n - tau]; b = xb[tau:]
        joint = np.zeros((n_bins, n_bins))
        np.add.at(joint, (a, b), 1.0)
        joint = joint / joint.sum()
        p_a = joint.sum(axis=1, keepdims=True)
        p_b = joint.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = joint / (p_a * p_b)
            mi = np.where(joint > 0, joint * np.log(ratio + 1e-300), 0.0).sum()
        out[tau - 1] = float(mi)
    return out

def false_nearest_neighbours(x: np.ndarray, max_m: int = 15, tau: int = 2,
                             R_tol: float = 15.0, A_tol: float = 2.0) -> np.ndarray:
    """Kennel-Brown-Abarbanel FNN fraction for m = 1..max_m at fixed tau."""
    from scipy.spatial import cKDTree
    sigma = float(np.std(x))
    out = np.empty(max_m)
    for m in range(1, max_m + 1):
        n = len(x) - m * tau
        if n <= 50:
            out[m - 1] = np.nan
            continue
        Xm = np.stack([x[i:i + n] for i in range(0, m * tau, tau)], axis=1)
        nv = x[m * tau:m * tau + n]
        tree = cKDTree(Xm)
        d, idx = tree.query(Xm, k=2)
        d_m = d[:, 1]; nn = idx[:, 1]
        d_m1 = np.sqrt(d_m**2 + (nv - nv[nn])**2)
        with np.errstate(divide="ignore", invalid="ignore"):
            num = np.sqrt(np.maximum(d_m1**2 - d_m**2, 0.0))
            crit1 = (num / np.where(d_m > 0, d_m, 1e-12)) > R_tol
            crit2 = (d_m1 / sigma) > A_tol
        out[m - 1] = float((crit1 | crit2).mean())
    return out


def _embedding_one(pair: "Pair") -> Optional[Dict]:
    try:
        x, sf, _ = _load_eeg(pair.psg_path)
        # 30-min middle segment, then within-epoch ::2 like main TDA
        L = int(30 * 60 * sf)
        start = max(0, (len(x) - L) // 2)
        seg = x[start:start + L][::2]
        ami = avg_mutual_info(seg, max_tau=20, n_bins=16)
        fnn = false_nearest_neighbours(seg, max_m=15, tau=EMBED_TAU)
        return {"subject": pair.subject, "psg_file": pair.psg_path.name,
                "ami": ami.tolist(), "fnn": fnn.tolist()}
    except Exception:
        return None


def step_embedding_diagnostics(pairs, n_jobs, force, n: "Narrator", n_subset: int = 30):
    out_ami = OUT_DIR / "embedding_ami.csv"
    out_fnn = OUT_DIR / "embedding_fnn.csv"
    out_summary = OUT_DIR / "embedding_diagnostics_summary.csv"
    if _exists(out_ami, out_fnn, out_summary) and not force:
        n.log("  ✓ skipping (outputs exist)")
        return
    rng = np.random.default_rng(RNG_SEED)
    sub_pairs = list(pairs)
    if n_subset and n_subset < len(sub_pairs):
        sub_pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), size=n_subset, replace=False))]
        n.log(f"  · running on a {len(sub_pairs)}-night random subset of {len(pairs)} (RNG_SEED={RNG_SEED}); pass --sensitivity-n-subset 0 to use all nights")
    else:
        n.log(f"  · running on all {len(sub_pairs)} nights")
    results = _parallel_map(_embedding_one, sub_pairs, n_jobs, "embedding diagnostics", n)
    rows_a, rows_f = [], []
    for r in results:
        if r is None:
            continue
        for tau, v in enumerate(r["ami"], start=1):
            rows_a.append({"subject": r["subject"], "psg_file": r["psg_file"],
                           "tau": int(tau), "ami": float(v) if v == v else np.nan})
        for m, v in enumerate(r["fnn"], start=1):
            rows_f.append({"subject": r["subject"], "psg_file": r["psg_file"],
                           "m": int(m), "fnn_fraction": float(v) if v == v else np.nan})
    df_a = pd.DataFrame(rows_a); df_f = pd.DataFrame(rows_f)
    _save(df_a, out_ami, n); _save(df_f, out_fnn, n)
    rows_s = []
    if not df_a.empty:
        mu = df_a.groupby("tau")["ami"].mean()
        opt_tau = None
        for i in range(1, len(mu) - 1):
            if mu.iloc[i] < mu.iloc[i - 1] and mu.iloc[i] < mu.iloc[i + 1]:
                opt_tau = int(mu.index[i]); break
        rows_s.append({"diagnostic": "AMI_first_local_min",
                       "suggested_value": opt_tau, "current_value": EMBED_TAU,
                       "applies_to": "tau"})
    if not df_f.empty:
        mu = df_f.groupby("m")["fnn_fraction"].mean()
        opt_m = None
        for m, f in mu.items():
            if f < 0.05:
                opt_m = int(m); break
        rows_s.append({"diagnostic": "FNN_first_below_5pct",
                       "suggested_value": opt_m, "current_value": EMBED_M,
                       "applies_to": "m"})
    _save(pd.DataFrame(rows_s), out_summary, n)


# ----- (f) Pz-Oz multi-channel ----------------------------------------------

def worker_main_tda_pz_oz(pair: "Pair", sub_seed: np.random.SeedSequence) -> List[dict]:
    rng = np.random.default_rng(sub_seed)
    try:
        raw = mne.io.read_raw_edf(str(pair.psg_path), preload=False, verbose="ERROR")
        if EEG_SECONDARY not in raw.ch_names:
            return []
        raw.pick([EEG_SECONDARY]); raw.load_data()
        raw.filter(LOWCUT, HIGHCUT, verbose="ERROR")
        raw.resample(TARGET_SFREQ, verbose="ERROR")
        x = raw.get_data()[0].astype(np.float64); sf = float(raw.info["sfreq"])
        intervals = load_intervals(pair.hyp_path)
        epoch_len = int(EPOCH_SEC * sf); n_epochs = len(x) // epoch_len
        by_stage = {s: [] for s in STAGES_MAIN}
        for e in range(n_epochs):
            s = stage_at(intervals, (e + 0.5) * epoch_len / sf)
            if s in by_stage:
                by_stage[s].append(e)
        out = []; psg_tag = _tag(_prefix(pair.psg_path))
        for stage, idxs in by_stage.items():
            if not idxs: continue
            if len(idxs) > MAX_EPOCHS_MAIN:
                idxs = list(rng.choice(idxs, size=MAX_EPOCHS_MAIN, replace=False))
            for e in idxs:
                seg = x[e * epoch_len:(e + 1) * epoch_len]
                feats = persistence_features(seg, EMBED_M, EMBED_TAU, MAXDIM,
                                             MIN_EMBED_POINTS_MAIN, downsample=True)
                if feats is None: continue
                out.append({"subject": pair.subject, "psg_tag": psg_tag,
                            "psg_file": pair.psg_path.name, "hyp_file": pair.hyp_path.name,
                            "channel": EEG_SECONDARY, "stage": stage, "epoch_index": int(e),
                            **feats})
        return out
    except Exception as ex:
        return [{"_ERROR_": f"{pair.psg_path.name} :: {type(ex).__name__}: {ex}"}]


def step_main_tda_pz_oz(pairs, n_jobs, force, n: "Narrator"):
    out_e = OUT_DIR / "tda_epoch_features_pz_oz.csv"
    out_omni = OUT_DIR / "tda_pz_oz_mixedlm_omnibus.csv"
    out_pc = OUT_DIR / "tda_pz_oz_mixedlm_planned_contrasts.csv"
    if _exists(out_e, out_omni, out_pc) and not force:
        n.log("  ✓ skipping (outputs exist)")
        return
    results = _parallel_map(worker_main_tda_pz_oz, pairs, n_jobs,
                            "Pz-Oz TDA per night", n, with_seed=True)
    rows = [x for r in results for x in r if "_ERROR_" not in x]
    df = pd.DataFrame(rows)
    _save(df, out_e, n)
    if df.empty:
        n.log("  ! Pz-Oz not present in any night")
        return
    df["K0_tot"] = within_subject_z(df, "H1_totpers")
    ss = df.groupby(["subject", "stage"], as_index=False)["K0_tot"].mean()
    try:
        res, lr, df_diff, p_lr = fit_mixedlm_stage(ss, "K0_tot", STAGES_MAIN, method="powell")
        omni = pd.DataFrame([{"channel": EEG_SECONDARY, "metric": "K0_tot",
                              "LR": float(lr), "df": int(df_diff), "p": float(p_lr),
                              "n_subjects": int(ss["subject"].nunique())}])
        _save(omni, out_omni, n)
        pc = planned_contrasts(res, "K0_tot", PLANNED_MAIN, STAGES_MAIN)
        pc["channel"] = EEG_SECONDARY
        _save(pc, out_pc, n)
    except Exception as ex:
        n.log(f"  ! Pz-Oz mixed-LM failed: {ex}")


# ----- (g) preprocessing sensitivity ----------------------------------------

def _prep_one_combo(pair_seed, low, high, sf_target):
    pair, sub_seed = pair_seed
    rng = np.random.default_rng(sub_seed)
    try:
        raw = mne.io.read_raw_edf(str(pair.psg_path), preload=True, verbose="ERROR")
        ch = EEG_PRIMARY if EEG_PRIMARY in raw.ch_names else next(
            (c for c in raw.ch_names if "EEG" in c.upper()), None)
        if ch is None:
            return []
        raw.pick([ch])
        raw.filter(low, high, verbose="ERROR")
        raw.resample(sf_target, verbose="ERROR")
        x = raw.get_data()[0]; sf = float(raw.info["sfreq"])
        intervals = load_intervals(pair.hyp_path)
        epoch_len = int(EPOCH_SEC * sf); n_epochs = len(x) // epoch_len
        by_stage = {s: [] for s in STAGES_MAIN}
        for e in range(n_epochs):
            s = stage_at(intervals, (e + 0.5) * epoch_len / sf)
            if s in by_stage:
                by_stage[s].append(e)
        rows = []
        for stage, idxs in by_stage.items():
            if not idxs: continue
            if len(idxs) > MAX_EPOCHS_MAIN:
                idxs = list(rng.choice(idxs, size=MAX_EPOCHS_MAIN, replace=False))
            for e in idxs:
                seg = x[e * epoch_len:(e + 1) * epoch_len]
                feats = persistence_features(seg, EMBED_M, EMBED_TAU, MAXDIM,
                                             MIN_EMBED_POINTS_MAIN, downsample=True)
                if feats is None: continue
                rows.append({"subject": pair.subject, "stage": stage,
                             "H1_totpers": feats["H1_totpers"]})
        return rows
    except Exception:
        return []


def step_preprocessing_sensitivity(pairs, n_jobs, force, n: "Narrator", n_subset: int = 30):
    out = OUT_DIR / "preprocessing_sensitivity_contrasts.csv"
    if _exists(out) and not force:
        n.log("  ✓ skipping (output exists)")
        return
    rng = np.random.default_rng(RNG_SEED)
    sub_pairs = list(pairs)
    if n_subset and n_subset < len(sub_pairs):
        sub_pairs = [pairs[i] for i in sorted(
            rng.choice(len(pairs), size=n_subset, replace=False))]
        n.log(f"  · running on a {len(sub_pairs)}-night random subset of {len(pairs)} (RNG_SEED={RNG_SEED}); pass --sensitivity-n-subset 0 to use all nights")
    else:
        n.log(f"  · running on all {len(sub_pairs)} nights (no subset)")
    bp_grid = [(0.5, 30.0), (0.5, 40.0), (0.5, 45.0), (1.0, 40.0)]
    sf_grid = [50.0, 100.0, 128.0]
    seeds = list(np.random.SeedSequence(RNG_SEED).spawn(len(sub_pairs)))
    parts = []
    for low, high in bp_grid:
        for sf_target in sf_grid:
            n.log(f"  · bandpass=({low},{high}) Hz, sfreq={sf_target} Hz")
            items = list(zip(sub_pairs, seeds))
            t0 = time.time()
            from functools import partial
            worker = partial(_prep_one_combo, low=low, high=high, sf_target=sf_target)
            results = _parallel_bar(worker, items, n_jobs,
                                    f"preproc bp=({low},{high}) sf={sf_target}")
            n.log(f"    done in {time.time()-t0:0.1f}s")
            rows = [x for r in results for x in r]
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["K0_tot"] = within_subject_z(df, "H1_totpers")
            ss = df.groupby(["subject", "stage"], as_index=False)["K0_tot"].mean()
            try:
                res, _, _, _ = fit_mixedlm_stage(ss, "K0_tot", STAGES_MAIN, method="powell")
                pc = planned_contrasts(res, "K0_tot", PLANNED_MAIN, STAGES_MAIN)
                pc["bandpass_low"] = low; pc["bandpass_high"] = high
                pc["sfreq_target"] = sf_target
                pc["n_subjects"] = int(ss["subject"].nunique())
                parts.append(pc)
            except Exception:
                continue
    out_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    _save(out_df, out, n)


# ----- (h) statistical assumption diagnostics -------------------------------

def step_diagnostics(force, n: "Narrator"):
    out = OUT_DIR / "statistical_diagnostics.csv"
    if _exists(out) and not force:
        n.log("  ✓ skipping (output exists)")
        return
    tda = OUT_DIR / "tda_epoch_features_all.csv"
    if not _exists(tda):
        n.log("  ! prerequisite missing or empty: tda_epoch_features_all.csv")
        return
    df = _safe_read_csv(tda)
    if df.empty:
        n.log("  ! prerequisite empty: tda_epoch_features_all.csv")
        return
    if df.empty:
        n.log(f"  ! prerequisite empty: tda_epoch_features_all.csv")
        return
    df = df[df["stage"].isin(STAGES_MAIN)].copy()
    df["K0_tot"] = within_subject_z(df, "H1_totpers")
    ss = df.groupby(["subject", "stage"], as_index=False)["K0_tot"].mean()
    rows = []
    try:
        res, _, _, _ = fit_mixedlm_stage(ss, "K0_tot", STAGES_MAIN, method="powell")
        residuals = np.asarray(res.resid)
        n_res = int(np.minimum(5000, residuals.size))
        sample = residuals if residuals.size <= 5000 else \
            residuals[np.random.default_rng(RNG_SEED).choice(
                residuals.size, size=5000, replace=False)]
        if sample.size >= 3:
            sw_stat, sw_p = stats.shapiro(sample)
            rows.append({"diagnostic": "shapiro_wilk_residuals",
                         "metric": "K0_tot", "statistic": float(sw_stat),
                         "p": float(sw_p), "n_used": int(sample.size)})
        groups = [ss.loc[ss["stage"] == s, "K0_tot"].values
                  for s in STAGES_MAIN if (ss["stage"] == s).any()]
        if len(groups) >= 2:
            lev_stat, lev_p = stats.levene(*groups, center="median")
            rows.append({"diagnostic": "levene_homoscedasticity_across_stages",
                         "metric": "K0_tot", "statistic": float(lev_stat),
                         "p": float(lev_p), "n_used": int(sum(len(g) for g in groups))})
        # Skewness / kurtosis of residuals
        rows.append({"diagnostic": "skewness_residuals", "metric": "K0_tot",
                     "statistic": float(stats.skew(residuals)),
                     "p": np.nan, "n_used": int(residuals.size)})
        rows.append({"diagnostic": "kurtosis_residuals_excess", "metric": "K0_tot",
                     "statistic": float(stats.kurtosis(residuals, fisher=True)),
                     "p": np.nan, "n_used": int(residuals.size)})
    except Exception as ex:
        rows.append({"diagnostic": "error", "metric": "K0_tot",
                     "statistic": np.nan, "p": np.nan, "error": str(ex)})
    _save(pd.DataFrame(rows), out, n)


# ----- (i) classification analysis (LOSO) -----------------------------------

def step_classification(force, n: "Narrator"):
    """LOSO classification: K0 alone vs band power alone vs combined,
    for REM-vs-W, REM-vs-NREM, REM-vs-other targets, with logistic + RF."""
    out_loso = OUT_DIR / "classification_loso_metrics.csv"
    out_summary = OUT_DIR / "classification_summary.csv"
    if _exists(out_loso, out_summary) and not force:
        n.log("  ✓ skipping (outputs exist)")
        return
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.base import clone
    except Exception as ex:
        n.log(f"  ! scikit-learn missing ({ex}); skipping classification")
        n.log("    pip install scikit-learn>=1.3")
        return

    tda = OUT_DIR / "tda_epoch_features_all.csv"
    base = OUT_DIR / "baseline_epoch_features_all.csv"
    if not (tda.exists() and base.exists()):
        n.log("  ! prerequisites missing")
        return
    t = _safe_read_csv(tda); b = _safe_read_csv(base)
    if t.empty or b.empty:
        n.log(f"  ! prerequisite empty: tda={t.empty}, baseline={b.empty}")
        return
    keys = ["subject", "psg_file", "hyp_file", "epoch_index"]
    for d in (t, b):
        d["epoch_index"] = d["epoch_index"].astype(int)
    t_small = t[keys + ["stage", "H1_totpers"]]
    b_small = b[keys + BANDPOWER_COLS]
    df = t_small.merge(b_small, on=keys, how="inner")
    df = df[df["stage"].isin(STAGES_MAIN)].copy()
    df["K0_tot"] = df.groupby("subject")["H1_totpers"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0)

    targets = {
        "REM_vs_W":     (df["stage"] == "REM", df["stage"] == "W"),
        "REM_vs_NREM":  (df["stage"] == "REM", df["stage"].isin(["N1", "N2", "N3"])),
        "REM_vs_other": (df["stage"] == "REM", df["stage"].isin(["W", "N1", "N2", "N3"])),
    }
    feature_sets = {
        "K0_only":        ["K0_tot"],
        "bandpower_only": list(BANDPOWER_COLS),
        "combined":       ["K0_tot"] + list(BANDPOWER_COLS),
    }
    models = {
        "logistic":      LogisticRegression(max_iter=2000, C=1.0),
        "random_forest": RandomForestClassifier(n_estimators=200,
                                                random_state=RNG_SEED, n_jobs=1),
    }

    fold_rows, summary_rows = [], []
    subjects = sorted(df["subject"].unique())
    total_fits = len(targets) * len(feature_sets) * len(models) * len(subjects)
    pbar = _inner_bar(total_fits, f"LOSO classification ({len(subjects)} subjects × 9 configs)")

    for tname, (pos_mask, neg_mask) in targets.items():
        sub_df = df[pos_mask | neg_mask].copy()
        sub_df["y"] = pos_mask[sub_df.index].astype(int).values
        for fs_name, fs_cols in feature_sets.items():
            for mname, mclf in models.items():
                fm = []
                for held in subjects:
                    if pbar is not None:
                        pbar.update(1)
                    tr = sub_df["subject"] != held
                    te = sub_df["subject"] == held
                    if te.sum() < 5 or len(np.unique(sub_df.loc[te, "y"])) < 2:
                        continue
                    X_tr = sub_df.loc[tr, fs_cols].values
                    y_tr = sub_df.loc[tr, "y"].values
                    X_te = sub_df.loc[te, fs_cols].values
                    y_te = sub_df.loc[te, "y"].values
                    pipe = Pipeline([("scaler", StandardScaler()),
                                     ("clf", clone(mclf))])
                    try:
                        pipe.fit(X_tr, y_tr)
                        if hasattr(pipe[-1], "predict_proba"):
                            y_score = pipe.predict_proba(X_te)[:, 1]
                        else:
                            y_score = pipe.decision_function(X_te)
                        y_pred = pipe.predict(X_te)
                        try:
                            auc = float(roc_auc_score(y_te, y_score))
                        except ValueError:
                            auc = np.nan
                        bacc = float(balanced_accuracy_score(y_te, y_pred))
                        f1 = float(f1_score(y_te, y_pred, zero_division=0))
                        tp = int(((y_pred == 1) & (y_te == 1)).sum())
                        fn = int(((y_pred == 0) & (y_te == 1)).sum())
                        tn = int(((y_pred == 0) & (y_te == 0)).sum())
                        fp = int(((y_pred == 1) & (y_te == 0)).sum())
                        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
                        fm.append({
                            "target": tname, "feature_set": fs_name, "model": mname,
                            "held_out_subject": held,
                            "n_test_epochs": int(te.sum()),
                            "n_pos": int((y_te == 1).sum()),
                            "n_neg": int((y_te == 0).sum()),
                            "auc": auc, "balanced_accuracy": bacc, "f1": f1,
                            "sensitivity": float(sens), "specificity": float(spec),
                        })
                    except Exception:
                        continue
                fold_rows.extend(fm)
                if fm:
                    fdf = pd.DataFrame(fm)
                    summary_rows.append({
                        "target": tname, "feature_set": fs_name, "model": mname,
                        "n_folds": int(len(fdf)),
                        "auc_mean": float(fdf["auc"].mean()),
                        "auc_sd": float(fdf["auc"].std(ddof=1)),
                        "bacc_mean": float(fdf["balanced_accuracy"].mean()),
                        "bacc_sd": float(fdf["balanced_accuracy"].std(ddof=1)),
                        "f1_mean": float(fdf["f1"].mean()),
                        "f1_sd": float(fdf["f1"].std(ddof=1)),
                        "sens_mean": float(fdf["sensitivity"].mean()),
                        "spec_mean": float(fdf["specificity"].mean()),
                    })
        n.log(f"  · target {tname} done")

    if pbar is not None:
        pbar.close()
    _save(pd.DataFrame(fold_rows), out_loso, n)
    _save(pd.DataFrame(summary_rows), out_summary, n)




# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

ALL_STEPS = [
    "main_tda", "robustness_grid", "baselines",
    "wake_subclasses", "wake_qc",
    "tda_wake_subclasses",
    "corrected_eeg",
    "tda_track3",
    "wake_subclass_robustness",
    "main_robustness_mixedlm", "baseline_mixedlm",
    "wake_subclass_mixedlm", "baseline_wake_mixedlm",
    "track3_mixedlm",
    "incremental_glm",
    "review_wake_subclass", "review_baseline_wake_subclass",
    "review_incremental",
    "comparison_table", "supplementary_table",
    # Revision-round additions (reviewer comments)
    "all_pairwise", "cohort_replication", "subsampling_stability",
    "bootstrap_contrasts", "embedding_diagnostics", "main_tda_pz_oz",
    "preprocessing_sensitivity", "diagnostics", "classification",
]

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Sleep-EDF TDA full pipeline")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Parallel processes for per-night TDA (1=sequential).")
    p.add_argument("--force", action="store_true",
                   help="Re-run every step even if outputs exist.")
    p.add_argument("--only", type=str, default="",
                   help="Comma-separated subset of steps to run.")
    p.add_argument("--log", type=str, default=None,
                   help="Path to log file (default outputs/logs/pipeline_<ts>.log).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N PSG/Hypnogram pairs (test-run mode).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the step plan and exit without running anything.")
    p.add_argument("--sensitivity-n-subset", type=int, default=30,
                   help="Number of nights used for preprocessing_sensitivity and "
                        "embedding_diagnostics sub-sampled steps (default 30; "
                        "set 0 or negative to use all nights).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else LOG_DIR / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
    n = Narrator(log_path)

    n.banner("Topological Recurrence in EEG Dynamics — full analysis pipeline")
    n.log(f"  · project root : {PROJECT_ROOT}")
    n.log(f"  · n_jobs       : {args.n_jobs} ({'parallel' if args.n_jobs>1 else 'sequential'})")
    n.log(f"  · force re-run : {args.force}")
    n.log(f"  · log file     : {log_path}")
    if args.n_jobs > 1 and not HAS_JOBLIB:
        n.log("  ! joblib not installed; falling back to sequential")
        args.n_jobs = 1

    data_root = resolve_data_root()
    n.log(f"  · dataset root : {data_root}")
    pairs = discover_pairs(data_root)
    n.log(f"  · paired nights: {len(pairs)}")
    if not pairs:
        raise SystemExit("No PSG/Hypnogram pairs found.")
    if args.limit and args.limit < len(pairs):
        pairs = pairs[:args.limit]
        n.log(f"  · TEST RUN: limited to first {len(pairs)} pairs (--limit {args.limit})")
        n.log(f"    (results will not be statistically meaningful)")

    only = set(s.strip() for s in args.only.split(",") if s.strip())
    def run(name: str):
        return not only or name in only

    plan = [
        ("main_tda",                  "Per-epoch persistent homology on EEG Fpz-Cz across W/N1/N2/N3/REM. Up to 30 epochs/stage/night, embedding (m=10, τ=2)."),
        ("robustness_grid",           "Same TDA across the m∈{6,8,10,12} × τ∈{1,2,4} grid; max 25 epochs/stage."),
        ("baselines",                 "Per-epoch Welch band power (δ/θ/α/σ/β), spectral entropy, permutation entropy, LZ76."),
        ("wake_subclasses",           "Track 2 wake labels: per-recording 75th/25th-pctile + 6-vote rule → W_quiet / W_active_ocular / W_bad."),
        ("wake_qc",                   "Track 3 richer wake QC table: per-subject MAD-based thresholds + 3-vote ocular rule."),
        ("tda_wake_subclasses",       "Track 2 TDA on REM + W_quiet + W_active_ocular epochs."),
        ("corrected_eeg",             "EOG-corrected EEG epoch bundles: full-recording linear regression β = cov(eeg,eog)/var(eog)."),
        ("tda_track3",                "Track 3 TDA on raw / EOG-corrected / EOG-channel for wake subclasses (artefact control)."),
        ("wake_subclass_robustness",  "Robustness grid for wake subclasses on raw EEG (m∈{8,10,12} × τ∈{1,2,3})."),
        ("main_robustness_mixedlm",   "Mixed-LM (Powell, REML=False) + planned contrasts on the main robustness grid."),
        ("baseline_mixedlm",          "Mixed-LM + planned contrasts (REM−W, REM−N3, N1−N3) on baseline metrics."),
        ("wake_subclass_mixedlm",     "Mixed-LM + planned contrasts on wake-subclass TDA (Powell)."),
        ("baseline_wake_mixedlm",     "Mixed-LM + planned contrasts on baselines restricted to wake subclasses."),
        ("track3_mixedlm",            "Mixed-LM + planned contrasts on raw / corrected / EOG track-3 TDA tables (lbfgs)."),
        ("incremental_glm",           "Binomial GLM with subject FE testing whether K0_tot adds info beyond band power."),
        ("review_wake_subclass",      "Wake-subclass review tables (stage means, paired effect sizes, summary)."),
        ("review_baseline_wake_subclass", "Baseline-wake review tables."),
        ("review_incremental",        "Incremental-GLM review tables."),
        ("comparison_table",          "REM-vs-wake-subclasses comparison table (TDA + baselines)."),
        ("supplementary_table",       "Supplementary table = wake-subclass contrasts + incremental K0 beyond band power."),
        # Revision-round additions
        ("all_pairwise",              "All 10 pairwise stage contrasts on K0 + per-stage descriptives + monotonicity (Spearman)."),
        ("cohort_replication",        "Refit headline contrasts on Sleep-Cassette vs Sleep-Telemetry separately."),
        ("subsampling_stability",     "Post-hoc resampling at caps {5,10,15,20,25,30} × 10 replicates."),
        ("bootstrap_contrasts",       "Subject-level non-parametric bootstrap (1000x) for headline contrast 95% CIs."),
        ("embedding_diagnostics",     "Per-recording AMI(τ) and FNN(m) to justify the m=10, τ=2 choice."),
        ("main_tda_pz_oz",            "Replicate the main TDA on the Pz-Oz channel where available."),
        ("preprocessing_sensitivity", "K0 contrast across bandpass {0.5-30, 0.5-40, 0.5-45, 1-40} × sfreq {50,100,128} on a 30-night subset."),
        ("diagnostics",               "Mixed-LM assumption checks: Shapiro-Wilk, Levene's, residual skew/kurtosis."),
        ("classification",            "LOSO logistic + random-forest classifiers on K0 / band power / combined for REM-vs-W, REM-vs-NREM, REM-vs-other."),
    ]
    plan = [(name, exp) for name, exp in plan if run(name)]
    total = len(plan)

    funcs = {
        "main_tda":                 lambda: step_main_tda(pairs, args.n_jobs, args.force, n),
        "robustness_grid":          lambda: step_robustness_grid(pairs, args.n_jobs, args.force, n),
        "baselines":                lambda: step_baselines(pairs, args.n_jobs, args.force, n),
        "wake_subclasses":          lambda: step_wake_subclasses(pairs, args.n_jobs, args.force, n),
        "wake_qc":                  lambda: step_wake_qc(pairs, args.n_jobs, args.force, n),
        "tda_wake_subclasses":      lambda: step_tda_wake_subclasses(pairs, args.n_jobs, args.force, n),
        "corrected_eeg":            lambda: step_corrected_eeg(pairs, args.n_jobs, args.force, n),
        "tda_track3":               lambda: step_tda_track3(pairs, args.n_jobs, args.force, n),
        "wake_subclass_robustness": lambda: step_wake_subclass_robustness(pairs, args.n_jobs, args.force, n),
        "main_robustness_mixedlm":  lambda: step_main_robustness_mixedlm(args.force, n),
        "baseline_mixedlm":         lambda: step_baseline_mixedlm(args.force, n),
        "wake_subclass_mixedlm":    lambda: step_wake_subclass_mixedlm(args.force, n),
        "baseline_wake_mixedlm":    lambda: step_baseline_wake_mixedlm(args.force, n),
        "track3_mixedlm":           lambda: step_track3_mixedlm(args.force, n),
        "incremental_glm":          lambda: step_incremental_glm(args.force, n),
        "review_wake_subclass":     lambda: step_review_wake_subclass(args.force, n),
        "review_baseline_wake_subclass": lambda: step_review_baseline_wake_subclass(args.force, n),
        "review_incremental":       lambda: step_review_incremental(args.force, n),
        "comparison_table":         lambda: step_comparison_table(args.force, n),
        "supplementary_table":      lambda: step_supplementary_table(args.force, n),
        # Revision-round additions
        "all_pairwise":              lambda: step_all_pairwise(args.force, n),
        "cohort_replication":        lambda: step_cohort_replication(args.force, n),
        "subsampling_stability":     lambda: step_subsampling_stability(args.force, n),
        "bootstrap_contrasts":       lambda: step_bootstrap_contrasts(args.force, n),
        "embedding_diagnostics":     lambda: step_embedding_diagnostics(pairs, args.n_jobs, args.force, n, n_subset=(args.sensitivity_n_subset if args.sensitivity_n_subset > 0 else None)),
        "main_tda_pz_oz":            lambda: step_main_tda_pz_oz(pairs, args.n_jobs, args.force, n),
        "preprocessing_sensitivity": lambda: step_preprocessing_sensitivity(pairs, args.n_jobs, args.force, n, n_subset=(args.sensitivity_n_subset if args.sensitivity_n_subset > 0 else None)),
        "diagnostics":               lambda: step_diagnostics(args.force, n),
        "classification":            lambda: step_classification(args.force, n),
    }

    if args.dry_run:
        n.banner("DRY RUN — plan only, no work performed")
        for i, (name, exp) in enumerate(plan, 1):
            n.log(f"  {i:2d}. {name:30s}  {exp}")
        return 0

    t_pipeline = time.time()
    failures = []
    for i, (name, exp) in enumerate(plan, 1):
        n.step(i, total, name, exp)
        try:
            t0 = time.time()
            funcs[name]()
            n.log(f"  ⏱  {time.time()-t0:0.1f}s")
        except Exception as ex:
            n.log(f"  ✗ STEP FAILED: {type(ex).__name__}: {ex}")
            failures.append((name, repr(ex)))

    n.banner("PIPELINE COMPLETE")
    n.log(f"  total time: {time.time()-t_pipeline:0.1f}s")
    n.log(f"  log: {log_path}")
    if failures:
        n.log("  failures:")
        for name, err in failures:
            n.log(f"    - {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
