from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import mne
from ripser import ripser
from persim import plot_diagrams

# Reuse your existing helpers (these are in src/scan_dataset.py)
from scan_dataset import (
    read_env_var_from_file,
    prefix_before_dash,
    subject_id,
    tag,
    tag_lead_letter,
)

# Map Sleep-EDF labels -> standard stages
STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",  # merge 3+4
    "Sleep stage R": "REM",
    "Sleep stage ?": "UNK",
}


def time_delay_embedding(x: np.ndarray, m: int = 10, tau: int = 2) -> np.ndarray | None:
    """
    x: (n_samples,)
    returns X: (n_points, m)
    """
    n = x.shape[0] - (m - 1) * tau
    if n <= 20:
        return None
    return np.stack([x[i : i + n] for i in range(0, m * tau, tau)], axis=1)


def zscore_columns(X: np.ndarray) -> np.ndarray:
    """Z-score each coordinate (column) to match your pipeline."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd


def pick_matching_pair(data_root: Path) -> tuple[Path, Path]:
    """
    Same matching logic: PSG + the hypnogram whose tag lead-letter matches PSG tag lead-letter.
    """
    psg_files = sorted(data_root.rglob("*-PSG.edf"))
    hyp_files = sorted(data_root.rglob("*-Hypnogram.edf"))

    hyp_by_subject: dict[str, list[Path]] = {}
    for h in hyp_files:
        hp = prefix_before_dash(h)
        subj = subject_id(hp)
        hyp_by_subject.setdefault(subj, []).append(h)

    p = psg_files[0]  # first PSG as a reproducible example
    pp = prefix_before_dash(p)
    subj = subject_id(pp)
    want = tag_lead_letter(tag(pp))

    matches = [h for h in hyp_by_subject.get(subj, [])
               if tag_lead_letter(tag(prefix_before_dash(h))) == want]
    matches = sorted(matches, key=lambda x: x.name)
    if not matches:
        raise RuntimeError("Could not find a matching hypnogram for the example PSG.")
    return p, matches[0]


def build_epoch_stage_labels(ann: mne.Annotations, sfreq: float, n_epochs: int, epoch_len: int) -> list[str | None]:
    """
    Assign stage per epoch by sampling stage at epoch midpoint, using hypnogram intervals.
    """
    intervals = []
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        stage = STAGE_MAP.get(desc, desc)
        intervals.append((float(onset), float(onset + dur), stage))

    def stage_at(t_sec: float) -> str | None:
        for (a, b, s) in intervals:
            if a <= t_sec < b:
                return s
        return None

    labels: list[str | None] = []
    for e in range(n_epochs):
        mid_t = (e * epoch_len + 0.5 * epoch_len) / sfreq
        labels.append(stage_at(mid_t))
    return labels


def first_epoch_index(labels: list[str | None], stage: str) -> int:
    for i, s in enumerate(labels):
        if s == stage:
            return i
    raise RuntimeError(f"No epoch found for stage={stage}")


def plot_trajectory_3d(X: np.ndarray, title: str) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(X[:, 0], X[:, 1], X[:, 2], lw=0.5)
    ax.set_title(title)
    ax.set_xlabel(r"$x(t)$")
    ax.set_ylabel(r"$x(t+\tau)$")
    ax.set_zlabel(r"$x(t+2\tau)$")
    return fig


def plot_h1_pd(H1: np.ndarray, title: str) -> plt.Figure:
    fig = plt.figure()
    plot_diagrams([H1], show=False)
    plt.title(title)
    plt.xlabel("Birth")
    plt.ylabel("Death")
    return fig


def main():
    # --- parameters (match paper/pipeline) ---
    epoch_sec = 30
    sfreq_target = 50.0
    use_channel = "EEG Fpz-Cz"

    m = 10
    tau = 2          # in samples at sfreq_target
    maxdim = 1       # compute H0 and H1, we will plot H1 only

    # --- locate dataset via config.env ---
    project_root = Path(__file__).resolve().parents[1]
    env_file = project_root / "config.env"
    rel_root = read_env_var_from_file(env_file, "SLEEP_EDF_ROOT")
    if rel_root is None:
        raise SystemExit(
            "Missing SLEEP_EDF_ROOT in config.env.\n"
            "Expected something like:\n"
            "SLEEP_EDF_ROOT=data/sleep-edfx/sleep-edf-database-expanded-1.0.0\n"
        )
    data_root = (project_root / rel_root).resolve()
    if not data_root.exists():
        raise SystemExit(f"Data root does not exist: {data_root}")

    # --- choose example PSG + matching hypnogram ---
    psg_path, hyp_path = pick_matching_pair(data_root)
    print("PSG:", psg_path.name)
    print("HYP:", hyp_path.name)

    # --- load EEG ---
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")

    # select channel (fallback to first EEG channel)
    if use_channel not in raw.ch_names:
        eegs = [ch for ch in raw.ch_names if "EEG" in ch.upper()]
        if not eegs:
            raise SystemExit(f"No EEG channels found. Channels: {raw.ch_names}")
        use_channel = eegs[0]
    raw.pick([use_channel])

    # filter + resample
    raw.filter(l_freq=0.5, h_freq=40.0, verbose="ERROR")
    raw.resample(sfreq_target, verbose="ERROR")

    x = raw.get_data()[0]
    sf = float(raw.info["sfreq"])
    epoch_len = int(epoch_sec * sf)
    n_epochs = len(x) // epoch_len

    # --- load hypnogram annotations ---
    ann = mne.read_annotations(hyp_path)
    labels = build_epoch_stage_labels(ann, sfreq=sf, n_epochs=n_epochs, epoch_len=epoch_len)

    wake_e = first_epoch_index(labels, "W")
    rem_e = first_epoch_index(labels, "REM")

    wake_epoch = x[wake_e * epoch_len : (wake_e + 1) * epoch_len]
    rem_epoch  = x[rem_e  * epoch_len : (rem_e  + 1) * epoch_len]

    # --- delay embedding ---
    X_wake = time_delay_embedding(wake_epoch, m=m, tau=tau)
    X_rem  = time_delay_embedding(rem_epoch,  m=m, tau=tau)
    if X_wake is None or X_rem is None:
        raise RuntimeError("Embedding failed (epoch too short after embedding).")

    # standardise per coordinate
    X_wake = zscore_columns(X_wake)
    X_rem  = zscore_columns(X_rem)

    # --- persistence diagrams ---
    dgms_w = ripser(X_wake, maxdim=maxdim)["dgms"]
    dgms_r = ripser(X_rem,  maxdim=maxdim)["dgms"]
    H1_wake = dgms_w[1]
    H1_rem  = dgms_r[1]

    # --- plots ---
    figA = plot_trajectory_3d(X_wake[:, :3], f"Wake (delay-embedded), m={m}, tau={tau}")
    figB = plot_trajectory_3d(X_rem[:, :3],  f"REM (delay-embedded), m={m}, tau={tau}")

    figC = plot_h1_pd(H1_wake, "Wake $H_1$ persistence")
    figD = plot_h1_pd(H1_rem,  "REM $H_1$ persistence")

    out_dir = project_root / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    pA = out_dir / "fig2A_wake_trajectory.png"
    pB = out_dir / "fig2B_rem_trajectory.png"
    pC = out_dir / "fig2C_wake_h1_pd.png"
    pD = out_dir / "fig2D_rem_h1_pd.png"

    figA.savefig(pA, dpi=300, bbox_inches="tight")
    figB.savefig(pB, dpi=300, bbox_inches="tight")
    figC.savefig(pC, dpi=300, bbox_inches="tight")
    figD.savefig(pD, dpi=300, bbox_inches="tight")

    plt.close(figA); plt.close(figB); plt.close(figC); plt.close(figD)

    print("\nSaved panels:")
    print(" ", pA)
    print(" ", pB)
    print(" ", pC)
    print(" ", pD)

    # Optional: also save a combined 2x2 panel using matplotlib (handy for Overleaf)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # We’ll load the saved PNGs into the grid to keep it simple.
    import matplotlib.image as mpimg
    imgs = [mpimg.imread(pA), mpimg.imread(pB), mpimg.imread(pC), mpimg.imread(pD)]
    titles = ["A: Wake trajectory", "B: REM trajectory", "C: Wake $H_1$", "D: REM $H_1$"]
    for ax, im, t in zip(axes.ravel(), imgs, titles):
        ax.imshow(im)
        ax.set_title(t)
        ax.axis("off")
    combined_path = out_dir / "fig2_panels.png"
    fig.tight_layout()
    fig.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(" ", combined_path)


if __name__ == "__main__":
    main()
