from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import mne
from ripser import ripser

from scan_dataset import read_env_var_from_file, prefix_before_dash, subject_id, tag, tag_lead_letter


# -----------------------------
# Stage mapping (Sleep-EDF)
# -----------------------------
STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",   # merge 3+4
    "Sleep stage R": "REM",
    "Sleep stage ?": "UNK",
}


# -----------------------------
# TDA helpers
# -----------------------------
def time_delay_embedding(x: np.ndarray, m: int, tau: int) -> np.ndarray | None:
    """
    x: (n_samples,)
    returns X: (n_points, m)
    """
    n = x.shape[0] - (m - 1) * tau
    if n <= 20:
        return None
    return np.stack([x[i:i + n] for i in range(0, m * tau, tau)], axis=1)


def dgm_summaries(dgm: np.ndarray) -> dict:
    if dgm.size == 0:
        return {"count": 0, "tot_pers": 0.0, "max_pers": 0.0}

    births = dgm[:, 0]
    deaths = dgm[:, 1]
    finite = np.isfinite(deaths)
    pers = (deaths - births)[finite]

    return {
        "count": int(dgm.shape[0]),
        "tot_pers": float(np.sum(pers)) if pers.size else 0.0,
        "max_pers": float(np.max(pers)) if pers.size else 0.0,
    }


# -----------------------------
# Pairing logic (same as your final scanner)
# -----------------------------
def build_pairs(data_root: Path) -> list[tuple[Path, Path]]:
    psg_files = sorted(data_root.rglob("*-PSG.edf"))
    hyp_files = sorted(data_root.rglob("*-Hypnogram.edf"))

    hyp_by_subject = defaultdict(list)
    for h in hyp_files:
        hp = prefix_before_dash(h)
        hyp_by_subject[subject_id(hp)].append(h)

    pairs: list[tuple[Path, Path]] = []
    for p in psg_files:
        pp = prefix_before_dash(p)
        subj = subject_id(pp)
        want = tag_lead_letter(tag(pp))  # E/F/G/J...

        matches = [
            h for h in hyp_by_subject.get(subj, [])
            if tag_lead_letter(tag(prefix_before_dash(h))) == want
        ]
        matches = sorted(matches, key=lambda x: x.name)

        if not matches:
            # should not happen given your scan result, but keep safe
            continue

        pairs.append((p, matches[0]))

    return pairs


# -----------------------------
# Main batch run
# -----------------------------
def main() -> None:
    # ----- config you can tweak -----
    epoch_sec = 30
    sfreq_target = 50          # downsample for speed
    maxdim = 1                 # H0 and H1
    embed_m = 10
    embed_tau = 2
    max_epochs_per_stage = 30  # per night per stage (start smaller; increase later)
    use_channel = "EEG Fpz-Cz" # default; will fall back to first EEG if missing

    stages_used = ["W", "N1", "N2", "N3", "REM"]  # we ignore UNK

    rng = np.random.default_rng(0)

    # ----- paths -----
    project_root = Path(__file__).resolve().parents[1]
    rel_root = read_env_var_from_file(project_root / "config.env", "SLEEP_EDF_ROOT")
    if rel_root is None:
        raise SystemExit("Missing SLEEP_EDF_ROOT in config.env")

    data_root = (project_root / rel_root).resolve()

    out_dir = project_root / "outputs"
    out_dir.mkdir(exist_ok=True)

    out_epochs = out_dir / "tda_epoch_features_all.csv"
    out_summary = out_dir / "tda_stage_summary_all.csv"
    out_errors = out_dir / "tda_errors.log"

    # ----- build pairs -----
    pairs = build_pairs(data_root)
    if not pairs:
        raise SystemExit("No PSG/Hypnogram pairs found.")

    print(f"Pairs found: {len(pairs)}")

    all_rows: list[dict] = []
    errors: list[str] = []

    # ----- process each night -----
    for i, (psg_path, hyp_path) in enumerate(pairs, start=1):
        try:
            print(f"[{i}/{len(pairs)}] {psg_path.name}")

            # Load PSG
            raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="ERROR")

            # choose channel
            ch = use_channel
            if ch not in raw.ch_names:
                eegs = [c for c in raw.ch_names if "EEG" in c.upper()]
                if not eegs:
                    raise RuntimeError(f"No EEG channels. Channels={raw.ch_names}")
                ch = eegs[0]

            raw.pick([ch])
            raw.load_data()
            raw.filter(l_freq=0.5, h_freq=40.0, verbose="ERROR")
            raw.resample(sfreq_target, verbose="ERROR")

            x = raw.get_data()[0]
            sf = float(raw.info["sfreq"])
            epoch_len = int(epoch_sec * sf)

            # Load hypnogram
            ann = mne.read_annotations(hyp_path)

            intervals: list[tuple[float, float, str]] = []
            for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
                stage = STAGE_MAP.get(desc, desc)
                intervals.append((float(onset), float(onset + dur), stage))

            def stage_at(t_sec: float) -> str | None:
                for a, b, s in intervals:
                    if a <= t_sec < b:
                        return s
                return None

            n_epochs = len(x) // epoch_len

            # Collect epochs by stage
            stage_to_epochs = {s: [] for s in stages_used}
            for e in range(n_epochs):
                mid_t = (e * epoch_len + 0.5 * epoch_len) / sf
                s = stage_at(mid_t)
                if s in stage_to_epochs:
                    stage_to_epochs[s].append(e)

            # Compute persistence features
            subj = subject_id(prefix_before_dash(psg_path))
            psg_tag = tag(prefix_before_dash(psg_path))  # e.g. E0, F0, G1...

            for stage in stages_used:
                idxs = stage_to_epochs[stage]
                if not idxs:
                    continue

                # sample epochs
                if len(idxs) > max_epochs_per_stage:
                    idxs = list(rng.choice(idxs, size=max_epochs_per_stage, replace=False))

                for e in idxs:
                    seg = x[e * epoch_len:(e + 1) * epoch_len]
                    seg = seg[::2]  # further downsample within epoch

                    X = time_delay_embedding(seg, m=embed_m, tau=embed_tau)
                    if X is None:
                        continue

                    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)

                    dgms = ripser(X, maxdim=maxdim)["dgms"]
                    h0 = dgm_summaries(dgms[0])
                    h1 = dgm_summaries(dgms[1]) if len(dgms) > 1 else {"count": 0, "tot_pers": 0.0, "max_pers": 0.0}

                    all_rows.append({
                        "subject": subj,
                        "psg_tag": psg_tag,
                        "psg_file": psg_path.name,
                        "hyp_file": hyp_path.name,
                        "channel": ch,
                        "stage": stage,
                        "epoch_index": int(e),
                        "H0_count": h0["count"],
                        "H0_totpers": h0["tot_pers"],
                        "H0_maxpers": h0["max_pers"],
                        "H1_count": h1["count"],
                        "H1_totpers": h1["tot_pers"],
                        "H1_maxpers": h1["max_pers"],
                    })

        except Exception as ex:
            msg = f"{psg_path.name} :: {type(ex).__name__}: {ex}"
            print("  ERROR:", msg)
            errors.append(msg)
            continue

    # ----- write outputs -----
    if errors:
        out_errors.write_text("\n".join(errors), encoding="utf-8")
        print(f"Wrote errors to: {out_errors}")

    if not all_rows:
        raise SystemExit("No rows were produced. Check errors log and parameters.")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_epochs, index=False)

    numeric_cols = [
        "H0_count", "H0_totpers", "H0_maxpers",
        "H1_count", "H1_totpers", "H1_maxpers",
    ]

    # Overall summary by stage (across all subjects/nights)
    summary = df.groupby("stage")[numeric_cols].agg(["mean", "std", "count"])
    summary.to_csv(out_summary)

    print("\nSaved:")
    print(" ", out_epochs)
    print(" ", out_summary)


if __name__ == "__main__":
    main()
