from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import mne
from ripser import ripser

from scan_dataset import (
    read_env_var_from_file,
    prefix_before_dash,
    subject_id,
    tag,
    tag_lead_letter,
)

STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Sleep stage ?": "UNK",
}

STAGES = ["W", "N1", "N2", "N3", "REM"]


def pick_pairs_all(data_root: Path) -> list[tuple[Path, Path]]:
    psg_files = sorted(data_root.rglob("*-PSG.edf"))
    hyp_files = sorted(data_root.rglob("*-Hypnogram.edf"))

    hyp_by_subject: dict[str, list[Path]] = {}
    for h in hyp_files:
        hp = prefix_before_dash(h)
        subj = subject_id(hp)
        hyp_by_subject.setdefault(subj, []).append(h)

    pairs: list[tuple[Path, Path]] = []
    for p in psg_files:
        pp = prefix_before_dash(p)
        subj = subject_id(pp)
        want = tag_lead_letter(tag(pp))
        matches = [
            h for h in hyp_by_subject.get(subj, [])
            if tag_lead_letter(tag(prefix_before_dash(h))) == want
        ]
        matches = sorted(matches, key=lambda x: x.name)
        if matches:
            pairs.append((p, matches[0]))
    return pairs


def stage_intervals(hyp_path: Path):
    ann = mne.read_annotations(hyp_path)
    intervals = []
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        s = STAGE_MAP.get(desc, desc)
        intervals.append((float(onset), float(onset + dur), s))
    return intervals


def stage_at(intervals, t_sec: float) -> str | None:
    for a, b, s in intervals:
        if a <= t_sec < b:
            return s
    return None


def time_delay_embedding(x: np.ndarray, m: int, tau: int) -> np.ndarray | None:
    n = x.shape[0] - (m - 1) * tau
    if n <= 30:
        return None
    return np.stack([x[i : i + n] for i in range(0, m * tau, tau)], axis=1)


def dgm_h1_summaries(dgm: np.ndarray) -> tuple[int, float, float]:
    # returns (count, totpers, maxpers)
    if dgm.size == 0:
        return 0, 0.0, 0.0
    births = dgm[:, 0]
    deaths = dgm[:, 1]
    finite = np.isfinite(deaths)
    pers = (deaths - births)[finite]
    tot = float(np.sum(pers)) if pers.size else 0.0
    mx = float(np.max(pers)) if pers.size else 0.0
    return int(dgm.shape[0]), tot, mx


def main():
    # ---- config ----
    epoch_sec = 30
    sfreq_target = 50.0
    filter_l, filter_h = 0.5, 40.0
    maxdim = 1

    # robustness grid
    ms = [6, 8, 10, 12]
    taus = [1, 2, 4]

    # sampling per stage per night
    max_epochs_per_stage = 25
    rng = np.random.default_rng(0)

    # channels to test (we’ll use whichever exists)
    channels_to_try = ["EEG Fpz-Cz", "EEG Pz-Oz"]

    project_root = Path(__file__).resolve().parents[1]
    rel_root = read_env_var_from_file(project_root / "config.env", "SLEEP_EDF_ROOT")
    if rel_root is None:
        raise SystemExit("Missing SLEEP_EDF_ROOT in config.env")

    data_root = (project_root / rel_root).resolve()
    out_dir = project_root / "outputs"
    out_dir.mkdir(exist_ok=True)

    pairs = pick_pairs_all(data_root)
    print("Paired nights:", len(pairs))

    rows = []
    for i, (psg_path, hyp_path) in enumerate(pairs, start=1):
        try:
            raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")

            # pick a channel (prefer Fpz-Cz, else Pz-Oz, else first EEG)
            ch = None
            for candidate in channels_to_try:
                if candidate in raw.ch_names:
                    ch = candidate
                    break
            if ch is None:
                eegs = [c for c in raw.ch_names if "EEG" in c.upper()]
                if not eegs:
                    continue
                ch = eegs[0]

            raw.pick([ch])
            raw.filter(l_freq=filter_l, h_freq=filter_h, verbose="ERROR")
            raw.resample(sfreq_target, verbose="ERROR")

            x = raw.get_data()[0]
            sf = float(raw.info["sfreq"])
            epoch_len = int(epoch_sec * sf)
            n_epochs = len(x) // epoch_len

            intervals = stage_intervals(hyp_path)

            # collect epoch indices by stage
            stage_to_idxs = {s: [] for s in STAGES}
            for e in range(n_epochs):
                mid_t = (e * epoch_len + 0.5 * epoch_len) / sf
                s = stage_at(intervals, mid_t)
                if s in stage_to_idxs:
                    stage_to_idxs[s].append(e)

            # sample indices per stage per night
            sampled = {}
            for s, idxs in stage_to_idxs.items():
                if not idxs:
                    continue
                if len(idxs) > max_epochs_per_stage:
                    sampled[s] = rng.choice(idxs, size=max_epochs_per_stage, replace=False).tolist()
                else:
                    sampled[s] = idxs

            subj = subject_id(prefix_before_dash(psg_path))

            for s, idxs in sampled.items():
                for e in idxs:
                    seg = x[e * epoch_len : (e + 1) * epoch_len]

                    # optional further speed-up (keep consistent across grid)
                    seg = seg[::2]

                    for m in ms:
                        for tau in taus:
                            X = time_delay_embedding(seg, m=m, tau=tau)
                            if X is None:
                                continue

                            # z-score each dimension
                            X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)

                            dgms = ripser(X, maxdim=maxdim)["dgms"]
                            h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
                            h1_count, h1_tot, h1_max = dgm_h1_summaries(h1)

                            rows.append({
                                "subject": subj,
                                "psg_file": psg_path.name,
                                "hyp_file": hyp_path.name,
                                "channel": ch,
                                "stage": s,
                                "epoch_index": int(e),
                                "m": int(m),
                                "tau": int(tau),
                                "H1_count": h1_count,
                                "H1_totpers": h1_tot,
                                "H1_maxpers": h1_max,
                            })

            if i % 10 == 0:
                print(f"...processed {i}/{len(pairs)} nights")

        except Exception as ex:
            print("Skip", psg_path.name, "->", type(ex).__name__, ex)

    df = pd.DataFrame(rows)
    out_path = out_dir / "tda_robustness_grid_epochs.csv"
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
