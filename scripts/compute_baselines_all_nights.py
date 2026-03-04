from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import mne

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

    hyp_by_subject = {}
    for h in hyp_files:
        hp = prefix_before_dash(h)
        subj = subject_id(hp)
        hyp_by_subject.setdefault(subj, []).append(h)

    pairs = []
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


def bandpower(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return float("nan")
    return float(np.trapezoid(psd[mask], freqs[mask]))


def spectral_entropy(psd: np.ndarray) -> float:
    p = psd / (psd.sum() + 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def permutation_entropy(x: np.ndarray, order: int = 5, delay: int = 1) -> float:
    # Simple, fast permutation entropy
    n = len(x) - (order - 1) * delay
    if n <= 10:
        return float("nan")
    patterns = np.empty(n, dtype=np.int64)
    for i in range(n):
        window = x[i : i + order * delay : delay]
        patterns[i] = np.argsort(window).dot((order ** np.arange(order)).astype(np.int64))
    _, counts = np.unique(patterns, return_counts=True)
    p = counts / counts.sum()
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def lz_complexity_binary(x: np.ndarray) -> float:
    # Coarse-grain by median, then LZ76 complexity (normalised)
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
        if s[i : i + k] == s[l : l + k]:
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
    # Normalise by n/log2(n) (common choice)
    return float(c * np.log2(n) / n)


def main():
    epoch_sec = 30
    sfreq_target = 50.0
    use_channel = "EEG Fpz-Cz"

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
            if use_channel not in raw.ch_names:
                eegs = [ch for ch in raw.ch_names if "EEG" in ch.upper()]
                if not eegs:
                    continue
                ch = eegs[0]
            else:
                ch = use_channel

            raw.pick([ch])
            raw.filter(l_freq=0.5, h_freq=40.0, verbose="ERROR")
            raw.resample(sfreq_target, verbose="ERROR")

            x = raw.get_data()[0]
            sf = float(raw.info["sfreq"])
            epoch_len = int(epoch_sec * sf)
            n_epochs = len(x) // epoch_len

            intervals = stage_intervals(hyp_path)

            # precompute freqs for welch
            for e in range(n_epochs):
                mid_t = (e * epoch_len + 0.5 * epoch_len) / sf
                stage = stage_at(intervals, mid_t)
                if stage not in STAGES:
                    continue

                seg = x[e * epoch_len : (e + 1) * epoch_len]

                # Welch PSD
                psd, freqs = mne.time_frequency.psd_array_welch(
                    seg,
                    sfreq=sf,
                    fmin=0.5,
                    fmax=40.0,
                    n_fft=min(2048, len(seg)),
                    verbose="ERROR",
                )

                d = bandpower(psd, freqs, 0.5, 4.0)
                t = bandpower(psd, freqs, 4.0, 8.0)
                a = bandpower(psd, freqs, 8.0, 12.0)
                s = bandpower(psd, freqs, 12.0, 15.0)
                b = bandpower(psd, freqs, 15.0, 30.0)

                rows.append(
                    {
                        "subject": subject_id(prefix_before_dash(psg_path)),
                        "psg_file": psg_path.name,
                        "hyp_file": hyp_path.name,
                        "channel": ch,
                        "epoch_index": int(e),
                        "stage": stage,
                        "log_delta": float(np.log(d + 1e-12)),
                        "log_theta": float(np.log(t + 1e-12)),
                        "log_alpha": float(np.log(a + 1e-12)),
                        "log_sigma": float(np.log(s + 1e-12)),
                        "log_beta": float(np.log(b + 1e-12)),
                        "spec_entropy": spectral_entropy(psd),
                        "perm_entropy": permutation_entropy(seg, order=5, delay=1),
                        "lz_complexity": lz_complexity_binary(seg),
                    }
                )

            if i % 10 == 0:
                print(f"...processed {i}/{len(pairs)} nights")

        except Exception as ex:
            print("Skip", psg_path.name, "->", type(ex).__name__, ex)

    df = pd.DataFrame(rows)
    out_epochs = out_dir / "baseline_epoch_features_all.csv"
    df.to_csv(out_epochs, index=False)
    print("Saved:", out_epochs)


if __name__ == "__main__":
    main()
