from pathlib import Path
from collections import defaultdict

def read_env_var_from_file(env_path: Path, key: str) -> str | None:
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip()
    return None

def prefix_before_dash(p: Path) -> str:
    return p.name.split("-", 1)[0]

def subject_id(prefix: str) -> str:
    return prefix[:6]

def tag(prefix: str) -> str:
    # "SC4261F0" -> "F0", "SC4001EC" -> "EC"
    return prefix[6:]

def tag_lead_letter(tag_str: str) -> str | None:
    # "F0" -> "F", "EC" -> "E"
    return tag_str[0] if tag_str else None

def main():
    project_root = Path(__file__).resolve().parents[1]
    env_file = project_root / "config.env"
    rel_root = read_env_var_from_file(env_file, "SLEEP_EDF_ROOT")

    if rel_root is None:
        raise SystemExit(
            "Missing SLEEP_EDF_ROOT in config.env.\n"
            "Expected:\n"
            "SLEEP_EDF_ROOT=data/sleep-edfx/sleep-edf-database-expanded-1.0.0\n"
        )

    data_root = (project_root / rel_root).resolve()
    if not data_root.exists():
        raise SystemExit(f"Data root does not exist: {data_root}")

    psg_files = sorted(data_root.rglob("*-PSG.edf"))
    hyp_files = sorted(data_root.rglob("*-Hypnogram.edf"))

    # Group hypnograms by subject
    hyp_by_subject = defaultdict(list)
    for h in hyp_files:
        hp = prefix_before_dash(h)
        hyp_by_subject[subject_id(hp)].append(h)

    pairs = []
    missing_hyp = []
    ambiguous = []

    for p in psg_files:
        pp = prefix_before_dash(p)       # e.g. SC4261F0
        subj = subject_id(pp)            # e.g. SC4261
        p_tag = tag(pp)                  # e.g. F0
        want_letter = tag_lead_letter(p_tag)  # e.g. F

        candidates = hyp_by_subject.get(subj, [])
        if not candidates or want_letter is None:
            missing_hyp.append(p)
            continue

        matches = []
        for h in candidates:
            h_tag = tag(prefix_before_dash(h))   # e.g. FC, FJ, FM...
            if tag_lead_letter(h_tag) == want_letter:
                matches.append(h)

        if not matches:
            missing_hyp.append(p)
        else:
            if len(matches) > 1:
                ambiguous.append((p, matches))
            matches_sorted = sorted(matches, key=lambda x: x.name)
            pairs.append((p, matches_sorted[0]))

    print("=== Sleep-EDF dataset scan ===")
    print(f"Data root: {data_root}")
    print(f"PSG files found: {len(psg_files)}")
    print(f"Hypnogram files found: {len(hyp_files)}")
    print(f"Paired nights: {len(pairs)}")
    print(f"PSG missing hypnogram: {len(missing_hyp)}")
    print(f"Ambiguous (multiple hypnograms match): {len(ambiguous)}")

    if pairs:
        print("\nExample pair:")
        print(f"  PSG: {pairs[0][0]}")
        print(f"  HYP: {pairs[0][1]}")

    if missing_hyp:
        print("\nFirst 10 PSG files missing hypnograms:")
        for p in missing_hyp[:10]:
            print(f"  {p}")

    if ambiguous:
        print("\nFirst 3 ambiguous examples (PSG -> hyp choices):")
        for p, ms in ambiguous[:3]:
            print(f"  {p.name}")
            for h in sorted(ms, key=lambda x: x.name)[:6]:
                print(f"    - {h.name}")

if __name__ == "__main__":
    main()
