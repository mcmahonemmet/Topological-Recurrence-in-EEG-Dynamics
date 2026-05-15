#!/usr/bin/env python3
"""
demographics_breakdown.py
=========================

Standalone script that produces a demographic breakdown of the analysis
cohort (197 nights from 100 unique subjects in Sleep-EDF Expanded).

Inputs (read-only):
  - data/sleep-edfx/sleep-edf-database-expanded-1.0.0/SC-subjects.xls
  - data/sleep-edfx/sleep-edf-database-expanded-1.0.0/ST-subjects.xls
  - The PSG/Hypnogram pairs discovered by pipeline.discover_pairs()

Outputs (written to outputs/ and outputs/figures/):
  - demographics_per_night.csv      one row per analysed night with the
                                    joined age, sex, and (for Telemetry)
                                    drug protocol assignment for that night.
  - demographics_per_subject.csv    one row per unique person, with cohort,
                                    age, sex, number of nights in the
                                    analysis, and drug-protocol info.
  - demographics_summary.csv        cohort-level (Cassette / Telemetry /
                                    Total) summary: n subjects, n nights,
                                    age range / mean / SD / median, sex
                                    distribution, drug-protocol counts.
  - figures/demographics_age_distribution.png
                                    histogram of age stratified by cohort,
                                    plus a simple sex-by-cohort bar panel.

Subject-ID conventions (Sleep-EDF Expanded):
  - SC4XYn-PSG.edf  → Sleep-Cassette, person XY (zero-padded), night n.
  - ST7XYn-PSG.edf  → Sleep-Telemetry, person XY, night n.
  Person identity is therefore the (cohort, XY) pair, not the 6-char file
  prefix used as the random-effect grouping in the main pipeline.

Sex coding in the source spreadsheets:
  - SC-subjects.xls : column "sex (F=1)" → 1 = Female, 2 = Male.
  - ST-subjects.xls : column "M1/F2"     → 1 = Male,   2 = Female.
  We canonicalise to {"F", "M"} in the joined tables to avoid the mismatch.

Note on study design (Sleep-Telemetry):
  The Telemetry sub-cohort participated in a randomised within-subject
  Temazepam-versus-placebo crossover. Each subject contributed exactly two
  nights, one on Temazepam and one on placebo, with the order counter-
  balanced across subjects. The script reports both per-subject assignment
  and per-night drug labels so the design is transparent in the demographics
  output.

Usage:
  python scripts/demographics_breakdown.py
  python scripts/demographics_breakdown.py --no-figure   # CSVs only
  python scripts/demographics_breakdown.py --force       # overwrite outputs
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# UTF-8 stdio for Windows consoles.
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

import pipeline as P  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUT_DIR = P.OUT_DIR
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_PER_NIGHT   = OUT_DIR / "demographics_per_night.csv"
OUT_PER_SUBJECT = OUT_DIR / "demographics_per_subject.csv"
OUT_SUMMARY     = OUT_DIR / "demographics_summary.csv"
OUT_FIG         = FIG_DIR / "demographics_age_distribution.png"


# ──────────────────────────────────────────────────────────────────────────────
# Tiny logger
# ──────────────────────────────────────────────────────────────────────────────

def banner(s: str):
    bar = "═" * max(len(s) + 4, 60)
    print(f"\n{bar}\n  {s}\n{bar}", flush=True)

def log(s: str):
    print(s, flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Filename → cohort / person / night decoding
# ──────────────────────────────────────────────────────────────────────────────

def parse_psg_filename(name: str) -> Optional[Dict]:
    """SC4XYn-PSG.edf  → cohort=Cassette,  person_id=XY (int), night=n (int).
    ST7XYn-PSG.edf  → cohort=Telemetry, person_id=XY,        night=n.

    Returns None if the filename does not match either of the two known
    Sleep-EDF Expanded prefixes.
    """
    stem = name.split("-")[0]                # SC4001E0 / ST7011J0
    if len(stem) < 6:
        return None
    if stem.startswith("SC4"):
        cohort = "Cassette"
    elif stem.startswith("ST7"):
        cohort = "Telemetry"
    else:
        return None
    try:
        person_id = int(stem[3:5])
        night = int(stem[5])
    except ValueError:
        return None
    return {"cohort": cohort, "person_id": int(person_id), "night": int(night),
            "psg_file": name}


# ──────────────────────────────────────────────────────────────────────────────
# Spreadsheet readers (canonicalise the schema)
# ──────────────────────────────────────────────────────────────────────────────

def _canon_sex(value, coding: str) -> str:
    """Canonicalise the source-specific sex code to {'F', 'M'}.

    coding == 'F=1, M=2' (Cassette spreadsheet)
    coding == 'M=1, F=2' (Telemetry spreadsheet)
    """
    try:
        v = int(value)
    except (TypeError, ValueError):
        return ""
    if coding == "F=1, M=2":
        return {1: "F", 2: "M"}.get(v, "")
    if coding == "M=1, F=2":
        return {1: "M", 2: "F"}.get(v, "")
    return ""


def read_sc_subjects(xls_path: Path) -> pd.DataFrame:
    """Return per-night Cassette demographics with canonical column names:
    cohort, person_id, night, age, sex, lights_off."""
    df = pd.read_excel(xls_path)
    # Expected columns: ['subject', 'night', 'age', 'sex (F=1)', 'LightsOff']
    rename = {
        "subject": "person_id",
        "night": "night",
        "age": "age",
        "sex (F=1)": "_sex_raw",
        "LightsOff": "lights_off",
    }
    missing = [c for c in rename if c not in df.columns]
    if missing:
        raise RuntimeError(f"SC-subjects.xls missing columns: {missing}")
    df = df.rename(columns=rename)
    df["cohort"] = "Cassette"
    df["sex"] = df["_sex_raw"].apply(lambda v: _canon_sex(v, "F=1, M=2"))
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce").astype("Int64")
    df["night"] = pd.to_numeric(df["night"], errors="coerce").astype("Int64")
    df["lights_off"] = df["lights_off"].astype(str)
    df["drug"] = "n/a"
    return df[["cohort", "person_id", "night", "age", "sex", "lights_off", "drug"]]


def read_st_subjects(xls_path: Path) -> pd.DataFrame:
    """Return per-night Telemetry demographics with canonical columns:
    cohort, person_id, night, age, sex, lights_off, drug. The Telemetry
    spreadsheet has a multi-row header that we collapse into a flat schema."""
    raw = pd.read_excel(xls_path, header=None)
    # Row 0 is headers like "Nr / Age / M1F2 / Placebo night, lights off /
    # Temazepam night, lights off". The first useful header sits at index 0;
    # actual data starts at row 2 (idx 1 carries the sub-headers night-nr /
    # lights-off). We resolve by hard-coded column positions.
    body = raw.iloc[2:].copy()
    body.columns = ["nr", "age", "sex_raw",
                    "placebo_night_nr", "placebo_lights_off",
                    "temazepam_night_nr", "temazepam_lights_off"]
    body = body.dropna(subset=["nr", "age"])
    rows = []
    for _, r in body.iterrows():
        try:
            nr = int(r["nr"]); age = int(r["age"])
        except (TypeError, ValueError):
            continue
        sex = _canon_sex(r["sex_raw"], "M=1, F=2")
        for drug, ncol, lcol in (
            ("placebo", "placebo_night_nr", "placebo_lights_off"),
            ("temazepam", "temazepam_night_nr", "temazepam_lights_off"),
        ):
            try:
                night_nr = int(r[ncol])
            except (TypeError, ValueError):
                continue
            rows.append({
                "cohort": "Telemetry",
                "person_id": nr,
                "night": night_nr,
                "age": age,
                "sex": sex,
                "lights_off": str(r[lcol]),
                "drug": drug,
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Join with the analysis cohort and build per-subject / summary tables
# ──────────────────────────────────────────────────────────────────────────────

def build_per_night(pairs, demo_sc: pd.DataFrame, demo_st: pd.DataFrame) -> pd.DataFrame:
    rows = []
    miss = []
    for pair in pairs:
        info = parse_psg_filename(pair.psg_path.name)
        if info is None:
            miss.append(pair.psg_path.name)
            continue
        rec = dict(info)
        rec["subject_key"] = pair.subject
        rec["hyp_file"] = pair.hyp_path.name
        if rec["cohort"] == "Cassette":
            m = demo_sc[(demo_sc["person_id"] == rec["person_id"])
                        & (demo_sc["night"] == rec["night"])]
        else:
            m = demo_st[(demo_st["person_id"] == rec["person_id"])
                        & (demo_st["night"] == rec["night"])]
        if m.empty:
            rec.update({"age": np.nan, "sex": "", "lights_off": "", "drug": ""})
        else:
            r = m.iloc[0]
            rec.update({"age": r["age"], "sex": r["sex"],
                        "lights_off": r["lights_off"], "drug": r["drug"]})
        rows.append(rec)
    df = pd.DataFrame(rows)
    if miss:
        log(f"  ! {len(miss)} PSG file(s) did not match SC/ST naming convention; "
            f"first: {miss[0]}")
    return df


def build_per_subject(per_night: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per unique person."""
    rows = []
    for (cohort, pid), g in per_night.groupby(["cohort", "person_id"]):
        # Age and sex should be invariant within a person; take the first
        # non-missing value but flag any inconsistency for debug.
        ages = g["age"].dropna().unique()
        sexes = [s for s in g["sex"].dropna().unique() if s]
        drugs = sorted(d for d in g["drug"].fillna("").unique() if d and d != "n/a")
        nights = sorted(int(n) for n in g["night"].dropna().unique())
        rows.append({
            "cohort": cohort,
            "person_id": int(pid),
            "age": float(ages[0]) if ages.size else np.nan,
            "sex": sexes[0] if sexes else "",
            "n_nights_in_analysis": int(g.shape[0]),
            "nights": ",".join(str(n) for n in nights),
            "drug_protocol_nights": ",".join(drugs) if drugs else "n/a",
            "had_placebo": ("placebo" in drugs),
            "had_temazepam": ("temazepam" in drugs),
            "age_inconsistent": int(ages.size > 1),
            "sex_inconsistent": int(len(sexes) > 1),
        })
    return pd.DataFrame(rows).sort_values(["cohort", "person_id"]).reset_index(drop=True)


def build_summary(per_subject: pd.DataFrame, per_night: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cohorts = sorted(per_subject["cohort"].unique().tolist())
    cohorts_with_total = cohorts + ["Total"]
    for cohort in cohorts_with_total:
        if cohort == "Total":
            ps = per_subject; pn = per_night
        else:
            ps = per_subject[per_subject["cohort"] == cohort]
            pn = per_night[per_night["cohort"] == cohort]
        ages = ps["age"].dropna()
        sex_counts = ps["sex"].value_counts().to_dict()
        nF = int(sex_counts.get("F", 0)); nM = int(sex_counts.get("M", 0))
        n_total = nF + nM
        # Drug-protocol counts apply only to Telemetry (and the Total row)
        n_subj_both = int((ps["had_placebo"] & ps["had_temazepam"]).sum())
        n_placebo_nights = int((pn["drug"] == "placebo").sum())
        n_temazepam_nights = int((pn["drug"] == "temazepam").sum())
        rows.append({
            "cohort": cohort,
            "n_subjects": int(ps.shape[0]),
            "n_nights":   int(pn.shape[0]),
            "age_min":    float(ages.min())    if ages.size else np.nan,
            "age_max":    float(ages.max())    if ages.size else np.nan,
            "age_mean":   float(ages.mean())   if ages.size else np.nan,
            "age_sd":     float(ages.std(ddof=1)) if ages.size > 1 else np.nan,
            "age_median": float(ages.median()) if ages.size else np.nan,
            "age_iqr_low":  float(ages.quantile(0.25)) if ages.size else np.nan,
            "age_iqr_high": float(ages.quantile(0.75)) if ages.size else np.nan,
            "n_female": nF, "n_male": nM,
            "pct_female": (100.0 * nF / n_total) if n_total else np.nan,
            "pct_male":   (100.0 * nM / n_total) if n_total else np.nan,
            "n_subjects_with_both_placebo_and_temazepam": n_subj_both,
            "n_placebo_nights":   n_placebo_nights,
            "n_temazepam_nights": n_temazepam_nights,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────

def make_figure(per_subject: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Panel 1: age histogram stratified by cohort
    ax = axes[0]
    palette = {"Cassette": "#1f77b4", "Telemetry": "#d62728"}
    bins = np.arange(15, 105, 5)
    for cohort in ["Cassette", "Telemetry"]:
        ages = per_subject.loc[per_subject["cohort"] == cohort, "age"].dropna().values
        if ages.size == 0:
            continue
        ax.hist(ages, bins=bins, alpha=0.55, color=palette[cohort],
                label=f"{cohort} (n = {ages.size}, age {ages.min():.0f}–{ages.max():.0f})",
                edgecolor="white")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of subjects")
    ax.set_title("Age distribution by cohort")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    # Panel 2: sex × cohort bar panel
    ax = axes[1]
    pivot = (per_subject.assign(_n=1)
             .pivot_table(index="cohort", columns="sex", values="_n", aggfunc="sum",
                          fill_value=0)
             .reindex(columns=["F", "M"], fill_value=0))
    pivot = pivot.reindex(["Cassette", "Telemetry"])
    x = np.arange(len(pivot.index))
    w = 0.38
    ax.bar(x - w/2, pivot["F"].values, width=w, color="#bf5b8c",
           label="Female", edgecolor="white")
    ax.bar(x + w/2, pivot["M"].values, width=w, color="#5b8cbf",
           label="Male", edgecolor="white")
    for i, cohort in enumerate(pivot.index):
        nF, nM = int(pivot.loc[cohort, "F"]), int(pivot.loc[cohort, "M"])
        ax.text(i - w/2, nF + 0.5, str(nF), ha="center", va="bottom", fontsize=9)
        ax.text(i + w/2, nM + 0.5, str(nM), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Number of subjects")
    ax.set_title("Sex distribution by cohort")
    ax.legend(frameon=False, fontsize=9)
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
        description="Demographic breakdown of the Sleep-EDF Expanded analysis cohort"
    )
    p.add_argument("--force", action="store_true",
                   help="Rerun and overwrite outputs even if they already exist.")
    p.add_argument("--no-figure", action="store_true",
                   help="Skip the demographics figure.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    banner("Sleep-EDF Expanded demographics breakdown")

    if (OUT_PER_NIGHT.exists() and OUT_PER_SUBJECT.exists() and OUT_SUMMARY.exists()
            and (args.no_figure or OUT_FIG.exists()) and not args.force):
        log("  ✓ outputs exist; use --force to recompute")
        return

    data_root = P.resolve_data_root()
    log(f"Dataset root: {data_root}")
    sc_xls = data_root / "SC-subjects.xls"
    st_xls = data_root / "ST-subjects.xls"
    if not sc_xls.exists() or not st_xls.exists():
        # Try one level up (common when the data root resolves to a subfolder).
        for cand in [data_root.parent, data_root.parent.parent]:
            if (cand / "SC-subjects.xls").exists() and (cand / "ST-subjects.xls").exists():
                sc_xls = cand / "SC-subjects.xls"
                st_xls = cand / "ST-subjects.xls"
                break
    if not sc_xls.exists() or not st_xls.exists():
        sys.exit(f"Missing demographics spreadsheets: {sc_xls}, {st_xls}")

    log(f"  · reading {sc_xls.name}")
    demo_sc = read_sc_subjects(sc_xls)
    log(f"  · reading {st_xls.name}")
    demo_st = read_st_subjects(st_xls)
    log(f"    SC rows: {len(demo_sc)}; ST rows: {len(demo_st)}")

    pairs = P.discover_pairs(data_root)
    log(f"  · discovered {len(pairs)} PSG/Hypnogram analysis pairs")

    per_night = build_per_night(pairs, demo_sc, demo_st)
    per_subject = build_per_subject(per_night)
    summary = build_summary(per_subject, per_night)

    per_night.to_csv(OUT_PER_NIGHT, index=False)
    log(f"  · wrote {OUT_PER_NIGHT.name} ({len(per_night)} rows)")
    per_subject.to_csv(OUT_PER_SUBJECT, index=False)
    log(f"  · wrote {OUT_PER_SUBJECT.name} ({len(per_subject)} rows)")
    summary.to_csv(OUT_SUMMARY, index=False)
    log(f"  · wrote {OUT_SUMMARY.name} ({len(summary)} rows)")

    # Console summary
    log("")
    log("  Cohort summary:")
    for _, r in summary.iterrows():
        log(f"    {r['cohort']:>10s}: "
            f"n_subjects = {int(r['n_subjects']):3d}, "
            f"n_nights = {int(r['n_nights']):3d}, "
            f"age = {r['age_min']:.0f}–{r['age_max']:.0f} "
            f"(mean {r['age_mean']:.1f} ± {r['age_sd']:.1f}, "
            f"median {r['age_median']:.0f}), "
            f"F/M = {int(r['n_female'])}/{int(r['n_male'])} "
            f"({r['pct_female']:.0f}% F)")
    nT = summary.loc[summary["cohort"] == "Telemetry"]
    if not nT.empty:
        rT = nT.iloc[0]
        log(f"    Telemetry drug protocol: "
            f"{int(rT['n_subjects_with_both_placebo_and_temazepam'])} subjects "
            f"contributed both placebo and Temazepam nights "
            f"({int(rT['n_placebo_nights'])} placebo + "
            f"{int(rT['n_temazepam_nights'])} Temazepam = "
            f"{int(rT['n_placebo_nights']) + int(rT['n_temazepam_nights'])} nights).")

    if not args.no_figure:
        make_figure(per_subject, OUT_FIG)

    log("\nDone.")


if __name__ == "__main__":
    main()
