#!/usr/bin/env python3
"""
run.py
======

Top-level entry point. From a shell at the project root:

    python run.py                    # interactive: prompts for cores
    python run.py --n-jobs 8         # skip the prompt
    python run.py --skip-figures     # pipeline + collate + extras, no figures
    python run.py --skip-extras      # pipeline + collate + figures, no extras
    python run.py --only-collate     # rebuild results.xlsx from existing CSVs
    python run.py --only-figures     # rebuild figures from existing CSVs
    python run.py --only-extras      # rebuild the standalone analyses only

The script runs four stages end-to-end: (1) scripts/pipeline.py, (2)
scripts/collate.py, (3) scripts/figures.py, and (4) the five standalone
supplementary-analysis scripts (demographics_breakdown.py,
stratified_effects.py, sensitivity_power_analysis.py, supplementary_round3.py,
supplementary_round4.py). It narrates each stage as it runs, mirrors all
output to a master log under ``outputs/logs/``, and saves outputs
incrementally so a crash mid-run never costs you completed work — just rerun
the same command.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Force UTF-8 on stdout/stderr so Unicode banner characters render on Windows
# consoles (default cp1252 codec) without UnicodeEncodeError. This also makes
# the subprocess pipes inherit a UTF-8 environment via PYTHONIOENCODING below.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUT_DIR / "logs"


# ──────────────────────────────────────────────────────────────────────────────
# Console + log helpers
# ──────────────────────────────────────────────────────────────────────────────

class Log:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n[run.py started] {datetime.now()}\n{'='*80}\n")

    def write(self, msg: str = ""):
        print(msg, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def banner(self, title: str):
        bar = "═" * 78
        self.write("")
        self.write(bar)
        self.write(f"║  {title}")
        self.write(bar)


def stream_subprocess(cmd: List[str], log: Log, cwd: Path = None, capture: bool = True) -> int:
    """Run a subprocess.

    capture=True  → pipe stdout/stderr through the master log (good for short scripts).
    capture=False → child inherits the real console stdio, so tqdm progress bars
                    render correctly with carriage-return updates. Use for long
                    interactive runs (pipeline.py). The child still writes its own
                    log under outputs/logs/, so nothing is lost.
    """
    log.write(f"$ {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    if not capture:
        log.write("  (live progress streamed directly to console; child writes its own log under outputs/logs/)")
        rc = subprocess.call(cmd, cwd=str(cwd or PROJECT_ROOT), env=env)
        return int(rc)

    proc = subprocess.Popen(
        cmd, cwd=str(cwd or PROJECT_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True,
        encoding="utf-8", errors="replace", env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        log.write(line)
    return int(proc.wait())


# ──────────────────────────────────────────────────────────────────────────────
# Environment checks
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_MODULES = [
    "numpy", "pandas", "scipy", "mne", "ripser", "persim",
    "matplotlib", "statsmodels", "openpyxl",
    "sklearn",   # scikit-learn — used by the LOSO classification step
]
OPTIONAL_MODULES = ["joblib"]

def check_environment(log: Log, allow_missing_optional: bool = True) -> bool:
    log.banner("Environment check")
    if sys.version_info < (3, 10):
        log.write(f"  ✗ Python {sys.version_info.major}.{sys.version_info.minor} detected; need ≥ 3.10")
        return False
    log.write(f"  ✓ Python {sys.version.split()[0]}")
    missing_required: List[str] = []
    for mod in REQUIRED_MODULES:
        try:
            __import__(mod)
            log.write(f"  ✓ {mod}")
        except ImportError:
            missing_required.append(mod)
            log.write(f"  ✗ {mod} (required)")
    for mod in OPTIONAL_MODULES:
        try:
            __import__(mod)
            log.write(f"  ✓ {mod} (optional, enables --n-jobs > 1)")
        except ImportError:
            log.write(f"  · {mod} not installed (parallelism will be disabled)")
    if missing_required:
        log.write("")
        log.write("Install missing packages with:")
        log.write("  pip install -r requirements.txt")
        return False
    return True


def check_dataset(log: Log) -> Optional[Path]:
    """Locate the Sleep-EDF Expanded dataset using config.env or auto-detect."""
    log.banner("Dataset check")
    env_path = PROJECT_ROOT / "config.env"
    rel = None
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "SLEEP_EDF_ROOT":
                rel = v.strip().strip("'").strip('"')
                break
    candidates = []
    if rel:
        candidates.append((PROJECT_ROOT / rel).resolve())
    candidates += [
        PROJECT_ROOT / "data" / "sleep-edfx" / "sleep-edf-database-expanded-1.0.0",
        PROJECT_ROOT / "data" / "sleep-edfx" / "sleep-edf-database-expanded-1.0.0" / "sleep-cassette",
    ]
    for cand in candidates:
        cand = cand.resolve()
        if cand.exists() and any(cand.rglob("*.edf")):
            n_psg = len(list(cand.rglob("*-PSG.edf")))
            n_hyp = len(list(cand.rglob("*-Hypnogram.edf")))
            log.write(f"  ✓ found dataset at {cand}")
            log.write(f"    {n_psg} PSG files, {n_hyp} hypnogram files")
            return cand
    log.write("  ✗ Sleep-EDF dataset not found.")
    log.write("    Either:")
    log.write("    1. Set SLEEP_EDF_ROOT in config.env (relative to project root), or")
    log.write("    2. Place the dataset under data/sleep-edfx/sleep-edf-database-expanded-1.0.0/")
    log.write("")
    log.write("    Download from https://physionet.org/content/sleep-edfx/")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Interactive prompt
# ──────────────────────────────────────────────────────────────────────────────

def prompt_n_jobs(log: Log) -> int:
    cpu = mp.cpu_count()
    suggested = max(1, min(cpu - 1, 8))
    log.write("")
    log.write(f"This machine reports {cpu} CPU cores available.")
    log.write("")
    log.write("Per-night TDA can run in parallel for a large speed-up.")
    log.write("Recommendation: leave 1 core free for the OS.")
    log.write("")
    while True:
        raw = input(f"How many parallel workers? [1-{cpu}, default {suggested}]: ").strip()
        if not raw:
            return suggested
        try:
            n = int(raw)
            if 1 <= n <= cpu:
                return n
        except ValueError:
            pass
        print(f"  Please enter an integer between 1 and {cpu}.")


# ──────────────────────────────────────────────────────────────────────────────
# Stage runners
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(n_jobs: int, force: bool, log: Log, extra_args=None) -> int:
    log.banner("STAGE 1 / 4 — running scripts/pipeline.py")
    log.write("This is the heavy lifting:")
    log.write("  · pair PSG/Hypnogram files")
    log.write("  · per-epoch persistent homology on Fpz-Cz (main TDA)")
    log.write("  · TDA across the m × τ robustness grid")
    log.write("  · Welch band power, spectral / permutation entropy, LZ baselines")
    log.write("  · wake QC, wake subclass labels, EOG-corrected EEG bundles")
    log.write("  · TDA on raw / corrected / EOG channels (artefact controls)")
    log.write("  · all mixed-LM stats with planned contrasts (Holm-corrected)")
    log.write("  · binomial GLM testing K0 incremental over band power")
    log.write("  · review / comparison / supplementary tables")
    log.write("")
    log.write("Each step writes its CSV to outputs/ as soon as it finishes,")
    log.write("so re-running this command will pick up where a crash left off.")
    log.write("")
    cmd = [sys.executable, "-u", str(SCRIPTS_DIR / "pipeline.py"), "--n-jobs", str(n_jobs)]
    if force:
        cmd.append("--force")
    if extra_args:
        cmd.extend(extra_args)
    # capture=False: let pipeline.py write directly to the terminal so tqdm
    # progress bars render. The pipeline still writes its own log to
    # outputs/logs/pipeline_<ts>.log via its Narrator class.
    return stream_subprocess(cmd, log, capture=False)


def run_collate(log: Log) -> int:
    log.banner("STAGE 2 / 4 — running scripts/collate.py")
    log.write("Bundling every CSV under outputs/ into a single Excel workbook")
    log.write("with one sheet per logical results section (see the README sheet).")
    log.write("")
    cmd = [sys.executable, "-u", str(SCRIPTS_DIR / "collate.py")]
    return stream_subprocess(cmd, log)


def run_figures(log: Log) -> int:
    log.banner("STAGE 3 / 4 — running scripts/figures.py")
    log.write("Generating manuscript and supplementary figures into outputs/figures/.")
    log.write("Figure 2 (example trajectories + persistence diagrams) re-processes a")
    log.write("single PSG/Hypnogram pair; pass --skip-fig2 below to skip it.")
    log.write("")
    cmd = [sys.executable, "-u", str(SCRIPTS_DIR / "figures.py")]
    return stream_subprocess(cmd, log)


def run_extras(n_jobs: int, force: bool, log: Log) -> int:
    """Stage 4: the five standalone supplementary-analysis scripts.

    These are not part of pipeline.py because they depend on its output CSVs.
    Order matters for the first three — stratified_effects.py and
    sensitivity_power_analysis.py both read demographics_per_night.csv, which
    demographics_breakdown.py produces. supplementary_round3.py and
    supplementary_round4.py depend only on the pipeline's output CSVs (and,
    for round-3 Section C, re-process raw EDFs for the ICA control), so they
    run last.

    supplementary_round3.py and supplementary_round4.py both accept --n-jobs
    and honour the same worker count as the pipeline.

    The recovery utility refit_wake_robustness_stats.py is intentionally NOT
    run here: it is only needed when the pipeline's wake_subclass_robustness
    step wrote its grid CSV but the inner mixed-LM fits failed.
    """
    log.banner("STAGE 4 / 4 — running standalone supplementary-analysis scripts")
    log.write("Five standalone scripts that depend on the pipeline's output CSVs:")
    log.write("  · demographics_breakdown.py      — cohort age / sex / drug-protocol tables")
    log.write("  · stratified_effects.py          — REM-Wake K0 by lifespan × sex and by drug")
    log.write("  · sensitivity_power_analysis.py  — observed d_z, minimum detectable effect")
    log.write("  · supplementary_round3.py        — diagnostic embedding, LOSO CIs, ICA, PE-order")
    log.write("  · supplementary_round4.py        — regime-specific AUCs, EOG-corrected band power")
    log.write("")
    log.write("demographics_breakdown.py runs first: the next two read")
    log.write("demographics_per_night.csv, which it produces. supplementary_round3.py")
    log.write("Section C re-processes raw EDFs for the ICA control, so it (and")
    log.write("supplementary_round4.py) honour the same --n-jobs setting as the pipeline.")
    log.write("")
    # (script name, whether it accepts --n-jobs)
    extras = [
        ("demographics_breakdown.py",     False),
        ("stratified_effects.py",         False),
        ("sensitivity_power_analysis.py", False),
        ("supplementary_round3.py",       True),
        ("supplementary_round4.py",       True),
    ]
    for name, accepts_n_jobs in extras:
        log.write(f"  → {name}")
        cmd = [sys.executable, "-u", str(SCRIPTS_DIR / name)]
        if accepts_n_jobs:
            cmd += ["--n-jobs", str(n_jobs)]
        if force:
            cmd.append("--force")
        rc = stream_subprocess(cmd, log)
        if rc != 0:
            log.write(f"\n  ! {name} exited with code {rc}.")
            return rc
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Top-level shell orchestrator for the EEG TDA analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-jobs", type=int, default=None,
                   help="Parallel workers for the pipeline (default: prompt).")
    p.add_argument("--force", action="store_true",
                   help="Re-run every pipeline step even if outputs exist.")
    p.add_argument("--skip-figures", action="store_true",
                   help="Run pipeline + collate + extras, skip figure generation.")
    p.add_argument("--skip-extras", action="store_true",
                   help="Run pipeline + collate + figures, skip the standalone "
                        "supplementary-analysis scripts (stage 4).")
    p.add_argument("--only-collate", action="store_true",
                   help="Skip pipeline, figures & extras, just rebuild results.xlsx.")
    p.add_argument("--only-figures", action="store_true",
                   help="Skip pipeline, collate & extras, just rebuild figures.")
    p.add_argument("--only-extras", action="store_true",
                   help="Skip pipeline, collate & figures, just rebuild the "
                        "standalone supplementary-analysis outputs (stage 4).")
    p.add_argument("--no-env-check", action="store_true",
                   help="Skip the environment / dataset check.")
    p.add_argument("--limit", type=int, default=None,
                   help="TEST RUN: process only the first N PSG/Hypnogram pairs.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the pipeline step plan and exit without running anything.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    log = Log(log_path)

    log.banner("Topological Recurrence in EEG Dynamics — full analysis run")
    log.write(f"  · project root : {PROJECT_ROOT}")
    log.write(f"  · log file     : {log_path}")
    log.write(f"  · started      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.only_collate:
        rc = run_collate(log)
        log.write(f"\nFinished in {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return rc
    if args.only_figures:
        rc = run_figures(log)
        log.write(f"\nFinished in {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return rc
    if args.only_extras:
        rc = run_extras(args.n_jobs or 1, args.force, log)
        log.write(f"\nFinished in {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return rc

    if not args.no_env_check:
        if not check_environment(log):
            return 1
        if check_dataset(log) is None:
            return 1

    if args.n_jobs is None:
        try:
            n_jobs = prompt_n_jobs(log)
        except (EOFError, KeyboardInterrupt):
            log.write("\n  ! aborted at prompt")
            return 130
    else:
        n_jobs = args.n_jobs
    log.write(f"\n  · using n_jobs = {n_jobs}")

    t0 = time.time()
    extra = []
    if args.limit:
        extra += ["--limit", str(args.limit)]
    if args.dry_run:
        extra.append("--dry-run")
    rc = run_pipeline(n_jobs, args.force, log, extra_args=extra)
    if rc != 0:
        log.write(f"\n  ! pipeline.py exited with code {rc}.")
        log.write("  Re-run this script after addressing the error; completed steps will be skipped.")
        return rc

    rc = run_collate(log)
    if rc != 0:
        log.write(f"\n  ! collate.py exited with code {rc}.")
        return rc

    if not args.skip_figures:
        rc = run_figures(log)
        if rc != 0:
            log.write(f"\n  ! figures.py exited with code {rc}.")
            return rc

    if not args.skip_extras:
        rc = run_extras(n_jobs, args.force, log)
        if rc != 0:
            log.write(f"\n  ! a stage-4 script exited with code {rc}.")
            log.write("  Re-run this script after addressing the error; "
                      "completed steps will be skipped.")
            return rc

    elapsed = time.time() - t0
    log.banner("ALL DONE")
    log.write(f"  · elapsed       : {elapsed/60:0.1f} minutes")
    log.write(f"  · results       : outputs/results.xlsx")
    log.write(f"  · figures       : outputs/figures/")
    log.write(f"  · demographics  : outputs/demographics_*.csv")
    log.write(f"  · stratified    : outputs/strat_*.csv")
    log.write(f"  · power         : outputs/sensitivity_power_*.csv")
    log.write(f"  · supplementary : outputs/supp_*.csv")
    log.write(f"  · log           : {log_path}")
    log.write(f"  · outputs/      : every intermediate CSV is preserved here")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
