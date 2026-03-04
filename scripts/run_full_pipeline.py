#!/usr/bin/env python3
"""
run_full_pipeline.py

One-command runner for the Sleep-EDF Expanded analysis pipeline.

Designed for VS Code / terminal usage. It:
  1) Scans dataset pairing sanity
  2) Runs TDA extraction (all nights)
  3) Runs TDA robustness grid extraction
  4) Summarises robustness with mixed models + planned contrasts
  5) Runs baseline feature extraction
  6) Runs baseline mixed-model benchmarks
  7) (Optional) Generates planned-contrast heatmap
  8) (Optional) Generates Figure 2 example panels

This script assumes the other scripts live in the same folder: ./scripts/

Notes:
- We do NOT commit raw data or outputs to git. This runner just generates them locally.
- It expects a .env-like file (default: ./config/config.env) containing SLEEP_EDF_ROOT
  (or you can set SLEEP_EDF_ROOT in your shell environment).

Usage examples (PowerShell):
  python .\scripts\run_full_pipeline.py
  python .\scripts\run_full_pipeline.py --env .\config\config.env --python python
  python .\scripts\run_full_pipeline.py --skip-robustness
  python .\scripts\run_full_pipeline.py --make-heatmap --heatmap-contrast "REM - W" --heatmap-metric K0_tot

"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path | None = None,
    verbose: bool = True,
) -> int:
    if verbose:
        print("\n$ " + " ".join(cmd))

    # Stream to console + optional log file
    if log_path is None:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env)
        return int(proc.returncode)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"COMMAND: {' '.join(cmd)}\n")
        f.write(f"TIME: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write("=" * 100 + "\n")

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            if verbose:
                print(line, end="")
            f.write(line)

        return int(proc.wait())


def _load_env_file(env_file: Path) -> dict[str, str]:
    """
    Minimal .env parser:
      KEY=VALUE
      # comments allowed
    Values may be quoted.
    """
    out: dict[str, str] = {}
    if not env_file.exists():
        return out

    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k:
            out[k] = v
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run the full Sleep-EDF analysis pipeline (code-only repo).",
        epilog=textwrap.dedent(
            """
            Recommended:
              - Ensure config/config.env exists (or export SLEEP_EDF_ROOT).
              - Create a venv and install requirements.
              - Run from repo root:
                    python scripts/run_full_pipeline.py
            """
        ),
    )

    parser.add_argument(
        "--env",
        type=str,
        default=str(Path("config") / "config.env"),
        help="Path to config.env (default: config/config.env).",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use for subprocess calls (default: current interpreter).",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Path to repository root (default: .).",
    )

    # toggles
    parser.add_argument("--skip-scan", action="store_true", help="Skip dataset scan step.")
    parser.add_argument("--skip-tda", action="store_true", help="Skip main TDA extraction step.")
    parser.add_argument("--skip-robustness", action="store_true", help="Skip robustness grid + summary steps.")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline extraction + stats steps.")
    parser.add_argument("--make-heatmap", action="store_true", help="Run make_planned_contrast_heatmap.py after robustness.")
    parser.add_argument("--make-fig2", action="store_true", help="Run make_figure2_example.py to generate example panels.")

    # heatmap args
    parser.add_argument("--heatmap-contrast", type=str, default="REM - W", help='Heatmap contrast label (default: "REM - W").')
    parser.add_argument("--heatmap-metric", type=str, default="K0_tot", help='Heatmap metric (default: "K0_tot").')
    parser.add_argument("--heatmap-channel", type=str, default="", help="Optional channel name if needed (e.g., 'EEG Fpz-Cz').")

    # output/logging
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Outputs directory (default: outputs).")
    parser.add_argument("--logs-dir", type=str, default="outputs/logs", help="Logs directory (default: outputs/logs).")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first failing step.")
    parser.add_argument("--quiet", action="store_true", help="Less console output (still logs to file).")

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    scripts_dir = repo_root / "scripts"
    outputs_dir = repo_root / args.outputs_dir
    logs_dir = repo_root / args.logs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Build environment for subprocesses
    env = os.environ.copy()
    env_file = Path(args.env)
    if not env_file.is_absolute():
        env_file = (repo_root / env_file).resolve()

    env_from_file = _load_env_file(env_file)
    env.update(env_from_file)

    # Validate required env var
    sleep_root = env.get("SLEEP_EDF_ROOT", "").strip()
    if not sleep_root:
        print(
            "ERROR: SLEEP_EDF_ROOT is not set.\n"
            f" - Put it in {env_file} as: SLEEP_EDF_ROOT=/path/to/sleep-edf...\n"
            " - Or set it in your environment before running.\n",
            file=sys.stderr,
        )
        return 2

    # Validate scripts exist
    required = [
        "scan_dataset.py",
        "run_tda_all_nights.py",
        "compute_tda_robustness_grid.py",
        "summarise_tda_robustness_grid.py",
        "compute_baselines_all_nights.py",
        "run_baseline_benchmarks_stats.py",
        "make_planned_contrast_heatmap.py",
        "make_figure2_example.py",
        "review_outputs.py",
    ]
    missing = [s for s in required if not (scripts_dir / s).exists()]
    if missing:
        print("ERROR: Missing expected script(s) in ./scripts:\n  - " + "\n  - ".join(missing), file=sys.stderr)
        return 3

    py = args.python
    verbose = not args.quiet
    stamp = _now_stamp()
    master_log = logs_dir / f"run_full_pipeline_{stamp}.log"

    steps: list[tuple[str, list[str], Path | None]] = []

    # Step 1: scan
    if not args.skip_scan:
        steps.append(
            (
                "scan_dataset",
                [py, str(scripts_dir / "scan_dataset.py")],
                master_log,
            )
        )

    # Step 2: main TDA all nights
    if not args.skip_tda:
        steps.append(
            (
                "run_tda_all_nights",
                [py, str(scripts_dir / "run_tda_all_nights.py")],
                master_log,
            )
        )

    # Step 3+4: robustness grid + summarise
    if not args.skip_robustness:
        steps.append(
            (
                "compute_tda_robustness_grid",
                [py, str(scripts_dir / "compute_tda_robustness_grid.py")],
                master_log,
            )
        )
        steps.append(
            (
                "summarise_tda_robustness_grid",
                [py, str(scripts_dir / "summarise_tda_robustness_grid.py")],
                master_log,
            )
        )

        if args.make_heatmap:
            # You can tweak paths/flags here if your heatmap script expects different CLI args.
            heatmap_out = outputs_dir / f"heatmap_{args.heatmap_metric}_{args.heatmap_contrast.replace(' ', '')}_{stamp}.png"
            cmd = [
                py,
                str(scripts_dir / "make_planned_contrast_heatmap.py"),
                "--csv",
                str(outputs_dir / "tda_robustness_mixedlm_planned_contrasts.csv"),
                "--out",
                str(heatmap_out),
                "--contrast",
                args.heatmap_contrast,
                "--metric",
                args.heatmap_metric,
            ]
            if args.heatmap_channel:
                cmd += ["--channel", args.heatmap_channel]
            steps.append(("make_planned_contrast_heatmap", cmd, master_log))

    # Step 5+6: baselines
    if not args.skip_baselines:
        steps.append(
            (
                "compute_baselines_all_nights",
                [py, str(scripts_dir / "compute_baselines_all_nights.py")],
                master_log,
            )
        )
        steps.append(
            (
                "run_baseline_benchmarks_stats",
                [py, str(scripts_dir / "run_baseline_benchmarks_stats.py")],
                master_log,
            )
        )

    # Optional: fig2 panels
    if args.make_fig2:
        steps.append(
            (
                "make_figure2_example",
                [py, str(scripts_dir / "make_figure2_example.py")],
                master_log,
            )
        )

    # Final: review outputs table summaries (prints + logs)
    steps.append(
        (
            "review_outputs",
            [py, str(scripts_dir / "review_outputs.py")],
            master_log,
        )
    )

    # Run steps
    if verbose:
        print(f"Repo root: {repo_root}")
        print(f"SLEEP_EDF_ROOT: {sleep_root}")
        print(f"Outputs dir: {outputs_dir}")
        print(f"Log: {master_log}")

    # Record environment snapshot for reproducibility
    env_snapshot = logs_dir / f"env_snapshot_{stamp}.txt"
    try:
        with open(env_snapshot, "w", encoding="utf-8") as f:
            f.write(f"TIME: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"PYTHON: {py}\n")
            f.write(f"REPO_ROOT: {repo_root}\n")
            f.write(f"ENV_FILE: {env_file}\n")
            f.write(f"SLEEP_EDF_ROOT: {sleep_root}\n")
    except Exception:
        pass

    for name, cmd, log in steps:
        if verbose:
            print("\n" + "#" * 100)
            print(f"# STEP: {name}")
            print("#" * 100)

        rc = _run(cmd, cwd=repo_root, env=env, log_path=log, verbose=verbose)
        if rc != 0:
            msg = f"Step '{name}' failed with exit code {rc}."
            print("\nERROR: " + msg, file=sys.stderr)
            if args.fail_fast:
                return rc

    if verbose:
        print("\nDONE. Pipeline completed.")
        print(f"Master log: {master_log}")
        print(f"Outputs are in: {outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())