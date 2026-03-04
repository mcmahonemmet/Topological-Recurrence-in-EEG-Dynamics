#!/usr/bin/env python3
"""
make_planned_contrast_heatmap.py

Plot a robustness heatmap from mixedlm planned contrasts.

Input CSV columns (as in your file):
  contrast, estimate, metric, channel, m, tau, ...

This script filters:
  metric == --metric (default: K0_tot)
  contrast == --contrast (default: "REM - W")
  channel == --channel (optional; if omitted and multiple channels exist, errors)

Then pivots estimate into an (m x tau) grid and saves a PNG.

Example:
  python .\src\make_planned_contrast_heatmap.py `
    --csv .\outputs\tda_robustness_mixedlm_planned_contrasts.csv `
    --out .\outputs\robustness_heatmap_K0tot_REMminusW.png `
    --metric K0_tot `
    --contrast "REM - W" `
    --channel "EEG Fpz-Cz"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _numsort(vals):
    try:
        return sorted(vals, key=lambda x: float(x))
    except Exception:
        return sorted(vals, key=lambda x: str(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="planned_contrasts CSV")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--metric", default="K0_tot", help="metric to plot (default: K0_tot)")
    ap.add_argument("--contrast", default="REM - W", help='contrast label (default: "REM - W")')
    ap.add_argument("--channel", default=None, help='channel (e.g., "EEG Fpz-Cz"). Required if CSV has >1 channel.')
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    df = pd.read_csv(Path(args.csv))

    required = ["contrast", "estimate", "metric", "channel", "m", "tau"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: Missing columns {missing}. Found: {list(df.columns)}")

    d = df[(df["metric"] == args.metric) & (df["contrast"] == args.contrast)].copy()
    if d.empty:
        raise SystemExit(
            f"ERROR: No rows after filtering metric={args.metric!r}, contrast={args.contrast!r}.\n"
            f"Available metrics: {sorted(df['metric'].dropna().unique().tolist())}\n"
            f"Available contrasts (example): {sorted(df['contrast'].dropna().unique().tolist())[:20]}"
        )

    chans = d["channel"].dropna().unique().tolist()
    if args.channel is None:
        if len(chans) != 1:
            raise SystemExit(
                "ERROR: Multiple channels present. Please specify --channel.\n"
                f"Channels found: {chans}"
            )
        channel = chans[0]
    else:
        channel = args.channel
        d = d[d["channel"] == channel]
        if d.empty:
            raise SystemExit(
                f"ERROR: No rows for channel={channel!r} after filtering.\n"
                f"Channels available (for this metric/contrast): {chans}"
            )

    # Pivot to grid (m x tau)
    table = d.pivot_table(index="m", columns="tau", values="estimate", aggfunc="mean")
    table = table.reindex(_numsort(table.index), axis=0)
    table = table.reindex(_numsort(table.columns), axis=1)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(table.values, aspect="auto", origin="lower")

    ax.set_xlabel("tau")
    ax.set_ylabel("m")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([str(x) for x in table.columns])
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([str(x) for x in table.index])

    title = args.title if args.title else f"{args.metric}: {args.contrast} estimate across (m,τ)\n{channel}"
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("estimate")

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Wrote: {out_path}")
    print(f"metric={args.metric} | contrast={args.contrast} | channel={channel}")
    print(f"grid: {table.shape[0]} m-values x {table.shape[1]} tau-values")


if __name__ == "__main__":
    main()
