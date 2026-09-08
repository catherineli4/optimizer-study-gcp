"""Per-token h/kappa distributions, one panel per config.

The summary figures reduce each config to a median and two percentiles; this
shows the whole distribution behind those numbers. One figure per model size,
one panel per chinchilla, adamw and muon overlaid -- so the question the summary
plots raise ("is the between-optimizer gap real, given how wide the spread is?")
can be answered by looking at the shapes rather than at three order statistics.

Needs the per-token arrays, i.e. margin_stats_sweep run with --save-per-token.
Only lambda > 0 positions are shown: sigma* := 0 elsewhere and the closed form
is stated only there, matching the summary aggregate.

    python -m new_utils.plot_margin_hist --margins-dir /mnt/localssd/margins-pt
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same validated adamw/muon hues as every other figure.
COLOR = {"adamw": "#2a78d6", "muon": "#eb6834"}
INK, MUTED = "#0b0b0b", "#52514e"
SIZES = ["30M", "60M", "100M", "300M", "600M"]


def load_size(margins_dir, size):
    """{chinchilla: {optimizer: 1-D array of h/kappa over lambda>0 positions}}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(margins_dir, size, "*.npz"))):
        name = os.path.basename(f)[:-4]
        m = re.search(r"chinchilla-([0-9.]+)-(adamw|muon)", name)
        if not m:
            continue
        d = np.load(f)
        if "h_over_kappa" not in d:
            print(f"{name}: summary only, rerun with --save-per-token")
            continue
        r = d["h_over_kappa"]
        out.setdefault(float(m.group(1)), {})[m.group(2)] = r[np.isfinite(r)]
    return out


def plot_size(size, cells, out, bins=70):
    chins = sorted(cells)
    if not chins:
        return
    ncol = min(5, len(chins))
    nrow = (len(chins) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.5 * nrow),
                             squeeze=False, sharex=True)

    # One common bin edge set per figure, so panels are visually comparable and
    # a shifted distribution reads as a shift rather than a rebinning artefact.
    allv = np.concatenate([v for c in chins for v in cells[c].values()])
    lo, hi = np.percentile(allv, [0.2, 99.8])
    edges = np.linspace(lo, hi, bins + 1)

    for i, chin in enumerate(chins):
        ax = axes[i // ncol][i % ncol]
        for opt in ("adamw", "muon"):
            v = cells[chin].get(opt)
            if v is None or not v.size:
                continue
            # density=True: the two arms have different lambda>0 counts, so raw
            # counts would compare shapes on different scales.
            ax.hist(v, bins=edges, density=True, color=COLOR[opt], alpha=0.30,
                    linewidth=0, zorder=1)
            ax.hist(v, bins=edges, density=True, histtype="step",
                    color=COLOR[opt], linewidth=1.6, label=opt, zorder=3)
            ax.axvline(np.median(v), color=COLOR[opt], linewidth=1.1,
                       linestyle=(0, (4, 2)), zorder=4)
        ax.set_title(f"chinchilla {chin:g}", fontsize=10, color=INK)
        ax.grid(True, alpha=0.2, linewidth=0.6)
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.set_yticks([])                     # density units carry no meaning here
        if i % ncol == 0:
            ax.set_ylabel("density", fontsize=9, color=MUTED)
    for j in range(len(chins), nrow * ncol):      # blank the unused cells
        axes[j // ncol][j % ncol].axis("off")

    # x labels go on the lowest VISIBLE panel of each column: with sharex, a
    # column whose bottom cell is blanked would otherwise lose its axis
    # entirely (the chinchilla-4 panel had no tick labels at all).
    for col in range(ncol):
        rows = [r for r in range(nrow) if r * ncol + col < len(chins)]
        if not rows:
            continue
        ax = axes[max(rows)][col]
        ax.set_xlabel(r"$h/\kappa$", fontsize=9.5, color=MUTED)
        ax.tick_params(labelbottom=True, labelsize=8, colors=MUTED)

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{size}: per-token $h/\\kappa$ distribution "
                 f"($\\lambda>0$ positions; dashed = median)",
                 fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf")


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--margins-dir", default="/mnt/localssd/margins-pt")
    p.add_argument("--out-dir", default=os.path.join(repo, "colm-moss-latex"))
    p.add_argument("--sizes", nargs="+", default=SIZES)
    p.add_argument("--bins", type=int, default=70)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for size in a.sizes:
        cells = load_size(a.margins_dir, size)
        if not cells:
            print(f"{size}: no per-token arrays found, skipping")
            continue
        n = sum(len(v) for v in cells.values())
        print(f"{size}: {n} config(s) over {len(cells)} chinchilla(s)")
        plot_size(size, cells, os.path.join(a.out_dir, f"margins-hist-{size}"),
                  bins=a.bins)


if __name__ == "__main__":
    main()
