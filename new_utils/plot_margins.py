"""Plot h/kappa (the critical-noise ratio) across size x chinchilla x optimizer.

Two figures from the summary JSONs written by margin_stats_sweep:

  margins-h-kappa        h/kappa vs chinchilla, one panel per size, adamw vs muon.
                         Median line with the p10-p90 band, since h/kappa is a
                         per-token distribution and its spread is wider than the
                         between-optimizer gap.
  margins-h-kappa-delta  muon minus adamw vs chinchilla, all sizes on one panel
                         against a zero rule -- the crossover is the result, and
                         it is invisible when each size sits in its own panel.

sigma* = f(h/kappa) with f strictly increasing, so higher h/kappa = larger
critical weight noise = more robust.

    python -m new_utils.plot_margins --margins-dir /mnt/localssd/margins
"""

import argparse
import json
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Same adamw/muon hues as the rest of the figures. Validated on the six checks:
# light mode passes all, CVD dE 24.7 (target 8), normal-vision dE 33.6 (floor 15).
COLOR = {"adamw": "#2a78d6", "muon": "#eb6834"}
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#8c8b87"
SIZES = ["30M", "60M", "100M", "300M", "600M"]


def load(margins_dir):
    """{size: {chinchilla: {optimizer: summary}}}"""
    out = {}
    for f in glob.glob(os.path.join(margins_dir, "*", "*.json")):
        size = os.path.basename(os.path.dirname(f))
        name = os.path.basename(f)[:-5]
        m = re.search(r"chinchilla-([0-9.]+)-(adamw|muon)", name)
        if not m:
            print(f"unparsed, skipping: {name}")
            continue
        with open(f) as fh:
            out.setdefault(size, {}).setdefault(
                float(m.group(1)), {})[m.group(2)] = json.load(fh)
    return out


def plot_panels(data, out):
    sizes = [s for s in SIZES if s in data]
    fig, axes = plt.subplots(1, len(sizes), figsize=(2.9 * len(sizes), 4.3),
                             squeeze=False, sharey=True)
    for i, size in enumerate(sizes):
        ax = axes[0][i]
        for opt in ("adamw", "muon"):
            chins = sorted(c for c in data[size] if opt in data[size][c])
            if not chins:
                continue
            mean = [data[size][c][opt]["h_over_kappa_mean"] for c in chins]
            med = [data[size][c][opt]["h_over_kappa_median"] for c in chins]
            lo = [data[size][c][opt]["h_over_kappa_p10"] for c in chins]
            hi = [data[size][c][opt]["h_over_kappa_p90"] for c in chins]
            ax.fill_between(chins, lo, hi, color=COLOR[opt], alpha=0.13,
                            linewidth=0, zorder=1)
            # Mean and median both shown: h/kappa is right-skewed (see the
            # histograms), so the mean sits above the median and the two can
            # in principle order the optimizers differently.
            ax.plot(chins, mean, "o-", color=COLOR[opt], linewidth=1.8,
                    markersize=5, label=f"{opt} mean", zorder=3)
            ax.plot(chins, med, "--", color=COLOR[opt], linewidth=1.3,
                    alpha=0.85, label=f"{opt} median", zorder=2)
        ax.set_xscale("log")
        # A narrow chinchilla span (300M, 600M) leaves the log minor ticks
        # (2x, 3x, 4x, 6x) close enough to collide into unreadable mush.
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.xaxis.set_major_locator(mticker.LogLocator(numticks=4))
        ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())
        ax.set_title(f"{size}", fontsize=11, color=INK)
        ax.set_xlabel("chinchilla", fontsize=9, color=MUTED)
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.tick_params(labelsize=8, colors=MUTED)
        if i == 0:
            ax.set_ylabel(r"$h/\kappa$   (mean, median, p10–p90)", fontsize=10, color=INK)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(r"Critical-noise ratio $h/\kappa$ on held-out DCLM   —   "
                 r"higher = larger $\sigma^*$ = more robust",
                 fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf")


def plot_delta(data, out):
    sizes = [s for s in SIZES if s in data]
    # Model size is ORDERED, so it gets a single-hue sequential ramp (light ->
    # dark), not categorical hues.
    ramp = plt.cm.viridis([0.08, 0.3, 0.5, 0.68, 0.86])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0, color=RULE, linewidth=1.2, zorder=2)
    for i, size in enumerate(sizes):
        chins, d = [], []
        for c in sorted(data[size]):
            cell = data[size][c]
            if "adamw" in cell and "muon" in cell:
                chins.append(c)
                d.append(cell["muon"]["h_over_kappa_mean"]
                         - cell["adamw"]["h_over_kappa_mean"])
        if chins:
            ax.plot(chins, d, "o-", color=ramp[i], linewidth=1.9, markersize=5.5,
                    label=size, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("chinchilla (token budget)", fontsize=10, color=MUTED)
    ax.set_ylabel(r"$h/\kappa$:  muon $-$ adamw", fontsize=10.5, color=INK)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.tick_params(labelsize=9, colors=MUTED)
    # Legend below the axes: inside the plot it collided with the sign
    # annotations, which sit in the only two free corners.
    ax.legend(frameon=False, fontsize=9.5, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.13))
    # Name what each side of the zero rule means, so the sign is not left to
    # the reader to reconstruct from the formula.
    ax.text(0.985, 0.965, "muon more robust", transform=ax.transAxes,
            fontsize=9, color=COLOR["muon"], va="top", ha="right")
    ax.text(0.985, 0.035, "adamw more robust", transform=ax.transAxes,
            fontsize=9, color=COLOR["adamw"], va="bottom", ha="right")
    ax.set_title(r"Crossover in $h/\kappa$ with token budget",
                 fontsize=12.5, color=INK, pad=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf")


def write_csv(data, out):
    with open(f"{out}.csv", "w") as fh:
        fh.write("model_size,chinchilla,optimizer,h_over_kappa_mean,"
                 "h_over_kappa_median,p10,p90,h_mean,kappa_mean,"
                 "top1_accuracy,frac_lambda_positive,n_positions\n")
        for size in [s for s in SIZES if s in data]:
            for c in sorted(data[size]):
                for opt in ("adamw", "muon"):
                    s = data[size][c].get(opt)
                    if not s:
                        continue
                    fh.write(f"{size},{c:g},{opt},{s['h_over_kappa_mean']:.4f},"
                             f"{s['h_over_kappa_median']:.4f},{s['h_over_kappa_p10']:.4f},"
                             f"{s['h_over_kappa_p90']:.4f},{s['h_mean']:.4f},"
                             f"{s['kappa_mean']:.4f},{s['top1_accuracy']:.4f},"
                             f"{s['frac_lambda_positive']:.4f},{s['n_positions']}\n")
    print(f"wrote {out}.csv")


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--margins-dir", default="/mnt/localssd/margins")
    p.add_argument("--out-dir", default=os.path.join(repo, "colm-moss-latex"))
    a = p.parse_args()
    data = load(a.margins_dir)
    n = sum(len(v) for s in data.values() for v in s.values())
    print(f"{n} cells over {len(data)} size(s)")
    os.makedirs(a.out_dir, exist_ok=True)
    plot_panels(data, os.path.join(a.out_dir, "margins-h-kappa"))
    plot_delta(data, os.path.join(a.out_dir, "margins-h-kappa-delta"))
    write_csv(data, os.path.join(a.out_dir, "margins-h-kappa"))


if __name__ == "__main__":
    main()
