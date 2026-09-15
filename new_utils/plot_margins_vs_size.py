"""h/kappa as a function of MODEL SIZE, one chinchilla per row.

The transpose of margins-h-kappa-{median,mean}: those hold size fixed per panel
and sweep the token budget, which answers "at what budget does adamw overtake
muon". Holding the budget fixed per row and sweeping size instead answers the
other question -- whether that overtake also happens as models get larger at a
fixed budget, and at which size.

One row per chinchilla, adamw vs muon, x = non-embedding-free parameter count on
a log axis. The dashed rule marks the first size at which adamw's h/kappa
exceeds muon's, exactly as in the by-chinchilla figures, and is annotated "not
sustained" when the ordering reverts at a larger size.

Rows with fewer than two paired sizes are dropped: a single point cannot show a
trend, and c=128 exists at 60M alone.

    python -m new_utils.plot_margins_vs_size --margins-dir /mnt/localssd/margins
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from new_utils.plot_margins import COLOR, INK, MUTED, SIZES, load

# Total parameter counts behind the size labels, so the x axis is a real
# quantity on a log scale rather than five equally spaced categories -- 30M to
# 600M is a 20x span and spacing it evenly would misstate every slope.
SIZE_PARAMS = {"30M": 0.03e9, "60M": 0.06e9, "100M": 0.1e9,
               "300M": 0.3e9, "600M": 0.6e9}

# Annotations land inside the axes and the y ranges here are tight, so they
# overlap the curves without a ground behind them.
BOX = dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82,
           edgecolor="none")


def _rows(data, stat, min_sizes=2):
    """[(chinchilla, {optimizer: (xs, ys)}, [sizes paired])] for plottable rows."""
    by_chin = {}
    for size in [s for s in SIZES if s in data]:
        for chin, cell in data[size].items():
            if "adamw" in cell and "muon" in cell:
                by_chin.setdefault(chin, {})[size] = cell

    rows = []
    for chin in sorted(by_chin):
        sizes = [s for s in SIZES if s in by_chin[chin]]
        if len(sizes) < min_sizes:
            print(f"c={chin:g}: only {len(sizes)} paired size(s), skipping row")
            continue
        series = {
            opt: ([SIZE_PARAMS[s] for s in sizes],
                  [by_chin[chin][s][opt][f"h_over_kappa_{stat}"] for s in sizes])
            for opt in ("adamw", "muon")
        }
        rows.append((chin, series, sizes))
    return rows


def plot(data, out, stat="median", ncol=2):
    rows = _rows(data, stat)
    if not rows:
        print("nothing to plot")
        return
    nrow = (len(rows) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.9 * nrow),
                             squeeze=False, sharex=True)

    for i, (chin, series, sizes) in enumerate(rows):
        ax = axes[i // ncol][i % ncol]
        for opt in ("adamw", "muon"):
            xs, ys = series[opt]
            ax.plot(xs, ys, "o-", color=COLOR[opt], linewidth=1.9,
                    markersize=5.5, label=opt, zorder=3)

        # First size where adamw overtakes muon -- the same event the
        # by-chinchilla figures mark, read along the other axis.
        flags = [a > m for a, m in zip(series["adamw"][1], series["muon"][1])]
        cross = next((k for k, f in enumerate(flags) if f), None)
        if cross is not None:
            ax.axvline(SIZE_PARAMS[sizes[cross]], color=INK, linewidth=1.2,
                       linestyle=(0, (5, 3)), alpha=0.75, zorder=2)
            txt = f"adamw ahead\nfrom {sizes[cross]}"
            if not all(flags[cross:]):
                txt += "\n(not sustained)"
            ax.annotate(txt, xy=(SIZE_PARAMS[sizes[cross]], 0.97),
                        xycoords=("data", "axes fraction"),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=INK, va="top", ha="left",
                        bbox=BOX)
        else:
            ax.annotate("muon ahead throughout", xy=(0.5, 0.97),
                        xycoords="axes fraction", fontsize=8,
                        color=COLOR["muon"], va="top", ha="center",
                        bbox=BOX)

        ax.set_xscale("log")
        # Label the five real model sizes; the default log locator would put
        # decade ticks at 1e8 where no model exists.
        ax.set_xticks([SIZE_PARAMS[s] for s in SIZES])
        ax.set_xticklabels(SIZES)
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_title(f"chinchilla {chin:g}", fontsize=10.5, color=INK)
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.tick_params(labelsize=8, colors=MUTED)
        if i % ncol == 0:
            ax.set_ylabel(f"{stat} " + r"$h/\kappa$", fontsize=10, color=INK)

    for j in range(len(rows), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    # With sharex, a column whose bottom cell is blanked loses its tick labels,
    # so put the x label on the lowest VISIBLE panel of each column.
    for col in range(ncol):
        vis = [r for r in range(nrow) if r * ncol + col < len(rows)]
        if not vis:
            continue
        ax = axes[max(vis)][col]
        ax.set_xlabel("model size (parameters)", fontsize=9.5, color=MUTED)
        ax.tick_params(labelbottom=True, labelsize=8, colors=MUTED)

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(f"{stat.capitalize()} " + r"$h/\kappa$ vs model size at fixed "
                 r"token budget   —   dashed rule: first size where adamw "
                 r"overtakes muon", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf")


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--margins-dir", default="/mnt/localssd/margins")
    p.add_argument("--out-dir", default=os.path.join(repo, "colm-moss-latex"))
    p.add_argument("--ncol", type=int, default=2)
    a = p.parse_args()
    data = load(a.margins_dir)
    os.makedirs(a.out_dir, exist_ok=True)
    for stat in ("median", "mean"):
        plot(data, os.path.join(a.out_dir, f"margins-h-kappa-vs-size-{stat}"),
             stat, ncol=a.ncol)


if __name__ == "__main__":
    main()
