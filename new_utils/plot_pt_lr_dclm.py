"""Pretrain LR vs held-out DCLM loss for the 60M adamw bs1M/wd0.1 sweep.

Reads the ModelEvaluation JSONs (already downloaded locally, or fetched from
GCS with --fetch) and writes two figures:

  pt-lr-dclm-by-chinchilla.png   one subplot per chinchilla, LR (log x) vs loss
  pt-lr-dclm-best.png            loss of the best-LR model vs chinchilla

Both naming schemas are plotted as separate series: a config trained under both
``MuonExpt3-...`` and ``PTSweep...-wd0.1-bs1M`` is two independent runs of the
same recipe, and their spread is the sweep's noise floor.
"""

import argparse
import glob
import json
import os
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# dataviz categorical slots 1 (blue) and 2 (orange); ink stays neutral.
COLOR = {"MuonExpt3": "#2a78d6", "PTSweep": "#eb6834"}
INK, MUTED, BEST = "#0b0b0b", "#52514e", "#008300"
LABEL = "DCLM_heldout"

RUN_RE = re.compile(
    r"^(MuonExpt3|PTSweep\w*?)-[\d.]+B-chinchilla-([\d.]+)-adamw"
    r"-lr([0-9.e+\-]+?)-(?:wd0\.1-bs1M-)?wsd-eval\.json$")


def load(eval_dir):
    """{chinchilla: {(lr, schema): loss}} plus the token count evals used."""
    data, ntok = {}, set()
    for path in sorted(glob.glob(os.path.join(eval_dir, "*-eval.json"))):
        m = RUN_RE.match(os.path.basename(path))
        if not m:
            continue
        schema, chin, lr = m.group(1), float(m.group(2)), float(m.group(3))
        cell = json.load(open(path)).get("by_label", {}).get(LABEL)
        if cell is None:
            continue
        schema = "MuonExpt3" if schema == "MuonExpt3" else "PTSweep"
        data.setdefault(chin, {})[(lr, schema)] = cell["loss"]
        ntok.add(cell["num_tokens"])
    return data, ntok


def fetch(eval_dir, bucket, runs_file):
    os.makedirs(eval_dir, exist_ok=True)
    urls = [f"{bucket}/ModelEvaluation/{r.strip()}-eval.json"
            for r in open(runs_file) if r.strip()]
    subprocess.run(["gsutil", "-m", "cp"] + urls + [eval_dir],
                   stderr=subprocess.DEVNULL)


def plot_grid(data, ntok, out):
    chins = sorted(data)
    ncol = 5
    nrow = -(-len(chins) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.4 * nrow),
                             squeeze=False)
    for i, chin in enumerate(chins):
        ax = axes[i // ncol][i % ncol]
        cell = data[chin]
        for schema in ("MuonExpt3", "PTSweep"):
            pts = sorted((lr, v) for (lr, s), v in cell.items() if s == schema)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                        color=COLOR[schema], markersize=4.5, linewidth=1.6,
                        label=schema, zorder=3)
        (blr, _), bval = min(cell.items(), key=lambda kv: kv[1])
        ax.scatter([blr], [bval], s=150, facecolors="none", edgecolors=BEST,
                   linewidths=2, zorder=4)
        # Above the marker: the best point is always at the bottom of the
        # range, so a label below it would land on the x-axis.
        ax.annotate(f"lr {blr:.3g}\n{bval:.3f}", (blr, bval),
                    textcoords="offset points", xytext=(0, 14),
                    ha="center", va="bottom", fontsize=8, color=BEST)
        ax.margins(y=0.22)
        ax.set_xscale("log")
        ax.set_title(f"chinchilla = {chin:g}   ({len(cell)} runs)",
                     fontsize=11, color=INK)
        ax.set_xlabel("pretrain LR", fontsize=9, color=MUTED)
        if i % ncol == 0:
            ax.set_ylabel(f"{LABEL} loss", fontsize=10, color=INK)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=8, colors=MUTED)
    for j in range(len(chins), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    # One figure-level legend: not every chinchilla has both schemas, so a
    # per-axes legend would silently drop a series.
    handles = [plt.Line2D([], [], color=COLOR[s], marker="o", markersize=4.5,
                          linewidth=1.6, label=s) for s in COLOR]
    handles.append(plt.Line2D([], [], color=BEST, marker="o", markersize=9,
                              markerfacecolor="none", markeredgewidth=2,
                              linestyle="none", label="best LR"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    tok = f"{min(ntok):,}" if ntok else "?"
    fig.suptitle("60M adamw (wd 0.1, batch 1M): pretrain LR vs held-out DCLM "
                 f"loss  —  {tok} eval tokens per point", fontsize=14, color=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):          # pdf is the vector copy for LaTeX
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf")


def plot_best(data, out):
    chins = sorted(data)
    best = [min(data[c].items(), key=lambda kv: kv[1]) for c in chins]
    losses = [b[1] for b in best]
    lrs = [b[0][0] for b in best]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(chins, losses, "o-", color=COLOR["MuonExpt3"], markersize=7,
            linewidth=2, zorder=3)
    for c, loss, lr in zip(chins, losses, lrs):
        ax.annotate(f"lr {lr:.3g}", (c, loss), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_xticks(chins)
    ax.set_xticklabels([f"{c:g}" for c in chins])
    ax.minorticks_off()
    ax.set_xlabel("chinchilla multiplier (token budget)", fontsize=11, color=INK)
    ax.set_ylabel(f"best {LABEL} loss", fontsize=11, color=INK)
    ax.set_title("60M adamw: loss of the best-LR model vs token budget\n"
                 "(annotation = the LR that won that budget)",
                 fontsize=12, color=INK)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=9, colors=MUTED)
    fig.tight_layout()
    for ext in ("png", "pdf"):          # pdf is the vector copy for LaTeX
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default="/mnt/localssd/allevals")
    p.add_argument("--out-dir",
                   default=os.path.join(os.path.dirname(os.path.dirname(
                       os.path.abspath(__file__))), "colm-moss-latex"))
    p.add_argument("--fetch", metavar="RUNS_FILE",
                   help="file of run names to download from GCS first")
    p.add_argument("--bucket",
                   default="gs://cmu-gpucloud-catheri4/Optim-60M-tuning")
    a = p.parse_args()

    if a.fetch:
        fetch(a.eval_dir, a.bucket, a.fetch)
    os.makedirs(a.out_dir, exist_ok=True)
    data, ntok = load(a.eval_dir)
    if not data:
        raise SystemExit(f"no matching eval JSONs in {a.eval_dir}")
    print(f"loaded {sum(len(v) for v in data.values())} runs over "
          f"{len(data)} chinchilla(s); eval tokens: {sorted(ntok)}")
    plot_grid(data, ntok, os.path.join(a.out_dir, "pt-lr-dclm-by-chinchilla"))
    plot_best(data, os.path.join(a.out_dir, "pt-lr-dclm-best"))


if __name__ == "__main__":
    main()
