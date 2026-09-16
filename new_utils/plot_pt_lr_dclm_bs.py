"""Pretrain LR vs held-out DCLM loss across GLOBAL BATCH SIZE.

Companion to plot_pt_lr_dclm_all.py, which fixes the batch at the 1M reference.
Here the batch is the extra axis: rows = optimizer, columns = chinchilla, one
line per batch size. PTSweep-named runs only, since the batch variants exist
only under that schema.

Note the LR axis is the CELL learning rate, which the sweep scales by sqrt of
the batch multiplier — so a bs4M curve sits at 2x the LRs of its bs1M
counterpart by construction, not by accident.

    python -m new_utils.plot_pt_lr_dclm_bs --sizes 60M
"""

import argparse
import collections
import json
import os
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BUCKET = "gs://cmu-gpucloud-catheri4"
LABEL = "DCLM_heldout"
MODEL_TYPE = {"30M": "0.03B", "60M": "0.06B", "100M": "0.1B",
              "300M": "0.3B", "600M": "0.6B"}
INK, MUTED = "#0b0b0b", "#52514e"
# Batch size is an ordered magnitude -> single-hue sequential ramp, light to
# dark, rather than categorical colors.
BATCHES = ["1M", "2M", "4M"]


def patterns(size):
    mt = re.escape(MODEL_TYPE[size])
    return [
        (re.compile(rf"^PTSweep{size}-{mt}-chinchilla-([0-9.]+)-adamw"
                    rf"-lr([0-9.e+\-]+)-wd0\.1-bs(\d+M)-wsd$"), "adamw"),
        (re.compile(rf"^PTSweep{size}-{mt}-chinchilla-([0-9.]+)-muon"
                    rf"-muonlr([0-9.e+\-]+)-adamwlr[0-9.e+\-]+"
                    rf"-wd0\.1-bs(\d+M)-wsd$"), "muon"),
    ]


def listing(size, cache):
    p = os.path.join(cache, f"names_{size}.txt")
    if not os.path.exists(p):
        out = subprocess.run(
            ["gsutil", "ls", f"{BUCKET}/Optim-{size}-tuning/ModelEvaluation/"],
            capture_output=True, text=True).stdout
        names = [l.strip().rsplit("/", 1)[-1].replace("-eval.json", "")
                 for l in out.splitlines() if l.strip().endswith("-eval.json")]
        open(p, "w").write("\n".join(names))
    return [n for n in open(p).read().split() if n]


def fetch(size, names, cache):
    d = os.path.join(cache, size)
    os.makedirs(d, exist_ok=True)
    todo = [n for n in names if not os.path.exists(os.path.join(d, n + "-eval.json"))]
    for i in range(0, len(todo), 200):
        urls = [f"{BUCKET}/Optim-{size}-tuning/ModelEvaluation/{n}-eval.json"
                for n in todo[i:i + 200]]
        subprocess.run(["gsutil", "-m", "cp"] + urls + [d],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return d


def load(size, cache, main_cache="/mnt/localssd/evalcache", sync_main=True):
    """{chinchilla: {optimizer: {batch: {lr: loss}}}}

    The batch-1M series is taken from plot_pt_lr_dclm_all.load VERBATIM -- both
    naming schemas, the tuned-adamw-component filter for muon, the per-LR
    schema mean, EXCLUDE and SKIP_CELLS -- so the 1M points here are the same
    points, by construction, as the size x chinchilla grid. The 2M/4M runs
    exist only under the PTSweep schema and each (chinchilla, batch) was run
    at a single adamw component, so they need no filter; they get the same
    per-LR mean (never the min) and the same exclusions.
    """
    from new_utils import plot_pt_lr_dclm_all as A
    out, ntok = {}, set()

    # --- batch 1M: the main grid's own loader ---------------------------------
    d1 = (A.sync(size, main_cache) if sync_main
          else os.path.join(main_cache, size))
    main, ntok1 = A.load(size, d1, tuned_component=A.tuned_adamw_lrs(size))
    ntok |= ntok1
    for chin, per_opt in main.items():
        for opt, series in per_opt.items():
            out.setdefault(chin, {}).setdefault(opt, {})["1M"] = {
                lr: sum(v for _, v in runs) / len(runs)
                for lr, runs in series.items()}

    # --- batches 2M / 4M: PTSweep names, same rules -----------------------------
    pats = patterns(size)
    hits = []
    for n in listing(size, cache):
        for rx, opt in pats:
            m = rx.match(n)
            if m and m.group(3) != "1M":
                hits.append((n, opt, float(m.group(1)), float(m.group(2)),
                             m.group(3)))
                break
    d = fetch(size, [h[0] for h in hits], cache)
    acc = {}
    for n, opt, chin, lr, bs in hits:
        if (size, chin) in A.SKIP_CELLS or (size, chin, opt, lr) in A.EXCLUDE:
            continue
        p = os.path.join(d, n + "-eval.json")
        if not os.path.exists(p):
            continue
        try:
            cell = json.load(open(p)).get("by_label", {}).get(LABEL)
        except (json.JSONDecodeError, OSError):
            continue
        if cell:
            acc.setdefault((chin, opt, bs), {}).setdefault(lr, []).append(cell["loss"])
            ntok.add(cell["num_tokens"])
    for (chin, opt, bs), series in acc.items():
        out.setdefault(chin, {}).setdefault(opt, {})[bs] = {
            lr: sum(v) / len(v) for lr, v in series.items()}
    return out, ntok


def plot(size, data, ntok, out):
    chins = [c for c in sorted(data)
             if any(len(data[c].get(o, {})) > 1 for o in ("adamw", "muon"))]
    if not chins:
        print(f"{size}: no chinchilla has more than one batch size, skipping")
        return
    ramp = plt.cm.Blues([0.42, 0.68, 0.95])
    fig, axes = plt.subplots(2, len(chins), figsize=(3.5 * len(chins), 6.4),
                             squeeze=False)
    for r, opt in enumerate(("adamw", "muon")):
        for c, chin in enumerate(chins):
            ax = axes[r][c]
            series = data[chin].get(opt, {})
            for bi, bs in enumerate(BATCHES):
                pts = sorted(series.get(bs, {}).items())
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                            color=ramp[bi], markersize=4.5, linewidth=1.7,
                            label=f"batch {bs}", zorder=3)
                    blr, bval = min(pts, key=lambda p: p[1])
                    ax.scatter([blr], [bval], s=120, facecolors="none",
                               edgecolors=ramp[bi], linewidths=1.8, zorder=4)
            ax.set_xscale("log")
            ax.set_title(f"chinchilla {chin:g} — {opt}", fontsize=10, color=INK)
            ax.grid(True, alpha=0.25, linewidth=0.6)
            ax.tick_params(labelsize=8, colors=MUTED)
            if r == 1:
                ax.set_xlabel("cell LR  (muon: muon_lr)", fontsize=9, color=MUTED)
            if c == 0:
                ax.set_ylabel(f"{LABEL} loss", fontsize=9.5, color=INK)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(h), frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.015))
    tok = f"{min(ntok):,}" if ntok else "?"
    fig.suptitle(f"{size} ({MODEL_TYPE[size]}), wd 0.1: pretrain LR vs held-out "
                 f"DCLM loss across global batch size  —  {tok} eval tokens/point",
                 fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.015, 1, 0.985))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf")


def write_table(size, data, out):
    """Best LR per (chinchilla x batch x optimizer) as CSV and a rendered table.

    The CSV is the machine-readable copy; the figure is for dropping straight
    into the paper next to pt-lr-dclm-bs-*.
    """
    cols = [(o, b) for o in ("adamw", "muon") for b in BATCHES]
    chins = sorted(data)

    with open(f"{out}.csv", "w") as fh:
        fh.write("chinchilla,optimizer,batch,best_lr,dclm_loss,n_lrs\n")
        for chin in chins:
            for opt, bs in cols:
                s = data[chin].get(opt, {}).get(bs, {})
                if not s:
                    continue
                lr, v = min(s.items(), key=lambda kv: kv[1])
                fh.write(f"{chin:g},{opt},{bs},{lr:g},{v:.4f},{len(s)}\n")
    print(f"wrote {out}.csv")

    cell, colour = [], []
    for chin in chins:
        row, crow = [], []
        for opt, bs in cols:
            s = data[chin].get(opt, {}).get(bs, {})
            if not s:
                row.append("--")
                crow.append("#ffffff")
            else:
                lr, v = min(s.items(), key=lambda kv: kv[1])
                row.append(f"{lr:g}\n{v:.3f}")
                # Tint by optimizer so the two halves read as blocks.
                crow.append("#eaf1fb" if opt == "adamw" else "#fdeee7")
        cell.append(row)
        colour.append(crow)

    fig, ax = plt.subplots(figsize=(1.35 * len(cols) + 1.6, 0.62 * len(chins) + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=cell, cellColours=colour,
                   rowLabels=[f"{c:g}" for c in chins],
                   colLabels=[f"{o}\nbs {b}" for o, b in cols],
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 2.0)
    for (r, c), cl in tbl.get_celld().items():
        cl.set_edgecolor("#d9d8d3")
        if r == 0 or c == -1:
            cl.set_text_props(fontweight="bold", color=INK)
    ax.set_title(f"{size} ({MODEL_TYPE[size]}) best pretrain LR by batch size "
                 f"and token budget\ncell = best LR / its {LABEL} loss; "
                 f"row = chinchilla; wd 0.1",
                 fontsize=11.5, color=INK, pad=16)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf")


def plot_grid(all_data, all_ntok, out, chin_range=(1.0, 4.0)):
    """Every size on one page: rows = (size, optimizer), columns = chinchilla,
    one line per global batch size in each cell -- the same panel plot() draws,
    built from the same loaded data. Only chinchillas with more than one batch
    size at that size get a cell; axes are independent per cell because loss
    levels differ by >1 nat across sizes and the LR range moves with size.
    """
    sizes = [s for s in MODEL_TYPE if s in all_data]
    rows = []
    for size in sizes:
        chins = [c for c in sorted(all_data[size])
                 if chin_range[0] <= c <= chin_range[1]
                 and any(len(all_data[size][c].get(o, {})) > 1
                         for o in ("adamw", "muon"))]
        if chins:
            for opt in ("adamw", "muon"):
                rows.append((size, opt, chins))
    if not rows:
        print("grid: no size has a multi-batch chinchilla, skipping")
        return
    chins_all = sorted({c for _, _, cs in rows for c in cs})
    ramp = plt.cm.Blues([0.42, 0.68, 0.95])
    fig, axes = plt.subplots(len(rows), len(chins_all),
                             figsize=(3.1 * len(chins_all), 2.7 * len(rows)),
                             squeeze=False)
    for r, (size, opt, chins) in enumerate(rows):
        first = None
        for c, chin in enumerate(chins_all):
            ax = axes[r][c]
            series = all_data[size].get(chin, {}).get(opt, {})
            if chin not in chins or not series:
                ax.axis("off")
                continue
            first = c if first is None else first
            for bi, bs in enumerate(BATCHES):
                pts = sorted(series.get(bs, {}).items())
                if not pts:
                    continue
                ax.plot([q[0] for q in pts], [q[1] for q in pts], "o-",
                        color=ramp[bi], markersize=4, linewidth=1.5,
                        label=f"batch {bs}", zorder=3)
                blr, bval = min(pts, key=lambda q: q[1])
                ax.scatter([blr], [bval], s=95, facecolors="none",
                           edgecolors=ramp[bi], linewidths=1.6, zorder=4)
            ax.set_xscale("log")
            # A narrow LR span leaves the log minor labels (2x, 3x, 4x, 6x)
            # colliding into mush; label majors only and cap their count.
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            ax.xaxis.set_major_locator(mticker.LogLocator(numticks=3))
            ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.tick_params(labelsize=7, colors=MUTED)
            if r == 0 or rows[r - 1][0] != size:
                ax.set_title(f"chinchilla {chin:g}", fontsize=9.5, color=INK)
            if r == len(rows) - 1:
                ax.set_xlabel("cell LR  (muon: muon_lr)", fontsize=8, color=MUTED)
        if first is not None:
            axes[r][first].set_ylabel(f"{size} ({MODEL_TYPE[size]}) — {opt}\n"
                                      f"{LABEL} loss", fontsize=8.5, color=INK)
    handles = [plt.Line2D([], [], color=ramp[i], marker="o", markersize=4,
                          linewidth=1.5, label=f"batch {b}")
               for i, b in enumerate(BATCHES)]
    handles.append(plt.Line2D([], [], color=MUTED, marker="o", markersize=8,
                              markerfacecolor="none", linestyle="none",
                              label="best LR"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    toks = {t for s in sizes for t in all_ntok.get(s, ())}
    tok = f"{min(toks):,}" if toks else "?"
    fig.suptitle(f"Pretrain LR vs held-out DCLM loss across global batch size, "
                 f"chinchilla {chin_range[0]:g}–{chin_range[1]:g} (wd 0.1)  —  "
                 f"{tok} eval tokens/point",
                 fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.975))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf  ({len(rows)} rows x {len(chins_all)} cols)")


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", default=["30M", "60M", "100M", "600M"])
    p.add_argument("--grid-chinchillas", nargs=2, type=float, default=[1, 4],
                   metavar=("LO", "HI"),
                   help="chinchilla range shown in the combined grid (default 1 4)")
    p.add_argument("--refresh", action="store_true",
                   help="re-list GCS instead of using the cached names_<size>.txt")
    p.add_argument("--cache", default="/mnt/localssd/bscache")
    p.add_argument("--out-dir", default=os.path.join(repo, "colm-moss-latex"))
    a = p.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    all_data, all_ntok = {}, {}
    for size in a.sizes:
        if a.refresh:
            lst = os.path.join(a.cache, f"names_{size}.txt")
            if os.path.exists(lst):
                os.remove(lst)
        data, ntok = load(size, a.cache)
        n = sum(len(v) for c in data.values() for b in c.values() for v in b.values())
        print(f"{size}: {n} runs over {len(data)} chinchilla(s)")
        if not data:
            continue
        all_data[size], all_ntok[size] = data, ntok
        plot(size, data, ntok, os.path.join(a.out_dir, f"pt-lr-dclm-bs-{size}"))
        write_table(size, data,
                    os.path.join(a.out_dir, f"pt-lr-dclm-bs-{size}-besttable"))
    plot_grid(all_data, all_ntok, os.path.join(a.out_dir, "pt-lr-dclm-bs-grid"),
              chin_range=tuple(a.grid_chinchillas))


if __name__ == "__main__":
    main()
