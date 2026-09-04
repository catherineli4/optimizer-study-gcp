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


def load(size, cache):
    """{chinchilla: {optimizer: {batch: {lr: loss}}}}"""
    pats = patterns(size)
    hits = []
    for n in listing(size, cache):
        for rx, opt in pats:
            m = rx.match(n)
            if m:
                hits.append((n, opt, float(m.group(1)), float(m.group(2)),
                             m.group(3)))
                break
    d = fetch(size, [h[0] for h in hits], cache)
    out, ntok = {}, set()
    for n, opt, chin, lr, bs in hits:
        p = os.path.join(d, n + "-eval.json")
        if not os.path.exists(p):
            continue
        try:
            cell = json.load(open(p)).get("by_label", {}).get(LABEL)
        except (json.JSONDecodeError, OSError):
            continue
        if cell:
            # KEEP THE MINIMUM, never overwrite. A muon cell can have several
            # runs at the same muon_lr with different adamw components; plain
            # assignment let whichever was read last win, which reported a
            # non-optimal LR as "best".
            slot = out.setdefault(chin, {}).setdefault(opt, {}).setdefault(bs, {})
            if lr not in slot or cell["loss"] < slot[lr]:
                slot[lr] = cell["loss"]
            ntok.add(cell["num_tokens"])
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


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", default=["60M"])
    p.add_argument("--cache", default="/mnt/localssd/bscache")
    p.add_argument("--out-dir", default=os.path.join(repo, "colm-moss-latex"))
    a = p.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    for size in a.sizes:
        data, ntok = load(size, a.cache)
        n = sum(len(v) for c in data.values() for b in c.values() for v in b.values())
        print(f"{size}: {n} runs over {len(data)} chinchilla(s)")
        plot(size, data, ntok, os.path.join(a.out_dir, f"pt-lr-dclm-bs-{size}"))
        write_table(size, data,
                    os.path.join(a.out_dir, f"pt-lr-dclm-bs-{size}-besttable"))


if __name__ == "__main__":
    main()
