"""One best-LR table across model size x chinchilla x batch x optimizer.

plot_pt_lr_dclm_bs.py writes one table per size; this stacks all of them into
a single lookup so the LR trends across sizes are readable in one place.
Reads the per-size *-besttable.csv it already wrote, so the numbers here and
in those figures can never disagree.

The muon column holds muon_lr (its adamw component is a separate axis and is
not tuned here) -- the same convention as the per-size tables.

    python -m new_utils.plot_best_lr_table --sizes 30M 60M 100M
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BATCHES = ["1M", "2M", "4M"]
OPTS = ["adamw", "muon"]
INK, MUTED = "#0b0b0b", "#52514e"
TINT = {"adamw": "#eaf1fb", "muon": "#fdeee7"}


def load(sizes, out_dir):
    """{(size, chinchilla): {(opt, batch): (lr, loss, n_lrs)}}, plus row order."""
    cells, rows = {}, []
    for size in sizes:
        p = os.path.join(out_dir, f"pt-lr-dclm-bs-{size}-besttable.csv")
        if not os.path.exists(p):
            print(f"{size}: no {os.path.basename(p)}, skipping "
                  f"(run plot_pt_lr_dclm_bs first)")
            continue
        seen = set()
        for r in csv.DictReader(open(p)):
            chin = float(r["chinchilla"])
            key = (size, chin)
            cells.setdefault(key, {})[(r["optimizer"], r["batch"])] = (
                float(r["best_lr"]), float(r["dclm_loss"]), int(r["n_lrs"]))
            seen.add(chin)
        rows += [(size, c) for c in sorted(seen)]
    return cells, rows


def write(cells, rows, sizes, out, with_loss):
    cols = [(o, b) for o in OPTS for b in BATCHES]

    with open(f"{out}.csv", "w") as fh:
        fh.write("model_size,chinchilla,optimizer,batch,best_lr,dclm_loss,n_lrs\n")
        for size, chin in rows:
            for opt, bs in cols:
                v = cells[(size, chin)].get((opt, bs))
                if v:
                    fh.write(f"{size},{chin:g},{opt},{bs},"
                             f"{v[0]:g},{v[1]:.4f},{v[2]}\n")
    print(f"wrote {out}.csv")

    text, colour = [], []
    for size, chin in rows:
        trow, crow = [], []
        for opt, bs in cols:
            v = cells[(size, chin)].get((opt, bs))
            if not v:
                trow.append("--")
                crow.append("#ffffff")
            else:
                trow.append(f"{v[0]:g}\n{v[1]:.3f}" if with_loss else f"{v[0]:g}")
                # A cell tuned on <=3 LRs is thin evidence; flag rather than
                # silently presenting it as equal to a 7-point sweep.
                crow.append("#fff8dc" if v[2] <= 3 else TINT[opt])
        text.append(trow)
        colour.append(crow)

    # Size the figure from the row count and then force the table to fill the
    # axes: matplotlib centres a table at its natural height, which on a tall
    # figure leaves most of the canvas blank.
    row_in = 0.40 if with_loss else 0.28
    fig, ax = plt.subplots(figsize=(1.25 * len(cols) + 2.4,
                                    row_in * (len(rows) + 1) + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=text, cellColours=colour,
                   rowLabels=[f"{s}  c{c:g}" for s, c in rows],
                   colLabels=[f"{o}\nbs {b}" for o, b in cols],
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    cell_h = 1.0 / (len(rows) + 1)
    for (r, c), cl in tbl.get_celld().items():
        cl.set_height(cell_h)
        cl.set_edgecolor("#d9d8d3")
        if r == 0 or c == -1:
            cl.set_text_props(fontweight="bold", color=INK)
        # No size-block separators: every row label already carries its size,
        # and a ruled-off row reads as emphasis on that row rather than as a
        # boundary before it.
    sub = "best pretrain LR" + (" / its DCLM loss" if with_loss else "")
    ax.set_title(f"Best pretrain LR by size, token budget, batch and optimizer\n"
                 f"cell = {sub};  muon = muon_lr;  wd 0.1;  "
                 f"cream = \u22643 LRs swept",
                 fontsize=10.5, color=INK, pad=12)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf")


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", default=["30M", "60M", "100M"])
    p.add_argument("--out-dir", default=os.path.join(repo, "colm-moss-latex"))
    p.add_argument("--no-loss", action="store_true",
                   help="LR only, no second line")
    a = p.parse_args()
    cells, rows = load(a.sizes, a.out_dir)
    if not rows:
        return
    print(f"{len(rows)} (size, chinchilla) row(s) over {len(cells)} cell group(s)")
    write(cells, rows, a.sizes,
          os.path.join(a.out_dir, "best-lr-by-size-chin-bs-opt"), not a.no_loss)


if __name__ == "__main__":
    main()
