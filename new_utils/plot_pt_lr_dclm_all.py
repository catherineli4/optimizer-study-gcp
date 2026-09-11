"""Pretrain LR vs held-out DCLM loss, adamw AND muon on the same axes.

One figure per model size; one subplot per chinchilla; x is the swept LR
(``learning_rate`` for adamw, ``muon_lr`` for muon, which is the axis each
sweep actually varies) on a log scale. Plus a cross-size summary of the
best-LR loss vs token budget.

Usage (evals must already exist on GCS):
    python -m new_utils.plot_pt_lr_dclm_all --sizes 30M 60M 100M 300M 600M
"""

import argparse
import json
import os
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# dataviz categorical slots: 1 blue = adamw, 2 orange = muon. Ink stays neutral.
COLOR = {"adamw": "#2a78d6", "muon": "#eb6834"}
INK, MUTED, BEST = "#0b0b0b", "#52514e", "#008300"
LABEL = "DCLM_heldout"
BUCKET = "gs://cmu-gpucloud-catheri4"

MODEL_TYPE = {"30M": "0.03B", "60M": "0.06B", "100M": "0.1B",
              "300M": "0.3B", "600M": "0.6B"}

# group 1 chinchilla, 2 swept LR. Both naming schemas, both optimizers.
def _patterns(size):
    mt = re.escape(MODEL_TYPE[size])
    return [
        (re.compile(rf"^MuonExpt3-{mt}-chinchilla-([0-9.]+)-adamw"
                    rf"-lr([0-9.e+\-]+)-wsd-eval\.json$"), "adamw"),
        (re.compile(rf"^PTSweep{size}-{mt}-chinchilla-([0-9.]+)-adamw"
                    rf"-lr([0-9.e+\-]+)-wd0\.1-bs1M-wsd-eval\.json$"), "adamw"),
        (re.compile(rf"^MuonExpt3-{mt}-chinchilla-([0-9.]+)-muon"
                    rf"-muonlr([0-9.e+\-]+)-adamwlr([0-9.e+\-]+)-wsd-eval\.json$"), "muon"),
        (re.compile(rf"^PTSweep{size}-{mt}-chinchilla-([0-9.]+)-muon"
                    rf"-muonlr([0-9.e+\-]+)-adamwlr([0-9.e+\-]+)"
                    rf"-wd0\.1-bs1M-wsd-eval\.json$"), "muon"),
    ]


def tuned_adamw_lrs(size):
    """PT_LR_BY_MODEL[model_type]["wsd"]["adamw"] read straight from the source.

    Parsed with ast rather than imported: pulling in pretraining_matrix would
    pull in the whole launcher (Project.init, GCS listings) just for a literal.
    """
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "launch_jolmo", "pretraining_matrix.py")
    import ast
    tree = ast.parse(open(src).read())
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
        if target == "PT_LR_BY_MODEL":
            table = ast.literal_eval(node.value)
            cells = table.get(MODEL_TYPE[size], {}).get("wsd", {}).get("adamw", {})
            return {float(k): float(v) for k, v in cells.items()}
    return {}


def tuned_lrs(size):
    """The table's declared optimum per optimizer: {opt: {chinchilla: lr}}.

    muon cells are ``(muon_lr, adamw_component_lr)``; the swept axis is
    ``muon_lr``, so that is what we key on.
    """
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "launch_jolmo", "pretraining_matrix.py")
    import ast
    tree = ast.parse(open(src).read())
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
        if target == "PT_LR_BY_MODEL":
            wsd = ast.literal_eval(node.value).get(MODEL_TYPE[size], {}).get("wsd", {})
            out = {}
            for opt in ("adamw", "muon"):
                cells = wsd.get(opt, {})
                out[opt] = {float(k): float(v[0] if isinstance(v, tuple) else v)
                            for k, v in cells.items()}
            return out
    return {"adamw": {}, "muon": {}}


def table_summary(size, data):
    """{opt: {chinchilla: (table_lr, loss)}} for the LRs the table declares.

    Unlike the measured summary this does NOT take the argmin — it reports the
    loss of whatever LR the table names, so the two figures can be compared.
    """
    want, out, missing = tuned_lrs(size), {}, []
    for opt in ("adamw", "muon"):
        for chin, lr in sorted(want.get(opt, {}).items()):
            series = data.get(chin, {}).get(opt, {})
            hit = next((v for x, v in series.items() if abs(x - lr) <= 1e-12), None)
            if hit is None:
                missing.append(f"{opt} c{chin:g} lr{lr:g}")
                continue
            out.setdefault(opt, {})[chin] = (lr, sum(v for _, v in hit) / len(hit))
    if missing:
        print(f"  {size}: no eval for table cell(s): {', '.join(missing)}")
    return out


def sync(size, cache):
    """Mirror the size's ModelEvaluation prefix locally (once)."""
    d = os.path.join(cache, size)
    os.makedirs(d, exist_ok=True)
    subprocess.run(
        ["gsutil", "-m", "rsync", "-x", r".*-CPT-.*|.*-EWC-.*|.*_perturbed_.*|.*-typo-.*",
         f"{BUCKET}/Optim-{size}-tuning/ModelEvaluation/", d],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return d


# Sizes where the MuonExpt3-named runs are NOT the same recipe as the PTSweep
# ones and must not be mixed in.
#
# 100M was listed here, wrongly. At overlapping LRs the two families' ADAMW
# runs agree to 0.009-0.024 nats (c2/c4 at lr 1e-2) — i.e. run-to-run noise,
# same recipe. The ~0.8-nat gap that prompted the exclusion was confined to
# MUON, and its cause is the adamw COMPONENT: those runs use adamwlr1.0e-3,
# far below the tuned value, which the component filter already excludes. The
# two families simply sweep disjoint LR ranges (MuonExpt3 3e-4..1.4e-2,
# PTSweep 1.4e-2..5.6e-2), so a steep loss curve between them looked like an
# offset when the schemas were drawn as separate lines.
FOREIGN_LEGACY = set()


def load(size, d, only_schema=None, tuned_component=None):
    """{chinchilla: {optimizer: {lr: [losses]}}} — a list per LR because the
    two naming schemas are independent runs of the same recipe."""
    pats, out, ntok = _patterns(size), {}, set()
    for fn in sorted(os.listdir(d)):
        for rx, opt in pats:
            m = rx.match(fn)
            if not m:
                continue
            cell = json.load(open(os.path.join(d, fn))).get(
                "by_label", {}).get(LABEL)
            if cell:
                schema = "MuonExpt3" if fn.startswith("MuonExpt3") else "PTSweep"
                if only_schema and schema != only_schema:
                    break
                # Keep only muon runs whose adamw COMPONENT is the tuned adamw
                # LR for that budget, so the muon curve is a clean muon_lr
                # sweep rather than a mix of component settings.
                if tuned_component is not None and opt == "muon":
                    want = tuned_component.get(float(m.group(1)))
                    # No tuned adamw LR at this budget means there is nothing to
                    # filter against — keep the run. Dropping it would make a
                    # whole chinchilla vanish from the figure just because the
                    # table has not been filled in for it yet.
                    if want is not None and abs(float(m.group(3)) - want) > 1e-12:
                        break
                (out.setdefault(float(m.group(1)), {})
                    .setdefault(opt, {})
                    .setdefault(float(m.group(2)), [])
                    .append((schema, cell["loss"])))
                ntok.add(cell["num_tokens"])
            break
    return out, ntok


SCHEMA_STYLE = {"MuonExpt3": dict(linestyle="-", marker="o"),
                "PTSweep": dict(linestyle="--", marker="^")}


def plot_size(size, data, ntok, out, split_schema=False):
    """One subplot per chinchilla.

    By default the two naming schemas are averaged per LR (they are independent
    runs of the same recipe) with their spread drawn as a vertical bar. With
    ``split_schema`` each schema becomes its own line — color still carries the
    optimizer, line style carries the schema.
    """
    chins = sorted(data)
    if not chins:
        return None
    ncol = min(5, len(chins))
    nrow = -(-len(chins) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.5 * nrow),
                             squeeze=False)
    summary = {}
    for i, chin in enumerate(chins):
        ax = axes[i // ncol][i % ncol]
        for opt in ("adamw", "muon"):
            series = data[chin].get(opt)
            if not series:
                continue
            lrs = sorted(series)
            if split_schema:
                for schema, st in SCHEMA_STYLE.items():
                    pts = sorted((x, v) for x in lrs
                                 for s, v in series[x] if s == schema)
                    if pts:
                        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                                color=COLOR[opt], markersize=4.5,
                                linewidth=1.6, zorder=3, **st)
            else:
                mean = [sum(v for _, v in series[x]) / len(series[x])
                        for x in lrs]
                ax.plot(lrs, mean, "o-", color=COLOR[opt], markersize=4.5,
                        linewidth=1.6, zorder=3)
                for x in lrs:
                    if len(series[x]) > 1:
                        vals = [v for _, v in series[x]]
                        ax.plot([x, x], [min(vals), max(vals)],
                                color=COLOR[opt], linewidth=1.0, alpha=0.55,
                                zorder=2)
            def _mean(x):
                return sum(v for _, v in series[x]) / len(series[x])
            blr = min(lrs, key=_mean)
            bval = _mean(blr)
            summary.setdefault(opt, {})[chin] = (blr, bval)
            ax.scatter([blr], [bval], s=140, facecolors="none",
                       edgecolors=COLOR[opt], linewidths=2.0, zorder=4)
            # adamw label above its marker, muon below: the two optima often sit
            # at a similar LR, and muon's is the lower curve.
            dy, va = ((13, "bottom") if opt == "adamw" else (-15, "top"))
            ax.annotate(f"{blr:.3g}", (blr, bval), textcoords="offset points",
                        xytext=(0, dy), ha="center", va=va, fontsize=8.5,
                        fontweight="bold", color=COLOR[opt], zorder=5)
        ax.set_xscale("log")
        # A narrow LR span leaves matplotlib's log minor ticks (2x, 3x, 4x,
        # 6x) close enough to collide into unreadable mush, as on c32. Label
        # only the majors and cap how many of those print.
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.xaxis.set_major_locator(mticker.LogLocator(numticks=4))
        ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())
        ax.set_title(f"chinchilla = {chin:g}", fontsize=11, color=INK)
        ax.set_xlabel("swept LR  (muon: muon_lr)", fontsize=8.5, color=MUTED)
        if i % ncol == 0:
            ax.set_ylabel(f"{LABEL} loss", fontsize=10, color=INK)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.margins(y=0.24)
    for j in range(len(chins), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles = [plt.Line2D([], [], color=COLOR[o], marker="o", markersize=4.5,
                          linewidth=1.6, label=o) for o in ("adamw", "muon")]
    if split_schema:
        handles += [plt.Line2D([], [], color=MUTED, linewidth=1.6,
                               markersize=4.5, label=s, **st)
                    for s, st in SCHEMA_STYLE.items()]
    handles.append(plt.Line2D([], [], color=MUTED, marker="o", markersize=9,
                              markerfacecolor="none", linestyle="none",
                              label="best LR"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.03))
    tok = f"{min(ntok):,}" if ntok else "?"
    note = " — naming schemas shown separately" if split_schema else ""
    fig.suptitle(f"{size} ({MODEL_TYPE[size]}), wd 0.1 / batch 1M: pretrain LR "
                 f"vs held-out DCLM loss  —  {tok} eval tokens per point{note}",
                 fontsize=14, color=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf")
    return summary


def plot_best(all_summary, out, title=None, ylabel=None):
    """Loss vs token budget at a chosen LR, one panel per size."""
    sizes = [s for s in all_summary if all_summary[s]]
    if not sizes:
        return
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 4.0),
                             squeeze=False)
    for i, size in enumerate(sizes):
        ax = axes[0][i]
        for opt in ("adamw", "muon"):
            pts = sorted(all_summary[size].get(opt, {}).items())
            if not pts:
                continue
            ax.plot([c for c, _ in pts], [v for _, (_, v) in pts], "o-",
                    color=COLOR[opt], markersize=6, linewidth=1.8, label=opt)
            dy, va = ((11, "bottom") if opt == "adamw" else (-13, "top"))
            for c, (lr, v) in pts:
                ax.annotate(f"{lr:.3g}", (c, v), textcoords="offset points",
                            xytext=(0, dy), ha="center", va=va, fontsize=7.5,
                            color=COLOR[opt])
        ax.set_xscale("log")
        chins = sorted({c for o in all_summary[size].values() for c in o})
        ax.set_xticks(chins)
        ax.set_xticklabels([f"{c:g}" for c in chins], fontsize=8)
        ax.minorticks_off()
        ax.set_title(f"{size} ({MODEL_TYPE[size]})", fontsize=11, color=INK)
        ax.set_xlabel("chinchilla multiplier", fontsize=9.5, color=MUTED)
        if i == 0:
            ax.set_ylabel(ylabel or f"best {LABEL} loss", fontsize=10, color=INK)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.margins(y=0.20)
        if i == 0:
            ax.legend(fontsize=9, frameon=False)
    fig.suptitle(title or "Best-LR held-out DCLM loss vs token budget "
                 "(wd 0.1, batch 1M)", fontsize=13, color=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf")


# Categorical slots 1-5, assigned small -> large so color tracks model size.
SIZE_COLOR = {"30M": "#2a78d6", "60M": "#eb6834", "100M": "#1baf7a",
              "300M": "#eda100", "600M": "#e87ba4"}


def plot_best_combined(all_summary, out, annotate=True):
    """All sizes on ONE axes: best-LR loss vs budget.

    Composite encoding — color carries model size, line style carries the
    optimizer — so neither dimension needs a hue beyond the 5 palette slots.
    """
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    style = {"adamw": dict(linestyle="--", marker="s", markersize=5),
             "muon": dict(linestyle="-", marker="o", markersize=6)}
    for size in [s for s in SIZE_COLOR if s in all_summary]:
        for opt in ("adamw", "muon"):
            pts = sorted(all_summary[size].get(opt, {}).items())
            if not pts:
                continue
            xs = [c for c, _ in pts]
            ys = [v for _, (_, v) in pts]
            ax.plot(xs, ys, color=SIZE_COLOR[size], linewidth=1.8,
                    zorder=3, **style[opt])
            if annotate:
                # Diagonal, opposite corners per optimizer: the two optima
                # often sit at nearly the same loss, so stacking the labels
                # vertically would overlap them.
                off, ha, va = ((-7, 8), "right", "bottom") if opt == "adamw" \
                    else ((7, -9), "left", "top")
                for c, (lr, v) in pts:
                    ax.annotate(f"{lr:.3g}", (c, v),
                                textcoords="offset points", xytext=off,
                                ha=ha, va=va, fontsize=6.5,
                                color=SIZE_COLOR[size])
    ax.set_xscale("log")
    ticks = sorted({c for s in all_summary.values()
                    for o in s.values() for c in o})
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{c:g}" for c in ticks], fontsize=9)
    ax.minorticks_off()
    ax.set_xlabel("chinchilla multiplier (token budget)", fontsize=11, color=INK)
    ax.set_ylabel(f"best {LABEL} loss", fontsize=11, color=INK)
    ax.set_title("Best-LR held-out DCLM loss vs token budget, all model sizes\n"
                 "(wd 0.1, batch 1M; annotation = the winning LR)",
                 fontsize=13, color=INK)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=9, colors=MUTED)
    ax.margins(y=0.06)

    size_h = [plt.Line2D([], [], color=SIZE_COLOR[s], linewidth=2.4, label=s)
              for s in SIZE_COLOR if s in all_summary]
    opt_h = [plt.Line2D([], [], color=MUTED, linewidth=1.8, label=o,
                        **style[o]) for o in ("adamw", "muon")]
    first = ax.legend(handles=size_h, title="model size", fontsize=9,
                      title_fontsize=9, frameon=False, loc="upper right")
    ax.add_artist(first)
    ax.legend(handles=opt_h, title="optimizer", fontsize=9, title_fontsize=9,
              frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf")


def plot_best_panels(all_summary, out):
    """Best-LR loss vs budget, ONE subplot per model size.

    Within a panel the only varying dimension is the optimizer, so it takes the
    validated adamw/muon hues (plus the dashed/solid + square/circle composite,
    so identity never rests on colour alone) instead of the per-size colours
    the single-axes version needs. y is NOT shared: loss scales differ ~1 nat
    between 30M and 600M, and a shared axis would flatten the small sizes' gap.
    """
    sizes = [s for s in SIZE_COLOR if s in all_summary]
    if not sizes:
        return
    color = {"adamw": COLOR["adamw"], "muon": COLOR["muon"]}
    style = {"adamw": dict(linestyle="--", marker="s", markersize=5),
             "muon": dict(linestyle="-", marker="o", markersize=6)}
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.4 * len(sizes), 4.2),
                             squeeze=False)
    for i, size in enumerate(sizes):
        ax = axes[0][i]
        chins_all = set()
        for opt in ("adamw", "muon"):
            pts = sorted(all_summary[size].get(opt, {}).items())
            if not pts:
                continue
            xs = [c for c, _ in pts]
            ys = [v for _, (_, v) in pts]
            chins_all.update(xs)
            ax.plot(xs, ys, color=color[opt], linewidth=1.8, label=opt,
                    zorder=3, **style[opt])
            # adamw labels above-left, muon below-right: the two optima often
            # sit at nearly the same loss, so stacking would overlap them.
        ax.set_xscale("log")
        ticks = sorted(chins_all)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{c:g}" for c in ticks], fontsize=7.5,
                           rotation=45 if len(ticks) > 6 else 0)
        ax.minorticks_off()
        ax.set_title(size, fontsize=11.5, color=INK)
        ax.set_xlabel("chinchilla", fontsize=9, color=MUTED)
        if i == 0:
            ax.set_ylabel(f"best {LABEL} loss", fontsize=10, color=INK)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.tick_params(axis="y", labelsize=8, colors=MUTED)
        ax.margins(x=0.08, y=0.08)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Best-LR held-out DCLM loss vs token budget  (wd 0.1, batch 1M)",
                 fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf")


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+",
                   default=["30M", "60M", "100M", "300M", "600M"])
    p.add_argument("--cache", default="/mnt/localssd/evalcache")
    p.add_argument("--out-dir", default=os.path.join(repo, "colm-moss-latex"))
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("--all-components", action="store_true",
                   help="plot every muon run, not just those whose adamw "
                        "component equals the tuned adamw LR for that budget")
    p.add_argument("--mix-legacy", action="store_true",
                   help="include MuonExpt3-named runs even at sizes where they "
                        "are a different recipe (default: exclude at 100M)")
    p.add_argument("--split-schema", nargs="*", default=["100M"],
                   metavar="SIZE",
                   help="sizes whose figure plots each naming schema as its "
                        "own line instead of averaging them (default: 100M)")
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    all_summary, all_table = {}, {}
    for size in a.sizes:
        d = (os.path.join(a.cache, size) if a.no_sync
             else sync(size, a.cache))
        if not os.path.isdir(d):
            print(f"{size}: no eval cache at {d}, skipping")
            continue
        keep = "PTSweep" if size in FOREIGN_LEGACY and not a.mix_legacy else None
        tc = None if a.all_components else tuned_adamw_lrs(size)
        data, ntok = load(size, d, only_schema=keep, tuned_component=tc)
        if keep:
            print(f"{size}: using {keep}-named runs only "
                  f"(MuonExpt3 at {size} is a different recipe)")
        runs = sum(len(l) for c in data.values() for v in c.values()
                   for l in v.values())
        cells = sum(len(v) for c in data.values() for v in c.values())
        print(f"{size}: {runs} runs ({cells} distinct LR cells) over "
              f"{len(data)} chinchilla(s)")
        s = plot_size(size, data, ntok,
                      os.path.join(a.out_dir, f"pt-lr-dclm-{size}"),
                      split_schema=size in (a.split_schema or []))
        if s:
            all_summary[size] = s
        ts = table_summary(size, data)
        if ts:
            all_table[size] = ts
    plot_best(all_summary, os.path.join(a.out_dir, "pt-lr-dclm-best-all-sizes"))
    plot_best_combined(all_summary,
                       os.path.join(a.out_dir, "pt-lr-dclm-best-combined"))
    plot_best_panels(all_summary,
                     os.path.join(a.out_dir, "pt-lr-dclm-best-panels"))
    plot_best(all_table,
              os.path.join(a.out_dir, "pt-lr-dclm-best-all-sizes-table"),
              title="Held-out DCLM loss at the PT_LR_BY_MODEL table's LRs "
                    "(wd 0.1, batch 1M)",
              ylabel=f"{LABEL} loss at table LR")


if __name__ == "__main__":
    main()
