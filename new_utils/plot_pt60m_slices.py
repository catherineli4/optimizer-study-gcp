"""PTSweep<size>: held-out DCLM loss against ONE hyperparameter, everything else fixed.

The 60M sweep is the only size that varies weight decay and batch size as well
as chinchilla and learning rate, so it is the only place a clean one-at-a-time
slice can be cut. For each axis this finds the (optimizer, remaining-hparams)
combinations that hold every OTHER axis constant and draws loss against that
axis alone -- so a trend cannot be an artifact of some co-varying setting.

Only slices with at least --min-points distinct x values are drawn; the sweep is
ragged and a two-point "trend" is not one.

Axes: chinchilla, lr (the swept LR: adamw_lr for adamw, muon_lr for muon),
weight_decay, batch_size.

    python -m new_utils.plot_pt60m_slices
    python -m new_utils.plot_pt60m_slices --size 30M --axis weight_decay --min-points 3
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
COLOR = {"adamw": "#2a78d6", "muon": "#eb6834"}
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#8c8b87"

MODEL_TYPE = {"30M": "0.03B", "60M": "0.06B", "100M": "0.1B",
              "300M": "0.3B", "600M": "0.6B"}


def patterns(size):
    """One eval JSON per run. The muon name carries BOTH LRs; muon_lr is the
    swept one and adamw_lr is the component, which must be part of the held-fixed
    key or a "muon LR sweep" would silently mix component settings."""
    mt = re.escape(MODEL_TYPE[size])
    return {
        # MuonExpt3 schema: same recipe, no wd/bs tags (= wd 0.1, bs 1M). Used
        # as a FALLBACK for a config with no PTSweep run, so a tuned base that
        # only exists under this name can still anchor a sweep built on it.
        "legacy-adamw": re.compile(
            rf"^MuonExpt3-{mt}-chinchilla-([0-9.]+)-adamw-lr([0-9.e+\-]+)"
            rf"-wsd-eval\.json$"),
        "legacy-muon": re.compile(
            rf"^MuonExpt3-{mt}-chinchilla-([0-9.]+)-muon-muonlr([0-9.e+\-]+)"
            rf"-adamwlr([0-9.e+\-]+)-wsd-eval\.json$"),
        "adamw": re.compile(
            rf"^PTSweep{size}-{mt}-chinchilla-([0-9.]+)-adamw-lr([0-9.e+\-]+)"
            rf"-wd([0-9.]+)-bs(\d+M)-wsd-eval\.json$"),
        # Optional -muonwd<x>: the Muon group's decay when only it was swept
        # and the adamw group stayed at the preceding wd<...> value.
        "muon": re.compile(
            rf"^PTSweep{size}-{mt}-chinchilla-([0-9.]+)-muon-muonlr([0-9.e+\-]+)"
            rf"-adamwlr([0-9.e+\-]+)-wd([0-9.]+)(?:-muonwd([0-9.]+))?"
            rf"-bs(\d+M)-wsd-eval\.json$"),
    }

AXES = ["chinchilla", "lr", "weight_decay", "batch_size"]
# Batch size is ordinal, not numeric-in-the-same-units as the rest.
BS_ORDER = {"1M": 1.0, "2M": 2.0, "4M": 4.0}


def sync(cache, size):
    os.makedirs(cache, exist_ok=True)
    subprocess.run(
        # -d makes the cache a true mirror. Without it a deletion on GCS never
        # propagates, so an eval that was removed as invalid keeps being plotted
        # from the stale local copy -- which is how a pre-fix wd=0.2 loss
        # survived into the figure after its GCS object was gone. The -x
        # patterns are applied to both sides, so excluded files are not deleted.
        ["gsutil", "-m", "rsync", "-d",
         "-x", r".*-CPT-.*|.*-EWC-.*|.*_perturbed_.*|.*-typo-.*",
         f"{BUCKET}/Optim-{size}-tuning/ModelEvaluation/", cache],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return cache


def load(cache, size):
    """[{optimizer, chinchilla, lr, adamw_component, weight_decay, batch_size,
         loss, num_tokens, name}]"""
    runs, legacy, pats = [], [], patterns(size)
    for fn in sorted(os.listdir(cache)):
        for kind, rx in pats.items():
            m = rx.match(fn)
            if not m:
                continue
            cell = json.load(open(os.path.join(cache, fn))).get(
                "by_label", {}).get(LABEL)
            if not cell:
                break
            g = m.groups()
            is_legacy = kind.startswith("legacy-")
            opt = kind.replace("legacy-", "")
            if is_legacy:
                # Re-shape to the PTSweep group layout: (chin, lr[, comp], wd, bs)
                g = (g[0], g[1], "0.1", "1M") if opt == "adamw" \
                    else (g[0], g[1], g[2], "0.1", None, "1M")
            if opt == "adamw":
                chin, lr, wd, bs = float(g[0]), float(g[1]), float(g[2]), g[3]
                comp, adamw_wd, tagged = None, wd, False
            else:
                chin, lr, comp, adamw_wd = (float(g[0]), float(g[1]),
                                            float(g[2]), float(g[3]))
                tagged = g[4] is not None
                # weight_decay is the axis value = the Muon group's decay. An
                # untagged run had both groups at that value; a tagged run kept
                # the adamw group at adamw_wd.
                wd = float(g[4]) if tagged else adamw_wd
                bs = g[5]
            rec = dict(optimizer=opt, chinchilla=chin, lr=lr,
                       adamw_component=comp, weight_decay=wd,
                       adamw_wd=adamw_wd, muon_only=tagged,
                       batch_size=bs, loss=cell["loss"],
                       num_tokens=cell["num_tokens"], name=fn)
            (legacy if is_legacy else runs).append(rec)
            break
    # Fallback only: a legacy run fills a config that has no PTSweep run. Where
    # both exist they are independent runs of one recipe, and the PTSweep one
    # is the artifact the sweeps were actually built on, so it wins outright
    # rather than being averaged with the legacy value.
    have = {(r["optimizer"], r["chinchilla"], r["lr"], r["adamw_component"],
             r["weight_decay"], r["adamw_wd"], r["batch_size"]) for r in runs}
    added = 0
    for r in legacy:
        k = (r["optimizer"], r["chinchilla"], r["lr"], r["adamw_component"],
             r["weight_decay"], r["adamw_wd"], r["batch_size"])
        if k not in have:
            runs.append(r); have.add(k); added += 1
    if added:
        print(f"  +{added} MuonExpt3-named run(s) used as wd0.1/bs1M fallback")
    return runs


def slices(runs, axis, min_points):
    """Group runs so that every hparam EXCEPT `axis` is identical."""
    keys = ["optimizer", "chinchilla", "lr", "adamw_component",
            "weight_decay", "batch_size"]
    others = [k for k in keys if k != axis]
    groups = collections.defaultdict(list)
    for r in runs:
        groups[tuple(r[k] for k in others)].append(r)

    if axis == "weight_decay":
        # Second family for the muon-only sweeps: runs where the adamw group
        # was PINNED. Keyed additionally by adamw_wd, and the untagged run at
        # that same value belongs too -- both groups at 0.1 is exactly "adamw
        # pinned at 0.1, muon at 0.1", i.e. the tuned base is the anchor. The
        # tied family above is left as is, so the older both-groups sweeps are
        # unchanged; here we drop tagged runs from it, since a tagged wd=0.2
        # point and a tied wd=0.2 point are different configurations.
        for key in list(groups):
            groups[key] = [r for r in groups[key] if not r["muon_only"]]
        pinned = collections.defaultdict(list)
        for r in runs:
            if r["optimizer"] != "muon":
                continue
            if r["muon_only"] or r["weight_decay"] == r["adamw_wd"]:
                pinned[tuple(r[k] for k in others) + ("adamw_wd", r["adamw_wd"])].append(r)
        for key, rs in pinned.items():
            if any(r["muon_only"] for r in rs):
                groups[key] = rs

    out = []
    for key, rs in groups.items():
        xs = {r[axis] for r in rs}
        if len(xs) < min_points:
            continue
        # Duplicate x within a slice = a re-run; average rather than draw a
        # vertical jump between two points that are the same configuration.
        by_x = collections.defaultdict(list)
        for r in rs:
            by_x[r[axis]].append(r["loss"])
        pts = sorted((x, sum(v) / len(v)) for x, v in by_x.items())
        spec = dict(zip(others, key[:len(others)]))
        if len(key) > len(others):          # pinned-adamw family
            spec["pinned_adamw_wd"] = key[-1]
        out.append((spec, pts))
    return out


def _xpos(axis, x):
    return BS_ORDER[x] if axis == "batch_size" else x


def _label(spec, axis):
    bits = [f"c={spec['chinchilla']:g}"] if axis != "chinchilla" else []
    if axis != "lr":
        bits.append(f"lr={spec['lr']:.1e}")
    if axis != "weight_decay":
        bits.append(f"wd={spec['weight_decay']:g}")
    if axis != "batch_size":
        bits.append(f"bs={spec['batch_size']}")
    if spec.get("adamw_component") is not None and axis != "lr":
        bits.append(f"acomp={spec['adamw_component']:.0e}")
    if "pinned_adamw_wd" in spec:
        bits.append(f"adamw-wd pinned {spec['pinned_adamw_wd']:g}")
    return "  ".join(bits)


def plot_axis(runs, axis, out, size, min_points=3, max_lines=14):
    sl = slices(runs, axis, min_points)
    if not sl:
        print(f"{axis}: no slice with >= {min_points} points, skipping")
        return
    # Longest slices first: those are the ones that actually show a trend, and
    # a legend with 40 entries is unreadable.
    sl.sort(key=lambda t: -len(t[1]))
    sl = sl[:max_lines]

    # Only optimizers that actually have a slice on this axis. 30M sweeps wd for
    # muon alone, and an empty adamw panel would take half the figure.
    opts = [o for o in ("adamw", "muon")
            if any(spec["optimizer"] == o for spec, _ in sl)]
    fig, axes = plt.subplots(1, len(opts), figsize=(6.25 * len(opts), 4.6),
                             squeeze=False, sharey=True)
    for c, opt in enumerate(opts):
        ax = axes[0][c]
        mine = [(s, p) for s, p in sl if s["optimizer"] == opt]
        cmap = plt.cm.viridis
        for i, (spec, pts) in enumerate(mine):
            shade = cmap(0.08 + 0.82 * (i / max(len(mine) - 1, 1)))
            ax.plot([_xpos(axis, x) for x, _ in pts], [y for _, y in pts],
                    "o-", color=shade, linewidth=1.6, markersize=4.5,
                    label=_label(spec, axis), zorder=3)
        if axis in ("chinchilla", "lr"):
            ax.set_xscale("log")
        if axis == "batch_size":
            ax.set_xscale("log")
            ax.set_xticks(list(BS_ORDER.values()))
            ax.set_xticklabels(list(BS_ORDER))
            ax.minorticks_off()
        ax.set_title(f"{size} — {opt}", fontsize=11, color=INK)
        ax.set_xlabel(axis.replace("_", " "), fontsize=9.5, color=MUTED)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=8, colors=MUTED)
        if mine:
            ax.legend(fontsize=6.6, frameon=False, ncol=1,
                      loc="best", handlelength=1.4)
        if c == 0:
            ax.set_ylabel(f"{LABEL} loss", fontsize=10, color=INK)

    fig.suptitle(f"PTSweep{size}: held-out DCLM loss vs {axis.replace('_', ' ')}"
                 f"   —   every other hyperparameter held constant per line",
                 fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf   ({len(sl)} slice(s))")


def plot_wd_pinned_all_sizes(sizes, cache_root, out, chinchilla=1.0,
                             min_points=2, sync_first=True):
    """One figure, one panel per size: DCLM loss vs the MUON group's weight
    decay with the AdamW group pinned at its default. Only the pinned family
    is drawn -- the older both-groups sweeps varied norm-gain decay too and
    are not the same experiment. Loss levels differ by >1 nat across sizes,
    so each panel keeps its own y axis; x is shared.
    """
    panels = []
    for size in sizes:
        cache = os.path.join(cache_root, size)
        if sync_first:
            sync(cache, size)
        runs = [r for r in load(cache, size)
                if r["chinchilla"] == chinchilla and r["num_tokens"] == 4193280]
        sl = [(spec, pts) for spec, pts in slices(runs, "weight_decay", min_points)
              if "pinned_adamw_wd" in spec and spec["optimizer"] == "muon"]
        if not sl:
            print(f"{size}: no pinned-adamw muon wd sweep at chinchilla {chinchilla:g}")
            continue
        sl.sort(key=lambda t: -len(t[1]))
        panels.append((size, sl))
    if not panels:
        print("nothing to plot"); return

    # One axis for every size, absolute final loss. Size is ordered, so the
    # lines take a single-hue sequential ramp rather than categorical colours.
    ramp = plt.cm.viridis([0.08, 0.32, 0.56, 0.8, 0.95][:len(panels)])
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for (size, sl), colour in zip(panels, ramp):
        spec, pts = sl[0]
        xs, ys = [x for x, _ in pts], [y for _, y in pts]
        ax.plot(xs, ys, "o-", color=colour, linewidth=1.9, markersize=5.5,
                zorder=3, label=f"{size}  (muon lr {spec['lr']:.2g})")
        k = [j for j, x in enumerate(xs) if x == spec["pinned_adamw_wd"]]
        if k:
            ax.scatter([xs[k[0]]], [ys[k[0]]], s=140, facecolors="none",
                       edgecolors=INK, linewidths=1.5, zorder=4)
    ax.set_xlabel("Muon-group weight decay", fontsize=10, color=MUTED)
    ax.set_ylabel(f"{LABEL} loss", fontsize=10, color=INK)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8.5, colors=MUTED)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK, marker="o", markersize=9,
                              markerfacecolor="none", linestyle="none"))
    labels.append("tuned base (wd 0.1)")
    # The band between the 30M and 60M curves is the only clear space.
    ax.legend(handles, labels, frameon=False, fontsize=9, loc="center right",
              bbox_to_anchor=(0.99, 0.74))
    ax.set_title(f"Held-out DCLM loss vs Muon weight decay at chinchilla "
                 f"{chinchilla:g}  —  AdamW group held at wd 0.1",
                 fontsize=11, color=INK, pad=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf  ({len(panels)} size(s))")


def plot_wd_cells(cells, cache_root, out, wds=(0.0, 0.1, 0.2), sync_first=True):
    """One panel per (size, chinchilla) cell: DCLM loss vs weight decay for
    BOTH optimizers at the table's tuned LRs. adamw varies its single wd;
    muon varies the Muon group's decay with the AdamW group pinned at 0.1
    (the -muonwd family), so wd 0.1 is the tuned base for both.
    """
    from new_utils.plot_pt_lr_dclm_all import tuned_lrs, tuned_adamw_lrs
    panels = []
    for size, chin in cells:
        cache = os.path.join(cache_root, size)
        if sync_first:
            sync(cache, size)
        runs = [r for r in load(cache, size)
                if r["chinchilla"] == chin and r["batch_size"] == "1M"
                and r["num_tokens"] == 4193280]
        alr = tuned_lrs(size)["adamw"].get(chin)
        mlr = tuned_lrs(size)["muon"].get(chin)
        comp = tuned_adamw_lrs(size).get(chin)
        series = {}
        for r in runs:
            if r["optimizer"] == "adamw" and alr is not None and r["lr"] == alr:
                series.setdefault("adamw", {})[r["weight_decay"]] = r["loss"]
            elif (r["optimizer"] == "muon" and mlr is not None
                  and r["lr"] == mlr and r["adamw_component"] == comp
                  and r["adamw_wd"] == 0.1
                  and (r["muon_only"] or r["weight_decay"] == 0.1)):
                series.setdefault("muon", {})[r["weight_decay"]] = r["loss"]
        for opt in list(series):
            series[opt] = {w: v for w, v in series[opt].items() if w in wds}
            missing = [w for w in wds if w not in series[opt]]
            if missing:
                print(f"{size} c{chin:g} {opt}: no eval at wd {missing}")
        if not series:
            print(f"{size} c{chin:g}: nothing at the table LRs")
            continue
        panels.append((size, chin, alr, (mlr, comp), series))
    if not panels:
        print("nothing to plot"); return

    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.9),
                             sharex=True)
    axes = list(axes) if len(panels) > 1 else [axes]
    for ax, (size, chin, alr, (mlr, comp), series) in zip(axes, panels):
        for opt in ("adamw", "muon"):
            if opt not in series:
                continue
            xs = sorted(series[opt]); ys = [series[opt][x] for x in xs]
            lab = f"adamw lr {alr:.2g}" if opt == "adamw" \
                else f"muon ({mlr:.2g}, {comp:.2g})"
            ax.plot(xs, ys, "o-", color=COLOR[opt], linewidth=1.8,
                    markersize=5.5, zorder=3, label=lab)
            if 0.1 in series[opt]:
                ax.scatter([0.1], [series[opt][0.1]], s=140, facecolors="none",
                           edgecolors=INK, linewidths=1.4, zorder=4)
        ax.set_title(f"{size}  chinchilla {chin:g}", fontsize=10, color=INK)
        ax.set_xticks(list(wds))
        ax.set_xlabel("weight decay", fontsize=9, color=MUTED)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.legend(frameon=False, fontsize=8, loc="best")
    axes[0].set_ylabel(f"{LABEL} loss", fontsize=10, color=INK)
    fig.suptitle("Held-out DCLM loss vs weight decay at the tuned LRs  —  "
                 "muon: Muon-group decay, AdamW group held at 0.1; ring = tuned base",
                 fontsize=11, color=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf  ({len(panels)} cell(s))")


def plot_comp_all_sizes(sizes, cache_root, out, chinchilla=1.0, sync_first=True):
    """DCLM loss vs the adamw COMPONENT of the muon pair, muon_lr held at the
    table's tuned value, one line per size on a single axis (absolute loss).
    The tuned component is ringed. wd 0.1 / batch 1M cells only.
    """
    from new_utils.plot_pt_lr_dclm_all import tuned_lrs, tuned_adamw_lrs
    lines = []
    for size in sizes:
        cache = os.path.join(cache_root, size)
        if sync_first:
            sync(cache, size)
        mlr = tuned_lrs(size)["muon"].get(chinchilla)
        comp = tuned_adamw_lrs(size).get(chinchilla)
        if mlr is None:
            print(f"{size}: no tuned muon pair at chinchilla {chinchilla:g}"); continue
        pts = {}
        for r in load(cache, size):
            if (r["optimizer"] == "muon" and r["chinchilla"] == chinchilla
                    and r["batch_size"] == "1M" and not r["muon_only"]
                    and r["weight_decay"] == 0.1 and abs(r["lr"] - mlr) < 1e-12
                    and r["num_tokens"] == 4193280):
                pts.setdefault(r["adamw_component"], []).append(r["loss"])
        # The component sweep is one sqrt(2) grid step either side of the
        # tuned value; older runs at other components are a different
        # experiment and would set the axis (100M has a comp-0.001 run at 4.33).
        if comp is not None:
            want = (comp / 2 ** 0.5, comp, comp * 2 ** 0.5)
            pts = {x: v for x, v in pts.items()
                   if any(abs(x - w) / w < 0.08 for w in want)}
        if len(pts) < 2:
            print(f"{size}: only {len(pts)} component(s) at muon lr {mlr:g}, skipping")
            continue
        xs = sorted(pts)
        ys = [sum(pts[x]) / len(pts[x]) for x in xs]
        lines.append((size, mlr, comp, xs, ys))
    if not lines:
        print("nothing to plot"); return

    ramp = plt.cm.viridis([0.08, 0.32, 0.56, 0.8, 0.95][:len(lines)])
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for (size, mlr, comp, xs, ys), colour in zip(lines, ramp):
        ax.plot(xs, ys, "o-", color=colour, linewidth=1.9, markersize=5.5,
                zorder=3, label=f"{size}  (muon lr {mlr:.2g})")
        k = [i for i, x in enumerate(xs) if comp is not None and abs(x - comp) < 1e-12]
        if k:
            ax.scatter([xs[k[0]]], [ys[k[0]]], s=140, facecolors="none",
                       edgecolors=INK, linewidths=1.5, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("adamw component LR  (norm gains, lm_head)", fontsize=10, color=MUTED)
    ax.set_ylabel(f"{LABEL} loss", fontsize=10, color=INK)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8.5, colors=MUTED)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK, marker="o", markersize=9,
                              markerfacecolor="none", linestyle="none"))
    labels.append("tuned component (table)")
    ax.legend(handles, labels, frameon=False, fontsize=9, loc="center right",
              bbox_to_anchor=(0.99, 0.74))
    ax.set_title(f"Held-out DCLM loss vs adamw component at chinchilla "
                 f"{chinchilla:g}  —  muon lr fixed at the tuned value, wd 0.1",
                 fontsize=11, color=INK, pad=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf  ({len(lines)} size(s))")
    for size, mlr, comp, xs, ys in lines:
        print(f"  {size}: " + "  ".join(f"comp {x:g} -> {y:.4f}{' *' if comp is not None and abs(x-comp)<1e-12 else ''}" for x, y in zip(xs, ys)))


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="60M", choices=list(MODEL_TYPE))
    p.add_argument("--cache", default=None,
                   help="default /mnt/localssd/pt-eval-cache/<size>")
    p.add_argument("--out-dir", default=os.path.join(repo, "colm-moss-latex"))
    p.add_argument("--axis", choices=AXES + ["all"], default="all")
    p.add_argument("--min-points", type=int, default=3)
    p.add_argument("--wd-all-sizes", nargs="*", metavar="SIZE",
                   help="instead of the per-size slice figures, draw one figure "
                        "of the pinned-adamw muon wd sweep across these sizes "
                        "(default 30M 60M 100M 300M) at --chinchilla (default 1)")
    p.add_argument("--comp-all-sizes", nargs="*", metavar="SIZE",
                   help="one figure of loss vs adamw component at the tuned "
                        "muon lr across these sizes (default 30M 60M 100M)")
    p.add_argument("--wd-cells", nargs="+", metavar="SIZE:CHIN",
                   help="one figure, one panel per (size, chinchilla) cell, "
                        "loss vs weight decay for both optimizers at the "
                        "table LRs, e.g. 30M:1 60M:0.5 60M:1 60M:2 100M:1")
    p.add_argument("--chinchilla", type=float, default=None,
                   help="restrict to one token budget, so a size whose sweep "
                        "covers several does not bury the cell of interest")
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("--eval-tokens", type=int, default=4193280,
                   help="expected DCLM_heldout size (DCLM_HELDOUT_INSTANCES x 4096)")
    p.add_argument("--keep-mismatched", action="store_true",
                   help="plot runs evaluated on a different held-out set anyway")
    a = p.parse_args()

    if a.wd_cells:
        cells = [(c.split(":")[0], float(c.split(":")[1])) for c in a.wd_cells]
        os.makedirs(a.out_dir, exist_ok=True)
        plot_wd_cells(cells, "/mnt/localssd/pt-eval-cache",
                      os.path.join(a.out_dir, "pt-dclm-vs-wd-cells"),
                      sync_first=not a.no_sync)
        return
    if a.comp_all_sizes is not None:
        sizes = a.comp_all_sizes or ["30M", "60M", "100M"]
        chin = 1.0 if a.chinchilla is None else a.chinchilla
        os.makedirs(a.out_dir, exist_ok=True)
        plot_comp_all_sizes(
            sizes, "/mnt/localssd/pt-eval-cache",
            os.path.join(a.out_dir, f"pt-dclm-vs-adamw-comp-c{chin:g}-all-sizes"),
            chinchilla=chin, sync_first=not a.no_sync)
        return
    if a.wd_all_sizes is not None:
        sizes = a.wd_all_sizes or ["30M", "60M", "100M", "300M"]
        chin = 1.0 if a.chinchilla is None else a.chinchilla
        os.makedirs(a.out_dir, exist_ok=True)
        plot_wd_pinned_all_sizes(
            sizes, "/mnt/localssd/pt-eval-cache",
            os.path.join(a.out_dir, f"pt-dclm-vs-muon-wd-c{chin:g}-all-sizes"),
            chinchilla=chin, min_points=a.min_points, sync_first=not a.no_sync)
        return
    if a.cache is None:
        a.cache = f"/mnt/localssd/pt-eval-cache/{a.size}"
    if not a.no_sync:
        sync(a.cache, a.size)
    runs = load(a.cache, a.size)
    print(f"{len(runs)} PTSweep{a.size} runs with a {LABEL} loss")
    if a.chinchilla is not None:
        runs = [r for r in runs if r["chinchilla"] == a.chinchilla]
        print(f"  {len(runs)} at chinchilla {a.chinchilla:g}")
    # DCLM_heldout is meant to be DCLM_HELDOUT_INSTANCES(1024) x 4096 tokens.
    # A handful of runs were evaluated against a larger held-out set, and their
    # losses are NOT comparable with the rest -- six of them are weight-decay
    # variants, which would have silently corrupted exactly the wd axis. Drop
    # them unless asked to keep them, and name them so they can be re-evaluated.
    odd = [r for r in runs if r["num_tokens"] != a.eval_tokens]
    if odd:
        print(f"\n{len(odd)} run(s) evaluated on a DIFFERENT held-out set "
              f"(expected {a.eval_tokens:,} tokens) -- "
              f"{'keeping' if a.keep_mismatched else 'EXCLUDED'}:")
        for r in sorted(odd, key=lambda x: x["name"]):
            print(f"    {r['num_tokens']:>10,}  {r['name'].replace('-eval.json','')}")
        if not a.keep_mismatched:
            runs = [r for r in runs if r["num_tokens"] == a.eval_tokens]
            print(f"  -> {len(runs)} comparable run(s) remain\n")
    os.makedirs(a.out_dir, exist_ok=True)
    for axis in (AXES if a.axis == "all" else [a.axis]):
        plot_axis(runs, axis,
                  os.path.join(a.out_dir,
                               f"pt{a.size.lower()}-dclm-vs-{axis}"
                               + (f"-c{a.chinchilla:g}"
                                  if a.chinchilla is not None else "")),
                  a.size, min_points=a.min_points)


if __name__ == "__main__":
    main()
