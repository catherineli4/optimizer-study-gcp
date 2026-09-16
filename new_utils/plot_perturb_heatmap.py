"""Muon - AdamW robustness to Gaussian weight perturbation, as heatmaps.

One heatmap per perturbation sigma (gamma). x = model size, y = chinchilla
(token budget), cell = perturbed held-out DCLM loss of the tuned MUON base
minus that of the tuned ADAMW base at the same (size, chinchilla).

    negative (blue)  -> muon is more robust at that cell
    positive (red)   -> adamw is more robust
    grey             -> no data (a base or its perturbed eval is missing)

The comparison is only meaningful between the tuned-LR bases, so the LRs come
from PT_LR_BY_MODEL rather than from an argmin over whatever happens to exist.

    python -m new_utils.plot_perturb_heatmap
"""

import argparse
import ast
import json
import os
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

BUCKET = "gs://cmu-gpucloud-catheri4"
LABEL = "DCLM_heldout"
SIZES = ["30M", "60M", "100M", "300M", "600M"]
MODEL_TYPE = {"30M": "0.03B", "60M": "0.06B", "100M": "0.1B",
              "300M": "0.3B", "600M": "0.6B"}
def _perturb_gammas():
    """Read PERTURB_WIDE_GAMMAS from the launcher source.

    Hardcoding the list here let it silently drift out of sync with the sweep
    that produced the artifacts, so new sigmas were plotted as missing.
    """
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "launch_jolmo", "pretraining_matrix.py")
    tree = ast.parse(open(src).read())
    for node in ast.walk(tree):
        t = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            t = node.target.id
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            t = node.targets[0].id
        if t == "PERTURB_WIDE_GAMMAS":
            return sorted(float(g) for g in ast.literal_eval(node.value))
    return [0.02, 0.05, 0.07, 0.1, 0.13]


GAMMAS = _perturb_gammas()

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"

# (size, chinchilla) cells excluded from every perturbation figure. 60M/c64 is
# held out: its muon-minus-adamw value is non-monotonic in gamma (-0.005 at
# 0.005, +0.067 at 0.01, +0.010 at 0.02) by more than the ~0.01 noise floor,
# and the row exists at no other size, so it cannot be cross-checked.
EXCLUDE_CELLS = {("60M", 64.0)}
# Diverging pair with a neutral midpoint: cool = muon better, warm = adamw
# better, grey at exactly zero. Never a rainbow, never a hue at the midpoint.
CMAP = "RdBu_r"


def gamma_tag(gamma):
    """Mirror training._perturbed_run_name's sigma formatting exactly."""
    return f"{gamma:.2e}".replace("e-0", "e-").replace("e+0", "e+").replace(".", "_")


def lr_tag(lr):
    return f"{lr:.1e}".replace("e-0", "e-")


def tuned_table(size):
    """{opt: {chinchilla: lr or (muon_lr, component)}} from PT_LR_BY_MODEL."""
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "launch_jolmo", "pretraining_matrix.py")
    tree = ast.parse(open(src).read())
    for node in ast.walk(tree):
        t = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            t = node.target.id
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            t = node.targets[0].id
        if t == "PT_LR_BY_MODEL":
            return ast.literal_eval(node.value).get(MODEL_TYPE[size], {}).get("wsd", {})
    return {}


def base_names(size, chin, opt, cell):
    """Both naming schemas for a tuned cell; caller keeps whichever exists."""
    mt = MODEL_TYPE[size]
    if opt == "adamw":
        tag = f"lr{lr_tag(float(cell))}"
    else:
        tag = f"muonlr{lr_tag(cell[0])}-adamwlr{lr_tag(cell[1])}"
    return [f"MuonExpt3-{mt}-chinchilla-{chin:g}-{opt}-{tag}-wsd",
            f"PTSweep{size}-{mt}-chinchilla-{chin:g}-{opt}-{tag}-wd0.1-bs1M-wsd"]


def listing(size, cache):
    """Cached listing of the size's ModelEvaluation prefix."""
    p = os.path.join(cache, f"evalnames_{size}.txt")
    if not os.path.exists(p):
        out = subprocess.run(
            ["gsutil", "ls", f"{BUCKET}/Optim-{size}-tuning/ModelEvaluation/"],
            capture_output=True, text=True).stdout
        names = [l.strip().rsplit("/", 1)[-1].replace("-eval.json", "")
                 for l in out.splitlines() if l.strip().endswith("-eval.json")]
        open(p, "w").write("\n".join(names))
    return set(open(p).read().split())


def fetch(size, names, cache):
    """Download the named eval JSONs (skipping ones already local)."""
    d = os.path.join(cache, size)
    os.makedirs(d, exist_ok=True)
    todo = [n for n in names if not os.path.exists(os.path.join(d, n + "-eval.json"))]
    for i in range(0, len(todo), 200):
        urls = [f"{BUCKET}/Optim-{size}-tuning/ModelEvaluation/{n}-eval.json"
                for n in todo[i:i + 200]]
        subprocess.run(["gsutil", "-m", "cp"] + urls + [d],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return d


def loss_of(d, name):
    p = os.path.join(d, name + "-eval.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p)).get("by_label", {}).get(LABEL, {}).get("loss")
    except (json.JSONDecodeError, OSError):
        return None


def collect(cache):
    """{gamma: {(size, chinchilla): muon_loss - adamw_loss}} plus the raw losses.

    ``raw`` is keyed (gamma, size, chinchilla, optimizer); gamma None holds the
    UNPERTURBED base loss, which the degradation plot subtracts.
    """
    diffs, raw, chins = {}, {}, set()
    for size in SIZES:
        table = tuned_table(size)
        have = listing(size, cache)
        wanted, resolved = set(), {}
        for opt in ("adamw", "muon"):
            for chin, cell in table.get(opt, {}).items():
                base = next((b for b in base_names(size, float(chin), opt, cell)
                             if b in have or any(n.startswith(b + "_perturbed_")
                                                 for n in have)), None)
                if base is None:
                    continue
                resolved[(float(chin), opt)] = base
                if base in have:            # the unperturbed reference
                    wanted.add(base)
                for g in GAMMAS:
                    n = f"{base}_perturbed_{gamma_tag(g)}"
                    if n in have:
                        wanted.add(n)
        d = fetch(size, wanted, cache)
        for (chin, opt), base in resolved.items():
            if (size, chin) in EXCLUDE_CELLS:
                continue
            chins.add(chin)
            v0 = loss_of(d, base)
            if v0 is not None:
                raw[(None, size, chin, opt)] = v0
            for g in GAMMAS:
                v = loss_of(d, f"{base}_perturbed_{gamma_tag(g)}")
                if v is not None:
                    raw[(g, size, chin, opt)] = v
        for g in GAMMAS:
            for chin in {c for c, _ in resolved}:
                a = raw.get((g, size, chin, "adamw"))
                m = raw.get((g, size, chin, "muon"))
                if a is not None and m is not None:
                    diffs.setdefault(g, {})[(size, chin)] = m - a
    return diffs, raw, sorted(chins)


def degradation_diffs(raw, relative=False):
    """{gamma: {(size, chinchilla): muon_degradation - adamw_degradation}}.

    Degradation is (perturbed - unperturbed) per optimizer, so this isolates
    the ROBUSTNESS difference from the baseline-quality difference: muon starts
    from a lower unperturbed loss, which flatters it in the absolute-loss view.

    relative=True divides each optimizer's degradation by its own unperturbed
    loss before differencing, and reports the result in PERCENT of that loss.
    A fixed 0.1-nat rise is a bigger relative hit to a 3.7-nat model than to a
    4.6-nat one, so this is the fairer cross-size comparison; it is also the
    same normalisation the tuned-LR tables use when they talk about "x% worse".
    """
    out = {}
    for (g, size, chin, opt), v in raw.items():
        if g is None or opt != "muon":
            continue
        m0 = raw.get((None, size, chin, "muon"))
        a = raw.get((g, size, chin, "adamw"))
        a0 = raw.get((None, size, chin, "adamw"))
        if None in (m0, a, a0):
            continue
        if relative:
            d = 100.0 * ((v - m0) / m0 - (a - a0) / a0)
        else:
            d = (v - m0) - (a - a0)
        out.setdefault(g, {})[(size, chin)] = d
    return out


def plot(diffs, chins, out, title=None, cbar_label=None, caption=None,
         gammas=None, clip_pct=97):
    # The colour scale is computed from the gammas actually shown, so a
    # restricted range rescales instead of being flattened by the large-gamma
    # values that dominate the full set.
    gs = [g for g in (gammas or GAMMAS) if diffs.get(g)]
    if not gs:
        raise SystemExit("no (size, chinchilla) cell has BOTH optimizers perturbed")
    vals = [v for g in gs for v in diffs[g].values()]
    # Robust limits: a single outlier (e.g. the 60M/c2 cell at gamma 0.01) would
    # otherwise set vmax for every panel and flatten all the real structure to
    # near-white. Clipped cells still carry their true value as text.
    mag = np.abs(vals)
    lim = float(np.percentile(mag, clip_pct))
    if lim <= 0:
        lim = float(mag.max()) or 1e-6
    n_clipped = int((mag > lim).sum())
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig, axes = plt.subplots(1, len(gs), figsize=(3.1 * len(gs) + 1.4, 4.6),
                             squeeze=False)
    for i, g in enumerate(gs):
        ax = axes[0][i]
        M = np.full((len(chins), len(SIZES)), np.nan)
        for (size, chin), v in diffs[g].items():
            M[chins.index(chin)][SIZES.index(size)] = v
        im = ax.imshow(M, cmap=CMAP, norm=norm, aspect="auto", origin="lower")
        ax.set_xticks(range(len(SIZES)))
        ax.set_xticklabels(SIZES, fontsize=8.5, rotation=45, ha="right")
        ax.set_yticks(range(len(chins)))
        ax.set_yticklabels([f"{c:g}" for c in chins], fontsize=8.5)
        ax.set_title(f"$\\gamma$ = {g:g}", fontsize=11.5, color=INK)
        ax.set_xlabel("model size", fontsize=9.5, color=MUTED)
        if i == 0:
            ax.set_ylabel("chinchilla (token budget)", fontsize=10, color=INK)
        # Direct labels: the grid is small enough that the number belongs in
        # the cell rather than only in the colorbar.
        for r in range(len(chins)):
            for c in range(len(SIZES)):
                if not np.isnan(M[r][c]):
                    ax.text(c, r, f"{M[r][c]:+.3f}", ha="center", va="center",
                            fontsize=7, color=INK)
        ax.set_xticks(np.arange(-.5, len(SIZES), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(chins), 1), minor=True)
        ax.grid(which="minor", color=GRID, linewidth=1)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(colors=MUTED)

    cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.03, pad=0.02)
    cb.set_label(cbar_label or "muon - adamw  perturbed DCLM loss",
                 fontsize=9.5, color=INK)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    fig.suptitle(title or
                 "Robustness to Gaussian weight perturbation: muon vs adamw",
                 fontsize=13.5, color=INK)
    # The interpretation sits under the panels: a second suptitle line collides
    # with the per-panel gamma titles.
    if n_clipped:
        fig.text(0.5, -0.085,
                 f"colour scale clipped at ±{lim:.3f} ({clip_pct}th pct); "
                 f"{n_clipped} cell(s) beyond it keep their printed value",
                 ha="center", fontsize=8, color=MUTED)
    fig.text(0.5, -0.04, caption or
             "cell = perturbed held-out DCLM loss, tuned muon base minus tuned "
             "adamw base    |    negative (blue) = muon degrades less    |    "
             "white = no data",
             ha="center", fontsize=9, color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf")


def plot_curves(raw, out, degradation=False):
    """Rows = model size, columns = optimizer; one line per sigma.

    sigma is an ordered magnitude, so the lines take a single-hue sequential
    ramp (light = small perturbation, dark = large) rather than categorical
    colors, which would imply the sigmas are unordered categories.
    """
    ramp = plt.cm.Blues(np.linspace(0.35, 1.0, len(GAMMAS)))
    rows = [s for s in SIZES
            if any(k[1] == s and k[0] is not None for k in raw)]
    fig, axes = plt.subplots(len(rows), 2, figsize=(10.5, 2.9 * len(rows)),
                             squeeze=False, sharex=False)
    for r, size in enumerate(rows):
        # A shared y-range per row makes adamw vs muon directly comparable.
        row_vals = []
        for opt in ("adamw", "muon"):
            for g in GAMMAS:
                for k, v in raw.items():
                    if k[0] == g and k[1] == size and k[3] == opt:
                        base = raw.get((None, size, k[2], opt))
                        if degradation and base is None:
                            continue
                        row_vals.append(v - base if degradation else v)
        for c, opt in enumerate(("adamw", "muon")):
            ax = axes[r][c]
            for gi, g in enumerate(GAMMAS):
                pts = []
                for k, v in raw.items():
                    if k[0] != g or k[1] != size or k[3] != opt:
                        continue
                    base = raw.get((None, size, k[2], opt))
                    if degradation:
                        if base is None:
                            continue
                        pts.append((k[2], v - base))
                    else:
                        pts.append((k[2], v))
                pts.sort()
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                            color=ramp[gi], markersize=4.5, linewidth=1.7,
                            label=f"$\\gamma$={g:g}", zorder=3)
            if not degradation:
                base_pts = sorted((k[2], v) for k, v in raw.items()
                                  if k[0] is None and k[1] == size and k[3] == opt)
                if base_pts:
                    ax.plot([p[0] for p in base_pts], [p[1] for p in base_pts],
                            "--", color=MUTED, linewidth=1.3, zorder=2,
                            label="unperturbed")
            ax.set_xscale("log")
            xs = sorted({k[2] for k in raw if k[1] == size})
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=8)
            ax.minorticks_off()
            ax.set_title(f"{size} ({MODEL_TYPE[size]}) — {opt}",
                         fontsize=10.5, color=INK)
            ax.grid(True, alpha=0.25, linewidth=0.6)
            ax.tick_params(labelsize=8, colors=MUTED)
            if row_vals:
                lo, hi = min(row_vals), max(row_vals)
                pad = 0.06 * (hi - lo) if hi > lo else 0.1
                ax.set_ylim(lo - pad, hi + pad)
            if r == len(rows) - 1:
                ax.set_xlabel("chinchilla (token budget)", fontsize=9.5,
                              color=MUTED)
            if c == 0:
                ax.set_ylabel("degradation (nats)" if degradation
                              else f"{LABEL} loss", fontsize=9.5, color=INK)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(h), frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.012))
    fig.suptitle(
        ("Loss degradation from Gaussian weight perturbation "
         "(perturbed - unperturbed)") if degradation else
        "Post-perturbation held-out DCLM loss",
        fontsize=13.5, color=INK)
    fig.tight_layout(rect=(0, 0.012, 1, 0.985))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf")



# Parameter counts behind the size labels, so the x axis is a real quantity on a
# log scale rather than five evenly spaced categories -- 30M to 600M is a 20x
# span and even spacing would misstate every slope.
SIZE_PARAMS = {"30M": 0.03e9, "60M": 0.06e9, "100M": 0.1e9,
               "300M": 0.3e9, "600M": 0.6e9}


def plot_curves_vs_size(raw, out, degradation=False, min_sizes=2):
    """Rows = chinchilla, columns = optimizer; x = model size, one line per sigma.

    The transpose of plot_curves: that one fixes the size per row and sweeps the
    token budget, answering "does a longer-trained model perturb worse". Fixing
    the budget per row and sweeping size instead asks whether the same holds as
    models grow at equal budget -- which is the axis the scaling claim is about
    and the one the by-size layout cannot show.

    Rows with fewer than ``min_sizes`` sizes are dropped: a single point is not
    a trend, and the budget grid is ragged (c=128 exists at 60M alone, c=64 at
    30M/60M only).
    """
    ramp = plt.cm.Blues(np.linspace(0.35, 1.0, len(GAMMAS)))

    chins = sorted({k[2] for k in raw})
    rows = []
    for chin in chins:
        sizes = [s for s in SIZES
                 if any(k[1] == s and k[2] == chin and k[0] is not None
                        for k in raw)]
        if len(sizes) < min_sizes:
            print(f"c={chin:g}: only {len(sizes)} size(s), skipping row")
            continue
        rows.append(chin)
    if not rows:
        print("nothing to plot")
        return

    fig, axes = plt.subplots(len(rows), 2, figsize=(10.5, 2.9 * len(rows)),
                             squeeze=False, sharex=True)
    for r, chin in enumerate(rows):
        # A shared y-range per row keeps adamw and muon directly comparable,
        # exactly as in plot_curves.
        row_vals = []
        for opt in ("adamw", "muon"):
            for k, v in raw.items():
                if k[0] is None or k[2] != chin or k[3] != opt:
                    continue
                base = raw.get((None, k[1], chin, opt))
                if degradation and base is None:
                    continue
                row_vals.append(v - base if degradation else v)

        for c, opt in enumerate(("adamw", "muon")):
            ax = axes[r][c]
            for gi, g in enumerate(GAMMAS):
                pts = []
                for k, v in raw.items():
                    if k[0] != g or k[2] != chin or k[3] != opt:
                        continue
                    base = raw.get((None, k[1], chin, opt))
                    if degradation:
                        if base is None:
                            continue
                        pts.append((SIZE_PARAMS[k[1]], v - base))
                    else:
                        pts.append((SIZE_PARAMS[k[1]], v))
                pts.sort()
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                            color=ramp[gi], markersize=4.5, linewidth=1.7,
                            label=f"$\\gamma$={g:g}", zorder=3)
            if not degradation:
                base_pts = sorted((SIZE_PARAMS[k[1]], v) for k, v in raw.items()
                                  if k[0] is None and k[2] == chin and k[3] == opt)
                if base_pts:
                    ax.plot([p[0] for p in base_pts], [p[1] for p in base_pts],
                            "--", color=MUTED, linewidth=1.3, zorder=2,
                            label="unperturbed")
            ax.set_xscale("log")
            # Tick only the five real sizes; the log locator would otherwise put
            # decade ticks at 1e8, where no model exists.
            ax.set_xticks([SIZE_PARAMS[s] for s in SIZES])
            ax.set_xticklabels(SIZES, fontsize=8)
            ax.minorticks_off()
            ax.set_title(f"chinchilla {chin:g} — {opt}", fontsize=10.5, color=INK)
            ax.grid(True, alpha=0.25, linewidth=0.6)
            ax.tick_params(labelsize=8, colors=MUTED)
            if row_vals:
                lo, hi = min(row_vals), max(row_vals)
                pad = 0.06 * (hi - lo) if hi > lo else 0.1
                ax.set_ylim(lo - pad, hi + pad)
            if r == len(rows) - 1:
                ax.set_xlabel("model size (parameters)", fontsize=9.5,
                              color=MUTED)
            if c == 0:
                ax.set_ylabel("degradation (nats)" if degradation
                              else f"{LABEL} loss", fontsize=9.5, color=INK)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(h), frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.012))
    fig.suptitle(
        ("Loss degradation from Gaussian weight perturbation "
         "(perturbed - unperturbed), by model size") if degradation else
        "Post-perturbation held-out DCLM loss, by model size",
        fontsize=13.5, color=INK)
    fig.tight_layout(rect=(0, 0.012, 1, 0.985))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf")


SIZE_MARKER = {"30M": "o", "60M": "s", "100M": "^", "300M": "D", "600M": "P"}
OPT_COLOR = {"adamw": "#2a78d6", "muon": "#eb6834"}
# (size, chinchilla, optimizer) bases left out of the degradation-vs-loss
# scatters. Each is a single cell sitting an order of magnitude off every
# neighbour; the evals stay on GCS and the heatmaps still show them.
VS_LOSS_EXCLUDE = {
    ("60M", 2.0, "adamw"),   # 7e-2 at gamma 0.01 vs 1-3e-3 for every other 60M adamw cell
}


def plot_degradation_vs_loss(raw, out, gamma=0.01, relative=False, facet=None):
    """Every tuned base as one point: x = its unperturbed held-out loss,
    y = its degradation at one gamma (log scale). Colour = optimizer, marker =
    size, so the question "is the muon/adamw robustness gap explained by
    where each model sits on the loss axis?" can be read directly: two clouds
    that separate vertically at equal x are a real optimizer effect; clouds
    lying on one curve are just the loss-level effect.
    relative=True divides the degradation by the unperturbed loss.
    """
    pts = []
    for (g, size, chin, opt), v in raw.items():
        if g is None or abs(g - gamma) > 1e-12:
            continue
        base = raw.get((None, size, chin, opt))
        if base is None or (size, chin, opt) in VS_LOSS_EXCLUDE:
            continue
        d = v - base
        if relative:
            d = d / base
        if d <= 0:
            print(f"  {size} c{chin:g} {opt}: non-positive degradation {d:.4g} at "
                  f"gamma {gamma:g}, cannot go on a log axis -- dropped")
            continue
        pts.append((size, chin, opt, base, d))
    if not pts:
        print(f"no points at gamma {gamma:g}")
        return
    ylab = (("relative degradation  (perturbed − base) / base" if relative
             else "degradation  perturbed − base  (nats)")
            + f"   at $\\gamma$ = {gamma:g}")
    what = f"{'Relative d' if relative else 'D'}egradation under Gaussian weight " \
           f"perturbation ($\\gamma$ = {gamma:g}) vs model quality"

    # facet=None: one axis. facet="size" / "chinchilla": one panel per value,
    # shared log-y so panels compare directly; x free, since the loss range
    # moves with size. The in-panel label is whichever of the two the panel
    # does NOT already fix.
    if facet is None:
        groups = [(None, pts)]
        label_key = 1                       # chinchilla
    elif facet == "size":
        groups = [(z, [q for q in pts if q[0] == z]) for z in SIZES
                  if any(q[0] == z for q in pts)]
        label_key = 1
    elif facet == "chinchilla":
        chins = sorted({q[1] for q in pts})
        groups = [(c, [q for q in pts if q[1] == c]) for c in chins]
        label_key = 0                       # size
    else:
        raise ValueError(facet)

    ncol = min(5, len(groups))
    nrow = -(-len(groups) // ncol)
    if facet is None:
        fig, axes = plt.subplots(1, 1, figsize=(7.2, 5.0), squeeze=False)
    else:
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow),
                                 squeeze=False, sharey=True)
    for i, (key, sel_all) in enumerate(groups):
        ax = axes[i // ncol][i % ncol]
        for size in SIZES:
            for opt in ("adamw", "muon"):
                sel = [q for q in sel_all if q[0] == size and q[2] == opt]
                if not sel:
                    continue
                ax.scatter([q[3] for q in sel], [q[4] for q in sel], s=48,
                           marker=SIZE_MARKER[size], color=OPT_COLOR[opt],
                           edgecolors="white", linewidths=0.6, alpha=0.9, zorder=3)
        for q in sel_all:
            lab = q[label_key]
            ax.annotate(f"{lab:g}" if isinstance(lab, float) else lab, (q[3], q[4]),
                        textcoords="offset points", xytext=(4, 3), fontsize=6,
                        color=OPT_COLOR[q[2]], alpha=0.8)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
        ax.tick_params(labelsize=8, colors=MUTED)
        if facet == "size":
            ax.set_title(f"{key} ({MODEL_TYPE[key]})", fontsize=10, color=INK)
        elif facet == "chinchilla":
            ax.set_title(f"chinchilla {key:g}", fontsize=10, color=INK)
        if i // ncol == nrow - 1 or facet is None:
            ax.set_xlabel("unperturbed held-out DCLM loss", fontsize=9, color=MUTED)
        if i % ncol == 0:
            # Faceted panels are short; the full label repeated per row
            # collides with itself, so use the short form (gamma is in the
            # suptitle).
            ax.set_ylabel(("relative degradation" if relative
                           else "degradation (nats)") if facet else ylab,
                          fontsize=9 if facet else 10, color=INK)
    for j in range(len(groups), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    # Optimizer = colour, size = marker shape. The optimizer entries use a
    # colour swatch rather than a circle, since a circle is also the 30M
    # marker; the size entries list only sizes actually drawn, and are
    # omitted when every panel is a single size (the title already says it).
    from matplotlib.patches import Patch
    handles = [Patch(color=OPT_COLOR[o], label=o) for o in ("adamw", "muon")]
    if facet != "size":
        handles += [plt.Line2D([], [], color=MUTED, marker=SIZE_MARKER[z],
                               linestyle="none", markersize=6.5, label=z)
                    for z in SIZES if any(q[0] == z for q in pts)]
    if facet is None:
        axes[0][0].legend(handles=handles, frameon=False, fontsize=8.5, ncol=2,
                          loc="upper left")
        axes[0][0].set_title(f"{what}\nevery tuned base; small label = chinchilla",
                             fontsize=11.5, color=INK)
        fig.tight_layout()
    else:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
        lab = "chinchilla" if facet == "size" else "size"
        fig.suptitle(f"{what}  —  one panel per {facet}; small label = {lab}",
                     fontsize=12, color=INK)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf  ({len(pts)} points)")


def plot_degradation_vs_loss_combined(raw, out, gamma=0.01, facet="size"):
    """Two rows sharing the facet columns: absolute degradation on top,
    relative (divided by the unperturbed loss) below. Each row has its own
    shared log-y; x is free per column, since the loss range moves with size.
    """
    def collect(relative):
        pts = []
        for (g, size, chin, opt), v in raw.items():
            if g is None or abs(g - gamma) > 1e-12:
                continue
            base = raw.get((None, size, chin, opt))
            if base is None or (size, chin, opt) in VS_LOSS_EXCLUDE:
                continue
            d = (v - base) / base if relative else v - base
            if d > 0:
                pts.append((size, chin, opt, base, d))
        return pts
    rows = [(False, collect(False)), (True, collect(True))]
    if not rows[0][1]:
        print(f"no points at gamma {gamma:g}")
        return
    if facet == "size":
        keys = [z for z in SIZES if any(q[0] == z for q in rows[0][1])]
        label_key, sel = 1, (lambda q, k: q[0] == k)
    elif facet == "chinchilla":
        keys = sorted({q[1] for q in rows[0][1]})
        label_key, sel = 0, (lambda q, k: q[1] == k)
    else:
        raise ValueError(facet)

    # Wrap at 5 columns: 9 chinchillas in one row are unreadable. The
    # absolute block occupies the first `per` rows, the relative block the
    # next `per`, so "absolute on top, relative below" still holds.
    ncol = min(5, len(keys))
    per = -(-len(keys) // ncol)
    fig, axes = plt.subplots(2 * per, ncol, figsize=(3.3 * ncol, 3.0 * 2 * per),
                             squeeze=False)
    for j in range(len(keys), per * ncol):          # blank unused cells
        for blk in range(2):
            axes[blk * per + j // ncol][j % ncol].axis("off")
    for r, (relative, pts) in enumerate(rows):
        for ci, key in enumerate(keys):
            ax = axes[r * per + ci // ncol][ci % ncol]
            c = ci % ncol
            here = [q for q in pts if sel(q, key)]
            for size in SIZES:
                for opt in ("adamw", "muon"):
                    s_ = [q for q in here if q[0] == size and q[2] == opt]
                    if s_:
                        ax.scatter([q[3] for q in s_], [q[4] for q in s_], s=44,
                                   marker=SIZE_MARKER[size], color=OPT_COLOR[opt],
                                   edgecolors="white", linewidths=0.6, alpha=0.9,
                                   zorder=3)
            for q in here:
                lab = q[label_key]
                ax.annotate(f"{lab:g}" if isinstance(lab, float) else lab,
                            (q[3], q[4]), textcoords="offset points",
                            xytext=(4, 3), fontsize=6, color=OPT_COLOR[q[2]],
                            alpha=0.8)
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
            ax.tick_params(labelsize=7.5, colors=MUTED)
            ax.set_title((f"{key} ({MODEL_TYPE[key]})" if facet == "size"
                          else f"chinchilla {key:g}")
                         + ("  (absolute)" if r == 0 else "  (relative)"),
                         fontsize=9, color=INK)
            if r == 1 and ci // ncol == per - 1:
                ax.set_xlabel("unperturbed DCLM loss", fontsize=8, color=MUTED)
            if c == 0:
                ax.set_ylabel("relative degradation\n(perturbed − base) / base"
                              if relative else "absolute degradation\nperturbed − base (nats)",
                              fontsize=8.5, color=INK)
        # Shared y within the metric block: same limits on every panel.
        block = [axes[r * per + i // ncol][i % ncol] for i in range(len(keys))]
        # Limits from the DATA with log-headroom, not from the panels'
        # autoscaled ranges: a panel whose points sit at the extreme (the
        # chinchilla-128 pair at 8e-2) otherwise lands them on the border.
        ys = [q[4] for q in pts]
        lo, hi = min(ys) / 1.6, max(ys) * 1.6
        for ax in block:
            ax.set_ylim(lo, hi)

    from matplotlib.patches import Patch
    handles = [Patch(color=OPT_COLOR[o], label=o) for o in ("adamw", "muon")]
    if facet != "size":
        handles += [plt.Line2D([], [], color=MUTED, marker=SIZE_MARKER[z],
                               linestyle="none", markersize=6.5, label=z)
                    for z in SIZES if any(q[0] == z for q in rows[0][1])]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.015))
    lab = "chinchilla" if facet == "size" else "size"
    fig.suptitle(f"Degradation under Gaussian weight perturbation ($\\gamma$ = "
                 f"{gamma:g}) vs model quality  —  one column per {facet}; "
                 f"top: absolute, bottom: relative; small label = {lab}",
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0.025, 1, 0.955))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf  ({2 * per} x {ncol})")


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="/mnt/localssd/perturbcache")
    p.add_argument("--out-dir", default=os.path.join(repo, "colm-moss-latex"))
    a = p.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)

    diffs, raw, chins = collect(a.cache)
    for g in GAMMAS:
        print(f"  gamma {g:<6g} {len(diffs.get(g, {})):3d} comparable cell(s)")
    plot(diffs, chins, os.path.join(a.out_dir, "perturb-muon-minus-adamw"))

    deg = degradation_diffs(raw)
    print("degradation-difference cells:",
          {f"{g:g}": len(v) for g, v in sorted(deg.items())})
    plot(deg, chins,
         os.path.join(a.out_dir, "perturb-degradation-diff-muon-minus-adamw"),
         title="Degradation under Gaussian weight perturbation: muon vs adamw",
         cbar_label="muon - adamw  loss degradation",
         caption="cell = (perturbed - unperturbed) for muon minus the same for "
                 "adamw    |    negative (blue) = muon degrades less    |    "
                 "white = no data")
    rel = degradation_diffs(raw, relative=True)
    REL_CAPTION = ("cell = 100 x (perturbed - unperturbed) / unperturbed for muon "
                   "minus the same for adamw    |    negative (blue) = muon "
                   "degrades less    |    white = no data")
    plot(rel, chins,
         os.path.join(a.out_dir, "perturb-reldegradation-diff-muon-minus-adamw"),
         title="Relative degradation under Gaussian weight perturbation: "
               "muon vs adamw",
         cbar_label="muon - adamw  degradation, % of own unperturbed loss",
         caption=REL_CAPTION)

    # Small-gamma versions: the large sigmas set the colour range for the full
    # figures, leaving the 0.005-0.05 regime nearly uniform. These rescale to it.
    small = [g for g in GAMMAS if g <= 0.02]
    plot(rel, chins,
         os.path.join(a.out_dir,
                      "perturb-reldegradation-diff-muon-minus-adamw-small"),
         title="Relative degradation under Gaussian weight perturbation: "
               "muon vs adamw ($\\gamma \\leq 0.02$)",
         cbar_label="muon - adamw  degradation, % of own unperturbed loss",
         caption=REL_CAPTION, gammas=small, clip_pct=85)
    plot(diffs, chins, os.path.join(a.out_dir, "perturb-muon-minus-adamw-small"),
         title="Robustness to Gaussian weight perturbation: muon vs adamw "
               "($\\gamma \\leq 0.02$)", gammas=small, clip_pct=85)
    plot(deg, chins,
         os.path.join(a.out_dir,
                      "perturb-degradation-diff-muon-minus-adamw-small"),
         title="Degradation under Gaussian weight perturbation: muon vs adamw "
               "($\\gamma \\leq 0.02$)",
         cbar_label="muon - adamw  loss degradation",
         caption="cell = (perturbed - unperturbed) for muon minus the same for "
                 "adamw    |    negative (blue) = muon degrades less    |    "
                 "white = no data",
         gammas=small, clip_pct=85)

    plot_curves(raw, os.path.join(a.out_dir, "perturb-loss-vs-chinchilla"))
    plot_curves(raw, os.path.join(a.out_dir, "perturb-degradation-vs-chinchilla"),
                degradation=True)

    for rel, stem in ((False, "perturb-degradation-vs-loss-g0.01"),
                      (True, "perturb-reldegradation-vs-loss-g0.01")):
        for facet, suffix in ((None, ""), ("size", "-by-size"),
                              ("chinchilla", "-by-chinchilla")):
            plot_degradation_vs_loss(raw, os.path.join(a.out_dir, stem + suffix),
                                     gamma=0.01, relative=rel, facet=facet)

    for facet in ("size", "chinchilla"):
        plot_degradation_vs_loss_combined(
            raw, os.path.join(a.out_dir, f"perturb-degradation-vs-loss-g0.01-by-{facet}-combined"),
            gamma=0.01, facet=facet)

    plot_curves_vs_size(raw, os.path.join(a.out_dir, "perturb-loss-vs-size"))
    plot_curves_vs_size(raw, os.path.join(a.out_dir, "perturb-degradation-vs-size"),
                        degradation=True)


if __name__ == "__main__":
    main()
