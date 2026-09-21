"""CPT learning-vs-forgetting at ONE chinchilla, one panel per model size.

    x = forgetting, as DEGRADATION: DCLM_heldout(CPT model) - DCLM_heldout(its
        tuned pretrain base)            (log axis; lower = forgot less)
    y = learning: the CPT dataset's validation loss   (lower = learned more)

Every point is one fine-tune (one CPT learning rate) of the tuned base for that
(size, chinchilla, pretrain optimizer). Colour = PRETRAIN optimizer, marker =
fine-tuning arm, and the line is the Pareto front (non-dominated set) per
pretrain optimizer -- so "which pretrain optimizer gives the better trade-off"
is read from which front sits lower-left. Plain fine-tunes only by default;
--include-replay adds the DCLM-replay variants.

Only evals on the standard held-out sets (4,193,280 tokens for both DCLM and
the dataset validation) are used; others are counted and dropped.

    python -m new_utils.plot_cpt_lf_bysize --dataset stackmathqa --chinchilla 2
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

from new_utils.plot_pt_lr_dclm_all import MODEL_TYPE, tuned_adamw_lrs, tuned_lrs

BUCKET = "gs://cmu-gpucloud-catheri4"
COLOR = {"adamw": "#2a78d6", "muon": "#eb6834"}
MARK = {"adamw": "o", "muon": "^"}
INK, MUTED = "#0b0b0b", "#52514e"
STD_TOKENS = 4193280


def _tag(lr):
    m, e = f"{lr:.1e}".split("e")
    return f"{m}e{int(e)}"


def base_names(size, chin, opt):
    """Both naming schemas for the tuned base; whichever has an eval is used."""
    mt, c = MODEL_TYPE[size], f"{chin:g}"
    a = tuned_lrs(size)["adamw"].get(chin)
    ml, comp = tuned_lrs(size)["muon"].get(chin), tuned_adamw_lrs(size).get(chin)
    if opt == "adamw":
        if a is None:
            return []
        core = f"adamw-lr{_tag(a)}"
    else:
        if ml is None or comp is None:
            return []
        core = f"muon-muonlr{_tag(ml)}-adamwlr{_tag(comp)}"
    return [f"PTSweep{size}-{mt}-chinchilla-{c}-{core}-wd0.1-bs1M-wsd",
            f"MuonExpt3-{mt}-chinchilla-{c}-{core}-wsd"]


def fetch(size, names, dataset, cache):
    d = os.path.join(cache, size)
    os.makedirs(d, exist_ok=True)
    pats = []
    for n in names:
        pats += [f"{BUCKET}/Optim-{size}-tuning/ModelEvaluation/{n}-eval.json",
                 f"{BUCKET}/Optim-{size}-tuning/ModelEvaluation/{n}-CPT-{dataset}-*-eval.json"]
    subprocess.run(["gsutil", "-q", "-m", "cp", "-n"] + pats + [d],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return d


def collect(size, chin, dataset, cache, include_replay, sync):
    """{pretrain_opt: [(degradation, val_loss, ft_arm, lr, replay)]}, dropped count"""
    out, dropped = {}, 0
    val_label = f"{dataset}-validation"
    for opt in ("adamw", "muon"):
        names = base_names(size, chin, opt)
        if not names:
            continue
        d = fetch(size, names, dataset, cache) if sync else os.path.join(cache, size)
        base = next((n for n in names
                     if os.path.exists(os.path.join(d, f"{n}-eval.json"))), None)
        if base is None:
            continue
        b = json.load(open(os.path.join(d, f"{base}-eval.json")))["by_label"].get("DCLM_heldout")
        if not b or b["num_tokens"] != STD_TOKENS:
            continue
        rx = re.compile(re.escape(base) + rf"-CPT-{dataset}-\d+M-(adamw|muon)-lr([0-9.e+\-]+)-exp"
                        r"(?:-muonlr[0-9.e+\-]+)?(?:-replay([0-9.]+))?-eval\.json$")
        for f in glob.glob(os.path.join(d, f"{base}-CPT-{dataset}-*-eval.json")):
            m = rx.match(os.path.basename(f))
            if not m:
                continue
            replay = float(m.group(3)) if m.group(3) else 0.0
            if replay and not include_replay:
                continue
            lab = json.load(open(f)).get("by_label", {})
            dc, vl = lab.get("DCLM_heldout"), lab.get(val_label)
            if not dc or not vl:
                continue
            if dc["num_tokens"] != STD_TOKENS or vl["num_tokens"] != STD_TOKENS:
                dropped += 1
                continue
            out.setdefault(opt, []).append(
                (dc["loss"] - b["loss"], vl["loss"], m.group(1), float(m.group(2)), replay))
    return out, dropped


def pareto(points):
    """Non-dominated (min x, min y) subset, sorted by x."""
    front, best_y = [], float("inf")
    for x, y in sorted(points):
        if y < best_y:
            front.append((x, y)); best_y = y
    return front


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="stackmathqa")
    p.add_argument("--chinchilla", type=float, default=2.0)
    p.add_argument("--sizes", nargs="+", default=list(MODEL_TYPE))
    p.add_argument("--cache", default="/mnt/localssd/cpt-lf-cache")
    p.add_argument("--out-dir", default=os.path.join(repo, "colm-moss-latex"))
    p.add_argument("--include-replay", action="store_true")
    p.add_argument("--no-sync", action="store_true")
    a = p.parse_args()

    panels = []
    for size in a.sizes:
        data, dropped = collect(size, a.chinchilla, a.dataset, a.cache,
                                a.include_replay, not a.no_sync)
        n = sum(len(v) for v in data.values())
        print(f"{size}: {n} fine-tune(s)" + (f", {dropped} dropped (non-standard eval set)" if dropped else ""))
        if n:
            panels.append((size, data))
    if not panels:
        raise SystemExit("no data")

    fig, axes = plt.subplots(1, len(panels), figsize=(3.9 * len(panels), 4.2), squeeze=False)
    for i, (size, data) in enumerate(panels):
        ax = axes[0][i]
        for opt in ("adamw", "muon"):
            pts = [q for q in data.get(opt, []) if q[0] > 0]
            nonpos = len(data.get(opt, [])) - len(pts)
            if nonpos:
                print(f"  {size} {opt}-pretrained: {nonpos} point(s) with degradation <= 0 "
                      f"cannot go on a log axis -- dropped")
            for arm in ("adamw", "muon"):
                sel = [q for q in pts if q[2] == arm]
                ax.scatter([q[0] for q in sel], [q[1] for q in sel], s=30,
                           marker=MARK[arm], color=COLOR[opt], alpha=0.75,
                           edgecolors="white", linewidths=0.5, zorder=3)
            fr = pareto([(q[0], q[1]) for q in pts])
            if len(fr) > 1:
                ax.plot([x for x, _ in fr], [y for _, y in fr], color=COLOR[opt],
                        linewidth=1.9, zorder=2)
        ax.set_xscale("log")
        ax.set_title(f"{size} ({MODEL_TYPE[size]})", fontsize=10.5, color=INK)
        ax.set_xlabel("DCLM degradation vs tuned base (nats)", fontsize=9, color=MUTED)
        if i == 0:
            ax.set_ylabel(f"{a.dataset} validation loss", fontsize=10, color=INK)
        ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
        ax.tick_params(labelsize=8, colors=MUTED)
    from matplotlib.patches import Patch
    handles = [Patch(color=COLOR[o], label=f"{o}-pretrained base (line = Pareto front)")
               for o in ("adamw", "muon")]
    handles += [plt.Line2D([], [], color=MUTED, marker=MARK[m], linestyle="none",
                           markersize=6.5, label=f"{m} fine-tune") for m in ("adamw", "muon")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"CPT on {a.dataset}: learning vs forgetting at chinchilla "
                 f"{a.chinchilla:g}, by model size"
                 + ("  (incl. replay)" if a.include_replay else ""),
                 fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"cpt-lf-bysize-deg-{a.dataset}-c{a.chinchilla:g}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {out}.png / .pdf")


if __name__ == "__main__":
    main()
