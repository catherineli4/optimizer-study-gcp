# ft-replay: what was evaluated, and how to plot it

Completed 2026-09-06T01:44:38Z. All four legs rc=0; 2161 tasks executed, zero
failures / OOMs / tracebacks.

## Where the numbers live

    gs://cmu-gpucloud-catheri4/Optim-{SIZE}-tuning/ModelEvaluation/{name}-eval.json

`{SIZE}` is `60M` or `100M`. Each JSON is:

```json
{"by_label": {"<label>": {"loss": float, "num_tokens": int}, ...},
 "overall": {...}, "model_type": "0.06B" | "0.1B"}
```

Labels present on every replay eval:

| label | tokens (100M sample) | use |
|---|---|---|
| `DCLM_heldout` | 4,193,280 | **pretrain-domain retention — the forgetting axis** |
| `<dataset>-validation` | ~230k–1M | **target-domain gain — e.g. `gsm8k-validation`** |
| `C4_val` `Books_val` `Wiki_val` `Reddit_val` | 0.8M–2.3M | secondary drift checks |

`DCLM_heldout` vs `<dataset>-validation` is the forgetting/plasticity
trade-off this sweep exists to measure; replay fraction is the knob.

## Design

8 pretrained bases per size = **chinchilla {1, 2, 4, 8} x pretrain optimizer
{adamw, muon}**. Every base is finetuned over the full cross product:

- **CPT dataset** (5): `starcoder` `musicpile` `alpaca` `gsm8k` `stackmathqa`
- **CPT LR** (8, per arm — see tags below)
- **replay fraction** (3): `0.1` `0.2` `0.3`
- **CPT optimizer arm**: adamw always; muon-*pretrained* bases additionally
  get a muon CPT arm

So an adamw-pretrained base yields 5x8x3 = **120** finetunes and a
muon-pretrained base **240**. Per size: 4x120 + 4x240 = **1440**.

CPT budget is 20M tokens, cosine schedule, seq len 1024, gbs 65,536.

### Pretrained bases (exact names)

60M:
```
MuonExpt3-0.06B-chinchilla-1-adamw-lr2.0e-2-wsd
PTSweep60M-0.06B-chinchilla-1-muon-muonlr1.4e-2-adamwlr2.0e-2-wd0.1-bs1M-wsd
MuonExpt3-0.06B-chinchilla-2-adamw-lr1.0e-2-wsd
MuonExpt3-0.06B-chinchilla-2-muon-muonlr1.0e-2-adamwlr1.0e-2-wsd
MuonExpt3-0.06B-chinchilla-4-adamw-lr1.0e-2-wsd
PTSweep60M-0.06B-chinchilla-4-muon-muonlr1.4e-2-adamwlr1.0e-2-wd0.1-bs1M-wsd
MuonExpt3-0.06B-chinchilla-8-adamw-lr1.4e-2-wsd
PTSweep60M-0.06B-chinchilla-8-muon-muonlr1.0e-2-adamwlr1.4e-2-wd0.1-bs1M-wsd
```

100M:
```
MuonExpt3-0.1B-chinchilla-1-adamw-lr1.4e-2-wsd
MuonExpt3-0.1B-chinchilla-1-muon-muonlr1.0e-2-adamwlr1.4e-2-wsd
MuonExpt3-0.1B-chinchilla-2-adamw-lr1.0e-2-wsd
MuonExpt3-0.1B-chinchilla-2-muon-muonlr1.0e-2-adamwlr1.0e-2-wsd
MuonExpt3-0.1B-chinchilla-4-adamw-lr1.0e-2-wsd
PTSweep100M-0.1B-chinchilla-4-muon-muonlr1.0e-2-adamwlr1.0e-2-wd0.1-bs1M-wsd
MuonExpt3-0.1B-chinchilla-8-adamw-lr7.0e-3-wsd
MuonExpt3-0.1B-chinchilla-8-muon-muonlr1.0e-2-adamwlr7.0e-3-wsd
```

**Both naming schemas appear in one sweep.** `MuonExpt3-<mt>-chinchilla-<c>-...-wsd`
and `PTSweep<SIZE>-<mt>-chinchilla-<c>-...-wd0.1-bs1M-wsd` are the SAME
configuration under two names. Parse the prefix out; never treat them as
separate series or you get phantom duplicate points.

## Name grammar

    <base>-CPT-<dataset>-20M-<arm>-replay<r>

adamw arm:  `-adamw-lr<LR>-exp`
muon arm:   `-muon-lr<ADAMW_COMPONENT>-exp-muonlr<MUON_LR>`

Eval file = that string + `-eval.json`.

### CPT LR tags actually present

adamw arm (8): `1.0e-4 2.0e-4 5.0e-4 8.0e-4 1.0e-3 2.0e-3 5.0e-3 1.0e-2`

muon arm (8), as `(adamw_component, muon_lr)` — note the component is NOT the
flat 1e-4 in `CPT_LR_SWEEP`; it is scaled per cell, so key on the pair:

    (1.5e-4, 6.0e-4)  (2.0e-4, 8.0e-4)  (2.5e-4, 1.0e-3)  (5.0e-4, 2.0e-3)
    (1.0e-3, 4.0e-3)  (2.0e-3, 8.0e-3)  (2.5e-3, 1.0e-2)  (5.0e-3, 2.0e-2)

## Parsing

```python
import re
BASE = r"(?:MuonExpt3|PTSweep\d+M)-(?P<mt>[\d.]+B)-chinchilla-(?P<chin>[\d.]+)-"
PT_ADAMW = BASE + r"adamw-lr(?P<ptlr>[\d.e-]+)"
PT_MUON  = BASE + r"muon-muonlr(?P<ptmuon>[\d.e-]+)-adamwlr(?P<ptcomp>[\d.e-]+)"
CPT = (r"-CPT-(?P<ds>[a-z0-9]+)-20M-"
       r"(?:adamw-lr(?P<alr>[\d.e-]+)-exp"
       r"|muon-lr(?P<mcomp>[\d.e-]+)-exp-muonlr(?P<mlr>[\d.e-]+))"
       r"-replay(?P<replay>[\d.]+)$")
```

Cell key: `(size, chinchilla, pt_optimizer, cpt_dataset, cpt_arm, cpt_lr, replay)`.

**Keep the minimum loss when a key collides.** Several muon cells share a
`muon_lr` at different adamw components; plain assignment lets whichever was
read last win, which previously reported a non-optimal LR as "best". This bit
me once already — take `min`, never overwrite.

## Suggested figures

1. **Forgetting/plasticity frontier.** x = `<dataset>-validation` loss,
   y = `DCLM_heldout` loss. One point per CPT LR, one line per replay
   fraction, colour by pretrain optimizer. Subplot grid = dataset x chinchilla.
   Down-left is better; the replay knob should trace the frontier.
2. **Retention vs replay.** x = replay in {0.1, 0.2, 0.3}, y = `DCLM_heldout`
   loss at each cell's best CPT LR. Line per pretrain optimizer, subplot per
   dataset, one figure per size. Tests whether muon needs more or less replay.
3. **Degradation.** Same as (2) but y = `DCLM_heldout(cpt) - DCLM_heldout(base)`.
   Subtracting the pretrained baseline matters: on the perturbation study muon
   looked better on raw loss and worse on degradation.
4. **Best-CPT-LR table** per (size, chinchilla, optimizer, dataset, replay),
   mirroring `plot_pt_lr_dclm_bs.py`'s `write_table`.

## Caveats

- Only chinchillas **{1, 2, 4, 8}** were run. `OPTIM_REPLAY_CHINCHILLAS`
  defaults to the full tuned ladder (60M has 10, up to 128; 100M has 8, up
  to 32) if you want to extend.
- **There is no r=0 control in this sweep.** The no-replay baseline is the
  plain CPT run without a `-replay` suffix, from the earlier `ft_sweep`
  stage — pull those separately for any "replay vs none" comparison.
- 60M was already complete before this run (1447/1448 existed); only one
  `stackmathqa` muon cell was newly trained. 100M is entirely fresh: 1080
  finetunes + 1080 evals.

Reference plotting code: `new_utils/plot_pt_lr_dclm_bs.py` (caching, GCS
listing, min-keeping, table rendering).
