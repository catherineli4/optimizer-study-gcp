# Existence checks — how the launcher decides what already exists on GCS

How every `runlocal` / `launch` / `drylaunch` determines which artifacts are
already trained before scheduling anything. Understanding this is what makes
every stage idempotent (rerun after failures = only the missing work runs) and
what makes run-name stability load-bearing.

## 1. The decision chain

```
executor.execute(stages)
  └─ build artifact tiers (dependency-ordered)
  └─ _filter_skipped_artifacts(tiers, rerun)          [experiments/executor.py]
       │   rerun=True  -> skip nothing (DANGER: includes dependency tiers —
       │                  always pair with --artifact <Class> on eval stages)
       │   otherwise, per tier, 32-way ThreadPoolExecutor over:
       └─ artifact.should_skip()                      [experiments/artifact.py]
            └─ default: return self.exists            (property, per class)
                 └─ blob_exists(<sentinel object>)    [launch_jolmo/utils.py]
```

The whole check happens BEFORE any task script is written. What survives the
filter becomes the task list ("Skipping N artifact(s) that already exist" is
this filter's output).

## 2. The sentinel object per artifact class

Existence is keyed on the artifact's FINAL output object — the last thing its
task uploads — never on a directory listing of intermediates:

| artifact | sentinel object checked |
| --- | --- |
| `JolmoModel` (pretrain) | `<project>/JolmoModel/<run_name>/final-unsharded/model.pt` |
| `CPTModel` (finetune) | `<project>/CPTModel/<dataset>/<run_name>/final-unsharded/model.pt` |
| `ModelEvaluation` | `<project>/ModelEvaluation/<run_name>[-<tag>]-eval.json` |
| `MultiSeedPerturbedEvaluation` | `<project>/ModelEvaluation/<run_name>_multiseed-eval.json` |

Because the sentinel is the LAST upload of a `set -e` task script, a partially
finished run (trained + saved sharded `final/` but died before unshard/upload)
counts as NOT existing and is rerun. Corollary: a truncated `final/` left by a
killed upload can break the rerun's resume — delete that artifact dir on GCS if
a resume errors on a missing shard.

## 3. `blob_exists` — the fast path            [launch_jolmo/utils.py]

One metadata GET against the GCS JSON API via a cached `google-cloud-storage`
client:

- ~60 ms steady state; cost scales with #artifacts checked, NOT with how many
  objects live in the directory (crucial for `ModelEvaluation/`, which holds
  thousands of JSONs).
- `timeout=30` on the request — one wedged connection can no longer hang a
  whole sweep's pre-flight (it used to).
- Positive results are memoised in a process-local `_exists_cache` set.
  Negative results are NOT cached (so a dependency built earlier in the same
  run is seen by later checks).
- Any client/credential error falls back to §4.

## 4. `check_exists_remote` — the gsutil fallback

- `cache_depth=0`: one `gsutil ls <exact path>` subprocess (30 s timeout);
  exit code 0 = exists.
- `cache_depth=N>0`: lists a parent prefix N levels up ONCE, caches every
  object found into `_exists_cache`, and records the prefix in
  `_listed_prefixes`; later paths covered by a recorded listing resolve with
  zero network. Used where many siblings will be probed.

## 5. Parallelism

`_filter_skipped_artifacts` runs `should_skip()` under a 32-worker thread pool
per tier (order-preserving). ~2,600 artifacts filter in seconds instead of
minutes. The storage client and the caches are shared across threads (reads
are HTTP GETs; the set/dict writes are GIL-atomic).

## 6. Name determinism is the contract

The sentinel path embeds `run_name`, so the check is only as good as name
stability:

- Every swept axis must be IN the name (LR, WD, batch tag, chinchilla, replay
  fraction, ...), formatted identically forever (`_lr_tag`, `_wd_tag`,
  `_bs_tag`, `_chin_tag`). A cosmetic rename silently retrains everything —
  the `LrBatchSweep` prefix MUST stay `PTSweep{SIZE}` for exactly this reason.
- Additive axes must leave the zero-case name unchanged: `replay_dclm=0`
  appends nothing, so all pre-replay CPT artifacts keep matching.
- `$OPTIM_SIZE` selects the project (bucket prefix) via
  `~/.experiments/projects/<Project>/project.json` — the same run name under a
  different size is a different object, which is why sweep objects carry a
  `size_ok` guard.

## 7. Discovery-style stages (the inverse direction)

`cpt-all-lrs` and `cpt-all` don't enumerate a grid and stat each cell — they
LIST `<project>/JolmoModel/` once (`_existing_jolmo_runs()`, cached per
process), parse run names back into model params by regex, and build artifacts
only for what exists. Listing scales with directory size but is one request;
use it when the question is "what do we have" rather than "does this cell
exist".

## 8. Interaction with `--rerun`, `--artifact`, `--head`

- `--rerun`: bypasses the filter entirely — INCLUDING dependency tiers, so a
  bare `--rerun` on an eval stage would retrain every model underneath it.
  Always combine with `--artifact <ClassName>` to restrict rerun to one
  artifact class (deps are then assumed satisfied).
- `--head N` / `--tail N`: applied AFTER the skip filter — "first N of what's
  actually missing", which is what makes `--head 1` a cheap smoke test.
