Run these commands from the repository root after `source activate isometry_pursuit`:

```sh
python infra/aws/manage.py build
python infra/aws/manage.py deploy
python infra/aws/manage.py status
python infra/aws/manage.py budget
python infra/aws/manage.py prepare
python infra/aws/manage.py start --run-id pilot-001
python infra/aws/manage.py download --run-id pilot-001
python infra/aws/plot.py results/cloud/pilot-001
```

Wait for stack creation to finish before initializing the budget or submitting work.
`prepare` uploads the existing real-data tensors and a pilot configuration.
`prepare --full` selects all 75 real replicates and 3250 saved synthetic seed specifications.
Neither command starts optimization; `start` submits the prepared configuration.

There is one preparation Lambda per dataset (Iris, Wine, Ethanol, synthetic),
and one shared optimizer Lambda. Preparation runs in parallel, followed by a
barrier. Optimization reads the saved matrices from S3. Real-data preparation
extracts the existing tensor replicates so comparisons use the paper's samples;
it does not repeat the original raw-data preprocessing. Synthetic preparation
uses the original generator and saved data seeds. Both optimizers start at zero;
optimizer seeds are not varied. SCS `scale` and direct ADMM `rho` remain separate,
strictly positive settings.

The pilot compares five parameter values (0.01, 0.1, 1, 10, 100) for each solver, for ten jobs per
sample, holding zero initialization and the search seed fixed. Results include raw
coefficients, thresholds, statuses, residuals, settings, versions, data hashes,
and code hashes. Searches exceeding 128 candidate columns are explicitly marked
`column_limit`; these results must not be counted as completed searches. Solves
have a 25-second limit, and worker invocations a 45-second limit. A cloud pilot
must establish suitable limits before using this for final scientific results.
No minimum-Frobenius guarantee is imposed.

The DynamoDB ledger reserves $0.0013 per optimization job plus $0.10 per dataset
preparation. Its cumulative $80 allowance never resets and never refunds unused
reservations. This conservatively covers the fixed worker timeouts, requests,
state transitions and small output writes. Failed batches consume their full
reservation. No task retries are configured. Redrive should not be used to
repeat work against an existing reservation; submit a new execution instead.

A separate $100 account cost budget covers a one-year custom billing period.
Its SNS notification at $75 locks this project's ledger, throttles its workers,
and stops its running workflows. Billing notifications are delayed, so this is
not a guaranteed account-wide hard cap. The workload allowance leaves $20 for
storage, logs and other charges. Other account activity, taxes and continuing
storage charges are outside the workload counter. Result objects expire after
90 days; download results locally before then. Data and deployment artifacts
are retained. There is no provisioned concurrency, NAT gateway or scheduled job.

Credentials stay local in the `isometry` AWS profile. They are never packaged
into the Lambda archive. The temporary `.env` copy is not used for deployment.

To validate the original implementation on the frozen simulation run 78 before
varying optimizer settings:

```sh
python infra/aws/reproduce.py prepare baseline
python infra/aws/manage.py start --run-id reproduce-run78-baseline
python infra/aws/manage.py download --run-id reproduce-run78-baseline
python infra/aws/reproduce.py verify baseline
python infra/aws/reproduce.py prepare sweep
python infra/aws/manage.py start --run-id reproduce-run78-zero-sweep
python infra/aws/manage.py download --run-id reproduce-run78-zero-sweep
python infra/aws/reproduce.py verify sweep
python infra/aws/plot.py results/cloud/reproduce-run78-zero-sweep
```

Wait for each execution to finish before downloading and verifying it. The
sweep configuration is gated on successful baseline verification. It compares
zero initialization, five SCS scales (0.01, 0.1, 1, 10, 100),
and five direct ADMM rho values (0.01, 0.1, 1, 10, 100), for ten jobs on the same saved
matrix. The data seed (1312) and downstream search seed remain fixed.
The baseline preserves the original stopping settings; the comparison uses
explicit absolute and relative tolerances of 1e-8. No support expansion or
minimum-Frobenius selection is applied.

The earlier `reproduce-run78-sweep` execution preserves the 18 random-start
results. To recheck those historical outputs, use
`python infra/aws/reproduce.py verify sweep --run-id reproduce-run78-sweep`.

`python infra/aws/smoke.py` prepares a solver-only smoke test on Iris, Wine,
Ethanol, and three synthetic matrices (the known P=10 failure, D=16/P=80,
and D=64/P=1072). It creates 60 zero-start jobs and does not launch them.
Searches are disabled for this convergence/runtime check. Launch the prepared
submission with `python infra/aws/manage.py start --run-id zero-start-smoke-001`.

The completed `zero-start-smoke-001` run contains the 60 solver-only tests.
`large-zero-smoke-001` contains 13 follow-up tests on a private version of the
same Lambda, using 3072 MB, a 300-second invocation timeout, a 240-second solver
limit and up to 500000 iterations. These were direct authenticated Lambda calls;
the normal Step Functions worker configuration was restored after publishing
the larger version. `large-zero-accuracy-001` is one tighter-tolerance ADMM control.

Audit downloaded outputs without running an optimizer:

```sh
python infra/aws/audit_smoke.py results/cloud/zero-start-smoke-001
python infra/aws/audit_smoke.py results/cloud/large-zero-smoke-001
python -m isometry_pursuit.exec.plot_optimizer_support_stability
```

The last command generates the supplemental figure from the saved real-data
checks and their originally reported supports. It includes the capped Ethanol
estimates and the longer reruns. Only one fixed replicate per real dataset was
checked. This supplemental analysis is the current scope; the full experiment
suite has not been rerun.

The `large-grid-20261116-001` batch uses one fixed replicate from each of the
65 synthetic D=32/64 conditions and all ten zero-start solver settings (650
jobs). Its configuration and execution record are in
`results/cloud/large-grid-setup/`. `launch_large_grid.py` reserves $13.10 in
total, including preparation, and pins the execution to an immutable workflow
version invoking `isometry-optimize:1` (3072 MB, 300-second invocation limit,
240-second solver limit, 500000 iterations). Greedy searches are disabled.
Normal workflow routing is restored after submission; the launched execution
retains its published definition. The existing lifetime ledger and billing
shutdown continue to apply. The script refuses to resubmit an existing local
execution record; do not retry a failed launch by deleting reservation records.

Greedy searches now have a dedicated private Lambda, `isometry-search`
(3072 MB, 300 seconds, reserved concurrency 8). The workflow accepts
`search_jobs_key` and `run_prefix` to run a search-only Distributed Map over
already frozen matrices. It does not rerun the optimizer. Each task runs one
forward or backward initializer and fixed-cardinality swap refinement, saving
original column indices, the Moore-Penrose row-norm loss, separate initializer
and swap timings, data/code hashes, and an explicit completion/omission status.
The worker has a 240-second search limit and uses the existing `1e9` backward
arithmetic-work budget; there is no 128-column limit. Budget shutdown includes
the new function, and it has no Function URL.

`large-search-20261116-001` joins the 650 large-grid optimizer results to all
four search methods. Identical matrix/candidate-support/search-seed combinations
share a search task: 278 unique tasks cover 2600 paired method evaluations.
The search seed is the original saved simulation seed. Synthetic searches use
the raw unit-column matrix, matching the original synthetic experiment. The
batch reserves $5.56, plus $0.04 for two cloud smoke tests.

```sh
python infra/aws/search_grid.py status
python infra/aws/search_grid.py download
python -m isometry_pursuit.exec.update_cloud_search_figures
python -m isometry_pursuit.exec.plot_synthetic_excess_loss_vs_dictionary \
  --input results/cloud/large-search-20261116-001/figure3_search_runs.json
```

The join command refuses partial search batches. Full-dictionary searches are
shown once per frozen sample; IP-restricted searches retain all ten optimizer
settings. Smaller synthetic dimensions and real-data comparisons remain the
original saved experiments. Optimizer limits and search omissions remain
explicit in the joined data.

The follow-up `backward-time-budget-20261116-001` reran the 28 work-budget
omissions with no arithmetic cutoff and a 245.37-second search limit, reserving
$0.56 plus a $0.04 smoke check. Of these, 23 completed, four stopped during
initialization, and one stopped during swaps. The subsequent
`swap-completion-20261116-001` reserves $0.06 to rerun that last case with the
cutoff applied only to initialization. The search worker now has a 900-second
Lambda limit to leave room for swap refinement; future launches reserve $0.06
per task. The search-only worker was updated directly; `stack.py` records the
same 900-second configuration for future CloudFormation deployments.
The swap rerun completed in 208.9 seconds of initialization and 36.7 seconds of
swaps. Its result is merged with the 28 replacements in
`results/cloud/swap-completion-20261116-001/merged_searches.json`; original raw
results remain intact. The cumulative workload reservation is $20.9065, not a
measurement of actual billed spend.

The main large-case figure uses only SCS at `scale=0.1`. The full parameter grid
and older small-D results are supplementary. Both main panels use the same 65
frozen Lambda matrices; separate local benchmark points are omitted from the
main runtime panel. Pass `--all-settings` to the figure updater for the full-grid
runtime plot; its default writes `figure3_main_search_runs.json` and the
fixed-setting runtime plot. Always pass the completed replacement batch via
`--search-overrides` when rebuilding, preserving original raw result files.

```sh
python -m isometry_pursuit.exec.update_cloud_search_figures \
  --search-overrides results/cloud/swap-completion-20261116-001/merged_searches.json
python -m isometry_pursuit.exec.plot_synthetic_excess_loss_vs_dictionary \
  --input results/cloud/large-search-20261116-001/figure3_main_search_runs.json
```

The full synthetic follow-up uses all 130 original conditions at
`D=2,4,8,16,32,64`, with 25 fixed data replicates per condition (3250 matrices).
`full_synthetic.py` runs the 3185 missing optimizations with the same published
`isometry-optimize:1` worker and SCS scale 0.1; the other 65 results are reused
from the large-grid batch. Frozen inputs and coefficients are downloaded and
SHA-256 checked by its `assets` command, then independently checked by
`audit_smoke.py` without refitting. Full-grid optimizer results, search results,
and aliases are joined in `results/cloud/full-synthetic-20261116-001/`.
Qualified job IDs retain the original source batch and raw job ID.

`full_synthetic_search.py` prepares 12086 new unique searches and reuses 260
search results for the 65 existing large matrices, providing all 13000 pipeline
links. Searches have no arithmetic or column-count cutoff. The
245.365741068-second limit applies only to initialization; swaps can use the
remaining portion of the 900-second Lambda invocation. Each invocation holds
its full cost exposure inside the already reserved $8 search allocation;
unused capacity can fund later invocations within that allocation. The lifetime
reservation is never refunded, and no new reservation is made during recovery.

The initial full search execution stopped after 20 completed scientific
searches because two concurrent budget-release transactions conflicted.
`results/cloud/full-search01-setup/execution_initial_failed.json` preserves its
control record. The recovery retries transaction conflicts with idempotency
tokens, guards lease release by its active state, skips the 20 completed
searches, and retains both old $0.06 holds. Its immutable worker is
`isometry-search:2` and its workflow version is 3. The original metered worker
and workflow versions remain intact. Recovery controls are saved in
`recovery.json`; `execution.json` selects the current execution for monitoring.
The `recover` command is a guarded one-time operation, not a general rerun or
budget-reset command.

```sh
source activate isometry_pursuit
python infra/aws/full_synthetic.py status
python infra/aws/full_synthetic_search.py status
python infra/aws/full_synthetic_search.py download
python -m isometry_pursuit.exec.update_cloud_search_figures \
  --full-grid-bundle results/cloud/full-synthetic-20261116-001 --audit-only
python infra/aws/audit_full_search.py
python -m isometry_pursuit.exec.update_cloud_search_figures \
  --full-grid-bundle results/cloud/full-synthetic-20261116-001
python -m isometry_pursuit.exec.plot_synthetic_excess_loss_vs_dictionary \
  --input results/cloud/full-synthetic-20261116-001/figure3_main_search_runs.json
python -m isometry_pursuit.exec.plot_optimizer_defeats \
  --synthetic-bundle results/cloud/full-synthetic-20261116-001
```

Publish the full main figure only after the coefficient audit and full-grid
join audit pass. The join rejects missing runs, unmatched hashes/seeds/shapes,
unexpected solver settings, unresolved search statuses, and incorrect candidate
supports. Main runtime observations use exactly the same fixed matrices, actual
observed IP iteration counts, and pure-initializer Lambda timings; swap timings
are omitted. The earlier parameter-grid figures remain supplementary.
