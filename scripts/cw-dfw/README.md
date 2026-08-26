# cw-dfw launch scripts

Site-specific launch wrappers for `cw-dfw-cs-001`. Not part of upstream AlpaGym; see
`ALPAGYM_RUNBOOK.md` for what each adaptation is for and why this cluster needs it.

`run_alpagym_clrl.sh` composes the Hydra overrides and runs the host CLI from
`.venv-host`. Run it from inside an allocation, outside the container:

```bash
srun --overlap --jobid <JOBID> bash -c 'bash <path>/run_alpagym_clrl.sh run'
```

`cfg` instead of `run` prints the composed config and exits.

`BASE` at the top is the only path to change for another user or cluster. The script
pins `max_num_steps=1` for a smoke run; override it for a longer one rather than editing
in place, so the committed copy stays the reference:

```bash
sed 's/max_num_steps=1/max_num_steps=5/' run_alpagym_clrl.sh > /tmp/run5.sh
```

## What it assumes exists

Everything under [What Persists Between Allocations](../../ALPAGYM_RUNBOOK.md#what-persists-between-allocations),
plus `imaginaire4` checked out beside `alpagym` — the script passes it as
`execution.slurm.posttrain_repo_root`, and the training entry is imported from there.
