> **Development freeze:** DABE is frozen as the publication artifact for the preprint at https://doi.org/10.5281/zenodo.20797432. Do not run new experiments or alter scientific claims unless the user explicitly unfreezes development. See [`FREEZE_NOTICE.md`](../FREEZE_NOTICE.md).

# Experiments

Each run creates a timestamped directory with:

- `config.yaml` — frozen experiment config
- `results.json` — final metrics and metadata
- `checkpoints/` — model checkpoints (Lightning `.ckpt`)
- `wandb/` — local W&B logs (if sync disabled)

## Running Experiments

See the main [README](../README.md#running-experiments).

## Directory Naming

```
{experiment_name}_{YYYYMMDD_HHMMSS}/
```

Example: `density_adaptive_20260101_120000/`
