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
