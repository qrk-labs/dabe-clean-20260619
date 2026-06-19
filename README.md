# DABE: Density-Adaptive Variable-Width Bitmask Encoding

A novel NLP architecture that replaces categorical token IDs with learned binary codes whose bit-width varies by information density.

## Architecture

```
Input Text: "The quick brown fox jumps over the lazy dog."
                    │
                    ▼
          ┌─────────────────────────┐
          │   Pass 1: Density Router │
          │   (entropy / learned /   │
          │    IB policy)            │
          │                          │
          │  Scores per-span density │
          └─────────┬───────────────┘
                    │
                    ▼
          ┌─────────────────────────┐
          │  Pass 2: Bitmask Encoder │
          │   (LFQ / semantic hash / │
          │    Gumbel)               │
          │                          │
          │  Span->binary code       │
          │  32b..8b by density      │
          └─────────┬───────────────┘
                    │
                    ▼
          ┌─────────────────────────┐
          │  BitmaskProjection       │
          │  (binary -> hidden dim)  │
          └─────────┬───────────────┘
                    │
                    ▼
          ┌─────────────────────────┐
          │  Transformer Backbone    │
          │  (causal, multi-layer)   │
          └─────────┬───────────────┘
                    │
                    ▼
          ┌─────────────────────────┐
          │  Output Head             │
          │  (LM / diffusion /       │
          │   contrastive)           │
          └─────────────────────────┘
```

**Key innovations:**
- **Density-adaptive**: high-density spans get wider codes (32-bit), low-density get narrower (8-bit)
- **Learnable bits**: each bit carries learned semantic meaning (not arbitrary indices)
- **Language-agnostic**: contrastive objectives align bitmasks across languages
- **Header bits**: reserved bits encode bit-width granularity for the backbone

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"
```

## Running Experiments

```bash
# Baseline BPE
python run_experiment.py --config-name baseline_bpe

# Fixed-width bitmask (ablation)
python run_experiment.py --config-name fixed_width_bitmask

# Full density-adaptive run
python run_experiment.py --config-name density_adaptive

# Ablations
python run_experiment.py --config-name ablation/density_adaptive_lfq_learned_router
python run_experiment.py --config-name ablation/density_adaptive_gumbel_encoder

# Local Apple Silicon smoke run (MPS/Metal)
PYTORCH_ENABLE_MPS_FALLBACK=1 WANDB_MODE=offline python run_experiment.py --config-name local_metal

# Single-track feasibility pipeline (TinyStories tokenizer + DABE LM + BPE baseline)
PYTORCH_ENABLE_MPS_FALLBACK=1 WANDB_MODE=offline python run_experiment.py --config-name feasibility

# OLMo-mix small-sample feasibility smoke (local)
PYTORCH_ENABLE_MPS_FALLBACK=1 WANDB_MODE=offline python run_experiment.py --config-name feasibility_modal_olmo_smoke

# OLMo-mix modular feasibility smoke (Modal remote, detached/background)
# Runs tokenizer -> dabe_lm -> bpe_baseline programmatically (no shell subprocess wrapper)
modal run -d scripts/modal_feasibility_smoke.py::run_pipeline

# Same run with explicit run_id (recommended for stage restarts)
modal run -d scripts/modal_feasibility_smoke.py::run_pipeline --run-id "olmo_smoke_001"

# Stage-level relaunches (share artifacts via --run-id)
modal run -d scripts/modal_feasibility_smoke.py::run_tokenizer_stage --run-id "olmo_smoke_001"
modal run -d scripts/modal_feasibility_smoke.py::run_dabe_stage --run-id "olmo_smoke_001"
modal run -d scripts/modal_feasibility_smoke.py::run_bpe_stage --run-id "olmo_smoke_001"

# OLMo-mix modular smoke with overrides (example)
modal run -d scripts/modal_feasibility_smoke.py::run_pipeline --overrides "feasibility.lm_data.train_samples=2048 feasibility.lm_data.val_samples=256"

# Optimization pass v3 config (compile/profile/cache + V4-inspired toggles)
PYTORCH_ENABLE_MPS_FALLBACK=1 WANDB_MODE=offline python run_experiment.py --config-name feasibility_opt_v3

# Stage-wise profiling harness (tokenizer / dabe_lm / bpe_baseline metrics + profiler traces)
PYTORCH_ENABLE_MPS_FALLBACK=1 WANDB_MODE=offline python scripts/profile_feasibility_stages.py --config-name feasibility_opt_v3 --run-id opt_v3_local_001

# Detached Modal launch with explicit launch contract JSON (command + app id + logs command)
python scripts/launch_modal_detached.py --function scripts/modal_feasibility_smoke.py::run_pipeline --config-name feasibility_modal_olmo_opt_v3 --run-id olmo_opt_v3_001

# Optional timeout estimator based on prior runtime metrics for a stage
modal run scripts/modal_feasibility_smoke.py::estimate_stage_timeout_seconds --run-id olmo_opt_v3_001 --stage dabe_lm

# Adapter-route feasibility (standard Transformer + DABE adapter + matched baseline)
PYTORCH_ENABLE_MPS_FALLBACK=1 WANDB_MODE=offline python run_experiment.py --config-name adapter_feasibility

# Stabilized adapter-route feasibility (zero-impact init + calibrated adapter LR)
PYTORCH_ENABLE_MPS_FALLBACK=1 WANDB_MODE=offline python run_experiment.py --config-name adapter_feasibility_stabilized

# Stabilized adapter-route feasibility with non-zero width-init probe
PYTORCH_ENABLE_MPS_FALLBACK=1 WANDB_MODE=offline python run_experiment.py --config-name adapter_feasibility_stabilized_widthinit

# Local fast debug run (no network dataset dependency)
PYTORCH_ENABLE_MPS_FALLBACK=1 WANDB_MODE=offline python run_experiment.py --config-name local_metal data.synthetic_only=true

# Shape-only validation (no training)
python run_experiment.py experiment.train=False
```

For Apple Silicon, keep `data.num_workers=0` and start with `configs/local_metal.yaml` before scaling up.

## Tests

```bash
pytest tests/ -v --cov=src
```

## Project Structure

```
dabe/
├── src/
│   ├── density_router/    # Pass 1: density scoring implementations
│   ├── bitmask_encoder/   # Pass 2: binary encoding implementations
│   ├── backbone/          # Transformer with bitmask projection
│   ├── training/          # Training loops & objectives
│   ├── evaluation/        # Benchmarks & analysis
│   └── data/              # Data loading & preprocessing
├── configs/               # Hydra YAML experiment configs
├── experiments/           # Run outputs & checkpoints
├── research/              # Literature notes & drafts
└── tests/                 # Unit tests
```

## Research Questions

1. Can end-to-end training learn both density router and bitmask encoder jointly?
2. What is the optimal bit-width schedule for different density levels?
3. Does bitmask encoding preserve semantic structure across languages?
4. How does variable-width masking affect attention patterns and training stability?
5. Does this outperform BPE on multilingual benchmarks at comparable parameter counts?
