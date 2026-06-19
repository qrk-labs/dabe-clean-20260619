import pytest

from src.training.modal_feasibility_pipeline import (
    compute_competitive_gate,
    parse_stage_list,
)


def test_parse_stage_list_accepts_csv():
    assert parse_stage_list("tokenizer,dabe_lm,bpe_baseline") == [
        "tokenizer",
        "dabe_lm",
        "bpe_baseline",
    ]


def test_parse_stage_list_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown feasibility stages"):
        parse_stage_list("tokenizer,unknown_stage")


def test_compute_competitive_gate_true_when_all_conditions_pass():
    assert compute_competitive_gate(
        dabe_val_loss=4.2,
        bpe_val_loss=4.0,
        compression_ratio_vs_bpe=1.15,
        dabe_stable=True,
        bpe_stable=True,
        loss_tolerance_ratio=0.1,
        min_compression_ratio=1.0,
    )


def test_compute_competitive_gate_false_when_missing_metrics():
    assert not compute_competitive_gate(
        dabe_val_loss=None,
        bpe_val_loss=4.0,
        compression_ratio_vs_bpe=1.15,
        dabe_stable=True,
        bpe_stable=True,
        loss_tolerance_ratio=0.1,
        min_compression_ratio=1.0,
    )
