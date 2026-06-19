import pytest

from scripts.run_fp16_hierarchical_interaction_probe import _validate_scored_decode_contract


def test_memory_lookup_scored_eval_requires_diagnostic_flag():
    with pytest.raises(RuntimeError, match="Scored evaluation with memory_lookup is disallowed"):
        _validate_scored_decode_contract(
            decode_mode="memory_lookup",
            allow_memory_lookup=True,
            memory_lookup_diagnostic=False,
        )


def test_memory_lookup_requires_explicit_opt_in():
    with pytest.raises(RuntimeError, match="memory_lookup decode mode is disabled by default"):
        _validate_scored_decode_contract(
            decode_mode="memory_lookup",
            allow_memory_lookup=False,
            memory_lookup_diagnostic=True,
        )


def test_mirror_decoder_allows_scored_eval():
    _validate_scored_decode_contract(
        decode_mode="mirror_decoder",
        allow_memory_lookup=False,
        memory_lookup_diagnostic=False,
    )
