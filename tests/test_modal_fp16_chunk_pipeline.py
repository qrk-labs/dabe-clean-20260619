import pytest

from src.training.modal_fp16_chunk_pipeline import parse_stage_list


def test_parse_stage_list_accepts_csv():
    assert parse_stage_list("preprocess,train") == ["preprocess", "train"]


def test_parse_stage_list_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown fp16 modal stages"):
        parse_stage_list("preprocess,unknown_stage")

