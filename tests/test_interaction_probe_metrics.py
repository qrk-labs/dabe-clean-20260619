from pathlib import Path

import numpy as np
import torch

from scripts.run_fp16_interaction_probe import (
    _chunk_rows,
    _extract_first_json_object,
    _generate_continuation,
    _aggregate_probe_quality,
    _load_fixed_prompt_set,
    _normalize_judge_result,
    _resolve_openrouter_api_key,
    _rubric_score,
    _strict_quality_score,
)


class _DummyProbeTokenizer:
    def encode(self, prompt, add_special_tokens=False):
        del prompt, add_special_tokens
        return [1, 2, 3]

    def decode(self, token_ids, clean_up_tokenization_spaces=False):
        del clean_up_tokenization_spaces
        return " ".join(str(token_id) for token_id in token_ids)


class _EmptyMultiScalarCompressor:
    tokenizer = _DummyProbeTokenizer()

    def compress_token_ids(self, token_ids):
        del token_ids
        return np.empty((0, 8), dtype=np.float32)


class _ShapeCheckingProbeModel:
    def __call__(self, input_scalars, mask=None):
        del mask
        assert input_scalars.shape == (1, 4, 8)
        return torch.zeros((1, 4, 8), dtype=torch.float32), torch.zeros((1, 4, 16))


class _DummyProbeModule:
    model = _ShapeCheckingProbeModel()


def test_strict_quality_flags_mojibake():
    strict_ok, stats_ok = _strict_quality_score(
        prompt="Lily saw a puppy in the park.",
        continuation="Lily smiled at the puppy, then she shared her snack and walked home happily.",
        base_coherent=True,
        base_stats={
            "unique_ratio": 0.70,
            "max_word_ratio": 0.08,
            "four_gram_diversity": 0.95,
        },
    )
    assert strict_ok is True
    assert stats_ok["mojibake_marker_hits"] == 0.0
    assert stats_ok["replacement_char_count"] == 0.0

    strict_bad, stats_bad = _strict_quality_score(
        prompt="Lily saw a puppy in the park.",
        continuation="Lily said â€œhelloâ€ to the puppy and smiled �.",
        base_coherent=True,
        base_stats={
            "unique_ratio": 0.70,
            "max_word_ratio": 0.08,
            "four_gram_diversity": 0.95,
        },
    )
    assert strict_bad is False
    assert stats_bad["mojibake_marker_hits"] > 0.0 or stats_bad["replacement_char_count"] > 0.0


def test_aggregate_probe_quality_gate_pass_and_fail():
    good_rows = []
    continuations = [
        "A fox found a berry and shared it.",
        "A child drew a kite and ran outside.",
        "The puppy chased a leaf in the wind.",
        "A rabbit hopped under the apple tree.",
        "Mina packed lunch and waved to mom.",
        "A bird sang while rain tapped the roof.",
        "A tiny boat floated down the stream.",
        "Tom fixed the wheel and rode home.",
        "Nia found a map and showed her friend.",
        "The class planted seeds in the garden.",
    ]
    for idx in range(20):
        good_rows.append({
            "continuation": continuations[idx % len(continuations)],
            "strict_coherent": idx < 14,
            "strict_stats": {
                "replacement_char_count": 0.0,
                "mojibake_marker_hits": 0.0,
            },
        })

    good = _aggregate_probe_quality(good_rows)
    assert good["strict_coherent_rate"] >= 0.60
    assert good["dominant_continuation_share"] <= 0.35
    assert good["unique_continuations"] >= 8
    assert good["strict_gate_pass"] == 1.0

    bad_rows = []
    for idx in range(20):
        bad_rows.append({
            "continuation": "Repeated continuation for collapse test." if idx < 12 else f"Alt {idx}",
            "strict_coherent": idx < 16,
            "strict_stats": {
                "replacement_char_count": 1.0 if idx < 3 else 0.0,
                "mojibake_marker_hits": 0.0,
            },
        })
    bad = _aggregate_probe_quality(bad_rows)
    assert bad["dominant_continuation_share"] > 0.35
    assert bad["mojibake_rows"] >= 3
    assert bad["strict_gate_pass"] == 0.0


def test_rubric_score_penalizes_mojibake():
    rubric_pass, rubric = _rubric_score(
        prompt="Tim played in the park.",
        continuation="Tim played in the park and shared his ball with his friend.",
        base_stats={
            "alpha_ratio": 0.95,
            "unique_ratio": 0.75,
            "max_word_ratio": 0.10,
            "four_gram_diversity": 0.92,
            "overlap_ratio": 0.20,
            "word_count": 12.0,
        },
        strict_stats={
            "non_ascii_ratio": 0.0,
            "replacement_char_count": 0.0,
            "mojibake_marker_hits": 0.0,
        },
    )
    assert rubric_pass is True
    assert rubric["format_integrity"] == 2.0

    rubric_bad_pass, rubric_bad = _rubric_score(
        prompt="Tim played in the park.",
        continuation="Tim said â€œhelloâ€ to his friend � and waved.",
        base_stats={
            "alpha_ratio": 0.90,
            "unique_ratio": 0.65,
            "max_word_ratio": 0.10,
            "four_gram_diversity": 0.90,
            "overlap_ratio": 0.20,
            "word_count": 11.0,
        },
        strict_stats={
            "non_ascii_ratio": 0.10,
            "replacement_char_count": 1.0,
            "mojibake_marker_hits": 2.0,
        },
    )
    assert rubric_bad_pass is False
    assert rubric_bad["format_integrity"] == 0.0


def test_load_fixed_prompt_set_file():
    prompt_set = Path("research/eval/fixed_prompt_set_v1.jsonl")
    prompts = _load_fixed_prompt_set(prompt_set, num_prompts=10)
    assert len(prompts) == 10
    assert all(isinstance(prompt, str) and prompt.strip() for prompt in prompts)


def test_generate_continuation_pads_empty_multi_scalar_prompt():
    continuation, latency_ms = _generate_continuation(
        prompt="short",
        model_module=_DummyProbeModule(),
        compressor=_EmptyMultiScalarCompressor(),
        train_scalars=np.zeros((2, 8), dtype=np.float32),
        train_chunk_tokens=np.asarray([[11, 12], [21, 22]], dtype=np.int32),
        compressed_seq_len=4,
        continuation_chunks=1,
        device=torch.device("cpu"),
    )
    assert continuation == "11 12"
    assert latency_ms >= 0.0


def test_chunk_rows_respects_chunk_size():
    rows = [{"idx": i} for i in range(11)]
    chunks = _chunk_rows(rows, chunk_size=4)
    assert [len(chunk) for chunk in chunks] == [4, 4, 3]
    assert chunks[0][0]["idx"] == 0
    assert chunks[-1][-1]["idx"] == 10


def test_extract_first_json_object_handles_fenced_output():
    raw = "Here is the result:\n```json\n{\"results\":[{\"idx\":1}]}\n```"
    parsed = _extract_first_json_object(raw)
    assert parsed["results"][0]["idx"] == 1


def test_normalize_judge_result_clamps_scores():
    normalized = _normalize_judge_result(
        row_idx=3,
        raw_item={
            "idx": 3,
            "format_integrity": 4,
            "prompt_relevance": -1,
            "fluency": 1.2,
            "diversity": "2",
            "total_score": "7.8",
            "pass": True,
        },
    )
    assert normalized["format_integrity"] == 2.0
    assert normalized["prompt_relevance"] == 0.0
    assert normalized["fluency"] == 1.0
    assert normalized["diversity"] == 2.0
    assert normalized["pass"] is True


def test_resolve_openrouter_api_key_from_primary_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    key = _resolve_openrouter_api_key(
        opencode_config_path=Path("/tmp/does-not-matter.json"),
        primary_env_name="OPENROUTER_API_KEY",
    )
    assert key == "test-key"


def test_resolve_openrouter_api_key_from_opencode_provider_env(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_FROM_CONFIG", "config-key")
    config_path = tmp_path / "opencode.json"
    config_path.write_text(
        """
{
  "provider": {
    "openrouter": {
      "api": "https://openrouter.ai/api/v1",
      "env": ["OPENROUTER_FROM_CONFIG"]
    }
  }
}
""".strip()
    )
    key = _resolve_openrouter_api_key(
        opencode_config_path=config_path,
        primary_env_name="OPENROUTER_API_KEY",
    )
    assert key == "config-key"
