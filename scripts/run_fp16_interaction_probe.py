#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
import torch
from datasets import load_dataset

from src.training.fp16_chunk_feasibility import (
    FP16ChunkCompressor,
    FP16ChunkLMModule,
    _extract_text_field,
    build_causal_mask,
)

OPENROUTER_CHAT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_DEFAULT_MODEL = "deepseek-v4-flash:free"


def _choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_train_config(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    train_stage = (summary.get("stages", {}).get("train") or {})
    config = train_stage.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(f"Train config missing in summary: {summary_path}")
    return config


def _load_probe_prompts(
    *,
    dataset_name: str,
    split: str,
    text_field: str,
    tokenizer: Any,
    num_prompts: int,
    skip_rows: int,
    prompt_tokens: int,
) -> list[str]:
    prompts: list[str] = []
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    for idx, row in enumerate(dataset):
        if idx < skip_rows:
            continue
        text = _extract_text_field(row, text_field=text_field)
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < prompt_tokens:
            continue
        prompt_ids = ids[:prompt_tokens]
        prompt = tokenizer.decode(prompt_ids, clean_up_tokenization_spaces=False).strip()
        if prompt:
            prompts.append(prompt)
        if len(prompts) >= num_prompts:
            break
    if len(prompts) < num_prompts:
        raise RuntimeError(
            f"Only collected {len(prompts)} prompts (requested {num_prompts})."
        )
    return prompts


def _load_fixed_prompt_set(prompt_set_file: Path, num_prompts: int) -> list[str]:
    if not prompt_set_file.exists():
        raise RuntimeError(f"Prompt set file not found: {prompt_set_file}")
    prompts: list[str] = []
    for line in prompt_set_file.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        prompt = str(payload.get("prompt", "")).strip()
        if prompt:
            prompts.append(prompt)
        if len(prompts) >= num_prompts:
            break
    if len(prompts) < num_prompts:
        raise RuntimeError(
            f"Prompt set contains {len(prompts)} prompts, requested {num_prompts}: {prompt_set_file}"
        )
    return prompts


def _word_list(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def _coherence_score(prompt: str, continuation: str) -> tuple[bool, dict[str, float]]:
    words = _word_list(continuation)
    prompt_words = set(_word_list(prompt))
    cont_words = set(words)
    non_space = [ch for ch in continuation if not ch.isspace()]
    alpha = [ch for ch in non_space if ch.isalpha()]
    alpha_ratio = float(len(alpha)) / float(len(non_space) or 1)

    counts = Counter(words)
    max_word_ratio = float(max(counts.values(), default=0)) / float(len(words) or 1)
    unique_ratio = float(len(cont_words)) / float(len(words) or 1)
    overlap = len(prompt_words & cont_words)
    overlap_ratio = float(overlap) / float(len(prompt_words) or 1)
    has_sentence_punct = any(ch in continuation for ch in ".!?")
    repeated_phrase = bool(re.search(r"(\b\w+\b)(?:\s+\1){4,}", continuation.lower()))

    if len(words) >= 8:
        four_grams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
        four_gram_diversity = float(len(set(four_grams))) / float(len(four_grams) or 1)
    else:
        four_gram_diversity = 1.0

    coherent = bool(
        len(words) >= 10
        and alpha_ratio >= 0.75
        and unique_ratio >= 0.45
        and max_word_ratio <= 0.14
        and has_sentence_punct
        and not repeated_phrase
        and four_gram_diversity >= 0.78
        and (overlap >= 2 or overlap_ratio >= 0.06)
    )
    return coherent, {
        "alpha_ratio": alpha_ratio,
        "unique_ratio": unique_ratio,
        "max_word_ratio": max_word_ratio,
        "four_gram_diversity": four_gram_diversity,
        "overlap": float(overlap),
        "overlap_ratio": overlap_ratio,
        "word_count": float(len(words)),
    }


def _strict_quality_score(
    *,
    prompt: str,
    continuation: str,
    base_coherent: bool,
    base_stats: dict[str, float],
) -> tuple[bool, dict[str, float]]:
    del prompt
    non_space = [ch for ch in continuation if not ch.isspace()]
    non_ascii_count = sum(1 for ch in non_space if ord(ch) > 127)
    non_ascii_ratio = float(non_ascii_count) / float(len(non_space) or 1)
    replacement_char_count = continuation.count("�")
    mojibake_marker_hits = sum(continuation.count(marker) for marker in ("â", "Ã", "€œ", "€\x9d"))

    strict_coherent = bool(
        base_coherent
        and replacement_char_count == 0
        and mojibake_marker_hits == 0
        and non_ascii_ratio <= 0.02
        and base_stats["unique_ratio"] >= 0.50
        and base_stats["max_word_ratio"] <= 0.12
        and base_stats["four_gram_diversity"] >= 0.82
    )
    return strict_coherent, {
        "non_ascii_ratio": non_ascii_ratio,
        "replacement_char_count": float(replacement_char_count),
        "mojibake_marker_hits": float(mojibake_marker_hits),
    }


def _rubric_score(
    *,
    prompt: str,
    continuation: str,
    base_stats: dict[str, float],
    strict_stats: dict[str, float],
) -> tuple[bool, dict[str, float]]:
    del prompt
    words = _word_list(continuation)
    has_sentence_punct = any(ch in continuation for ch in ".!?")
    repeated_phrase = bool(re.search(r"(\b\w+\b)(?:\s+\1){4,}", continuation.lower()))
    non_ascii_ratio = float(strict_stats.get("non_ascii_ratio", 1.0))
    replacement_char_count = float(strict_stats.get("replacement_char_count", 1.0))
    mojibake_marker_hits = float(strict_stats.get("mojibake_marker_hits", 1.0))

    format_integrity = 2
    if replacement_char_count > 0.0 or mojibake_marker_hits > 0.0 or non_ascii_ratio > 0.04:
        format_integrity = 0
    elif non_ascii_ratio > 0.015:
        format_integrity = 1

    overlap_ratio = float(base_stats.get("overlap_ratio", 0.0))
    prompt_relevance = 2 if overlap_ratio >= 0.10 else 1 if overlap_ratio >= 0.05 else 0

    alpha_ratio = float(base_stats.get("alpha_ratio", 0.0))
    word_count = float(base_stats.get("word_count", 0.0))
    fluency = 2
    if not has_sentence_punct or alpha_ratio < 0.80 or word_count < 12 or repeated_phrase:
        fluency = 0
    elif alpha_ratio < 0.90 or word_count < 20:
        fluency = 1

    unique_ratio = float(base_stats.get("unique_ratio", 0.0))
    four_gram_diversity = float(base_stats.get("four_gram_diversity", 0.0))
    max_word_ratio = float(base_stats.get("max_word_ratio", 1.0))
    diversity = 2
    if unique_ratio < 0.45 or four_gram_diversity < 0.72 or max_word_ratio > 0.18:
        diversity = 0
    elif unique_ratio < 0.55 or four_gram_diversity < 0.82 or max_word_ratio > 0.12:
        diversity = 1

    total_score = float(format_integrity + prompt_relevance + fluency + diversity)
    rubric_pass = bool(total_score >= 6.0 and format_integrity >= 1 and fluency >= 1)
    return rubric_pass, {
        "format_integrity": float(format_integrity),
        "prompt_relevance": float(prompt_relevance),
        "fluency": float(fluency),
        "diversity": float(diversity),
        "total_score": total_score,
    }


def _aggregate_probe_quality(rows: list[dict[str, Any]]) -> dict[str, float]:
    num_rows = len(rows)
    if num_rows == 0:
        return {
            "strict_coherent_count": 0.0,
            "strict_coherent_rate": 0.0,
            "unique_continuations": 0.0,
            "dominant_continuation_share": 0.0,
            "mojibake_rows": 0.0,
            "strict_gate_pass": 0.0,
            "rubric_avg_score": 0.0,
            "rubric_pass_rate": 0.0,
            "rubric_format_avg": 0.0,
            "rubric_relevance_avg": 0.0,
            "rubric_fluency_avg": 0.0,
            "rubric_diversity_avg": 0.0,
        }

    continuations = [str(row.get("continuation", "")) for row in rows]
    counts = Counter(continuations)
    dominant_share = float(max(counts.values(), default=0)) / float(num_rows)
    strict_count = sum(int(bool(row.get("strict_coherent", False))) for row in rows)
    strict_rate = float(strict_count) / float(num_rows)
    rubric_pass_count = 0
    rubric_total = 0.0
    rubric_format_sum = 0.0
    rubric_relevance_sum = 0.0
    rubric_fluency_sum = 0.0
    rubric_diversity_sum = 0.0
    mojibake_rows = 0
    for row in rows:
        strict_stats = row.get("strict_stats", {})
        replacement_count = float(strict_stats.get("replacement_char_count", 0.0))
        marker_hits = float(strict_stats.get("mojibake_marker_hits", 0.0))
        if replacement_count > 0.0 or marker_hits > 0.0:
            mojibake_rows += 1
        rubric = row.get("rubric", {})
        rubric_total += float(rubric.get("total_score", 0.0))
        rubric_format_sum += float(rubric.get("format_integrity", 0.0))
        rubric_relevance_sum += float(rubric.get("prompt_relevance", 0.0))
        rubric_fluency_sum += float(rubric.get("fluency", 0.0))
        rubric_diversity_sum += float(rubric.get("diversity", 0.0))
        rubric_pass_count += int(bool(row.get("rubric_pass", False)))

    strict_gate_pass = bool(
        strict_rate >= 0.60
        and dominant_share <= 0.35
        and len(counts) >= max(8, int(0.30 * num_rows))
        and mojibake_rows <= max(2, int(0.05 * num_rows))
    )
    return {
        "strict_coherent_count": float(strict_count),
        "strict_coherent_rate": strict_rate,
        "unique_continuations": float(len(counts)),
        "dominant_continuation_share": dominant_share,
        "mojibake_rows": float(mojibake_rows),
        "strict_gate_pass": float(int(strict_gate_pass)),
        "rubric_avg_score": float(rubric_total) / float(num_rows),
        "rubric_pass_rate": float(rubric_pass_count) / float(num_rows),
        "rubric_format_avg": float(rubric_format_sum) / float(num_rows),
        "rubric_relevance_avg": float(rubric_relevance_sum) / float(num_rows),
        "rubric_fluency_avg": float(rubric_fluency_sum) / float(num_rows),
        "rubric_diversity_avg": float(rubric_diversity_sum) / float(num_rows),
    }


def _resolve_openrouter_api_key(
    *,
    opencode_config_path: Path,
    primary_env_name: str,
) -> str:
    env_direct = os.getenv(primary_env_name, "").strip()
    if env_direct:
        return env_direct

    if not opencode_config_path.exists():
        raise RuntimeError(
            f"OpenRouter key not found: `{primary_env_name}` is empty and config is missing at {opencode_config_path}."
        )

    config = json.loads(opencode_config_path.read_text())
    providers = config.get("provider", {})
    if not isinstance(providers, dict):
        providers = {}

    def _candidate_from_provider(provider: dict[str, Any]) -> str:
        direct_key = provider.get("apiKey") or provider.get("api_key") or provider.get("key")
        if isinstance(direct_key, str) and direct_key.strip():
            return direct_key.strip()
        env_list = provider.get("env")
        if isinstance(env_list, list):
            for env_name in env_list:
                if not isinstance(env_name, str):
                    continue
                env_value = os.getenv(env_name, "").strip()
                if env_value:
                    return env_value
        return ""

    # Prefer explicit openrouter providers.
    for provider_name, provider in providers.items():
        if not isinstance(provider, dict):
            continue
        if "openrouter" in provider_name.lower():
            candidate = _candidate_from_provider(provider)
            if candidate:
                return candidate

    # Fallback: provider pointing to openrouter API host.
    for provider in providers.values():
        if not isinstance(provider, dict):
            continue
        api_host = str(provider.get("api", "")).lower()
        if "openrouter.ai" in api_host:
            candidate = _candidate_from_provider(provider)
            if candidate:
                return candidate

    # Last fallback: any configured provider env var that looks openrouter-related.
    for provider in providers.values():
        if not isinstance(provider, dict):
            continue
        env_list = provider.get("env")
        if not isinstance(env_list, list):
            continue
        for env_name in env_list:
            if not isinstance(env_name, str):
                continue
            if "openrouter" not in env_name.lower():
                continue
            env_value = os.getenv(env_name, "").strip()
            if env_value:
                return env_value

    raise RuntimeError(
        f"OpenRouter API key is unavailable. Set `{primary_env_name}` or configure an OpenRouter provider/env in {opencode_config_path}."
    )


def _chunk_rows(rows: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    size = max(1, int(chunk_size))
    return [rows[i : i + size] for i in range(0, len(rows), size)]


def _extract_first_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise RuntimeError("Judge response is empty.")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        parsed = json.loads(fenced.group(1))
        if isinstance(parsed, dict):
            return parsed

    start = text.find("{")
    if start < 0:
        raise RuntimeError("Judge response does not contain a JSON object.")
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(text[start : idx + 1])
                if isinstance(parsed, dict):
                    return parsed
                break
    raise RuntimeError("Judge response JSON extraction failed.")


def _normalize_judge_result(
    *,
    row_idx: int,
    raw_item: dict[str, Any],
) -> dict[str, Any]:
    def _score(key: str) -> float:
        value = raw_item.get(key, 0)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        return float(min(2.0, max(0.0, round(numeric))))

    format_integrity = _score("format_integrity")
    prompt_relevance = _score("prompt_relevance")
    fluency = _score("fluency")
    diversity = _score("diversity")
    raw_total = raw_item.get("total_score")
    if raw_total is None:
        total_score = float(format_integrity + prompt_relevance + fluency + diversity)
    else:
        try:
            total_score = float(raw_total)
        except (TypeError, ValueError):
            total_score = float(format_integrity + prompt_relevance + fluency + diversity)
    llm_pass = raw_item.get("pass")
    if isinstance(llm_pass, bool):
        judge_pass = llm_pass
    else:
        judge_pass = bool(total_score >= 6.0 and format_integrity >= 1.0 and fluency >= 1.0)
    return {
        "idx": int(raw_item.get("idx", row_idx)),
        "pass": judge_pass,
        "format_integrity": format_integrity,
        "prompt_relevance": prompt_relevance,
        "fluency": fluency,
        "diversity": diversity,
        "total_score": float(total_score),
        "notes": str(raw_item.get("notes", "")).strip(),
    }


def _openrouter_chat_completion(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> str:
    ssl_context: ssl.SSLContext | None = None
    try:
        import certifi  # type: ignore

        ssl_context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        # Fall back to system trust store when certifi is unavailable.
        ssl_context = None

    request_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(request_payload).encode("utf-8")
    request = urllib_request.Request(
        OPENROUTER_CHAT_ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    retries = max(0, int(max_retries))
    attempt = 0
    while True:
        try:
            with urllib_request.urlopen(
                request,
                timeout=int(timeout_seconds),
                context=ssl_context,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError("OpenRouter response missing choices.")
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("OpenRouter response content is empty.")
            return content
        except urllib_error.HTTPError as err:
            status = int(getattr(err, "code", 0))
            retriable = status in (408, 409, 425, 429, 500, 502, 503, 504)
            if attempt >= retries or not retriable:
                details = err.read().decode("utf-8", errors="replace") if hasattr(err, "read") else ""
                raise RuntimeError(
                    f"OpenRouter request failed with status={status}. Details: {details[:300]}"
                ) from err
        except urllib_error.URLError as err:
            if attempt >= retries:
                raise RuntimeError(f"OpenRouter network error after retries: {err}") from err
        attempt += 1
        sleep_seconds = float(retry_backoff_seconds) * float(2 ** max(0, attempt - 1))
        time.sleep(min(30.0, max(0.0, sleep_seconds)))


def _aggregate_llm_judge(rows: list[dict[str, Any]]) -> dict[str, float]:
    llm_rows = [row for row in rows if isinstance(row.get("llm_judge"), dict)]
    num_rows = len(llm_rows)
    if num_rows == 0:
        return {
            "llm_rubric_avg_score": 0.0,
            "llm_rubric_pass_rate": 0.0,
            "llm_rubric_format_avg": 0.0,
            "llm_rubric_relevance_avg": 0.0,
            "llm_rubric_fluency_avg": 0.0,
            "llm_rubric_diversity_avg": 0.0,
        }

    pass_count = 0
    total = 0.0
    format_sum = 0.0
    relevance_sum = 0.0
    fluency_sum = 0.0
    diversity_sum = 0.0
    for row in llm_rows:
        judge = row["llm_judge"]
        pass_count += int(bool(judge.get("pass", False)))
        total += float(judge.get("total_score", 0.0))
        format_sum += float(judge.get("format_integrity", 0.0))
        relevance_sum += float(judge.get("prompt_relevance", 0.0))
        fluency_sum += float(judge.get("fluency", 0.0))
        diversity_sum += float(judge.get("diversity", 0.0))
    denom = float(num_rows)
    return {
        "llm_rubric_avg_score": total / denom,
        "llm_rubric_pass_rate": float(pass_count) / denom,
        "llm_rubric_format_avg": format_sum / denom,
        "llm_rubric_relevance_avg": relevance_sum / denom,
        "llm_rubric_fluency_avg": fluency_sum / denom,
        "llm_rubric_diversity_avg": diversity_sum / denom,
    }


def _judge_rows_with_openrouter(
    *,
    rows: list[dict[str, Any]],
    api_key: str,
    model: str,
    chunk_size: int,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    inter_chunk_sleep_seconds: float,
) -> dict[str, float]:
    if not rows:
        return {"llm_judge_requests": 0.0, "llm_judge_fallback_rows": 0.0}

    system_prompt = (
        "You are a strict research evaluator for short text generations. "
        "Score each item from 0-2 on format_integrity, prompt_relevance, fluency, and diversity. "
        "Return JSON only with schema: {\"results\":[{\"idx\":int,\"format_integrity\":0|1|2,"
        "\"prompt_relevance\":0|1|2,\"fluency\":0|1|2,\"diversity\":0|1|2,"
        "\"total_score\":number,\"pass\":bool,\"notes\":string}]}. "
        "Do not omit any idx."
    )
    request_count = 0
    fallback_rows = 0
    row_by_idx: dict[int, dict[str, Any]] = {int(row["idx"]): row for row in rows}

    for chunk in _chunk_rows(rows, chunk_size=chunk_size):
        judge_items = [
            {
                "idx": int(row["idx"]),
                "prompt": str(row.get("prompt", "")),
                "continuation": str(row.get("continuation", "")),
            }
            for row in chunk
        ]
        user_prompt = (
            "Evaluate the following generated continuations.\n"
            "Output JSON only.\n"
            f"{json.dumps({'items': judge_items}, ensure_ascii=False)}"
        )
        try:
            raw_content = _openrouter_chat_completion(
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            request_count += 1
            parsed = _extract_first_json_object(raw_content)
            raw_results = parsed.get("results")
            if not isinstance(raw_results, list):
                raise RuntimeError("OpenRouter judge response missing `results` list.")
            normalized_by_idx: dict[int, dict[str, Any]] = {}
            for item in raw_results:
                if not isinstance(item, dict):
                    continue
                normalized = _normalize_judge_result(
                    row_idx=int(item.get("idx", -1)),
                    raw_item=item,
                )
                normalized_by_idx[int(normalized["idx"])] = normalized

            for row in chunk:
                idx = int(row["idx"])
                normalized = normalized_by_idx.get(idx)
                if normalized is None:
                    fallback_rows += 1
                    heuristic = row.get("rubric", {})
                    normalized = {
                        "idx": idx,
                        "pass": bool(row.get("rubric_pass", False)),
                        "format_integrity": float(heuristic.get("format_integrity", 0.0)),
                        "prompt_relevance": float(heuristic.get("prompt_relevance", 0.0)),
                        "fluency": float(heuristic.get("fluency", 0.0)),
                        "diversity": float(heuristic.get("diversity", 0.0)),
                        "total_score": float(heuristic.get("total_score", 0.0)),
                        "notes": "fallback:missing_idx_in_judge_response",
                    }
                row_by_idx[idx]["llm_judge"] = normalized
        except Exception as err:
            for row in chunk:
                idx = int(row["idx"])
                fallback_rows += 1
                heuristic = row.get("rubric", {})
                row_by_idx[idx]["llm_judge"] = {
                    "idx": idx,
                    "pass": bool(row.get("rubric_pass", False)),
                    "format_integrity": float(heuristic.get("format_integrity", 0.0)),
                    "prompt_relevance": float(heuristic.get("prompt_relevance", 0.0)),
                    "fluency": float(heuristic.get("fluency", 0.0)),
                    "diversity": float(heuristic.get("diversity", 0.0)),
                    "total_score": float(heuristic.get("total_score", 0.0)),
                    "notes": f"fallback:judge_error:{type(err).__name__}",
                }
        if inter_chunk_sleep_seconds > 0:
            time.sleep(float(inter_chunk_sleep_seconds))
    return {
        "llm_judge_requests": float(request_count),
        "llm_judge_fallback_rows": float(fallback_rows),
    }


def _generate_continuation(
    *,
    prompt: str,
    model_module: FP16ChunkLMModule,
    compressor: FP16ChunkCompressor,
    train_scalars: np.ndarray,
    train_chunk_tokens: np.ndarray,
    compressed_seq_len: int,
    continuation_chunks: int,
    device: torch.device,
) -> tuple[str, float]:
    prompt_ids = compressor.tokenizer.encode(prompt, add_special_tokens=False)
    prompt_scalars = compressor.compress_token_ids(prompt_ids).astype(np.float32, copy=False)
    num_scalars = 1 if train_scalars.ndim == 1 else int(train_scalars.shape[1])
    if num_scalars > 1:
        prompt_scalars = np.asarray(prompt_scalars, dtype=np.float32).reshape(-1, num_scalars)
    else:
        prompt_scalars = np.asarray(prompt_scalars, dtype=np.float32).reshape(-1)
    if prompt_scalars.shape[0] == 0:
        prompt_scalars = np.zeros((1, num_scalars) if num_scalars > 1 else (1,), dtype=np.float32)

    if prompt_scalars.shape[0] >= compressed_seq_len:
        context = prompt_scalars[-compressed_seq_len:].copy()
    else:
        pad = np.zeros((compressed_seq_len - prompt_scalars.shape[0], num_scalars) if num_scalars > 1 else (compressed_seq_len - prompt_scalars.shape[0],), dtype=np.float32)
        context = np.concatenate([pad, prompt_scalars], axis=0)

    generated_token_ids: list[int] = []
    start = time.perf_counter()
    for _ in range(continuation_chunks):
        input_scalars = torch.from_numpy(context[None, :]).to(device=device, dtype=torch.float32)
        mask = build_causal_mask(compressed_seq_len, device)
        with torch.no_grad():
            preds, _ = model_module.model(input_scalars, mask=mask)
        next_vector = preds[0, -1].cpu().numpy()
        if train_scalars.ndim == 1:
            nearest_idx = int(np.argmin(np.abs(train_scalars - next_vector)))
        else:
            distances = np.linalg.norm(train_scalars - next_vector, axis=1)
            nearest_idx = int(np.argmin(distances))
        chunk_ids = train_chunk_tokens[nearest_idx].astype(np.int64, copy=False).tolist()
        generated_token_ids.extend(chunk_ids)
        context = np.concatenate(
            [context[1:], np.asarray([next_vector], dtype=np.float32)],
            axis=0,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    continuation = compressor.tokenizer.decode(
        generated_token_ids,
        clean_up_tokenization_spaces=False,
    ).strip()
    return continuation, elapsed_ms


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        default="experiments/modal_downloads/exp028_modal_fp16_2m1b_001",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="best-step-0028000.ckpt",
    )
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--continuation-chunks", type=int, default=2)
    parser.add_argument("--skip-rows", type=int, default=-1)
    parser.add_argument("--prompt-set-file", default="")
    parser.add_argument("--output-name", default="interaction_probe_50.json")
    parser.add_argument("--openrouter-judge", action="store_true")
    parser.add_argument("--openrouter-model", default=OPENROUTER_DEFAULT_MODEL)
    parser.add_argument("--openrouter-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--opencode-config-path", default="~/.config/opencode/opencode.json")
    parser.add_argument("--judge-chunk-size", type=int, default=8)
    parser.add_argument("--judge-timeout-seconds", type=int, default=60)
    parser.add_argument("--judge-max-retries", type=int, default=5)
    parser.add_argument("--judge-retry-backoff-seconds", type=float, default=1.5)
    parser.add_argument("--judge-inter-chunk-sleep-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_root = Path(args.run_root)
    summary_path = run_root / "pipeline_summary.json"
    checkpoint_path = run_root / args.checkpoint_name
    train_scalars_path = run_root / "train_scalars.npy"
    train_chunk_tokens_path = run_root / "train_chunk_tokens.npy"
    output_path = run_root / args.output_name

    config = _load_train_config(summary_path)
    fp16_cfg = dict(config.get("fp16_chunk", {}))
    dataset_cfg = dict(fp16_cfg.get("dataset", {}))
    compression_cfg = dict(fp16_cfg.get("compression", {}))
    training_cfg = dict(fp16_cfg.get("training", {}))

    compressor = FP16ChunkCompressor(
        chunk_size_tokens=int(compression_cfg.get("chunk_size_tokens", 64)),
        window_overlap_tokens=int(compression_cfg.get("window_overlap", 0)),
        tokenizer_name=str(compression_cfg.get("tokenizer_name", "gpt2")),
        force_fast_tokenizer=bool(compression_cfg.get("force_fast_tokenizer", True)),
        dtype=str(compression_cfg.get("dtype", "float16")),
        scalars_per_chunk=int(compression_cfg.get("scalars_per_chunk", 1)),
    )

    if str(args.prompt_set_file).strip():
        prompts = _load_fixed_prompt_set(
            prompt_set_file=Path(args.prompt_set_file),
            num_prompts=int(args.num_prompts),
        )
    else:
        prompts = _load_probe_prompts(
            dataset_name=str(dataset_cfg.get("dataset_name", "roneneldan/TinyStories")),
            split=str(dataset_cfg.get("val_split", "validation")),
            text_field=str(dataset_cfg.get("text_field", "text")),
            tokenizer=compressor.tokenizer,
            num_prompts=int(args.num_prompts),
            skip_rows=(
                int(args.skip_rows)
                if int(args.skip_rows) >= 0
                else int(dataset_cfg.get("val_samples", 2048))
            ),
            prompt_tokens=int(args.prompt_tokens),
        )

    device = _choose_device()
    model_config = {
        **fp16_cfg,
        "diffusion": {
            **dict(fp16_cfg.get("diffusion", {})),
            "vocab_size": int(fp16_cfg.get("diffusion", {}).get("vocab_size", compressor.vocab_size)),
        },
    }
    model_module = FP16ChunkLMModule(config=model_config).to(device)
    model_module.eval()
    state = torch.load(checkpoint_path, map_location=device)
    model_module.load_state_dict(state["state_dict"], strict=True)

    train_scalars = np.load(train_scalars_path, mmap_mode="r").astype(np.float32, copy=False)
    train_chunk_tokens = np.load(train_chunk_tokens_path, mmap_mode="r")
    compressed_seq_len = int(training_cfg.get("compressed_seq_len", 8))
    continuation_chunks = int(args.continuation_chunks)

    rows: list[dict[str, Any]] = []
    coherent_count = 0
    strict_coherent_count = 0
    latencies: list[float] = []
    for idx, prompt in enumerate(prompts):
        continuation, latency_ms = _generate_continuation(
            prompt=prompt,
            model_module=model_module,
            compressor=compressor,
            train_scalars=train_scalars,
            train_chunk_tokens=train_chunk_tokens,
            compressed_seq_len=compressed_seq_len,
            continuation_chunks=continuation_chunks,
            device=device,
        )
        coherent, stats = _coherence_score(prompt, continuation)
        strict_coherent, strict_stats = _strict_quality_score(
            prompt=prompt,
            continuation=continuation,
            base_coherent=coherent,
            base_stats=stats,
        )
        rubric_pass, rubric = _rubric_score(
            prompt=prompt,
            continuation=continuation,
            base_stats=stats,
            strict_stats=strict_stats,
        )
        coherent_count += int(coherent)
        strict_coherent_count += int(strict_coherent)
        latencies.append(latency_ms)
        rows.append(
            {
                "idx": idx,
                "prompt": prompt,
                "continuation": continuation,
                "coherent": coherent,
                "strict_coherent": strict_coherent,
                "rubric_pass": rubric_pass,
                "latency_ms": latency_ms,
                "stats": stats,
                "strict_stats": strict_stats,
                "rubric": rubric,
            }
        )

    coherent_rate = float(coherent_count) / float(len(rows) or 1)
    strict_coherent_rate = float(strict_coherent_count) / float(len(rows) or 1)
    avg_latency_ms = float(np.mean(latencies)) if latencies else 0.0
    aggregate = _aggregate_probe_quality(rows)
    llm_aggregate = {
        "llm_rubric_avg_score": 0.0,
        "llm_rubric_pass_rate": 0.0,
        "llm_rubric_format_avg": 0.0,
        "llm_rubric_relevance_avg": 0.0,
        "llm_rubric_fluency_avg": 0.0,
        "llm_rubric_diversity_avg": 0.0,
    }
    llm_meta = {"llm_judge_requests": 0.0, "llm_judge_fallback_rows": 0.0}
    llm_judge_error = ""
    if bool(args.openrouter_judge):
        try:
            openrouter_api_key = _resolve_openrouter_api_key(
                opencode_config_path=Path(str(args.opencode_config_path)).expanduser(),
                primary_env_name=str(args.openrouter_api_key_env),
            )
            llm_meta = _judge_rows_with_openrouter(
                rows=rows,
                api_key=openrouter_api_key,
                model=str(args.openrouter_model),
                chunk_size=int(args.judge_chunk_size),
                timeout_seconds=int(args.judge_timeout_seconds),
                max_retries=int(args.judge_max_retries),
                retry_backoff_seconds=float(args.judge_retry_backoff_seconds),
                inter_chunk_sleep_seconds=float(args.judge_inter_chunk_sleep_seconds),
            )
            llm_aggregate = _aggregate_llm_judge(rows)
        except Exception as err:
            llm_judge_error = str(err)

    payload = {
        "run_id": run_root.name,
        "num_prompts": len(rows),
        "coherent_count": coherent_count,
        "coherent_rate": coherent_rate,
        "strict_coherent_count": strict_coherent_count,
        "strict_coherent_rate": strict_coherent_rate,
        "avg_decode_latency_ms": avg_latency_ms,
        "device": str(device),
        "continuation_chunks": continuation_chunks,
        "unique_continuations": int(aggregate["unique_continuations"]),
        "dominant_continuation_share": aggregate["dominant_continuation_share"],
        "mojibake_rows": int(aggregate["mojibake_rows"]),
        "strict_gate_pass": bool(int(aggregate["strict_gate_pass"])),
        "rubric_avg_score": aggregate["rubric_avg_score"],
        "rubric_pass_rate": aggregate["rubric_pass_rate"],
        "rubric_format_avg": aggregate["rubric_format_avg"],
        "rubric_relevance_avg": aggregate["rubric_relevance_avg"],
        "rubric_fluency_avg": aggregate["rubric_fluency_avg"],
        "rubric_diversity_avg": aggregate["rubric_diversity_avg"],
        "llm_judge_enabled": bool(args.openrouter_judge),
        "llm_judge_model": str(args.openrouter_model) if bool(args.openrouter_judge) else "",
        "llm_judge_chunk_size": int(args.judge_chunk_size),
        "llm_judge_requests": int(llm_meta["llm_judge_requests"]),
        "llm_judge_fallback_rows": int(llm_meta["llm_judge_fallback_rows"]),
        "llm_judge_error": llm_judge_error,
        "llm_rubric_avg_score": llm_aggregate["llm_rubric_avg_score"],
        "llm_rubric_pass_rate": llm_aggregate["llm_rubric_pass_rate"],
        "llm_rubric_format_avg": llm_aggregate["llm_rubric_format_avg"],
        "llm_rubric_relevance_avg": llm_aggregate["llm_rubric_relevance_avg"],
        "llm_rubric_fluency_avg": llm_aggregate["llm_rubric_fluency_avg"],
        "llm_rubric_diversity_avg": llm_aggregate["llm_rubric_diversity_avg"],
        "rows": rows,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in payload if k != "rows"}, indent=2))
    print(f"Saved probe: {output_path}")


if __name__ == "__main__":
    main()
