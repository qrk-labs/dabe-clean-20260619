#!/usr/bin/env python3
"""Build paper figures from DABE experiment artifacts.

The script intentionally uses only the Python standard library for plotting. It
renders SVG directly, then calls rsvg-convert to produce PDF and PNG outputs.
"""

from __future__ import annotations

import html
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "research" / "paper_drafts" / "figures"
RSVG_CONVERT = Path("/opt/homebrew/bin/rsvg-convert")

COLORS = {
    "dabe": "#1f6f68",
    "baseline": "#a4512b",
    "variable": "#9b4d9f",
    "fixed_lookup": "#3f6fb5",
    "adaptive": "#d08a21",
    "quality": "#205493",
    "cost": "#a23b3b",
    "grid": "#d8ddd7",
    "text": "#1e2724",
    "muted": "#65736f",
}


def load_json(path: str) -> dict[str, Any]:
    full = ROOT / path
    if not full.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    with full.open() as f:
        return json.load(f)


def unwrap_result(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result", payload)
    if isinstance(result, dict) and "train" in result:
        return result["train"]
    return result if isinstance(result, dict) else payload


def read_standard(path: str) -> dict[str, Any]:
    return unwrap_result(load_json(path))


def read_reconstructed_088(path: str) -> dict[str, Any]:
    payload = load_json(path)
    return payload["last_metrics"]


def read_reconstructed_089(path: str) -> dict[str, Any]:
    payload = load_json(path)
    last = payload["last_validation"]
    return {
        "val_token_acc_last": last["token_acc"],
        "val_chunk_deviation_mean_last": last["chunk_deviation_mean"],
        "val_chunk_deviation_p90_last": last["chunk_deviation_p90"],
        "val_observed_effective_bits_per_token_last": last["observed_effective_bits_per_token"],
        "val_lookup_active_k_mean_last": last.get("lookup_active_k_mean"),
    }


def fnum(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def metric(record: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = fnum(record.get(key))
        if value is not None:
            return value
    return None


def exact_block_avg(record: dict[str, Any]) -> float | None:
    vals = [metric(record, f"val_block{i}_exact_acc_last") for i in range(4)]
    if all(v is not None for v in vals):
        return sum(v for v in vals if v is not None) / 4.0
    return None


CatalogReader = Callable[[str], dict[str, Any]]

CATALOG: list[dict[str, Any]] = [
    {
        "id": "EXP-071",
        "label": "Fixed 1024b no lookup",
        "mechanism": "fixed-rate no lookup",
        "family": "baseline",
        "path": "experiments/modal_downloads/exp071_modal_dabe_hier_local_1024_001/stage_result.json",
        "reader": read_standard,
        "bits_override": 16.0
    },
    {
        "id": "EXP-077",
        "label": "Fixed lookup K=8",
        "mechanism": "fixed sparse lookup",
        "family": "fixed_lookup",
        "path": "experiments/modal_downloads/exp077_modal_dabe_hier1024_lookup_learned_selector_001/stage_result.json",
        "reader": read_standard,
    },
    {
        "id": "EXP-079",
        "label": "Fixed 1200b no lookup",
        "mechanism": "fixed-rate no lookup",
        "family": "baseline",
        "path": "experiments/modal_downloads/exp079_modal_dabe_hier1200_no_lookup_001/stage_result.json",
        "reader": read_standard
    },
    {
        "id": "EXP-080-K4",
        "label": "Fixed lookup K=4",
        "mechanism": "fixed sparse lookup",
        "family": "fixed_lookup",
        "path": "experiments/modal_downloads/exp080_modal_dabe_hier1024_lookup_learned_k4_001/stage_result.json",
        "reader": read_standard,
    },
    {
        "id": "EXP-080-K16",
        "label": "Fixed lookup K=16",
        "mechanism": "fixed sparse lookup",
        "family": "fixed_lookup",
        "path": "experiments/modal_downloads/exp080_modal_dabe_hier1024_lookup_learned_k16_001/stage_result.json",
        "reader": read_standard,
    },
    {
        "id": "EXP-083",
        "label": "Fixed lookup K=32",
        "mechanism": "fixed sparse lookup upper envelope",
        "family": "fixed_lookup",
        "path": "experiments/modal_downloads/exp083_084_modal_dabe_lookup_k32_pair_001/exp083_fixed_k32/stage_result.json",
        "reader": read_standard,
        "active_k_override": 32.0,
    },
    {
        "id": "EXP-084",
        "label": "Ranked halting Kmax=32",
        "mechanism": "adaptive ranked halting",
        "family": "adaptive",
        "path": "experiments/modal_downloads/exp083_084_modal_dabe_lookup_k32_pair_001/exp084_ranked_halting_k32/stage_result.json",
        "reader": read_standard,
    },
    {
        "id": "EXP-085-K12",
        "label": "Halting target K=12",
        "mechanism": "adaptive halting target",
        "family": "adaptive",
        "path": "experiments/modal_downloads/exp085_modal_dabe_halting_rate_sweep_001/exp085_halting_target_k12/stage_result.json",
        "reader": read_standard,
    },
    {
        "id": "EXP-085-K16",
        "label": "Halting target K=16",
        "mechanism": "adaptive halting target",
        "family": "adaptive",
        "path": "experiments/modal_downloads/exp085_modal_dabe_halting_rate_sweep_001/exp085_halting_target_k16/stage_result.json",
        "reader": read_standard,
    },
    {
        "id": "EXP-085-K20",
        "label": "Halting target K=20",
        "mechanism": "adaptive halting target",
        "family": "adaptive",
        "path": "experiments/modal_downloads/exp085_modal_dabe_halting_rate_sweep_001/exp085_halting_target_k20/stage_result.json",
        "reader": read_standard,
    },
    {
        "id": "EXP-086",
        "label": "Gist + residual router",
        "mechanism": "gist residual lookup",
        "family": "dabe",
        "path": "experiments/modal_downloads/exp086_modal_dabe_gist_residual_router_001/stage_result.json",
        "reader": read_standard,
    },
    {
        "id": "EXP-087",
        "label": "DABE quality anchor",
        "mechanism": "gist residual lookup",
        "family": "dabe",
        "path": "experiments/modal_downloads/exp087_modal_dabe_gist_residual_router_sweep_001/exp087_router_w0p2_cost0p02/stage_result.json",
        "reader": read_standard,
    },
    {
        "id": "EXP-088",
        "label": "Variable windows quantile",
        "mechanism": "variable token windows",
        "family": "variable",
        "path": "experiments/modal_downloads/exp088_modal_dabe_gist_residual_variable_windows_001/reconstructed_result.json",
        "reader": read_reconstructed_088,
    },
    {
        "id": "EXP-089",
        "label": "Variable windows action-value",
        "mechanism": "variable token windows",
        "family": "variable",
        "path": "experiments/modal_downloads/exp089_modal_dabe_action_window_router_001/reconstructed_result.json",
        "reader": read_reconstructed_089,
    },
    {
        "id": "EXP-090",
        "label": "Variable windows hard ST",
        "mechanism": "variable token windows",
        "family": "variable",
        "path": "experiments/modal_downloads/exp090_modal_dabe_straight_through_window_router_001/stage_result.json",
        "reader": read_standard,
    },
    {
        "id": "EXP-091-0.02",
        "label": "Cost 0.020",
        "mechanism": "cost-aware DABE",
        "family": "dabe",
        "path": "experiments/modal_downloads/exp091_modal_dabe_cost_aware_gist_residual_001/exp087_router_w0p2_cost0p02/stage_result.json",
        "reader": read_standard,
        "slot_cost": 0.02,
    },
    {
        "id": "EXP-091-0.025",
        "label": "Cost 0.025",
        "mechanism": "cost-aware DABE",
        "family": "dabe",
        "path": "experiments/modal_downloads/exp091_modal_dabe_cost_aware_gist_residual_001/exp087_router_w0p2_cost0p025/stage_result.json",
        "reader": read_standard,
        "slot_cost": 0.025,
    },
    {
        "id": "EXP-091-0.03",
        "label": "Cost 0.030",
        "mechanism": "cost-aware DABE",
        "family": "dabe",
        "path": "experiments/modal_downloads/exp091_modal_dabe_cost_aware_gist_residual_001/exp087_router_w0p2_cost0p03/stage_result.json",
        "reader": read_standard,
        "slot_cost": 0.03,
    },
    {
        "id": "EXP-092-0.0225",
        "label": "Cost 0.0225",
        "mechanism": "cost-aware DABE replicated",
        "family": "dabe",
        "path": "experiments/modal_downloads/exp092_modal_dabe_cost_knee_repl_001/exp087_router_w0p2_cost0p0225/stage_result.json",
        "reader": read_standard,
        "slot_cost": 0.0225,
    },
    {
        "id": "EXP-092-0.025",
        "label": "DABE cost knee",
        "mechanism": "cost-aware DABE replicated",
        "family": "dabe",
        "path": "experiments/modal_downloads/exp092_modal_dabe_cost_knee_repl_001/exp087_router_w0p2_cost0p025/stage_result.json",
        "reader": read_standard,
        "slot_cost": 0.025,
    },
    {
        "id": "EXP-092-0.0275",
        "label": "Cost 0.0275",
        "mechanism": "cost-aware DABE replicated",
        "family": "dabe",
        "path": "experiments/modal_downloads/exp092_modal_dabe_cost_knee_repl_001/exp087_router_w0p2_cost0p0275/stage_result.json",
        "reader": read_standard,
        "slot_cost": 0.0275,
    },
    {
        "id": "EXP-093",
        "label": "Fixed-rate 20bpt baseline",
        "mechanism": "fixed-rate no lookup",
        "family": "baseline",
        "path": "experiments/modal_downloads/exp093_modal_dabe_fixed_rate_20bpt_001/stage_result.json",
        "reader": read_standard,
    },
]


def normalize(entry: dict[str, Any]) -> dict[str, Any]:
    raw = entry["reader"] (entry["path"])
    bits = entry.get("bits_override")
    if bits is None:
        bits = metric(raw, "val_observed_effective_bits_per_token_last", "effective_bits_per_token", "bits_per_token")
    chunk_dev = entry.get("chunk_dev_override")
    if chunk_dev is None:
        chunk_dev = metric(raw, "val_chunk_deviation_mean_last")
    chunk_p90 = entry.get("chunk_p90_override")
    if chunk_p90 is None:
        chunk_p90 = metric(raw, "val_chunk_deviation_p90_last")
    token_acc = metric(raw, "val_token_acc_last")
    active_k = entry.get("active_k_override")
    if active_k is None:
        active_k = metric(raw, "val_lookup_active_k_mean_last", "val_lookup_budget_k_mean_last")
    return {
        "id": entry["id"],
        "label": entry["label"],
        "mechanism": entry["mechanism"],
        "family": entry["family"],
        "source_path": entry["path"],
        "bits_per_token": bits,
        "token_accuracy": token_acc,
        "chunk_deviation_mean": chunk_dev,
        "chunk_deviation_p90": chunk_p90,
        "active_k": active_k,
        "slot_cost": entry.get("slot_cost"),
        "exact_16token_block_avg": exact_block_avg(raw),
        "exact_64token_chunk_acc": metric(raw, "val_exact_chunk_acc_last"),
        "bit_density": metric(raw, "val_bit_density_last"),
    }


def require(records: Iterable[dict[str, Any]], fields: Iterable[str], figure: str) -> None:
    for rec in records:
        for field in fields:
            if rec.get(field) is None:
                raise ValueError(f"{figure}: {rec['id']} missing required field {field}")


def esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def svg_doc(width: int, height: int, body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fbfaf6"/>
<style>
  .title {{ font-family: Georgia, serif; font-size: 22px; font-weight: 700; fill: {COLORS['text']}; }}
  .subtitle {{ font-family: Georgia, serif; font-size: 13px; fill: {COLORS['muted']}; }}
  .axis {{ stroke: {COLORS['text']}; stroke-width: 1.2; }}
  .grid {{ stroke: {COLORS['grid']}; stroke-width: 1; }}
  .tick {{ font-family: Helvetica, Arial, sans-serif; font-size: 11px; fill: {COLORS['muted']}; }}
  .label {{ font-family: Helvetica, Arial, sans-serif; font-size: 12px; fill: {COLORS['text']}; }}
  .small {{ font-family: Helvetica, Arial, sans-serif; font-size: 10px; fill: {COLORS['muted']}; }}
</style>
{body}
</svg>
'''


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    return dst_min + ((value - src_min) / (src_max - src_min)) * (dst_max - dst_min)


def nice_range(values: list[float], pad: float = 0.08, floor_zero: bool = False) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    if floor_zero:
        lo = min(0.0, lo)
    span = hi - lo or max(abs(hi), 1.0)
    return lo - span * pad, hi + span * pad


def axis_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    return [lo + (hi - lo) * i / (count - 1) for i in range(count)]


def scatter_svg(
    title: str,
    subtitle: str,
    records: list[dict[str, Any]],
    x_field: str,
    y_field: str,
    x_label: str,
    y_label: str,
    out_name: str,
    invert_y: bool = False,
    label_offsets: dict[str, tuple[float, float, str]] | None = None,
) -> None:
    require(records, [x_field, y_field], out_name)
    w, h = 920, 620
    left, right, top, bottom = 90, 250, 82, 92
    plot_w, plot_h = w - left - right, h - top - bottom
    xs = [float(r[x_field]) for r in records]
    ys = [float(r[y_field]) for r in records]
    xlo, xhi = nice_range(xs)
    ylo, yhi = nice_range(ys)
    body = [f'<text x="{left}" y="38" class="title">{esc(title)}</text>', f'<text x="{left}" y="60" class="subtitle">{esc(subtitle)}</text>']
    for tick in axis_ticks(xlo, xhi):
        x = scale(tick, xlo, xhi, left, left + plot_w)
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" class="grid"/>')
        body.append(f'<text x="{x:.1f}" y="{top+plot_h+24}" text-anchor="middle" class="tick">{tick:.1f}</text>')
    for tick in axis_ticks(ylo, yhi):
        y = scale(tick, ylo, yhi, top + plot_h, top) if not invert_y else scale(tick, ylo, yhi, top, top + plot_h)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick:.2f}</text>')
    body.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>')
    body.append(f'<text x="{left+plot_w/2}" y="{h-30}" text-anchor="middle" class="label">{esc(x_label)}</text>')
    body.append(f'<text x="26" y="{top+plot_h/2}" transform="rotate(-90 26 {top+plot_h/2})" text-anchor="middle" class="label">{esc(y_label)}</text>')
    legend_y = top + 10
    seen: set[str] = set()
    for fam in ["dabe", "adaptive", "fixed_lookup", "baseline", "variable"]:
        if any(r["family"] == fam for r in records):
            color = COLORS.get(fam, "#555")
            body.append(f'<circle cx="{left+plot_w+38}" cy="{legend_y}" r="6" fill="{color}"/>')
            body.append(f'<text x="{left+plot_w+52}" y="{legend_y+4}" class="small">{esc(fam.replace("_", " "))}</text>')
            legend_y += 22
    for r in records:
        x = scale(float(r[x_field]), xlo, xhi, left, left + plot_w)
        y = scale(float(r[y_field]), ylo, yhi, top + plot_h, top) if not invert_y else scale(float(r[y_field]), ylo, yhi, top, top + plot_h)
        color = COLORS.get(r["family"], "#555")
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" fill-opacity="0.88" stroke="#fff" stroke-width="1.5"/>')
        label = r["id"].replace("EXP-", "")
        default_dy = -10 if label not in seen else 14
        seen.add(label)
        dx, dy, anchor = (label_offsets or {}).get(r["id"], (9.0, float(default_dy), "start"))
        body.append(f'<text x="{x+dx:.1f}" y="{y+dy:.1f}" text-anchor="{anchor}" class="small">{esc(label)}</text>')
    write_svg(out_name, svg_doc(w, h, "\n".join(body)))


def paired_bar_svg(records: list[dict[str, Any]]) -> None:
    require(records, ["token_accuracy", "chunk_deviation_mean"], "fig01_matched_20bpt_comparator")
    w, h = 920, 560
    body = [
        '<text x="70" y="40" class="title">Fig. 1. Matched 20bpt comparator</text>',
        '<text x="70" y="62" class="subtitle">Adaptive sparse repair wins at essentially the same bitrate.</text>',
    ]
    panels = [
        (70, 110, 330, 340, "Token accuracy", "token_accuracy", 1.0, False),
        (520, 110, 330, 340, "Mean chunk deviation", "chunk_deviation_mean", max(r["chunk_deviation_mean"] for r in records) * 1.12, True),
    ]
    for px, py, pw, ph, title, field, ymax, lower_better in panels:
        body.append(f'<text x="{px}" y="{py-25}" class="label">{esc(title)}</text>')
        body.append(f'<line x1="{px}" y1="{py+ph}" x2="{px+pw}" y2="{py+ph}" class="axis"/>')
        body.append(f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py+ph}" class="axis"/>')
        for tick in axis_ticks(0, ymax):
            y = scale(tick, 0, ymax, py + ph, py)
            body.append(f'<line x1="{px}" y1="{y:.1f}" x2="{px+pw}" y2="{y:.1f}" class="grid"/>')
            body.append(f'<text x="{px-8}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick:.2f}</text>')
        bar_w = 88
        gap = 42
        start = px + 64
        for i, r in enumerate(records):
            val = float(r[field])
            bh = scale(val, 0, ymax, 0, ph)
            x = start + i * (bar_w + gap)
            y = py + ph - bh
            color = COLORS[r["family"]]
            body.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{bh:.1f}" rx="5" fill="{color}"/>')
            body.append(f'<text x="{x+bar_w/2}" y="{y-8:.1f}" text-anchor="middle" class="label">{val:.3f}</text>')
            body.append(f'<text x="{x+bar_w/2}" y="{py+ph+23}" text-anchor="middle" class="small">{esc(r["id"])}</text>')
        if lower_better:
            body.append(f'<text x="{px+pw-6}" y="{py+18}" text-anchor="end" class="small">lower is better</text>')
    write_svg("fig01_matched_20bpt_comparator", svg_doc(w, h, "\n".join(body)))


def line_svg(title: str, subtitle: str, records: list[dict[str, Any]], x_field: str, y_fields: list[tuple[str, str, str]], out_name: str, x_label: str) -> None:
    for y_field, _, _ in y_fields:
        require(records, [x_field, y_field], out_name)
    w, h = 920, 620
    left, top, plot_w, plot_h = 86, 122, 650, 360
    xs = [float(r[x_field]) for r in records]
    xlo, xhi = nice_range(xs, pad=0.12)
    body = [f'<text x="70" y="40" class="title">{esc(title)}</text>', f'<text x="70" y="62" class="subtitle">{esc(subtitle)}</text>']
    for idx, (y_field, label, color) in enumerate(y_fields):
        ys = [float(r[y_field]) for r in records]
        ylo, yhi = nice_range(ys, pad=0.12)
        x0 = left + idx * 390
        pw = 300
        body.append(f'<text x="{x0}" y="{top-24}" class="label">{esc(label)}</text>')
        for tick in axis_ticks(xlo, xhi):
            x = scale(tick, xlo, xhi, x0, x0 + pw)
            body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" class="grid"/>')
            body.append(f'<text x="{x:.1f}" y="{top+plot_h+23}" text-anchor="middle" class="tick">{tick:.4g}</text>')
        for tick in axis_ticks(ylo, yhi):
            y = scale(tick, ylo, yhi, top + plot_h, top)
            body.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}" class="grid"/>')
            body.append(f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick:.2f}</text>')
        body.append(f'<line x1="{x0}" y1="{top+plot_h}" x2="{x0+pw}" y2="{top+plot_h}" class="axis"/>')
        body.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{top+plot_h}" class="axis"/>')
        pts=[]
        for r in sorted(records, key=lambda item: float(item[x_field])):
            x = scale(float(r[x_field]), xlo, xhi, x0, x0+pw)
            y = scale(float(r[y_field]), ylo, yhi, top+plot_h, top)
            pts.append((x,y,r))
        body.append('<polyline points="' + ' '.join(f'{x:.1f},{y:.1f}' for x,y,_ in pts) + f'" fill="none" stroke="{color}" stroke-width="2.5"/>')
        label_offsets = {
            "EXP-092-0.0225": (8.0, -10.0, "start"),
            "EXP-092-0.025": (0.0, 18.0, "middle"),
            "EXP-092-0.0275": (-8.0, -10.0, "end"),
        }
        for x,y,r in pts:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="#fff" stroke-width="1.5"/>')
            dx, dy, anchor = label_offsets.get(r["id"], (7.0, -8.0, "start"))
            label_text = r["label"].replace("Cost ", "")
            if r["id"] == "EXP-092-0.025":
                label_text = "0.025 knee"
            body.append(f'<text x="{x+dx:.1f}" y="{y+dy:.1f}" text-anchor="{anchor}" class="small">{esc(label_text)}</text>')
        body.append(f'<text x="{x0+pw/2}" y="{h-40}" text-anchor="middle" class="label">{esc(x_label)}</text>')
    write_svg(out_name, svg_doc(w, h, "\n".join(body)))


def sparse_repair_trace_svg() -> None:
    """Render a token-level sparse repair trace from EXP-094 diagnostics."""
    diagnostic_path = "experiments/modal_downloads/exp094_modal_dabe_repair_trace_diag_001/decode_diagnostics.json"
    diagnostic = load_json(diagnostic_path)
    corrected_samples = diagnostic.get("corrected_samples", [])
    if not corrected_samples:
        raise ValueError("fig07_sparse_repair_trace requires EXP-094 corrected_samples.")

    def active_corrected_count(sample: dict[str, Any]) -> int:
        return sum(
            1
            for item in sample.get("repair_trace", [])
            if item.get("repair_outcome") == "corrected" and item.get("lookup_active")
        )

    sample = max(
        corrected_samples,
        key=lambda item: (
            float(item.get("token_acc", 0.0)) >= 0.98,
            active_corrected_count(item),
            int(item.get("repair_corrected_count", 0)),
            float(item.get("token_acc", 0.0)),
        ),
    )
    corrected_rows = [
        item
        for item in sample.get("repair_trace", [])
        if item.get("repair_outcome") == "corrected" and item.get("lookup_active")
    ]
    if not corrected_rows:
        corrected_rows = [
            item for item in sample.get("repair_trace", []) if item.get("repair_outcome") == "corrected"
        ]
    rows = corrected_rows[:13]

    trace_payload = {
        "description": "Sparse repair trace from EXP-094 on the EXP-087 quality-anchor checkpoint.",
        "source": diagnostic_path,
        "sample_index": sample.get("sample_index"),
        "batch_index": sample.get("batch_index"),
        "row_index": sample.get("row_index"),
        "token_acc": sample.get("token_acc"),
        "gist_token_acc": sample.get("gist_token_acc"),
        "chunk_deviation": sample.get("chunk_deviation"),
        "repair_corrected_count": sample.get("repair_corrected_count"),
        "target_excerpt": sample.get("target_text", "")[:220],
        "gist_excerpt": (sample.get("gist_pred_text") or "")[:220],
        "final_excerpt": sample.get("pred_text", "")[:220],
        "rows": rows,
    }
    (OUT_DIR / "repair_trace_data.json").write_text(json.dumps(trace_payload, indent=2))

    w, h = 920, 585
    x0, y0 = 70, 118
    col_x = [x0, x0 + 54, x0 + 150, x0 + 292, x0 + 432, x0 + 585, x0 + 708]
    row_h = 30
    body = [
        '<text x="70" y="40" class="title">Fig. 7. Sparse repair trace from EXP-094</text>',
        '<text x="70" y="62" class="subtitle">The gist stream misses local lexical details; sparse repair restores selected tokens in the final decode.</text>',
        f'<text x="{x0}" y="86" class="small">Source: EXP-087 checkpoint diagnostic. Gist acc {float(sample.get("gist_token_acc", 0.0)):.3f} -> final acc {float(sample.get("token_acc", 0.0)):.3f}; {int(sample.get("repair_corrected_count", 0))} corrected positions.</text>',
    ]
    headers = ["pos", "target", "gist pred", "final pred", "slot p", "outcome", "why it matters"]
    for idx, header in enumerate(headers):
        body.append(f'<text x="{col_x[idx]}" y="{y0}" class="label">{esc(header)}</text>')
    body.append(f'<line x1="{x0}" y1="{y0+10}" x2="850" y2="{y0+10}" class="axis"/>')
    outcome_colors = {
        "corrected": COLORS["dabe"],
        "missed": COLORS["cost"],
        "preserved": COLORS["muted"],
        "damaged": COLORS["cost"],
    }
    for ridx, row in enumerate(rows):
        y = y0 + 38 + ridx * row_h
        fill = "#f4f1e8" if ridx % 2 == 0 else "#fbfaf6"
        body.append(f'<rect x="{x0-10}" y="{y-22}" width="790" height="30" fill="{fill}" rx="4"/>')
        body.append(f'<text x="{col_x[0]}" y="{y}" class="small">{int(row["position"])}</text>')
        body.append(f'<text x="{col_x[1]}" y="{y}" class="small">{esc(row["target_text"])}</text>')
        body.append(f'<text x="{col_x[2]}" y="{y}" class="small">{esc(row.get("gist_pred_text", ""))}</text>')
        body.append(f'<text x="{col_x[3]}" y="{y}" class="small">{esc(row.get("pred_text", ""))}</text>')
        keep_prob = row.get("lookup_keep_prob")
        keep_label = f'{float(keep_prob):.2f}' if keep_prob is not None else "n/a"
        body.append(f'<text x="{col_x[4]}" y="{y}" class="small">{esc(keep_label)}</text>')
        outcome = str(row.get("repair_outcome", ""))
        color = outcome_colors.get(outcome, COLORS["muted"])
        body.append(f'<circle cx="{col_x[5]+8}" cy="{y-4}" r="5" fill="{color}"/>')
        body.append(f'<text x="{col_x[5]+20}" y="{y}" class="small">{esc(outcome)}</text>')
        if row.get("target_text") in {" cloud", " monster", " Anna", "What"}:
            interp = "entity/dialogue detail"
        elif row.get("target_text") in {" sun", " sky", " darker", " covered"}:
            interp = "scene state"
        elif row.get("target_text") in {" like", " just"}:
            interp = "local syntax"
        else:
            interp = "lexical repair"
        body.append(f'<text x="{col_x[6]}" y="{y}" class="small">{esc(interp)}</text>')
    body.append(
        f'<text x="{x0}" y="{h-38}" class="label">'
        f'Target: {esc(sample.get("target_text", "").replace(chr(10), " ")[:150])}...'
        '</text>'
    )
    write_svg("fig07_sparse_repair_trace", svg_doc(w, h, "\n".join(body)))


def write_svg(name: str, content: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.svg").write_text(content)


def convert_outputs() -> None:
    if not RSVG_CONVERT.exists():
        raise FileNotFoundError(f"rsvg-convert not found at {RSVG_CONVERT}")
    for svg in sorted(OUT_DIR.glob("fig*.svg")):
        subprocess.run([str(RSVG_CONVERT), "-f", "pdf", "-o", str(svg.with_suffix(".pdf")), str(svg)], check=True)
        subprocess.run([str(RSVG_CONVERT), "-f", "png", "-o", str(svg.with_suffix(".png")), str(svg)], check=True)


def build() -> list[dict[str, Any]]:
    records = [normalize(entry) for entry in CATALOG]
    # Hard validation for the normalized pack.
    for rec in records:
        if rec["bits_per_token"] is None or rec["token_accuracy"] is None:
            raise ValueError(f"{rec['id']} missing required normalized bits/token or token accuracy")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figure_data.json").write_text(json.dumps(records, indent=2))

    by_id = {r["id"]: r for r in records}
    paired_bar_svg([by_id["EXP-093"], by_id["EXP-092-0.025"]])
    fig2_ids = ["EXP-071", "EXP-079", "EXP-093", "EXP-088", "EXP-089", "EXP-090", "EXP-087", "EXP-092-0.025"]
    scatter_svg(
        "Fig. 2. Rate-distortion frontier",
        "Higher token accuracy at a comparable or lower bitrate indicates a better tokenizer-autoencoder point.",
        [by_id[i] for i in fig2_ids],
        "bits_per_token",
        "token_accuracy",
        "bits per token",
        "token accuracy",
        "fig02_rate_distortion_frontier",
    )
    line_svg(
        "Fig. 3. Replicated cost-knee curve",
        "Slot cost 0.025 is the local quality knee; higher pressure saves bits but over-prunes.",
        [by_id[i] for i in ["EXP-092-0.0225", "EXP-092-0.025", "EXP-092-0.0275"]],
        "slot_cost",
        [("chunk_deviation_mean", "Mean chunk deviation", COLORS["quality"]), ("bits_per_token", "Observed bits/token", COLORS["cost"])],
        "fig03_cost_knee_curve",
        "lookup slot cost weight",
    )
    fig4_ids = ["EXP-080-K4", "EXP-077", "EXP-080-K16", "EXP-083", "EXP-084", "EXP-085-K12", "EXP-085-K16", "EXP-085-K20", "EXP-086", "EXP-087"]
    scatter_svg(
        "Fig. 4. Adaptive repair-budget ablation",
        "Better allocation recovers high-K quality without paying the full fixed-K cost.",
        [by_id[i] for i in fig4_ids],
        "bits_per_token",
        "token_accuracy",
        "observed/effective bits per token",
        "token accuracy",
        "fig04_adaptive_budget_ablation",
        label_offsets={
            "EXP-084": (8.0, -26.0, "start"),
            "EXP-085-K12": (-12.0, 15.0, "end"),
            "EXP-085-K16": (10.0, 20.0, "start"),
            "EXP-085-K20": (12.0, -8.0, "start"),
            "EXP-086": (-10.0, -16.0, "end"),
            "EXP-087": (12.0, 14.0, "start"),
        },
    )
    fig5_ids = ["EXP-088", "EXP-089", "EXP-090", "EXP-087"]
    scatter_svg(
        "Fig. 5. Variable-window ablation",
        "Changing token-window geometry did not beat fixed chunks plus residual repair.",
        [by_id[i] for i in fig5_ids],
        "bits_per_token",
        "chunk_deviation_mean",
        "observed bits per token",
        "mean chunk deviation (lower is better)",
        "fig05_variable_window_negative_result",
        invert_y=False,
    )
    fig6_records = [r for r in records if r["active_k"] is not None and r["chunk_deviation_mean"] is not None and r["id"] in {
        "EXP-084", "EXP-085-K12", "EXP-085-K16", "EXP-085-K20", "EXP-086", "EXP-087", "EXP-091-0.02", "EXP-091-0.025", "EXP-091-0.03", "EXP-092-0.0225", "EXP-092-0.025", "EXP-092-0.0275"
    }]
    scatter_svg(
        "Fig. 6. Active repair budget vs distortion",
        "Active lookup slots are useful, but router quality determines how much distortion each slot removes.",
        fig6_records,
        "active_k",
        "chunk_deviation_mean",
        "active lookup K",
        "mean chunk deviation (lower is better)",
        "fig06_active_k_vs_quality",
        label_offsets={
            "EXP-091-0.02": (10.0, -14.0, "start"),
            "EXP-091-0.025": (-12.0, 18.0, "end"),
            "EXP-091-0.03": (10.0, -12.0, "start"),
            "EXP-092-0.0225": (12.0, -24.0, "start"),
            "EXP-092-0.025": (12.0, 18.0, "start"),
            "EXP-092-0.0275": (10.0, -12.0, "start"),
        },
    )
    sparse_repair_trace_svg()
    convert_outputs()
    return records


if __name__ == "__main__":
    built = build()
    print(f"Wrote {len(built)} normalized records and {len(list(OUT_DIR.glob('fig*.svg')))} figures to {OUT_DIR}")
