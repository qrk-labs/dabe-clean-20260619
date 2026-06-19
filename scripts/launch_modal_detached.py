#!/usr/bin/env python3
"""Detached Modal launcher with explicit launch contract recording.

Records:
- exact command
- app id
- log-tail command
- status command
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def _list_apps() -> list[dict]:
    try:
        proc = subprocess.run(
            ["modal", "app", "list", "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    try:
        parsed = json.loads(proc.stdout)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def _pick_new_app_id(before_ids: set[str], after_apps: list[dict]) -> str | None:
    new_apps = [app for app in after_apps if str(app.get("App ID", "")) not in before_ids]
    if not new_apps:
        return None
    detached = [app for app in new_apps if "detached" in str(app.get("State", "")).lower()]
    candidates = detached or new_apps
    candidates.sort(key=lambda app: str(app.get("Created at", "")), reverse=True)
    app_id = str(candidates[0].get("App ID", "")).strip()
    return app_id or None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", default="scripts/modal_feasibility_smoke.py::run_pipeline")
    parser.add_argument("--config-name", default="feasibility_modal_olmo_smoke")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--stages", default="tokenizer,dabe_lm,bpe_baseline")
    parser.add_argument("--overrides", default="")
    parser.add_argument("--periodic-commit-seconds", type=int, default=300)
    parser.add_argument("--output-dir", default="experiments/modal_launches")
    parser.add_argument("--launch-timeout-seconds", type=int, default=25)
    args = parser.parse_args()

    command = [
        "modal",
        "run",
        "-d",
        args.function,
        "--config-name",
        args.config_name,
        "--stages",
        args.stages,
        "--periodic-commit-seconds",
        str(args.periodic_commit_seconds),
    ]
    if args.run_id:
        command.extend(["--run-id", args.run_id])
    if args.overrides:
        command.extend(["--overrides", args.overrides])

    before_apps = _list_apps()
    before_ids = {str(app.get("App ID", "")) for app in before_apps}

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=max(5, int(args.launch_timeout_seconds)))
    except subprocess.TimeoutExpired:
        timed_out = True
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
    output = f"{stdout}\n{stderr}".strip()
    app_id_match = re.search(r"\b(ap-[A-Za-z0-9]+)\b", output)
    app_id = app_id_match.group(1) if app_id_match else None
    if app_id is None:
        after_apps = _list_apps()
        app_id = _pick_new_app_id(before_ids, after_apps)

    launched_at = datetime.now(timezone.utc).isoformat()
    launch_contract = {
        "launched_at": launched_at,
        "command": _shell_join(command),
        "return_code": process.returncode,
        "timed_out_waiting_for_local_process": timed_out,
        "app_id": app_id,
        "log_tail_command": f"modal app logs {app_id}" if app_id else None,
        "status_command": "modal app list --json",
        "raw_output": output,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_label = args.run_id or "adhoc"
    out_path = output_dir / f"{stamp}_{run_label}.json"
    out_path.write_text(json.dumps(launch_contract, indent=2))

    print(json.dumps(launch_contract, indent=2))
    print(f"\nSaved launch contract to: {out_path}")

    return 0 if app_id else process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
