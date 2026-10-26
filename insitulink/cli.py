"""Headless JSON command-line entry points for InSituLink."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import tempfile

from .errors import InputError
from .storage import json_value


def _assignments(values, option):
    result = {}
    for value in values or []:
        key, separator, item = value.partition("=")
        if not separator or not key or not item or key in result:
            raise InputError(f"{option} requires unique ID=VALUE entries", "invalid_assignment", value=value)
        result[key] = item
    return result


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Align in-situ frames with external parameter records.")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("run", help="Run a versioned JSON experiment request")
    command.add_argument("--request", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    for name in ("inspect", "resume"):
        command = sub.add_parser(name, help="Verify saved artifacts" if name == "inspect" else "Revalidate inputs and resume stages")
        command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("demo", help="Create and run deterministic synthetic data; thermal intentionally needs confirmation")
    command.add_argument("--case", choices=("tensile", "thermal", "multi"), required=True)
    command.add_argument("--output", type=Path, help="New example directory; default is a unique temporary directory")
    command = sub.add_parser("prepare", help="Preview inputs and create an editable request; no clock is confirmed automatically")
    command.add_argument("--frames", type=Path, required=True, help="Frame CSV/XLSX/JSON table or HDF5 metadata file")
    command.add_argument("--source", action="append", required=True, metavar="ID=PATH", help="External device table; repeat for independent devices")
    command.add_argument("--output", type=Path, required=True, help="New import directory; existing directories are never overwritten")
    command.add_argument("--profile", type=Path, help="JSON profile with reusable column mappings, units and identity registration")
    command.add_argument("--experiment", choices=("generic", "tensile", "thermal"))
    command.add_argument("--frame-time-unit", help="Declared detector time unit, for example s or ms")
    command.add_argument("--frame-time-start", help="Declared detector ISO8601 origin; include timezone")
    command.add_argument("--source-time-unit", action="append", metavar="ID=UNIT", help="Declared external time unit; repeat per device")
    command.add_argument("--source-time-start", action="append", metavar="ID=ISO8601", help="Declared external clock origin; repeat per device")
    args = parser.parse_args(argv)
    try:
        # Optional native readers may print diagnostics. Stdout is a JSON protocol.
        with redirect_stdout(sys.stderr):
            if args.command != "prepare":
                from . import workflow
            if args.command == "run":
                result = workflow.run(args.request, args.output)
            elif args.command == "inspect":
                result = workflow.inspect(args.output)
            elif args.command == "resume":
                result = workflow.resume(args.output)
            elif args.command == "prepare":
                from .importing import prepare
                result = prepare(args.frames, _assignments(args.source, "--source"), args.output,
                                 profile_path=args.profile, experiment=args.experiment,
                                 frame_time_unit=args.frame_time_unit, frame_time_start=args.frame_time_start,
                                 source_time_units=_assignments(args.source_time_unit, "--source-time-unit"),
                                 source_time_starts=_assignments(args.source_time_start, "--source-time-start"))
            else:
                from .demo import create_demo
                directory = args.output or Path(tempfile.mkdtemp(prefix=f"insitulink-{args.case}-"))
                request = create_demo(args.case, directory)
                result = workflow.run(request, directory / "results")
    except (InputError, OSError, ValueError) as exc:
        result = {"schema": "insitulink.prepare.v1" if args.command == "prepare" else "insitulink.run.v1", "status": "needs_input", "issues": [exc.as_dict() if isinstance(exc, InputError)
                  else {"code": "invalid_input", "message": str(exc)}]}
    print(json.dumps(json_value(result), ensure_ascii=False, indent=2, allow_nan=False))
    return {"completed": 0, "running": 0, "needs_input": 2}.get(result["status"], 1)
