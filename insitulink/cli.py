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
    args = parser.parse_args(argv)
    try:
        # Optional native readers may print diagnostics. Stdout is a JSON protocol.
        with redirect_stdout(sys.stderr):
            from . import workflow
            if args.command == "run":
                result = workflow.run(args.request, args.output)
            elif args.command == "inspect":
                result = workflow.inspect(args.output)
            elif args.command == "resume":
                result = workflow.resume(args.output)
            else:
                from .demo import create_demo
                directory = args.output or Path(tempfile.mkdtemp(prefix=f"insitulink-{args.case}-"))
                request = create_demo(args.case, directory)
                result = workflow.run(request, directory / "results")
    except (InputError, OSError, ValueError) as exc:
        result = {"schema": "insitulink.run.v1", "status": "needs_input", "issues": [exc.as_dict() if isinstance(exc, InputError)
                  else {"code": "invalid_input", "message": str(exc)}]}
    print(json.dumps(json_value(result), ensure_ascii=False, indent=2, allow_nan=False))
    return {"completed": 0, "running": 0, "needs_input": 2}.get(result["status"], 1)
