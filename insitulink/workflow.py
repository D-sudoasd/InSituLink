"""Deterministic, resumable frame-to-parameter alignment and file handoff."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO
import json
import math
from pathlib import Path
import re
import sys
import traceback

import numpy as np

from . import __version__
from .errors import InputError
from .inputs import read_frames, read_source, referenced_paths
from .numerics import label_phases, resolve_clock, sample_channel
from .report import make_report
from .storage import FileDigests, Stages, atomic_text, digest, json_value, output_lock, write_json


SCHEMA = "insitulink.run.v1"


def _path(value, base):
    if not isinstance(value, (str, Path)) or not str(value):
        raise InputError("An input path must be a nonempty string", "invalid_path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[\w-]+", value):
        raise InputError("Source and channel identifiers must contain only letters, digits, '_' or '-'", "invalid_identifier")
    return value


def _csv_text(rows):
    if not rows:
        return ""
    names = list(dict.fromkeys(key for row in rows for key in row))
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, names)
    writer.writeheader()
    for row in json_value(rows):
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    return "\ufeff" + stream.getvalue()


def _inline_features(frames):
    if not any("eps_hkl" in row for row in frames):
        return {"rows": [], "signals": [], "metadata": {"method": "not_requested"}, "issues": []}
    values = []
    for frame in frames:
        try:
            values.append(float(frame.get("eps_hkl")))
        except (TypeError, ValueError):
            values.append(float("nan"))
    valid = [math.isfinite(value) and row.get("t_mid_s") is not None for row, value in zip(frames, values)]
    return {"rows": [], "signals": [{"id": "eps_hkl", "t_s": [r.get("t_mid_s") for r in frames], "values": values, "valid": valid}],
            "metadata": {"method": "supplied_frame_feature", "feature_column": "eps_hkl"}, "issues": []}


def _preview_dependencies(config, frames, base, files):
    paths = []
    if config.get("type") == "spectra":
        paths.extend(_path(item["path"], base) for item in config.get("files", []))
    else:
        paths.extend(_path(config[name], base) for name in ("geometry", "mask") if config.get(name))
        paths.extend(Path(row["source_file"]) for row in frames if row.get("source_file"))
    return [files.file(path) for path in dict.fromkeys(paths)]


def _frames(config, base, stages, files):
    # HDF5 metadata is re-read on resume, without touching detector pixels. Its
    # normalized values bind subsequent stages even when timestamps are restored.
    if config.get("type", "table") == "hdf5":
        rows, metadata = read_frames(config, base)
        binding = {"config": config, "rows": rows, "metadata": metadata}
        return stages.run("frames", binding, lambda: [rows, metadata], ["inputs.py"])
    dependencies = [files.file(_path(config["path"], base))] if config.get("path") else []
    return stages.run("frames", {"config": config, "dependencies": dependencies, "base": str(base)},
                      lambda: read_frames(config, base), ["inputs.py"])


def _load_source(config, base, stages, files):
    dependencies = [files.file(_path(config["path"], base))] if config.get("path") else []
    source = stages.run(f"source-{_id(config.get('id'))}", {"config": config, "dependencies": dependencies, "base": str(base)},
                        lambda: read_source(config, base), ["inputs.py"])
    source["t_s"] = np.asarray(source["t_s"], dtype=float)
    for name, channel in source["channels"].items():
        _id(name)
        channel["values"] = np.asarray(channel["values"], dtype=float)
    return source


def _identify(frames):
    seen = set()
    for sequence, row in enumerate(frames):
        identity = {key: row.get(key) for key in ("source_file", "dataset", "frame_index")}
        uid = digest(identity)[:24]
        if uid in seen:
            raise InputError("Two table rows refer to the same physical frame", "duplicate_frame", identity=identity)
        seen.add(uid)
        row["frame_uid"] = uid
        row["sequence_index"] = sequence


def _select(frames, analysis):
    if not isinstance(analysis, dict):
        raise InputError("analysis must be an object")
    indices = analysis.get("frame_indices")
    uids = analysis.get("frame_uids")
    if indices is not None and (not isinstance(indices, list) or any(type(i) is not int or i < 0 for i in indices)):
        raise InputError("analysis.frame_indices must contain zero-based nonnegative integers")
    if uids is not None and (not isinstance(uids, list) or any(not isinstance(i, str) for i in uids)):
        raise InputError("analysis.frame_uids must be a list of stable frame identifiers")
    if indices is not None and not set(indices).issubset({row["frame_index"] for row in frames}):
        raise InputError("analysis.frame_indices contains a nonexistent physical frame")
    if uids is not None and not set(uids).issubset({row["frame_uid"] for row in frames}):
        raise InputError("analysis.frame_uids contains a nonexistent frame")
    for row in frames:
        row["selected"] = (indices is None or row["frame_index"] in indices) and (uids is None or row["frame_uid"] in uids)


def _table(frames, sources, decisions, issues, experiment):
    table = []
    previous = None
    for frame in frames:
        flags = []
        start, end = frame.get("t_start_s"), frame.get("t_end_s")
        if start is None or end is None:
            flags.append("missing_frame_time_or_exposure")
        if start is not None:
            if previous is not None and start <= previous:
                flags.append("nonmonotonic_frame_clock")
            previous = start
        table.append({**frame, "frame_flags": flags, "valid_analysis": bool(frame["selected"] and not flags)})
    required_seen = False
    for source in sources:
        decision = decisions[source["id"]]
        resolved = decision["status"] == "resolved"
        for name, channel in source["channels"].items():
            prefix = f"{source['id']}.{name}"
            required_seen |= channel["required"]
            starts = [row.get("t_start_s") if row.get("t_start_s") is not None else np.nan for row in frames]
            ends = [row.get("t_end_s") if row.get("t_end_s") is not None else np.nan for row in frames]
            sampled = [{"mid": None, "mean": None, "min": None, "max": None, "coverage": 0., "valid": False,
                        "flags": ["missing_frame_time_or_exposure" if resolved else "clock_unresolved"]} for _ in frames]
            usable = [i for i, (a, b) in enumerate(zip(starts, ends)) if np.isfinite(a) and np.isfinite(b)]
            if resolved and usable:
                try:
                    samples = sample_channel(source["t_s"], channel["values"], [starts[i] for i in usable], [ends[i] for i in usable],
                                             decision["dt_s"], channel["kind"], channel.get("max_gap_s"))
                    for index, values in zip(usable, samples):
                        sampled[index] = values
                except InputError as exc:
                    issues.append({"source_id": source["id"], "channel": name, **exc.as_dict()})
            queries = [np.nan if row.get("t_mid_s") is None or not resolved else row["t_mid_s"] + decision["dt_s"] for row in frames]
            channel_experiment = channel.get("experiment", experiment)
            phases = label_phases(source["t_s"], channel["values"], queries, channel_experiment,
                                  channel["kind"], channel.get("phase_tolerance"))
            for row, values, phase in zip(table, sampled, phases):
                for key, value in values.items():
                    row[f"{prefix}.{key}"] = value
                row[f"{prefix}.unit"] = channel["unit"]
                row[f"{prefix}.phase"] = phase if values["valid"] else "unknown"
                if channel["required"] and not values["valid"]:
                    row["valid_analysis"] = False
    if not required_seen:
        issues.append({"code": "no_required_channels", "message": "All channels are optional; valid_analysis covers frame selection only."})
    return table


def _delivery(output, state, frames, table, sources, features, files):
    report = make_report(state, frames, table, sources, features)
    atomic_text(output / "frames.csv", _csv_text(table))
    atomic_text(output / "peaks.csv", _csv_text(features.get("rows", [])))
    write_json(output / "decisions.json", state.get("decisions", {}))
    manifest_rows = []
    for frame, row in zip(frames, table):
        manifest_rows.append({key: row.get(key) for key in ("frame_uid", "sequence_index", "source_file", "dataset", "frame_index", "frame_number", "original_id",
                                                           "t_start_s", "t_end_s", "t_mid_s", "exposure_s", "selected", "valid_analysis", "frame_flags")})
    write_json(output / "frame_manifest.json", {"schema": "insitulink.frames.v1", "index_base": 0, "display_number_base": 1,
                                                "metadata": state.get("frame_metadata", {}), "frames": manifest_rows})
    channels = [{"id": f"{source['id']}.{name}", "unit": channel["unit"], "kind": channel["kind"], "required": channel["required"]}
                for source in sources for name, channel in source["channels"].items()]
    source_available = {}
    raw_issues = []
    for row in manifest_rows:
        if not row["valid_analysis"]:
            continue
        path = row["source_file"]
        if path and path not in source_available:
            source_available[path] = Path(path).is_file()
        if not path or not source_available[path]:
            raw_issues.append({"frame_uid": row["frame_uid"], "source_file": path,
                               "reason": "raw_source_missing" if path else "raw_source_not_referenced"})
    handoff = {"schema": "insitulink.handoff.v1", "producer": "InSituLink", "version": __version__, "status": state["status"],
               "alignment_complete": state["status"] == "completed",
               "ready_for_integration": state["status"] == "completed" and state["n_valid_analysis"] > 0 and not raw_issues,
               "raw_source_issues": raw_issues,
               "consumer": "DIDASFlow", "consumer_integration": "file_contract_only", "raw_frames_copied": False,
               "parameter_table": "frames.csv", "frame_manifest": "frame_manifest.json", "decisions": "decisions.json", "index_base": 0,
               "run_status": "status.json", "request_sha256": state.get("request_sha256"),
               "channels": channels, "selected_frame_uids": [r["frame_uid"] for r in table if r["selected"]],
               "analysis_frame_uids": [r["frame_uid"] for r in table if r["valid_analysis"]], "frames": manifest_rows,
               "notes": ["Each frame remains independent. No frame combination is requested.",
                         "Metadata-only examples can complete alignment without raw detector images; those are not ready for real integration.",
                         "File existence and declared frame identity do not replace the consumer's calibration and frame preflight.",
                         "DIDASFlow does not automatically consume this schema yet. This package establishes the file contract.",
                         "Clock evidence and coverage flags must accompany the parameter table."]}
    atomic_text(output / "report.html", report)
    for name in ("frames.csv", "peaks.csv", "decisions.json", "frame_manifest.json", "report.html"):
        state["artifacts"][name] = files.file(output / name)
    # Publish readiness last; status.json is the final run commit marker.
    write_json(output / "handoff.json", handoff)
    state["artifacts"]["handoff.json"] = files.file(output / "handoff.json")


def _protect_inputs(request_path, output):
    """Check every declared input before creating or changing output artifacts."""
    if output == request_path or request_path.is_relative_to(output):
        raise InputError("Use a result directory separate from the request directory", "unsafe_output")
    try:
        request = json.loads(request_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return  # Normal input handling will produce a structured issue.
    if not isinstance(request, dict):
        return
    base = request_path.parent
    configs = [request.get("frames")] + (request.get("sources") if isinstance(request.get("sources"), list) else [])
    paths = []
    for config in configs:
        if not isinstance(config, dict):
            continue
        if config.get("path"):
            paths.append(_path(config["path"], base))
        if config is request.get("frames"):
            try:
                paths.extend(referenced_paths(config, base))
            except InputError:
                pass
    config = request.get("features")
    if isinstance(config, dict):
        paths.extend(_path(config[key], base) for key in ("geometry", "mask") if config.get(key))
        if isinstance(config.get("files"), list):
            paths.extend(_path(item["path"], base) for item in config["files"] if isinstance(item, dict) and item.get("path"))
    metadata = request.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("sync_markers_file"):
        paths.append(_path(metadata["sync_markers_file"], base))
    conflicts = [str(path) for path in paths if path.is_relative_to(output)]
    if conflicts:
        raise InputError("The output directory contains experiment inputs; choose a separate directory", "unsafe_output", paths=conflicts)


def _not_ready(output, status, issues):
    write_json(output / "handoff.json", {"schema": "insitulink.handoff.v1", "producer": "InSituLink", "status": status,
                                         "ready_for_integration": False, "issues": issues, "run_status": "status.json"})


def _verify_input_stability(files, output):
    for (name, signature), _ in files.values.items():
        path = Path(name)
        if path.is_relative_to(output):
            continue
        try:
            current = path.stat()
        except OSError as exc:
            raise InputError(f"Input disappeared during this run: {path}", "input_changed") from exc
        if signature != (current.st_size, current.st_mtime_ns, current.st_ctime_ns, current.st_ino):
            raise InputError(f"Input changed during this run: {path}; resume after acquisition is stable", "input_changed")


def run(request_path, output):
    """Run or reuse stages; return JSON-safe status. Never modify experiment inputs."""
    request_path, output = Path(request_path).resolve(), Path(output).resolve()
    _protect_inputs(request_path, output)
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        return _run_locked(request_path, output)


def _run_locked(request_path, output):
    state = {"schema": SCHEMA, "version": __version__, "status": "running", "request_path": str(request_path), "output": str(output),
             "started_at": datetime.now(timezone.utc).isoformat(), "stages": {}, "issues": [], "decisions": {}, "artifacts": {},
             "n_frames": 0, "n_selected": 0, "n_valid_analysis": 0}
    files = FileDigests()
    stages = Stages(output, state, files)
    frames, sources, table = [], [], []
    features = {"rows": [], "signals": [], "metadata": {}, "issues": []}
    _not_ready(output, "running", [])
    write_json(output / "status.json", state)
    atomic_text(output / "run.log", f"InSituLink {__version__}\nrequest: {request_path}\n")
    try:
        request = json.loads(request_path.read_text(encoding="utf-8-sig"))
        if not isinstance(request, dict) or request.get("schema_version") != 1:
            raise InputError("Request requires schema_version: 1", "unsupported_schema")
        experiment = request.get("experiment", "generic")
        if experiment not in ("tensile", "thermal", "generic"):
            raise InputError("experiment must be tensile, thermal or generic")
        state["experiment"] = experiment
        state["python_version"] = sys.version.split()[0]
        state["request_sha256"] = files.file(request_path)["sha256"]
        write_json(output / "request.snapshot.json", request)
        base = request_path.parent
        if not isinstance(request.get("frames"), dict) or not isinstance(request.get("sources"), list) or not request["sources"]:
            raise InputError("Request requires frames and a nonempty sources list")
        if any(not isinstance(source, dict) for source in request["sources"]):
            raise InputError("Each source must be an object")
        if request.get("features") is not None and not isinstance(request["features"], dict):
            raise InputError("features must be an object")
        for source in request["sources"]:
            if source.get("clock") is not None and not isinstance(source["clock"], dict):
                raise InputError("source.clock must be an object")
        ids = [_id(source.get("id")) for source in request["sources"]]
        if len(set(ids)) != len(ids):
            raise InputError("External device identifiers must be unique")
        frames, metadata = _frames(request["frames"], base, stages, files)
        if not frames:
            raise InputError("No physical frames found", "missing_frames")
        _identify(frames)
        _select(frames, request.get("analysis") or {})
        state.update(n_frames=len(frames), n_selected=sum(r["selected"] for r in frames), frame_metadata=metadata)
        state["frame_metadata_sha256"] = digest({"frames": frames, "metadata": metadata})
        if metadata.get("nonmonotonic_rows") or metadata.get("duplicate_time_rows"):
            state["issues"].append({"code": "nonmonotonic_frame_clock", "message": "Frame times repeat or decrease; acquisition order was preserved."})
        evidence_file = (request.get("metadata") or {}).get("sync_markers_file") if isinstance(request.get("metadata"), dict) else None
        state["evidence_files"] = [files.file(_path(evidence_file, base))] if evidence_file else []
        for config in request["sources"]:
            try:
                sources.append(_load_source(config, base, stages, files))
            except InputError as exc:
                state["issues"].append({"source_id": config["id"], **exc.as_dict()})
        features = _inline_features(frames)
        needs_features = any((s["clock"].get("method") == "event" or s["clock"].get("event")) for s in sources)
        if needs_features and request.get("features") and not features["signals"]:
            try:
                from .preview import extract_features
                config = request["features"]
                if config.get("files") is not None and (not isinstance(config["files"], list) or any(not isinstance(item, dict) or not item.get("path") for item in config["files"])):
                    raise InputError("features.files must contain objects with a path")
                binding = {"config": config, "frames": frames, "dependencies": _preview_dependencies(config, frames, base, files)}
                features = stages.run("features", binding, lambda: extract_features(config, frames, base, output / ".insitulink" / "preview"), ["preview.py"])
            except InputError as exc:
                state["issues"].append(exc.as_dict())
        elif not needs_features:
            state["stages"]["features"] = {"status": "not_needed", "reason": "No image or spectrum read for confirmed time relationships."}
        if features.get("signals"):
            state["features"] = {"n_signals": len(features["signals"]), "metadata": features.get("metadata"), "issues": features.get("issues", [])}
        for source in sources:
            try:
                if source["clock"].get("method") == "shared" and source["metadata"]["time_basis"] != metadata["time_basis"]:
                    raise InputError("Shared clocks need compatible time bases and origins; one record is absolute and the other relative", "incompatible_time_basis")
                decision = stages.run(f"clock-{source['id']}", {"frames": frames, "source": source, "features": features, "experiment": experiment, "evidence_files": state["evidence_files"]},
                                      lambda source=source: resolve_clock(frames, source, features, experiment), ["numerics.py"])
            except InputError as exc:
                decision = {"status": "needs_input", "method": source["clock"].get("method"), "dt_s": None, "candidate_dt_s": None,
                            "reason": str(exc), "evidence": exc.as_dict()}
            state["decisions"][source["id"]] = decision
            if decision["status"] != "resolved":
                state["issues"].append({"code": "clock_unresolved", "source_id": source["id"], "message": decision["reason"]})
        table = _table(frames, sources, state["decisions"], state["issues"], experiment)
        if len(sources) != len(request["sources"]):
            for row in table:
                row["valid_analysis"] = False
                row["frame_flags"].append("source_input_unavailable")
        state["n_valid_analysis"] = sum(row["valid_analysis"] for row in table)
        _verify_input_stability(files, output)
        if request["frames"].get("type") == "hdf5":
            checked_frames, checked_metadata = read_frames(request["frames"], base)
            _identify(checked_frames)
            _select(checked_frames, request.get("analysis") or {})
            if digest({"frames": checked_frames, "metadata": checked_metadata}) != state["frame_metadata_sha256"]:
                raise InputError("Detector metadata changed during alignment", "input_changed")
        state["status"] = "needs_input" if any(item.get("code") != "no_required_channels" for item in state["issues"]) else "completed"
        if not state["n_valid_analysis"] and state["status"] == "completed":
            state["status"] = "needs_input"
            state["issues"].append({"code": "no_analysis_frames", "message": "No selected frame has complete required parameter coverage."})
    except (InputError, OSError, ImportError, json.JSONDecodeError) as exc:
        state["status"] = "needs_input"
        if isinstance(exc, InputError) and exc.code == "input_changed":
            for row in table:
                row["valid_analysis"] = False
                row["frame_flags"].append("input_changed")
            state["n_valid_analysis"] = 0
        state["issues"].append(exc.as_dict() if isinstance(exc, InputError) else {"code": "unreadable_input", "message": str(exc)})
    except Exception as exc:
        state["status"] = "failed"
        state["issues"].append({"code": "runtime_error", "message": str(exc)})
        atomic_text(output / "error.log", traceback.format_exc())
    if state["status"] != "failed":
        try:
            if not table:
                table = [{**frame, "frame_flags": ["incomplete_run"], "valid_analysis": False} for frame in frames]
            _delivery(output, state, frames, table, sources, features, files)
        except Exception as exc:
            state["status"] = "failed"
            state["issues"].append({"code": "delivery_failed", "message": str(exc)})
            atomic_text(output / "error.log", traceback.format_exc())
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    state["current_stage"] = None
    if state["status"] == "failed":
        _not_ready(output, "failed", state["issues"])
        state["artifacts"] = {}
    state["input_files"] = [{"path": key[0], "sha256": value} for key, value in files.values.items()
                            if not Path(key[0]).is_relative_to(output) and not Path(key[0]).is_relative_to(Path(__file__).parent)]
    state["next_action"] = ("Review report.html and deliver handoff.json with frames.csv and frame_manifest.json." if state["status"] == "completed"
                            else "Review issues and decisions, correct or confirm experiment.json, then resume this output directory.")
    write_json(output / "status.json", state)
    with (output / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(f"status: {state['status']}\n")
        for issue in state["issues"]:
            stream.write(f"{issue['code']}: {issue.get('message', '')}\n")
    return json_value(state)


def inspect(output):
    """Read status and verify published artifacts, without opening experiment data."""
    output = Path(output).resolve()
    try:
        state = json.loads((output / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputError(f"Cannot read saved run: {exc}", "unreadable_status") from exc
    if not isinstance(state, dict) or state.get("schema") != SCHEMA:
        raise InputError("Unsupported run status schema", "unsupported_schema")
    if not isinstance(state.get("artifacts"), dict) or not isinstance(state.get("issues"), list):
        raise InputError("Run status has an invalid artifact or issue inventory", "invalid_status")
    files = FileDigests()
    required = {"frames.csv", "peaks.csv", "decisions.json", "frame_manifest.json", "handoff.json", "report.html"}
    problems = sorted(required - set(state.get("artifacts", {})))
    for name, saved in state.get("artifacts", {}).items():
        path = (output / name).resolve()
        if not path.is_relative_to(output):
            problems.append(name)
            continue
        try:
            if files.file(path)["sha256"] != saved["sha256"]:
                problems.append(name)
        except InputError:
            problems.append(name)
    state["artifact_verification"] = {"ok": not problems, "changed_or_missing": problems, "source_inputs_checked": False}
    if problems and state["status"] != "running":
        state["status"] = "failed"
        state["issues"].append({"code": "artifact_changed", "message": "Published artifacts changed or disappeared", "files": problems})
    return state


def resume(output):
    state = inspect(output)
    return run(state["request_path"], output)
