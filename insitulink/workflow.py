"""Deterministic, resumable frame-to-parameter alignment and file handoff."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import traceback
from uuid import uuid4

import numpy as np

from . import __version__
from .errors import InputError
from .inputs import read_frames, read_source
from .numerics import exposure_phases, sample_channel
from .report import make_report
from .storage import FileDigests, Stages, atomic_text, digest, json_value, output_lock, write_json
from .workflow_inputs import (frame_origin, input_path as _path, local_frames, prepare_inputs, resolve_source_clock,
                              table_snapshot, verify_input_stability as _verify_input_stability)
from .workflow_delivery import next_actions, write_delivery


SCHEMA = "insitulink.run.v1"


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[\w-]+", value):
        raise InputError("Source and channel identifiers must contain only letters, digits, '_' or '-'", "invalid_identifier")
    return value


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
    snapshots = {}
    storage = {"paths": [], "complete": True, "issues": [], "datasets": []}
    if config.get("type") == "spectra":
        paths.extend(_path(item["path"], base) for item in config.get("files", []))
    else:
        from .hdf5_sources import discover_dependencies

        paths.extend(_path(config[name], base) for name in ("geometry", "mask") if config.get(name))
        storage = discover_dependencies(frames)
        paths.extend(Path(path) for path in storage["paths"])
    records = []
    for path in dict.fromkeys(paths):
        if config.get("type") == "spectra":
            snapshots[str(path)], record = files.capture(path)
        else:
            record = files.file(path)
        records.append(record)
    return {"files": records, "storage": storage}, snapshots


def _frames(config, base, stages, files):
    # HDF5 metadata is re-read on resume, without touching detector pixels. Its
    # normalized values bind subsequent stages even when timestamps are restored.
    if config.get("type", "table") == "hdf5":
        rows, metadata = read_frames(config, base)
        binding = {"config": config, "rows": rows, "metadata": metadata}
        return stages.run("frames", binding, lambda: [rows, metadata], ["inputs.py", "time_coordinates.py"])
    table_bytes, dependencies = table_snapshot(config, base, files)
    rows, metadata = stages.run("frames", {"config": config, "dependencies": dependencies, "base": str(base)},
                                lambda: read_frames(config, base, table_bytes=table_bytes), ["inputs.py", "time_coordinates.py"])
    metadata["input_file"] = dependencies[0] if dependencies else None
    return rows, metadata


def _load_source(config, base, stages, files):
    table_bytes, dependencies = table_snapshot(config, base, files)
    source = stages.run(f"source-{_id(config.get('id'))}", {"config": config, "dependencies": dependencies, "base": str(base)},
                        lambda: read_source(config, base, table_bytes=table_bytes), ["inputs.py", "time_coordinates.py"])
    source["t_s"] = np.asarray(source["t_s"], dtype=float)
    source["t_rel_s"] = np.asarray(source.get("t_rel_s", source["t_s"]), dtype=float)
    for name, channel in source["channels"].items():
        _id(name)
        channel["values"] = np.asarray(channel["values"], dtype=float)
    source["input_file"] = dependencies[0] if dependencies else None
    return source


def _identify(frames, context=None):
    from .identity import identify_frames

    return identify_frames(frames, context=context)


def _select(frames, analysis):
    if not isinstance(analysis, dict):
        raise InputError("analysis must be an object", field="analysis")
    indices = analysis.get("frame_indices")
    uids = analysis.get("frame_uids")
    if indices is not None and (not isinstance(indices, list) or any(type(i) is not int or i < 0 for i in indices)):
        raise InputError("analysis.frame_indices must contain zero-based nonnegative integers", field="analysis.frame_indices")
    if uids is not None and (not isinstance(uids, list) or any(not isinstance(i, str) for i in uids)):
        raise InputError("analysis.frame_uids must be a list of stable frame identifiers", field="analysis.frame_uids")
    if indices is not None and not set(indices).issubset({row["frame_index"] for row in frames}):
        raise InputError("analysis.frame_indices contains a nonexistent physical frame", field="analysis.frame_indices")
    known_uids = {value for row in frames for value in (row["frame_uid"], row.get("legacy_frame_uid"))}
    if uids is not None and not set(uids).issubset(known_uids):
        raise InputError("analysis.frame_uids contains a nonexistent frame", field="analysis.frame_uids")
    for row in frames:
        row["selected"] = (indices is None or row["frame_index"] in indices) and (uids is None or row["frame_uid"] in uids or row.get("legacy_frame_uid") in uids)


def _table(frames, sources, decisions, issues, experiment):
    from .time_coordinates import clock_offset_local_s

    calculation_frames = local_frames(frames)
    table, previous = [], None
    for frame, local in zip(frames, calculation_frames):
        flags = []
        start, end = local.get("t_start_s"), local.get("t_end_s")
        if start is None or end is None:
            flags.append("missing_frame_time_or_exposure")
        if start is not None:
            if previous is not None and start <= previous:
                flags.append("nonmonotonic_frame_clock")
            previous = start
        if frame.get("identity_status") == "invalid":
            flags.append("invalid_raw_identity")
        table.append({**frame, "frame_flags": flags, "clock_mappings": {},
                      "valid_analysis": bool(frame.get("selected") and not flags)})
    starts = [row.get("t_start_s") if row.get("t_start_s") is not None else np.nan for row in calculation_frames]
    ends = [row.get("t_end_s") if row.get("t_end_s") is not None else np.nan for row in calculation_frames]
    usable = [i for i, (a, b) in enumerate(zip(starts, ends)) if np.isfinite(a) and np.isfinite(b)]
    required_seen = False
    for source in sources:
        decision = decisions[source["id"]]
        resolved = decision["status"] == "resolved"
        origin = frame_origin(frames)
        external_origin = source.get("time_origin_s", "0")
        offset = (decision["local_dt_s"] if decision.get("local_dt_s") is not None else
                  clock_offset_local_s(decision["dt_s"], origin, external_origin)) if resolved else None
        mapping = {"status": decision["status"], "published_offset_s": decision.get("dt_s"), "local_offset_s": offset,
                   "published_offset_s_decimal": decision.get("dt_s_decimal"),
                   "frame_origin_s": origin, "source_origin_s": external_origin,
                   "convention": "source_time = frame_time + offset"}
        for row, local in zip(table, calculation_frames):
            row["clock_mappings"][source["id"]] = {**mapping,
                **{f"source_{name}_rel_s": local.get(f"t_{name}_s") + offset
                   if resolved and local.get(f"t_{name}_s") is not None else None for name in ("start", "mid", "end")}}
        times = np.asarray(source.get("t_rel_s", source["t_s"]), dtype=float)
        original_rows = source.get("source_row_numbers", [])
        for name, channel in source["channels"].items():
            prefix = f"{source['id']}.{name}"
            required_seen |= channel["required"]
            sampled = [{"mid": None, "mean": None, "min": None, "max": None, "coverage": 0., "valid": False,
                        "flags": ["missing_frame_time_or_exposure" if resolved else "clock_unresolved"]} for _ in frames]
            phases = [{"midpoint": "unknown", "phases": [], "crosses_phase": False, "unsupported": True} for _ in frames]
            if resolved and usable:
                try:
                    samples = sample_channel(times, channel["values"], [starts[i] for i in usable], [ends[i] for i in usable],
                                             offset, channel["kind"], channel.get("max_gap_s"))
                    sampled_phases = exposure_phases(times, channel["values"], [starts[i] for i in usable], [ends[i] for i in usable],
                        offset, channel.get("experiment", experiment), channel["kind"], channel.get("phase_tolerance"), channel.get("max_gap_s"))
                    for index, values, phase in zip(usable, samples, sampled_phases):
                        sampled[index], phases[index] = values, phase
                except InputError as exc:
                    issues.append({"source_id": source["id"], "channel": name, **exc.as_dict()})
            for row, values, phase in zip(table, sampled, phases):
                provenance = values.setdefault("provenance", {"time_coordinate": "source_local_seconds"})
                provenance["mapping"] = mapping
                provenance["source"] = {"id": source["id"], "path": source.get("metadata", {}).get("source"),
                                         "input_file": source.get("input_file"),
                                         "value_column": channel.get("column", name), "unit": channel["unit"],
                                         "time_column": source.get("metadata", {}).get("time_provenance", {}).get("field")}
                for support in provenance.get("midpoint", {}).get("support", []):
                    index = support.get("row_index")
                    if isinstance(index, int) and 0 <= index < len(original_rows):
                        support["source_row_number"] = original_rows[index]
                for key, value in values.items():
                    row[f"{prefix}.{key}"] = value
                row[f"{prefix}.unit"] = channel["unit"]
                row[f"{prefix}.phase"] = phase["midpoint"] if values["valid"] else "unknown"
                row[f"{prefix}.phase_exposure"] = phase
                if channel["required"] and not values["valid"]:
                    row["valid_analysis"] = False
    if not required_seen:
        issues.append({"code": "no_required_channels", "message": "All channels are optional; valid_analysis covers frame selection only."})
    return table


def _delivery(output, state, frames, table, sources, features, files, verify=None):
    write_delivery(output, state, frames, table, sources, features, files, make_report, verify)


def _protect_inputs(request_path, output):
    return prepare_inputs(request_path, output)


def _not_ready(output, status, issues):
    write_json(output / "handoff.json", {"schema": "insitulink.handoff.v1", "producer": "InSituLink", "status": status,
                                         "ready_for_integration": False, "issues": issues, "run_status": "status.json"})


def _verify_run_inputs(files, output, request, base, frames, metadata, identity_context, expected, feature_storage):
    _verify_input_stability(files, output)
    if feature_storage is not None:
        from .hdf5_sources import discover_dependencies

        if digest(discover_dependencies(frames)) != digest(feature_storage):
            raise InputError("Detector storage dependencies changed during alignment", "input_changed")
    if request["frames"].get("type") == "hdf5":
        checked_frames, checked_metadata = read_frames(request["frames"], base)
    else:
        # Verify raw addressing without re-reading table bytes or detector pixels.
        checked_frames, checked_metadata = [dict(frame) for frame in frames], metadata
    _identify(checked_frames, identity_context)
    _select(checked_frames, request.get("analysis", {}))
    if digest({"frames": checked_frames, "metadata": checked_metadata}) != expected:
        raise InputError("Detector metadata or physical frame addressing changed during alignment", "input_changed")


def run(request_path, output):
    """Run or reuse stages; return JSON-safe status. Never modify experiment inputs."""
    request_path, output = Path(request_path).resolve(), Path(output).resolve()
    prepared = _protect_inputs(request_path, output)
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        return _run_locked(request_path, output, prepared)


def _run_locked(request_path, output, prepared):
    state = {"schema": SCHEMA, "version": __version__, "status": "running", "request_path": str(request_path), "output": str(output),
             "started_at": datetime.now(timezone.utc).isoformat(), "stages": {}, "issues": [], "decisions": {}, "artifacts": {},
             "run_id": str(uuid4()), "n_frames": 0, "n_selected": 0, "n_valid_analysis": 0,
             "ready_for_integration": False, "raw_source_issues": [], "identity_validation": {},
             "request_sha256": prepared.get("request_sha256"), "request_snapshot": None}
    files = prepared["files"]
    stages = Stages(output, state, files)
    frames, sources, table = [], [], []
    features = {"rows": [], "signals": [], "metadata": {}, "issues": []}
    feature_storage = None
    verify = None
    _not_ready(output, "running", [])
    write_json(output / "status.json", state)
    atomic_text(output / "run.log", f"InSituLink {__version__}\nrequest: {request_path}\n")
    try:
        if prepared["error"] is not None:
            raise prepared["error"]
        request = prepared["request"]
        write_json(output / "request.snapshot.json", request)
        state["request_snapshot"] = "request.snapshot.json"
        if not isinstance(request, dict) or request.get("schema_version") != 1:
            raise InputError("Request requires schema_version: 1", "unsupported_schema")
        experiment = request.get("experiment", "generic")
        if experiment not in ("tensile", "thermal", "generic"):
            raise InputError("experiment must be tensile, thermal or generic")
        state["experiment"] = experiment
        state["experiment_metadata"] = request.get("metadata") if isinstance(request.get("metadata"), dict) else {}
        state["python_version"] = sys.version.split()[0]
        state["request_sha256"] = prepared["request_sha256"]
        state["code_files"] = [files.file(path) for path in sorted(Path(__file__).parent.glob("*.py"))]
        state["code_version"] = digest([{**item, "path": Path(item["path"]).name} for item in state["code_files"]])
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
        identity_context = {**state["experiment_metadata"], "base_dir": str(base), "request_id": str(request_path),
                            "assets": request["frames"].get("assets", {})}
        state["identity_validation"] = _identify(frames, identity_context)
        state["issues"].extend(state["identity_validation"].get("issues", []))
        state["warnings"] = state["identity_validation"].get("warnings", [])
        state.update(n_frames=len(frames), frame_metadata=metadata)
        _select(frames, request.get("analysis", {}))
        state.update(n_frames=len(frames), n_selected=sum(r["selected"] for r in frames), frame_metadata=metadata)
        state["frame_metadata_sha256"] = digest({"frames": frames, "metadata": metadata})
        if metadata.get("frame_count_unresolved"):
            state["issues"].append({"code": "frame_count_unresolved", "message": metadata["frame_count_unresolved_reason"]})
        if metadata.get("nonmonotonic_rows") or metadata.get("duplicate_time_rows"):
            state["issues"].append({"code": "nonmonotonic_frame_clock", "message": "Frame times repeat or decrease; acquisition order was preserved."})
        evidence_file = (request.get("metadata") or {}).get("sync_markers_file") if isinstance(request.get("metadata"), dict) else None
        state["evidence_files"] = [files.file(_path(evidence_file, base))] if evidence_file else []
        for config in request["sources"]:
            try:
                sources.append(_load_source(config, base, stages, files))
            except InputError as exc:
                state["issues"].append({"source_id": config["id"], **exc.as_dict()})
        calculation_frames = local_frames(frames)
        features = _inline_features(calculation_frames)
        needs_features = any((s["clock"].get("method") == "event" or s["clock"].get("event")) for s in sources)
        if needs_features and request.get("features") and not features["signals"]:
            try:
                from .preview import extract_features
                config = request["features"]
                if config.get("files") is not None and (not isinstance(config["files"], list) or any(not isinstance(item, dict) or not item.get("path") for item in config["files"])):
                    raise InputError("features.files must contain objects with a path")
                dependencies, source_bytes = _preview_dependencies(config, frames, base, files)
                feature_storage = dependencies["storage"] if config.get("type") == "images" else None
                binding = {"config": config, "frames": frames, "dependencies": dependencies}
                modules = ["preview.py", "inputs.py", "time_coordinates.py", "workflow_inputs.py"] + (["hdf5_sources.py"] if feature_storage is not None else [])
                features = stages.run("features", binding, lambda: extract_features(config, calculation_frames, base, output / ".insitulink" / "preview",
                                      **({"source_bytes": source_bytes} if config.get("type") == "spectra" else {})),
                                      modules, cacheable=dependencies["storage"]["complete"])
                features["metadata"]["input_dependencies"] = dependencies
                if not dependencies["storage"]["complete"]:
                    state["issues"].append({"code": "unverified_feature_dependencies", "message": "Preanalysis was recomputed without cache reuse, but its storage dependencies remain unverified.",
                                             "details": dependencies["storage"]["issues"]})
            except InputError as exc:
                state["issues"].append(exc.as_dict())
        elif not needs_features:
            state["stages"]["features"] = {"status": "not_needed", "reason": "No image or spectrum read for confirmed time relationships."}
        if features.get("signals"):
            features.setdefault("metadata", {}).update(time_coordinate="frame_local_seconds", time_origin_s=frame_origin(frames))
            state["features"] = {"n_signals": len(features["signals"]), "metadata": features.get("metadata"), "issues": features.get("issues", [])}
        for source in sources:
            try:
                decision = stages.run(f"clock-{source['id']}", {"frames": frames, "source": source, "features": features, "experiment": experiment, "evidence_files": state["evidence_files"]},
                                      lambda source=source: resolve_source_clock(frames, source, features, experiment, metadata),
                                      ["numerics.py", "time_coordinates.py", "workflow_inputs.py"])
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
        verify = lambda: _verify_run_inputs(files, output, request, base, frames, metadata, identity_context,
                                            state["frame_metadata_sha256"], feature_storage)
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
    state["input_files"] = [{"path": key[0], "sha256": value} for key, value in files.values.items()
                            if not Path(key[0]).is_relative_to(output) and not Path(key[0]).is_relative_to(Path(__file__).parent)]
    state["input_validation"] = {"tables_and_request": "parsed_from_hashed_byte_snapshot",
                                 "bound_content": "SHA-256_rechecked_before_publication",
                                 "detector_pixels": "content_bound_for_requested_preview" if feature_storage is not None else "not_read",
                                 "raw_addressing": "metadata_only_rechecked_before_publication"}
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    if state["status"] != "failed":
        try:
            if not table:
                table = [{**frame, "selected": frame.get("selected", False), "frame_flags": ["incomplete_run"], "valid_analysis": False} for frame in frames]
            _delivery(output, state, frames, table, sources, features, files, verify)
        except InputError as exc:
            state["status"] = "needs_input"
            state["issues"].append(exc.as_dict())
            state["n_valid_analysis"] = 0
            for row in table:
                row["valid_analysis"] = False
                row["frame_flags"].append(exc.code)
            try:
                _delivery(output, state, frames, table, sources, features, files)
            except Exception as delivery_exc:
                state["status"] = "failed"
                state["issues"].append({"code": "delivery_failed", "message": str(delivery_exc)})
                atomic_text(output / "error.log", traceback.format_exc())
        except Exception as exc:
            state["status"] = "failed"
            state["issues"].append({"code": "delivery_failed", "message": str(exc)})
            atomic_text(output / "error.log", traceback.format_exc())
    state["current_stage"] = None
    if state["status"] == "failed":
        _not_ready(output, "failed", state["issues"])
        state["artifacts"] = {}
        state["ready_for_integration"] = False
    next_actions(state)
    write_json(output / "status.json", state)
    with (output / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(f"status: {state['status']}\n")
        for issue in state["issues"]:
            stream.write(f"{issue['code']}: {issue.get('message', '')}\n")
    return json_value(state)


def _read_status(output, *, inventory=True):
    output = Path(output).resolve()
    try:
        state = json.loads((output / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputError(f"Cannot read saved run: {exc}", "unreadable_status") from exc
    if not isinstance(state, dict) or state.get("schema") != SCHEMA:
        raise InputError("Unsupported run status schema", "unsupported_schema")
    if not isinstance(state.get("status"), str) or state["status"] not in {"running", "completed", "needs_input", "failed"}:
        raise InputError("Run status has an invalid status value", "invalid_status", field="status")
    if inventory and (not isinstance(state.get("artifacts"), dict) or not isinstance(state.get("issues"), list)
                      or any(not isinstance(issue, dict) for issue in state["issues"])):
        raise InputError("Run status has an invalid artifact or issue inventory", "invalid_status")
    return state


def inspect(output):
    """Read status and verify published artifacts, without opening experiment data."""
    output = Path(output).resolve()
    state = _read_status(output)
    files = FileDigests()
    required = {"frames.csv", "peaks.csv", "decisions.json", "frame_manifest.json", "handoff.json", "report.html"}
    if state.get("run_id"):
        required.add("frame_records.json")
    if state.get("request_snapshot"):
        required.add("request.snapshot.json")
    problems = sorted(required - set(state.get("artifacts", {})))
    for name, saved in state.get("artifacts", {}).items():
        if not isinstance(saved, dict) or not isinstance(saved.get("sha256"), str) or not re.fullmatch(r"[0-9a-fA-F]{64}", saved["sha256"]):
            problems.append(name)
            continue
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
        state["ready_for_integration"] = False
        state["issues"].append({"code": "artifact_changed", "message": "Published artifacts changed or disappeared", "files": problems})
    return state


def resume(output):
    # Generated output damage is repaired by rerunning delivery. Rehashing that
    # output before validating the original request adds no recovery guarantee.
    state = _read_status(output, inventory=False)
    if not isinstance(state.get("request_path"), str) or not state["request_path"].strip():
        raise InputError("Saved run has no valid request_path; run again with an explicit --request", "invalid_status", field="request_path")
    return run(state["request_path"], output)
