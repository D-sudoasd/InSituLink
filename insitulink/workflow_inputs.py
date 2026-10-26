"""Input snapshots, output separation and local clock-coordinate adapters."""

from __future__ import annotations

import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

from .errors import InputError
from .inputs import _records, canonical_source_file
from .storage import FileDigests


def input_path(value, base):
    if not isinstance(value, (str, Path)) or not str(value):
        raise InputError("An input path must be a nonempty string", "invalid_path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def table_snapshot(config, base, files):
    if config.get("path"):
        data, record = files.capture(input_path(config["path"], base))
        return data, [record]
    return None, []


def prepare_inputs(request_path, output):
    """Protect declared and indirect raw inputs before the first output write.

    Request and table bytes are retained so preflight, parsing and their hashes
    describe the same snapshot. HDF5 discovery reads object metadata, not pixels.
    """
    if output == request_path or request_path.is_relative_to(output):
        raise InputError("Use a result directory separate from the request directory", "unsafe_output")
    prepared = {"files": FileDigests(), "request": None, "error": None, "frame_bytes": None}
    try:
        data, record = prepared["files"].capture(request_path)
        prepared["request_sha256"] = record["sha256"]
        prepared["request"] = json.loads(data.decode("utf-8-sig"))
    except (InputError, ValueError) as exc:
        prepared["error"] = exc if isinstance(exc, InputError) else InputError(f"Cannot parse request JSON: {exc}", "invalid_request_json")
        return prepared
    request = prepared["request"]
    if not isinstance(request, dict):
        return prepared
    base = request_path.parent
    configs = [request.get("frames")] + (request.get("sources") if isinstance(request.get("sources"), list) else [])
    paths, selections = [], []
    frame_config = request.get("frames")
    for config in configs:
        if isinstance(config, dict) and config.get("path"):
            paths.append(input_path(config["path"], base))
    if isinstance(frame_config, dict):
        if frame_config.get("type", "table") == "hdf5" and frame_config.get("path"):
            source = str(input_path(frame_config["path"], base))
            # Include timing datasets too: external links may store metadata in
            # a different file from the detector dataset.
            for key, default in (("data_dataset", "/entry/data/data"),
                                 ("start_dataset", "/entry/instrument/detector/start_time"),
                                 ("exposure_dataset", "/entry/instrument/detector/count_time")):
                selections.append({"source_file": source, "dataset": frame_config.get(key, default)})
        else:
            try:
                prepared["frame_bytes"], _ = table_snapshot(frame_config, base, prepared["files"])
                records, _ = _records(frame_config, base, table_bytes=prepared["frame_bytes"])
                columns = frame_config.get("columns") or {}
                if not isinstance(columns, dict):
                    raise InputError("frames.columns must be an object")
                raw_base = input_path(frame_config["source_root"], base) if frame_config.get("source_root") else base
                for row in records:
                    source = canonical_source_file(row.get(columns.get("source_file", "source_file")), raw_base)
                    if source:
                        paths.append(Path(source))
                        selections.append({"source_file": source, "dataset": row.get(columns.get("dataset", "dataset"))})
            except InputError:
                # Normal parsing reports the precise problem. Explicit paths
                # collected above still receive the output-containment check.
                pass
    if selections:
        from .hdf5_sources import discover_dependencies

        storage = discover_dependencies(selections)
        paths.extend(Path(path) for path in storage["paths"])
    config = request.get("features")
    if isinstance(config, dict):
        paths.extend(input_path(config[key], base) for key in ("geometry", "mask") if config.get(key))
        if isinstance(config.get("files"), list):
            paths.extend(input_path(item["path"], base) for item in config["files"] if isinstance(item, dict) and item.get("path"))
    metadata = request.get("metadata")
    if isinstance(metadata, dict) and metadata.get("sync_markers_file"):
        paths.append(input_path(metadata["sync_markers_file"], base))
    conflicts = sorted({str(path) for path in paths if path.is_relative_to(output)})
    if conflicts:
        raise InputError("The output directory contains experiment inputs; choose a separate directory", "unsafe_output", paths=conflicts)
    return prepared


def verify_input_stability(files, output):
    """Rehash bound content; restored Windows timestamps are not proof of stability."""
    verifier = FileDigests()
    for (name, _), expected in list(files.values.items()):
        path = Path(name)
        if path.is_relative_to(output):
            continue
        try:
            current = verifier.file(path)["sha256"]
        except InputError as exc:
            raise InputError(f"Input disappeared or became unreadable during this run: {path}", "input_changed") from exc
        if current != expected:
            raise InputError(f"Input content changed during this run: {path}; resume after acquisition is stable", "input_changed", path=str(path))


def local_frames(frames):
    return [{**row, **{f"t_{name}_s": row.get(f"t_{name}_rel_s", row.get(f"t_{name}_s"))
                      for name in ("start", "end", "mid")}} for row in frames]


def frame_origin(frames):
    return next((row["time_origin_s"] for row in frames if row.get("time_origin_s") is not None), "0")


def local_source(source, frames):
    from .time_coordinates import clock_offset_local_s, time_relative_s

    result = {**source, "t_s": np.asarray(source.get("t_rel_s", source["t_s"]), dtype=float)}
    clock = dict(source.get("clock") or {})
    origin, external_origin = frame_origin(frames), source.get("time_origin_s", "0")
    method = clock.get("method")
    if method in {"offset", "shared"}:
        value = clock.get("offset_s") if method == "offset" else 0
        if value is not None:
            clock["offset_s"] = clock_offset_local_s(value, origin, external_origin)
        if method == "shared":
            clock["method"] = "offset"
    elif method == "anchors" and isinstance(clock.get("pairs"), list):
        clock["pairs"] = [{**pair,
                           "frame_time_s": time_relative_s(pair.get("frame_time_s"), origin),
                           "source_time_s": time_relative_s(pair.get("source_time_s"), external_origin)}
                          if isinstance(pair, dict) else pair for pair in clock["pairs"]]
    result["clock"] = clock
    return result


def public_decision(decision, source, frames):
    from .time_coordinates import clock_offset_public_s

    origin, external_origin = frame_origin(frames), source.get("time_origin_s", "0")
    result = dict(decision)
    for key in ("dt_s", "candidate_dt_s"):
        value = result.get(key)
        result["local_" + key] = value
        if value is not None:
            result[key] = clock_offset_public_s(value, origin, external_origin)
            with localcontext() as context:
                context.prec = 50
                result[key + "_decimal"] = str(Decimal(str(value)) - Decimal(str(origin)) + Decimal(str(external_origin)))
    if source.get("clock", {}).get("method") == "shared":
        result["method"] = "shared"
    evidence = dict(result.get("evidence") or {})
    evidence["time_coordinates"] = {"frame": "frame_local_seconds", "source": "source_local_seconds",
                                    "frame_origin_s": origin, "source_origin_s": external_origin,
                                    "published_offset_convention": "source_time = frame_time + dt_s"}
    if "anchor_offsets_s" in evidence:
        evidence["anchor_offsets_local_s"] = evidence["anchor_offsets_s"]
        evidence["anchor_offsets_s"] = [clock_offset_public_s(value, origin, external_origin)
                                        for value in evidence["anchor_offsets_s"]]
    for event in evidence.get("signal_events", []):
        if event.get("candidate_dt_s") is not None:
            event["candidate_local_dt_s"] = event["candidate_dt_s"]
            event["candidate_dt_s"] = clock_offset_public_s(event["candidate_dt_s"], origin, external_origin)
    result["evidence"] = evidence
    return result


def resolve_source_clock(frames, source, features=None, experiment="generic", frame_metadata=None):
    """One shared clock-validation and coordinate path for prepare and execution."""
    from .numerics import resolve_clock

    if (source.get("clock", {}).get("method") == "shared" and frame_metadata is not None
            and source.get("metadata", {}).get("time_basis") != frame_metadata.get("time_basis")):
        raise InputError("Shared clocks need compatible time bases and origins; one record is absolute and the other relative", "incompatible_time_basis")
    decision = resolve_clock(local_frames(frames), local_source(source, frames), features, experiment)
    return public_decision(decision, source, frames)
