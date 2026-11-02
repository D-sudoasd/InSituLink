"""Input snapshots, output separation and local clock-coordinate adapters."""

from __future__ import annotations

import json
import math
from decimal import Decimal, InvalidOperation, localcontext
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
    frame_config = request.get("frames")
    archived_frames = isinstance(frame_config, dict) and frame_config.get("archived") is True
    if archived_frames:
        # Do not open an archived frame path, including when that path is missing.
        prepared["archived_scan"] = True
    source_configs = request.get("sources") if isinstance(request.get("sources"), list) else []
    configs = source_configs if archived_frames else [frame_config, *source_configs]
    paths, selections = [], []
    for config in configs:
        if isinstance(config, dict) and config.get("path"):
            paths.append(input_path(config["path"], base))
    if isinstance(frame_config, dict) and not archived_frames:
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


_CONTINUATION_REASONS = frozenset({
    "onset_at_first_frame", "initially_changing", "baseline_missing",
    "onset_missing", "elastic_shape_weak", "event_missing",
})


def _finite_scalar(value):
    """Return a finite float, or None. Booleans and non-scalars do not count."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, np.ndarray):
        if value.shape != ():
            return None
        try:
            value = value.item()
        except ValueError:
            return None
        if isinstance(value, (bool, np.bool_)):
            return None
    if isinstance(value, (list, tuple, dict, set, bytes, bytearray)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _same_origin(left, right):
    if left == right:
        return True
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except (InvalidOperation, ValueError, TypeError):
        return False


def _event_channel_name(source, clock):
    for holder in (clock.get("event"), source.get("event")):
        if isinstance(holder, dict):
            name = holder.get("channel")
            if isinstance(name, str) and name:
                return name
    if clock.get("method") == "event":
        name = clock.get("channel")
        if isinstance(name, str) and name:
            return name
    return "force"


def _channel_record(source, name):
    channels = source.get("channels")
    if isinstance(channels, dict):
        return channels.get(name)
    if isinstance(channels, list):
        for item in channels:
            if isinstance(item, dict) and item.get("name") == name:
                return item
    return None


def _channel_values(channel):
    if isinstance(channel, dict):
        return channel.get("values")
    if isinstance(channel, (list, tuple, np.ndarray)):
        return channel
    return None


def _declared_preload(source, clock, parent, channel):
    """First explicit preload. Do not estimate one from the force trace."""
    holders = [parent, clock]
    for holder in (clock.get("event"), source.get("event")):
        if isinstance(holder, dict):
            holders.append(holder)
    if isinstance(channel, dict):
        holders.append(channel)
    holders.append(source)
    for holder in holders:
        if isinstance(holder, dict) and "preload" in holder:
            return holder["preload"]
    return None


def _join_force(source, values):
    """Force at the first local time >= 0. Samples before the record are not used."""
    if values is None:
        return None
    times = source["t_rel_s"] if "t_rel_s" in source else source.get("t_s")
    if times is None:
        return None
    try:
        t = np.asarray(times, dtype=float)
        y = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    if t.ndim != 1 or y.ndim != 1 or t.shape != y.shape or t.size == 0:
        return None
    eligible = np.flatnonzero(np.isfinite(t) & (t >= 0))
    if eligible.size == 0:
        return None
    index = int(eligible[int(np.argmin(t[eligible]))])
    sample = float(y[index])
    if not math.isfinite(sample):
        return None
    return sample


def apply_continuation(decision, source, frames):
    """Inherit a published parent wall span when this child record joins above preload.

    The offset is ``parent_local_dt_s + (child_wall_start_s - parent_wall_start_s)``.
    That wall-clock difference is the only gap term. Filenames and frame indices are not used.
    A rejected continuation is not written into ``candidate_dt_s``.
    """
    if not isinstance(decision, dict) or not isinstance(source, dict):
        return decision
    clock = source.get("clock")
    if not isinstance(clock, dict) or "continuation_of" not in clock:
        return decision
    continuation_of = clock.get("continuation_of")
    if not isinstance(continuation_of, str) or continuation_of == "":
        return decision
    parent = clock.get("parent_wall_span")
    if not isinstance(parent, dict) or parent.get("source_id") != continuation_of:
        return decision
    if decision.get("status") == "resolved":
        return decision
    reason = decision.get("reason")
    if not isinstance(reason, str):
        return decision
    bare = reason[len("source_"):] if reason.startswith("source_") else reason
    if bare not in _CONTINUATION_REASONS:
        return decision
    if parent.get("archived") is True:
        return decision
    if "uncertainty_s" not in parent:
        return decision
    parent_dt = _finite_scalar(parent.get("local_dt_s"))
    parent_uncertainty = _finite_scalar(parent.get("uncertainty_s"))
    parent_wall = _finite_scalar(parent.get("wall_start_s"))
    child_wall = _finite_scalar(clock.get("wall_start_s"))
    if None in (parent_dt, parent_uncertainty, parent_wall, child_wall):
        return decision
    channel_name = _event_channel_name(source, clock)
    channel = _channel_record(source, channel_name)
    preload = _finite_scalar(_declared_preload(source, clock, parent, channel))
    if preload is None:
        return decision
    join = _join_force(source, _channel_values(channel))
    if join is None or not join > preload:
        return decision
    existing = decision.get("evidence")
    if existing is not None and not isinstance(existing, dict):
        return decision
    # The wall-clock difference is the only gap; do not add a second gap term.
    local_dt = parent_dt + (child_wall - parent_wall)
    origin = frame_origin(frames) if isinstance(frames, (list, tuple)) else "0"
    external = source.get("time_origin_s", "0")
    if _same_origin(origin, external):
        dt_s = local_dt
        dt_decimal = str(local_dt)
    else:
        from .time_coordinates import clock_offset_public_s

        dt_s = clock_offset_public_s(local_dt, origin, external)
        with localcontext() as context:
            context.prec = 50
            dt_decimal = str(Decimal(str(local_dt)) - Decimal(str(origin)) + Decimal(str(external)))
    result = dict(decision)
    result.update(status="resolved", method="continuation", reason="continuation_of_published_wall_span",
                  dt_s=dt_s, local_dt_s=local_dt, dt_s_decimal=dt_decimal,
                  uncertainty_s=parent_uncertainty + 1.0)
    evidence = dict(existing or {})
    evidence.update(continuation_of=continuation_of, child_reason=reason, parent_dt_s=parent_dt,
                    parent_wall_start_s=parent_wall, child_wall_start_s=child_wall,
                    join_force=join, preload=preload, filename_used=False, frame_indices_merged=False)
    result["evidence"] = evidence
    return result


def resolve_source_clock(frames, source, features=None, experiment="generic", frame_metadata=None):
    """One shared clock-validation and coordinate path for prepare and execution."""
    from .numerics import resolve_clock

    if (source.get("clock", {}).get("method") == "shared" and frame_metadata is not None
            and source.get("metadata", {}).get("time_basis") != frame_metadata.get("time_basis")):
        raise InputError("Shared clocks need compatible time bases and origins; one record is absolute and the other relative", "incompatible_time_basis")
    decision = resolve_clock(local_frames(frames), local_source(source, frames), features, experiment)
    return apply_continuation(public_decision(decision, source, frames), source, frames)
