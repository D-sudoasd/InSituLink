"""Prepare editable requests from observed metadata, without confirming clocks."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
import json
import os
from pathlib import Path
import re
from uuid import uuid4

from .errors import InputError
from .inputs import _records, _time_settings, read_frames, read_source, referenced_paths
from .storage import atomic_text, json_value


SCHEMA = "insitulink.prepare.v1"
PROFILE_SCHEMA = "insitulink.import.profile.v1"
_HDF = {".h5", ".hdf5", ".nxs", ".nx5"}


def _load_profile(path):
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8-sig") as stream:
        profile = json.load(stream)
    if not isinstance(profile, dict) or profile.get("schema_version", 1) != 1:
        raise InputError("Import profile must be a version-1 JSON object", "invalid_import_profile")
    if profile.get("schema", PROFILE_SCHEMA) != PROFILE_SCHEMA:
        raise InputError("Unsupported import profile schema", "invalid_import_profile")
    for field in ("frames", "sources", "metadata"):
        if not isinstance(profile.get(field, {}), dict):
            raise InputError(f"Profile {field} must be an object", "invalid_import_profile", field=field)
    if any(not isinstance(value, dict) for value in profile.get("sources", {}).values()):
        raise InputError("Each profile source must be an object", "invalid_import_profile")
    for field in ("columns", "assets"):
        if field in profile.get("frames", {}) and not isinstance(profile["frames"][field], dict):
            raise InputError(f"Profile frames.{field} must be an object", "invalid_import_profile")
    for key, source in profile.get("sources", {}).items():
        if "clock" in source and not isinstance(source["clock"], dict):
            raise InputError(f"Profile sources.{key}.clock must be an object", "invalid_import_profile")
    return profile


def _safe_preview(value):
    """Keep observed decimal precision and normalize reader metadata for JSON."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _safe_preview(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_preview(item) for item in value]
    converted = json_value(value)
    # json_value also handles NumPy containers, whose contents can be Decimal.
    return _safe_preview(converted) if isinstance(converted, (dict, list)) else converted


def _table_preview(config):
    rows, origin = _records(config, Path.cwd())
    columns = list(dict.fromkeys(str(key) for row in rows for key in row))
    return {"path": origin, "format": Path(origin).suffix.lower(), "row_count": len(rows),
            "columns": columns, "sample_rows": _safe_preview(rows[:3])}


def _hdf_preview(config):
    try:
        import h5py
    except ImportError as exc:
        raise InputError("Reading HDF5 requires h5py", "missing_dependency") from exc
    paths = {"data_dataset": config.get("data_dataset", "/entry/data/data"),
             "start_dataset": config.get("start_dataset", "/entry/instrument/detector/start_time"),
             "exposure_dataset": config.get("exposure_dataset", "/entry/instrument/detector/count_time")}
    result = {"path": config["path"], "format": "hdf5", "datasets": {}, "available_datasets": []}
    with h5py.File(config["path"], "r") as handle:
        def observe(name, item):
            if isinstance(item, h5py.Dataset) and len(result["available_datasets"]) < 100:
                result["available_datasets"].append({"path": "/" + name, "shape": list(item.shape)})
        handle.visititems(observe)
        for key, name in paths.items():
            item = handle.get(name)
            if isinstance(item, h5py.Dataset):
                attrs = {key: item.attrs[key] for key in ("units", "start", "start_time", "timezone", "scaling_factor")
                         if key in item.attrs}
                result["datasets"][key] = {"path": name, "shape": list(item.shape), "attributes": json_value(attrs)}
                if key == "start_dataset":
                    settings = _time_settings(config, attrs)
                    # A reader default is not an observed unit declaration.
                    if "units" in attrs or (config.get("time") or {}).get("unit"):
                        config["time"] = settings
                    elif any(key in attrs for key in ("start", "start_time", "timezone", "scaling_factor")):
                        config["time"] = {key: value for key, value in settings.items() if key != "unit"}
                if key == "exposure_dataset" and not config.get("exposure_unit") and "units" in attrs:
                    config["exposure_unit"] = json_value(attrs["units"])
            else:
                result["datasets"][key] = {"path": name, "missing": True}
    return result


def _action(actions, code, field, source, message, **details):
    actions.append({"code": code, "field": field, "source": str(source), "message": message, **details})


def _time_config(config, unit, start):
    settings = config.get("time") or {}
    if not isinstance(settings, dict):
        raise InputError("Profile time settings must be objects", "invalid_import_profile")
    config["time"] = {key: value for key, value in settings.items() if value is not None}
    if unit is not None:
        config["time"]["unit"] = unit
    if start is not None:
        config["time"]["start"] = start


def _protect(output, paths):
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        if resolved == output or output in resolved.parents:
            raise InputError("Import output must not contain or replace an input file", "output_contains_input",
                             output=str(output), input=str(resolved))


def _channels(columns, time_column):
    result, used = [], set()
    for column in columns:
        if column == time_column:
            continue
        name = re.sub(r"[^\w-]", "_", column).strip("_") or "channel"
        base, suffix = name, 2
        while name in used:
            name, suffix = f"{base}_{suffix}", suffix + 1
        used.add(name)
        result.append({"name": name, "column": column, "unit": None, "kind": "continuous", "required": True})
    return result


def prepare(frames_path, source_paths, output, *, profile_path=None, experiment=None,
            frame_time_unit=None, frame_time_start=None, source_time_units=None, source_time_starts=None):
    """Write a new import package; all supplied physical settings remain explicit.

    Profiles use request-compatible ``frames`` settings and a ``sources`` object
    keyed by device ID. A profile may supply this experiment's clock evidence;
    absent evidence is never replaced by a zero offset or shared-clock claim.
    """
    requested_output = Path(output).expanduser()
    if requested_output.exists() or requested_output.is_symlink():
        raise InputError("prepare requires a new output directory; existing directories are not overwritten",
                         "output_exists", output=str(requested_output))
    output = requested_output.resolve()
    frames_path = Path(frames_path).expanduser().resolve()
    if any(not isinstance(key, str) or not re.fullmatch(r"[\w-]+", key) for key in source_paths):
        raise InputError("Source identifiers must contain only letters, digits, '_' or '-'", "invalid_identifier")
    sources = {key: Path(value).expanduser().resolve() for key, value in source_paths.items()}
    if not sources:
        raise InputError("At least one --source ID=PATH is required", "missing_sources")
    inputs = [frames_path, *sources.values()]
    if profile_path is not None:
        profile_path = Path(profile_path).expanduser().resolve()
        inputs.append(profile_path)
    _protect(output, inputs)
    for path in inputs:
        if not path.is_file():
            raise InputError("Input file does not exist", "missing_input", path=str(path))
    profile = _load_profile(profile_path)
    unit_overrides, start_overrides = source_time_units or {}, source_time_starts or {}
    unknown = (set(unit_overrides) | set(start_overrides)) - set(sources)
    if unknown:
        raise InputError("Time overrides reference unknown source IDs", "invalid_source_override", source_ids=sorted(unknown))
    actions, previews = [], {"frames": {}, "sources": {}}
    metadata = deepcopy(profile.get("metadata", {}))
    generated = []
    for name in ("experiment_id", "acquisition_id", "detector_id"):
        if not metadata.get(name):
            metadata[name] = str(uuid4())
            generated.append(name)
        elif not isinstance(metadata[name], str) or not metadata[name].strip():
            raise InputError(f"metadata.{name} must be a nonempty string", "invalid_import_profile")
    earlier_registration = metadata.get("import_registration") or {}
    earlier_generated = earlier_registration.get("generated_fields", []) if isinstance(earlier_registration, dict) else []
    metadata["import_registration"] = {"generated_fields": list(dict.fromkeys([*earlier_generated, *generated])),
                                        "newly_generated_fields": generated,
                                        "meaning": "Persistent registration IDs, not observations of experimental conditions"}
    request = {"schema_version": 1, "experiment": experiment or profile.get("experiment", "generic"),
               "metadata": metadata, "frames": deepcopy(profile.get("frames", {})), "sources": []}
    if request["experiment"] not in ("generic", "tensile", "thermal"):
        raise InputError("experiment must be generic, tensile or thermal", "invalid_import_profile")
    frame_config = request["frames"]
    frame_config.pop("data", None)
    frame_config["path"] = str(frames_path)
    frame_config.setdefault("type", "hdf5" if frames_path.suffix.lower() in _HDF else "table")
    if frame_config["type"] == "table":
        raw_root = Path(frame_config.get("source_root") or frames_path.parent)
        profile_base = profile_path.parent if profile_path else frames_path.parent
        frame_config["source_root"] = str((raw_root if raw_root.is_absolute() else profile_base / raw_root).resolve())
    _time_config(frame_config, frame_time_unit, frame_time_start)
    try:
        previews["frames"] = _hdf_preview(frame_config) if frame_config["type"] == "hdf5" else _table_preview(frame_config)
        if frame_config["type"] == "table":
            columns = frame_config.setdefault("columns", {})
            available = previews["frames"]["columns"]
            start_column = columns.get("start", "t_start_s")
            if start_column not in available and not (frame_config.get("constant_period") or {}).get("confirmed"):
                _action(actions, "missing_frame_start_column", "frames.columns.start", frames_path,
                        "Select the actual exposure-start column; reconstruction requires confirmed constant_period.", available_columns=available)
            if not frame_config["time"].get("unit") and start_column == "t_start_s" and start_column in available:
                frame_config["time"]["unit"] = "s"
                previews["frames"]["time_unit_source"] = "explicit t_start_s column name"
            exposure_column = columns.get("exposure", "exposure_s")
            if exposure_column not in available and not (frame_config.get("constant_period") or {}).get("confirmed"):
                _action(actions, "missing_exposure_column", "frames.columns.exposure", frames_path,
                        "Select the measured exposure-duration column or supply a confirmed constant period.", available_columns=available)
    except (InputError, OSError, ValueError, KeyError) as exc:
        _action(actions, "frame_preview_unavailable", "frames", frames_path, str(exc))
    if not frame_config["time"].get("unit"):
        frame_config["time"]["unit"] = None
        _action(actions, "missing_time_unit", "frames.time.unit", frames_path,
                "Declare the detector timestamp unit from acquisition metadata or the instrument configuration.")

    loaded_sources = []
    for index, (source_id, path) in enumerate(sources.items()):
        config = deepcopy(profile.get("sources", {}).get(source_id, {}))
        config.pop("data", None)
        config.update(id=source_id, path=str(path))
        _time_config(config, unit_overrides.get(source_id), start_overrides.get(source_id))
        prefix = f"sources[{index}]"
        before = len(actions)
        try:
            preview = _table_preview(config)
            previews["sources"][source_id] = preview
            available = preview["columns"]
            config.setdefault("time_column", "time" if "time" in available else None)
            if config["time_column"] not in available:
                _action(actions, "missing_time_column", prefix + ".time_column", path,
                        "Select this device's recorded timestamp column.", available_columns=available)
            config.setdefault("channels", _channels(available, config["time_column"]))
        except (InputError, OSError, ValueError, KeyError) as exc:
            config.setdefault("channels", [])
            _action(actions, "source_preview_unavailable", prefix, path, str(exc))
        if not config["time"].get("unit"):
            config["time"]["unit"] = None
            _action(actions, "missing_time_unit", prefix + ".time.unit", path,
                    "Declare this device's timestamp unit; no unit is inferred from numerical values.")
        if not isinstance(config.get("channels"), list) or not config["channels"]:
            _action(actions, "missing_channels", prefix + ".channels", path, "Declare at least one parameter column and its unit.")
        else:
            for channel_index, channel in enumerate(config["channels"]):
                if not isinstance(channel, dict):
                    _action(actions, "invalid_channel", f"{prefix}.channels[{channel_index}]", path, "A channel definition must be an object.")
                elif not isinstance(channel.get("unit"), str) or not channel["unit"].strip():
                    channel["unit"] = None
                    _action(actions, "missing_channel_unit", f"{prefix}.channels[{channel_index}].unit", path,
                            f"Declare the unit of column {channel.get('column', channel.get('name'))!r}; use '1' only for a dimensionless quantity.")
        config.setdefault("clock", {"method": "offset", "offset_s": None, "confirmed": False, "evidence": ""})
        request["sources"].append(config)
        if len(actions) == before:
            try:
                loaded_sources.append((index, read_source(config, output)))
            except InputError as exc:
                _action(actions, exc.code, prefix, path, str(exc), details=exc.details)

    frames, frame_metadata, manifest = [], {}, None
    if frame_config["time"].get("unit"):
        try:
            frames, frame_metadata = read_frames(frame_config, output)
            inputs.extend(referenced_paths(frame_config, output))
            from .hdf5_sources import discover_dependencies
            dependencies = discover_dependencies(frames)
            inputs.extend(dependencies["paths"])
            previews["frames"]["metadata"] = frame_metadata
            previews["frames"]["storage_dependencies"] = dependencies
            if not dependencies["complete"]:
                _action(actions, "frame_dependencies_unresolved", "frames", frames_path,
                        "Resolve the listed detector storage dependencies before using the frame manifest.", details=dependencies["issues"])
            if frame_metadata.get("frame_count_unresolved"):
                _action(actions, "frame_count_unresolved", "frames.frame_count", frames_path,
                        frame_metadata.get("frame_count_unresolved_reason", "Confirm the physical frame count."))
            if any(row.get("t_start_s") is None or row.get("exposure_s") is None for row in frames):
                _action(actions, "incomplete_frame_times", "frames", frames_path,
                        "Some frames lack measured time or exposure; supply the missing records or confirmed reconstruction settings.")
            if frame_metadata.get("nonmonotonic_rows") or frame_metadata.get("duplicate_time_rows"):
                _action(actions, "nonmonotonic_frame_clock", "frames", frames_path,
                        "Frame times repeat or decrease; acquisition order has been preserved.")
        except InputError as exc:
            _action(actions, exc.code, "frames", frames_path, str(exc), details=exc.details)
    else:
        # Protect raw paths even when units are not yet sufficient for decoding.
        try:
            inputs.extend(referenced_paths(frame_config, output))
        except InputError:
            pass
    if frame_config["type"] == "hdf5" and not frames:
        from .hdf5_sources import discover_dependencies
        dependencies = discover_dependencies([{"source_file": str(frames_path),
                                                "dataset": frame_config.get("data_dataset", "/entry/data/data")}])
        inputs.extend(dependencies["paths"])
        previews["frames"]["storage_dependencies"] = dependencies
    _protect(output, inputs)

    if frames:
        from .identity import identify_frames
        assets = frame_config.setdefault("assets", {})
        if not isinstance(assets, dict):
            raise InputError("frames.assets must map source paths to persistent asset IDs", "invalid_import_profile")
        profile_base = profile_path.parent if profile_path else output
        assets = {str((Path(key) if Path(key).is_absolute() else profile_base / key).resolve()): value
                  for key, value in assets.items()}
        known_assets = {os.path.normcase(key) for key in assets}
        def register_asset(path):
            key = os.path.normcase(path)
            if key not in known_assets:
                assets[path] = str(uuid4())
                known_assets.add(key)
        for row in frames:
            if row.get("source_file"):
                register_asset(row["source_file"])
        frame_config["assets"] = assets
        context = {**metadata, "assets": assets, "base_dir": output, "request_id": output / "experiment.json"}
        try:
            identity = identify_frames(frames, context=context)
            asset_count = len(assets)
            for row in frames:
                physical = row.get("physical_source") or {}
                if physical.get("source_file"):
                    register_asset(physical["source_file"])
            if len(assets) != asset_count:
                identity = identify_frames(frames, context=context)
            _protect(output, identity.get("storage_paths", []))
            for issue in identity.get("issues", []):
                _action(actions, issue["code"], "frames", frames_path, issue["message"],
                        details={key: value for key, value in issue.items() if key not in ("code", "message")})
            manifest = {"schema": "insitulink.frames.v1", "purpose": "import_preview", "alignment_complete": False,
                        "metadata": frame_metadata, "identity": identity, "frames": frames}
        except InputError as exc:
            if exc.code == "output_contains_input":
                raise
            _action(actions, exc.code, "frames", frames_path, str(exc), details=exc.details)

    loaded = dict(loaded_sources)
    for index, config in enumerate(request["sources"]):
        decision = None
        if index in loaded:
            from .workflow_inputs import resolve_source_clock
            try:
                decision = resolve_source_clock(frames, loaded[index], experiment=request["experiment"], frame_metadata=frame_metadata)
            except InputError as exc:
                _action(actions, exc.code, f"sources[{index}].clock", config["path"], str(exc), details=exc.details)
                previews["sources"].setdefault(config["id"], {})["clock_decision"] = {
                    "status": "needs_input", "reason": exc.code, "evidence": exc.as_dict()}
                continue
        if decision is None and config.get("clock") and config["clock"] != {"method": "offset", "offset_s": None, "confirmed": False, "evidence": ""}:
            _action(actions, "clock_not_evaluated", f"sources[{index}].clock", config["path"],
                    "Complete this device's missing input fields before its supplied clock settings can be evaluated.")
        elif decision is None or decision["status"] != "resolved":
            _action(actions, "clock_evidence_required", f"sources[{index}].clock", config["path"],
                    "Supply independent clock evidence: confirmed offset/shared origin, or corresponding event anchors. A waveform match alone is not confirmation.",
                    decision=decision)
        previews["sources"].setdefault(config["id"], {})["clock_decision"] = decision

    exported_profile = {"schema": PROFILE_SCHEMA, "schema_version": 1, "experiment": request["experiment"],
                        "metadata": metadata, "frames": deepcopy(frame_config),
                        "sources": {source["id"]: deepcopy(source) for source in request["sources"]}}
    for config in [exported_profile["frames"], *exported_profile["sources"].values()]:
        config.pop("path", None)
        config.pop("data", None)
    previews.update(schema="insitulink.import.preview.v1", generated_identity_fields=generated,
                    defaults={"experiment": "generic unless declared", "channel_kind": "continuous unless declared",
                              "required": "true unless declared"},
                    note="Preview rows are original observations. Interpolated parameters are produced only by run.")
    artifacts = {"request": output / "experiment.json", "profile": output / "profile.json",
                 "preview": output / "import_preview.json", "actions": output / "actions.json"}
    payloads = {"request": request, "profile": exported_profile, "preview": previews,
                "actions": {"schema": SCHEMA, "actions": actions}}
    if manifest is not None:
        artifacts["frame_manifest"] = output / "frame_manifest.json"
        payloads["frame_manifest"] = manifest
    result = _safe_preview({"schema": SCHEMA, "status": "needs_input" if actions else "completed", "output": str(output),
                            "artifacts": artifacts, "issues": actions, "n_frames": len(frames), "n_sources": len(sources),
                            "alignment_complete": False,
                            "next_action": "Complete the listed fields in experiment.json, then run the request." if actions else "Run the prepared request.",
                            "run_command": ["python", "-m", "insitulink", "run", "--request", str(artifacts["request"]),
                                            "--output", str(output / "results")]})
    # Validate every artifact and the CLI response before claiming a new output
    # directory. Serialization errors must not leave a partially written draft.
    encoded = {}
    for name, value in {**payloads, "response": result}.items():
        try:
            encoded[name] = json.dumps(_safe_preview(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        except (TypeError, ValueError) as exc:
            raise InputError(f"Cannot serialize import {name}: {exc}", "invalid_import_output", artifact=name) from exc
    output.mkdir(parents=True, exist_ok=False)
    for name, path in artifacts.items():
        atomic_text(path, encoded[name])
    return result
