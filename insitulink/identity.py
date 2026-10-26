"""Frame identities and metadata-only verification of physical image addresses.

Declared acquisition and asset identifiers are evidence supplied by the caller
or persisted by an import. Paths are locators, not globally permanent identities.
"""
from __future__ import annotations

import os
from pathlib import Path

from .errors import InputError
from .inputs import canonical_source_file
from .storage import digest


_HDF_SUFFIXES = {".h5", ".hdf5", ".hdf", ".nxs", ".nx5"}


def _asset_ids(context):
    values = context.get("assets") or {}
    if not isinstance(values, dict):
        raise InputError("frames.assets must map source paths to persistent asset_id strings", "invalid_asset_ids")
    assets = {}
    for path, value in values.items():
        if not isinstance(path, str) or not path.strip() or not isinstance(value, str) or not value.strip():
            raise InputError("Each frames.assets entry requires a nonempty source path and asset_id", "invalid_asset_ids")
        key = os.path.normcase(canonical_source_file(path, context.get("base_dir", Path.cwd())))
        if key in assets and assets[key] != value:
            raise InputError("Equivalent source paths have conflicting asset_id declarations", "invalid_asset_ids")
        assets[key] = value
    return assets


def _physical_keys(source):
    if source is None:
        return []
    if source.get("candidates"):
        return list(dict.fromkeys(key for item in source["candidates"] for key in _physical_keys(item)))
    return [(os.path.normcase(source["source_file"]), source.get("object_address", source.get("dataset")), source["frame_index"])]


def _uid(row, context, assets, namespace):
    source = row["physical_source"]
    if source is not None and not source.get("candidates"):
        asset = assets.get(os.path.normcase(source["source_file"]))
        component = {"object_address": source["object_address"]} if "object_address" in source else {"dataset": source.get("dataset")}
        if asset and namespace:
            row["identity_scope"] = "declared_asset"
            row["asset_id"] = asset
            row["identity_basis"] = "hdf5_object_address" if "object_address" in source else "file_frame_index"
            return digest({"namespace": namespace, "asset_id": asset, **component, "frame_index": source["frame_index"]})[:24]
        row["identity_scope"] = "physical_path"
        row["identity_basis"] = "hdf5_object_address" if "object_address" in source else "file_frame_index"
        return digest({"source_file": os.path.normcase(source["source_file"]), **component, "frame_index": source["frame_index"]})[:24]
    if row.get("source_file"):
        # An unverified/missing physical source must not acquire a stronger identity.
        row["identity_scope"] = "declared_path"
        row["identity_basis"] = "unverified_locator"
        return row["legacy_frame_uid"]
    original = row.get("original_id")
    original = None if original is None or str(original).strip() == "" else str(original)
    row["identity_scope"] = ("declared_acquisition_record" if namespace else
                             "request_record" if context.get("request_id") else "metadata_table_record")
    row["identity_basis"] = "original_id" if original is not None else "record_index"
    record = {"original_id": original} if original is not None else {"record_index": row.get("frame_index", row["sequence_index"])}
    return digest({"namespace": namespace or {"request_id": context.get("request_id")}, **record})[:24]


def identify_frames(frames: list[dict], *, context: dict | None = None) -> dict:
    """Annotate rows in place and return blocking issues, warnings and counts.

    ``context`` accepts experiment_id/acquisition_id/detector_id, assets (the
    request's frames.assets mapping), base_dir and request_id. A declared asset
    identity requires all three namespace fields plus a physical-source asset.
    Without them the explicitly named scope remains path/request-local. HDF5
    object addresses distinguish links within an unchanged file; repacking a
    file may change those addresses and requires a verified identity migration.

    Every row receives frame_uid, its exact previous legacy_frame_uid,
    sequence_index, identity_scope, identity_basis, identity_status,
    identity_issues, declared_source and physical_source. The four statuses are
    verified (HDF metadata address), unverified (e.g. image header not inspected),
    metadata_only (no raw source), and invalid. Pixels are never read here.
    """
    context = context or {}
    assets = _asset_ids(context)
    names = ("experiment_id", "acquisition_id", "detector_id")
    namespace = {name: context[name] for name in names if isinstance(context.get(name), str) and context[name].strip()}
    if len(namespace) != len(names):
        namespace = {}
    hdf_rows, hdf_positions = [], []
    source_paths, source_exists = {}, {}
    for position, row in enumerate(frames):
        declared = {key: row.get(key) for key in ("source_file", "dataset", "frame_index")}
        row.update(legacy_frame_uid=digest(declared)[:24], sequence_index=position,
                   identity_issues=[], physical_source=None, identity_status="metadata_only")
        row.pop("asset_id", None)
        row["declared_source"] = {**declared, "source_file": row.get("source_file_declared", declared["source_file"]),
                                  "dataset": row.get("dataset_declared", declared["dataset"])}
        index = row.get("frame_index")
        if type(index) is not int or index < 0:
            row["identity_issues"].append({"code": "invalid_frame_index", "message": "frame_index must be a zero-based nonnegative integer", "blocking": True})
            continue
        declared_path = row.get("source_file")
        if declared_path not in source_paths:
            source_paths[declared_path] = canonical_source_file(declared_path, context.get("base_dir", Path.cwd()))
        source = source_paths[declared_path]
        if source is None:
            continue
        if source not in source_exists:
            source_exists[source] = Path(source).is_file()
        if not source_exists[source]:
            row["identity_issues"].append({"code": "raw_source_missing", "message": "The declared raw source file is unavailable", "source_file": source, "blocking": True})
        elif row.get("dataset"):
            hdf_rows.append({"source_file": source, "dataset": str(row["dataset"]), "frame_index": index})
            hdf_positions.append(position)
        elif Path(source).suffix.lower() in _HDF_SUFFIXES:
            row["identity_issues"].append({"code": "missing_data_dataset", "message": "An HDF5 raw source requires an explicit dataset address", "blocking": True})
        else:
            row.update(identity_status="unverified", physical_source={"source_file": source, "dataset": None,
                       "frame_index": index, "storage_paths": [source], "mapping": []})
            row["identity_issues"].append({"code": "image_address_unverified", "message": "File exists; image header and frame count have not been inspected", "blocking": False})
    storage_paths = set()
    if hdf_rows:
        from .hdf5_sources import resolve_frame_sources

        checked = resolve_frame_sources(hdf_rows)
        storage_paths.update(checked["paths"])
        for position, result in zip(hdf_positions, checked["frames"]):
            row = frames[position]
            row["physical_source"] = result["physical_source"]
            row["identity_issues"].extend(result["issues"])
            row["identity_status"] = "unverified" if result["issues"] else "verified"
    for row in frames:
        row["frame_uid"] = _uid(row, context, assets, namespace)
        if row["physical_source"]:
            storage_paths.update(row["physical_source"]["storage_paths"])

    # Mark every conflicting row, not just the later occurrence. A previous
    # valid occurrence must never remain eligible for a contradictory mapping.
    physical_groups, uid_groups = {}, {}
    for position, row in enumerate(frames):
        for key in _physical_keys(row["physical_source"]):
            physical_groups.setdefault(key, []).append(position)
        uid_groups.setdefault(row["frame_uid"], []).append(position)
    conflicts = set(tuple(group) for group in [*physical_groups.values(), *uid_groups.values()] if len(group) > 1)
    for positions in sorted(conflicts):
        issue = {"code": "duplicate_frame", "message": "Multiple records identify the same physical frame or declared record identity",
                 "conflicting_rows": list(positions), "blocking": True}
        for position in positions:
            frames[position]["identity_issues"].append(dict(issue))
    issues, warnings = [], []
    for row in frames:
        if any(issue["blocking"] for issue in row["identity_issues"]):
            row["identity_status"] = "invalid"
        for issue in row["identity_issues"]:
            entry = {"frame_uid": row["frame_uid"], "sequence_index": row["sequence_index"], **issue}
            (issues if issue["blocking"] else warnings).append(entry)
    counts = {f"n_{status}": sum(row["identity_status"] == status for row in frames)
              for status in ("verified", "unverified", "metadata_only", "invalid")}
    return {"schema": "insitulink.identity.v1", **counts, "issues": issues, "warnings": warnings,
            "physical_identity_complete": bool(frames) and counts["n_verified"] == len(frames),
            "storage_paths": sorted(storage_paths), "namespace": namespace or None,
            "content_verified": False,
            "notes": ["Metadata validation does not verify pixel content or calibration.",
                      "HDF5 object addresses can change after file repacking; declared asset IDs alone do not verify a repacking migration."]}
