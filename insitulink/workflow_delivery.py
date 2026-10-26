"""Publish one alignment run with frame identity and bounded parameter provenance."""

from __future__ import annotations

import csv
from io import StringIO
import json
from pathlib import Path

from . import __version__
from .storage import atomic_text, json_value, write_json


FRAME_FIELDS = (
    "frame_uid", "legacy_frame_uid", "sequence_index", "source_file", "dataset", "frame_index",
    "frame_number", "original_id", "identity_scope", "identity_basis", "asset_id", "declared_source",
    "source_file_declared", "dataset_declared", "identity_status", "identity_issues", "physical_source",
    "t_start_s", "t_end_s", "t_mid_s", "time_origin_s", "t_start_rel_s", "t_end_rel_s", "t_mid_rel_s",
    "exposure_s", "time_provenance", "source_row_number", "clock_source", "raw_time_missing", "time_missing",
    "selected", "valid_analysis", "frame_flags", "clock_mappings",
)
SAMPLE_FIELDS = ("mid", "mean", "min", "max", "coverage", "valid", "flags", "unit", "phase", "phase_exposure", "provenance")


def csv_text(rows):
    if not rows:
        return ""
    names = list(dict.fromkeys(key for row in rows for key in row))
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, names)
    writer.writeheader()
    for row in json_value(rows):
        writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    return "\ufeff" + stream.getvalue()


def next_actions(state):
    command = f'python -m insitulink resume --output "{state["output"]}"'
    actions = []
    for issue in state.get("issues", []):
        code = issue.get("code", "invalid_input")
        if code == "no_required_channels":
            continue
        source = issue.get("source_id")
        field = issue.get("field") or (f"sources[id={source}].clock" if code == "clock_unresolved" else
                                       f"sources[id={source}]" if source else "frames" if "frame" in code or "identity" in code else "request")
        instruction = ("Wait until the input files are stable, then resume." if code == "input_changed" else
                       "Supply independent clock evidence and update offset/shared/anchors; do not confirm an event candidate automatically."
                       if code == "clock_unresolved" else "Correct the indicated input using the experiment records, then resume.")
        actions.append({"code": code, "field": field, "reason": issue.get("message", code),
                        "instruction": instruction, "command": command})
    state["actions"] = actions
    if actions:
        first = actions[0]
        state["next_action"] = f'{first["field"]}: {first["reason"]} {first["instruction"]} {command}'
    elif state.get("ready_for_integration"):
        state["next_action"] = "Review report.html and deliver handoff.json, frames.csv, frame_manifest.json and frame_records.json; the consumer must preflight raw frames and calibration."
    elif state.get("status") == "completed":
        state["next_action"] = "Alignment is complete. Review raw_source_issues and supply raw frame references before requesting integration."
    else:
        state["next_action"] = f"Review issues and correct the request, then run {command}"


def _sources(sources, state):
    hashes = {item["path"]: item["sha256"] for item in state.get("input_files", [])}
    result = {}
    for source in sources:
        metadata = source.get("metadata", {})
        result[source["id"]] = {"metadata": metadata, "time_origin_s": source.get("time_origin_s", "0"),
                                "time_provenance": metadata.get("time_provenance"),
                                "source_row_numbers": source.get("source_row_numbers"),
                                "input_file": source.get("input_file"),
                                "clock": state.get("decisions", {}).get(source["id"]),
                                "input_sha256": (source.get("input_file") or {}).get("sha256", hashes.get(metadata.get("source")))}
    return result


def _records(table, sources, state):
    result = []
    for row in table:
        record = {key: row.get(key) for key in FRAME_FIELDS}
        devices = {}
        for source in sources:
            decision = state.get("decisions", {}).get(source["id"], {})
            parameters = {}
            for name, channel in source["channels"].items():
                prefix = f"{source['id']}.{name}"
                parameters[name] = {key: row.get(f"{prefix}.{key}") for key in SAMPLE_FIELDS}
                parameters[name].update(kind=channel["kind"], required=channel["required"])
            devices[source["id"]] = {"source_ref": source["id"], "clock_status": decision.get("status"),
                                     "published_offset_s": decision.get("dt_s"), "local_offset_s": decision.get("local_dt_s"),
                                     "published_offset_s_decimal": decision.get("dt_s_decimal"),
                                     "source_origin_s": source.get("time_origin_s", "0"), "parameters": parameters}
        record["devices"] = devices
        result.append(record)
    return result


def prepare_delivery(state, table):
    raw_issues = []
    active = [row for row in table if row.get("valid_analysis")]
    for row in active:
        path = row.get("source_file")
        if not path or not Path(path).is_file():
            raw_issues.append({"frame_uid": row.get("frame_uid"), "source_file": path,
                               "reason": "raw_source_missing" if path else "raw_source_not_referenced"})
        elif row.get("identity_status") == "invalid":
            raw_issues.append({"frame_uid": row.get("frame_uid"), "source_file": path,
                               "reason": "raw_identity_invalid", "details": row.get("identity_issues", [])})
    state["raw_source_issues"] = raw_issues
    state["raw_identity_verified"] = bool(active) and all(row.get("identity_status") == "verified" for row in active)
    state["identity_verification"] = state.get("identity_validation", {})
    state["ready_for_integration"] = state["status"] == "completed" and bool(active) and not raw_issues
    next_actions(state)


def write_delivery(output, state, frames, table, sources, features, files, report_builder, verify=None):
    prepare_delivery(state, table)
    records = _records(table, sources, state)
    manifest = [{key: row.get(key) for key in FRAME_FIELDS} for row in table]
    for item, record in zip(manifest, records):
        item["device_mappings"] = {name: {key: value for key, value in device.items() if key != "parameters"}
                                   for name, device in record["devices"].items()}
        item["frame_record"] = {"artifact": "frame_records.json", "frame_uid": item["frame_uid"]}
    run = {key: state.get(key) for key in ("run_id", "version", "code_version", "request_sha256", "started_at", "finished_at")}
    report = report_builder(state, frames, table, sources, features)
    if verify is not None:
        verify()
    atomic_text(output / "frames.csv", csv_text(table))
    atomic_text(output / "peaks.csv", csv_text(features.get("rows", [])))
    write_json(output / "decisions.json", state.get("decisions", {}))
    write_json(output / "frame_manifest.json", {"schema": "insitulink.frames.v1", "index_base": 0, "display_number_base": 1,
                                                "run": run, "metadata": state.get("frame_metadata", {}), "frames": manifest})
    write_json(output / "frame_records.json", {"schema": "insitulink.frame-records.v1", "run": run,
                                               "request": state.get("request_snapshot"), "sources": _sources(sources, state),
                                               "input_validation": state.get("input_validation", {}),
                                               "frames": records})
    atomic_text(output / "report.html", report)
    channels = [{"id": f"{source['id']}.{name}", "unit": channel["unit"], "kind": channel["kind"], "required": channel["required"]}
                for source in sources for name, channel in source["channels"].items()]
    handoff = {"schema": "insitulink.handoff.v1", "producer": "InSituLink", "version": __version__, "run": run,
               "status": state["status"], "alignment_complete": state["status"] == "completed",
               "ready_for_integration": state["ready_for_integration"], "raw_identity_verified": state["raw_identity_verified"],
               "identity_verification": state["identity_verification"], "raw_source_issues": state["raw_source_issues"],
               "consumer": "DIDASFlow", "consumer_integration": "file_contract_only", "raw_frames_copied": False,
               "parameter_table": "frames.csv", "frame_manifest": "frame_manifest.json", "frame_records": "frame_records.json",
               "decisions": "decisions.json", "index_base": 0, "run_status": "status.json", "request_sha256": state.get("request_sha256"),
               "channels": channels, "selected_frame_uids": [row.get("frame_uid") for row in table if row.get("selected")],
               "analysis_frame_uids": [row.get("frame_uid") for row in table if row.get("valid_analysis")], "frames": manifest,
               "notes": ["Each frame remains independent; no frame combination is requested.",
                         "Raw addressing and calibration still require consumer preflight; metadata verification does not read detector pixels.",
                         "DIDASFlow does not automatically consume this schema yet.",
                         "Clock evidence, support intervals and coverage flags accompany the parameter table."]}
    for name in ("frames.csv", "peaks.csv", "decisions.json", "frame_manifest.json", "frame_records.json", "report.html"):
        state["artifacts"][name] = files.file(output / name)
    if state.get("request_snapshot"):
        state["artifacts"]["request.snapshot.json"] = files.file(output / "request.snapshot.json")
    write_json(output / "handoff.json", handoff)
    state["artifacts"]["handoff.json"] = files.file(output / "handoff.json")
