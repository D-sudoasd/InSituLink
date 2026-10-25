"""Deterministic synthetic clock examples, with separate validation truth.

These tables contain simulated metadata and signals, not detector images or
experimental measurements. All clocks obey source = detector + offset.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _thermal_temperature(t):
    # Two complete cycles revisit the same temperatures chronologically.
    if t < 12:
        return 20.0
    if t < 30:
        return 20.0 + 10.0 * (t - 12)
    if t < 42:
        return 200.0
    if t < 60:
        return 200.0 - 10.0 * (t - 42)
    if t < 68:
        return 20.0
    if t < 86:
        return 20.0 + 10.0 * (t - 68)
    if t < 96:
        return 200.0
    if t < 114:
        return 200.0 - 10.0 * (t - 96)
    return 20.0


def _force(t):
    if t < 12:
        return 0.0
    if t < 40:
        return 100.0 * (t - 12)
    if t < 53:
        return 2800.0
    if t < 81:
        return 2800.0 - 100.0 * (t - 53)
    return 0.0


def _times(start, stop, step):
    return [round(start + i * step, 8) for i in range(int((stop - start) / step) + 1)]


def _frames(count, response, seed):
    clock_rng = random.Random(seed)
    noise_rng = random.Random(seed + 1)
    rows, start = [], 0.0
    for index in range(count):
        exposure = round(clock_rng.uniform(0.24, 0.72), 6)
        # Integrate the independently sampled physical response over exposure.
        strain = sum(response(start + exposure * (j + 0.5) / 20) for j in range(20)) / 20
        rows.append({"frame_index": index, "frame_id": f"synthetic-{1003 + index * 7:05d}",
                     "source_file": "", "t_start_s": round(start, 6),
                     "exposure_s": exposure, "eps_hkl": round(strain + noise_rng.uniform(-2e-7, 2e-7), 10)})
        start += clock_rng.uniform(0.86, 1.14)
    return rows


def _channel(name, unit, *, required=True, kind="continuous", max_gap_s=1.0):
    return {"name": name, "column": name, "unit": unit, "kind": kind,
            "required": required, "max_gap_s": max_gap_s,
            "phase_tolerance": {"force": 5.0, "stress": 0.5, "temperature": 0.1}.get(name, 0.0)}


def _source(source_id, filename, channels, clock):
    return {"id": source_id, "path": filename, "time_column": "time", "time": {"unit": "s"},
            "channels": channels, "clock": clock}


def _event(channel):
    return {"method": "event", "channel": channel, "baseline_samples": 8,
            "min_run": 3, "k_sigma": 4, "direction": "rise"}


def _anchors(source_id, offset, marker_rows):
    pairs = []
    # Simulated observations of the same TTL edges on independent clock inputs.
    # They are explicit synchronization evidence, separate from response curves.
    for number, detector_time in enumerate((3.2, 72.7), 1):
        pair = {"marker_id": f"ttl-{number}", "frame_time_s": detector_time,
                "source_time_s": round(detector_time + offset, 8)}
        pairs.append(pair)
        marker_rows.append({"source_id": source_id, **pair})
    return {"method": "anchors", "pairs": pairs, "tolerance_s": 0.01,
            "evidence": "Common TTL edge timestamps recorded in sync_markers.csv"}


def create_demo(case: str, directory: Path) -> Path:
    """Write a ``tensile``, ``thermal`` or ``multi`` example and return its request.

    Existing generated filenames are rejected before writing any files. Other
    files in the destination are preserved. ``truth.json`` is solely a test and
    report reference; it is never included in the experiment request.
    """
    if case not in {"tensile", "thermal", "multi"}:
        raise ValueError("Demo case must be tensile, thermal or multi")
    directory = Path(directory)
    files = {"frames.csv", "experiment.json", "truth.json"}
    files.update({"tensile": {"mechanics.csv"}, "thermal": {"furnace.csv"},
                  "multi": {"mechanics.csv", "furnace.csv", "pressure.csv", "sync_markers.csv"}}[case])
    for filename in sorted(files):
        if (directory / filename).exists():
            raise FileExistsError(f"Refusing to replace existing demo file: {directory / filename}")
    directory.mkdir(parents=True, exist_ok=True)
    request = {"schema_version": 1, "experiment": "generic" if case == "multi" else case,
               "frames": {"type": "table", "path": "frames.csv", "time": {"unit": "s"},
                          "columns": {"start": "t_start_s", "exposure": "exposure_s",
                                      "frame_index": "frame_index", "source_file": "source_file", "id": "frame_id"}},
               "sources": [], "metadata": {"synthetic": True, "case": case,
                                            "detector_images": False}}
    truth = {"schema_version": 1, "synthetic": True, "case": case,
             "offset_convention": "source_time_s = detector_time_s + dt_s"}
    tables = {}
    if case == "tensile":
        offset = 7.35
        frames = _frames(100, lambda t: _force(t) / 1_000_000, 101)
        rng = random.Random(201)
        # External samples use an independent cadence and clock coordinate.
        tables["mechanics.csv"] = [
            {"time": t, "force": round(_force(t - offset) + rng.uniform(-0.01, 0.01), 6)}
            for t in _times(4.15, 111.95, 0.2)]
        request["sources"] = [_source("mechanics", "mechanics.csv", [_channel("force", "N")], _event("force"))]
        request["metadata"]["description"] = "Synthetic load and elastic lattice-strain onset with one external channel."
        truth.update(expected_status="resolved", offsets_s={"mechanics": offset},
                     phase_boundaries_detector_s={"force": [12.0, 40.0, 53.0, 81.0]})
    elif case == "thermal":
        offset, response_delay = -4.6, 3.2
        frames = _frames(123, lambda t: (_thermal_temperature(t - response_delay) - 20) * 1.2e-5, 301)
        tables["furnace.csv"] = [{"time": t, "temperature": round(_thermal_temperature(t - offset), 7)}
                                  for t in _times(-7.7, 122.3, 0.25)]
        request["sources"] = [_source("furnace", "furnace.csv", [_channel("temperature", "degC")], _event("temperature"))]
        request["metadata"].update(expected_status="needs_input", description=
            "Synthetic repeated heating, hold and cooling; lattice response has physical delay. "
            "A thermal response candidate cannot establish clock synchronization.")
        truth.update(expected_status="needs_input", offsets_s={"furnace": offset},
                     physical_response_delay_s=response_delay,
                     phase_boundaries_detector_s={"temperature": [12.0, 30.0, 42.0, 60.0, 68.0, 86.0, 96.0, 114.0]},
                     expected_warning="thermal_response_is_not_clock_evidence")
    else:
        offsets = {"mechanics": 6.25, "furnace": -4.75, "pressure": 2.4}
        frames = _frames(120, lambda t: _force(t) / 1_000_000 + (_thermal_temperature(t - 2) - 20) * 4e-6, 401)
        marker_rows = []
        rng = random.Random(501)
        tables["mechanics.csv"] = []
        for t in _times(2.05, 130.05, 0.16):
            force = _force(t - offsets["mechanics"]) + rng.uniform(-0.01, 0.01)
            # A channel-specific missing interval does not invalidate force.
            stress = "" if 36 < t - offsets["mechanics"] < 43 else round(force / 10.0, 6)
            tables["mechanics.csv"].append({"time": t, "force": round(force, 6), "stress": stress})
        tables["furnace.csv"] = [{"time": t, "temperature": round(_thermal_temperature(t - offsets["furnace"]), 7)}
                                  for t in _times(-8.15, 124.85, 0.35)
                                  if not 52 < t - offsets["furnace"] < 60]
        # The optional pressure recorder stops well before detector acquisition.
        tables["pressure.csv"] = [{"time": t, "pressure": 0.1 if t - offsets["pressure"] < 25 else
                                   0.3 if t - offsets["pressure"] < 52 else 0.1}
                                  for t in _times(0.0, 84.0, 0.6)]
        request["sources"] = [
            _source("mechanics", "mechanics.csv", [_channel("force", "N"), _channel("stress", "MPa")],
                    _anchors("mechanics", offsets["mechanics"], marker_rows)),
            _source("furnace", "furnace.csv", [_channel("temperature", "degC", max_gap_s=1.2)],
                    _anchors("furnace", offsets["furnace"], marker_rows)),
            _source("pressure", "pressure.csv", [_channel("pressure", "MPa", required=False, kind="step", max_gap_s=2)],
                    _anchors("pressure", offsets["pressure"], marker_rows))]
        tables["sync_markers.csv"] = marker_rows
        request["metadata"].update(sync_markers_file="sync_markers.csv", description=
            "Synthetic independent clocks synchronized by common TTL observations; "
            "temperature has a recording gap, stress has missing samples, and optional step pressure ends early.")
        truth.update(expected_status="resolved", offsets_s=offsets,
                     phase_boundaries_detector_s={"force": [12.0, 40.0, 53.0, 81.0],
                         "temperature": [12.0, 30.0, 42.0, 60.0, 68.0, 86.0, 96.0, 114.0], "pressure": [25.0, 52.0]},
                     missing_intervals_detector_s={"stress": [36.0, 43.0], "temperature": [52.0, 60.0]},
                     pressure_record_end_detector_s=81.6)
    _write_csv(directory / "frames.csv", frames)
    for filename, rows in tables.items():
        _write_csv(directory / filename, rows)
    _write_json(directory / "truth.json", truth)
    path = directory / "experiment.json"
    _write_json(path, request)
    return path
