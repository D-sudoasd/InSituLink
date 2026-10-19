"""Non-GUI entry. Returns a per-frame table and a decision record."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from strainlink.profiles import CLOCK_READERS, PROFILES, ClockError
from strainlink.time_align import blank_outside, detect_load_rise, resolve_event_dt


class AlignError(ValueError):
    """The alignment inputs cannot be sampled."""


def _round_num(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    number = float(value)
    if not np.isfinite(number):
        return None
    return round(number, 9)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return _round_num(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _tensile_arrays(record: dict) -> dict[str, np.ndarray]:
    if "t_s" not in record:
        raise AlignError("tensile record requires t_s")
    for key in ("force", "stress", "strain"):
        if key not in record:
            raise AlignError(f"tensile record requires {key}")
    t = np.asarray(record["t_s"], dtype=np.float64).reshape(-1)
    channels = {
        key: np.asarray(record[key], dtype=np.float64).reshape(-1)
        for key in ("force", "stress", "strain")
    }
    n = t.size
    if n < 2:
        raise AlignError("tensile record needs at least two samples")
    if not np.all(np.isfinite(t)):
        raise AlignError("tensile t_s must be finite")
    if np.any(np.diff(t) < 0):
        raise AlignError("tensile t_s must be non-decreasing")
    for key, values in channels.items():
        if values.shape != (n,):
            raise AlignError(f"tensile {key} length {values.size} != {n}")
    return {"t_s": t, **channels}


def _notes(decision: dict) -> list[str]:
    notes = [
        "Time alignment samples the tensile curve at the exposure midpoint plus Δt.",
        "It is not reference-curve interpolation and it is not a wall-clock lock.",
        "Use rows with valid_intact=1. Uncertainty is about one measured frame interval.",
    ]
    if decision["onset_unresolved"]:
        notes.append(
            f"Onset is unresolved ({decision['unresolved_reason']}). Published Δt is 0. "
            "No large lag was applied."
        )
    if decision["pre_record_frames"]:
        notes.append("Pre-record frames have empty force, stress, and tensile strain.")
    if decision["post_record_frames"]:
        notes.append("Post-record frames have empty force, stress, and tensile strain.")
    if decision["duration_mismatch"]:
        notes.append(
            "Photon span and tensile span disagree. Neither clock was stretched."
        )
    if decision["nominal_period_disagrees"]:
        notes.append(
            "Nominal frame_time disagrees with the measured start-to-start interval. "
            "The interval column is the measured difference."
        )
    if decision["nonmonotonic_frame_clock"]:
        notes.append("Finite start times are not strictly increasing. Frames were not reordered.")
    if decision["gui_minus_detector_s_ignored"] is not None:
        notes.append("The GUI-minus-detector offset was ignored.")
    return notes


def align_scan(frame_record: dict, tensile_record: dict, profile: str) -> dict[str, Any]:
    """Align one scan. ``profile`` is ``heps_id07_eiger`` or ``extracted_frame_clock``."""
    if profile not in PROFILES:
        known = ", ".join(sorted(PROFILES))
        raise AlignError(f"unknown profile {profile!r}; known: {known}")
    config = PROFILES[profile]
    try:
        clock = CLOCK_READERS[profile](frame_record)
    except ClockError as exc:
        raise AlignError(str(exc)) from exc
    tensile = _tensile_arrays(tensile_record)
    stats = clock["interval_stats"]
    frame_interval = stats["median_s"] if stats["median_s"] is not None else 1.53
    i_force, load_status = detect_load_rise(
        tensile["force"],
        baseline_n=config.load_baseline_n,
        k_sigma=config.load_k_sigma,
        min_run=config.load_min_run,
        abs_floor=config.load_abs_floor,
    )
    resolved = resolve_event_dt(
        clock["t_mid_s"],
        clock["eps_hkl"],
        tensile["t_s"],
        i_force,
        baseline_n=config.onset_baseline_n,
        k_sigma=config.onset_k_sigma,
        min_run=config.onset_min_run,
        force=tensile["force"],
        frame_interval_s=float(frame_interval),
        elastic_window_s=config.elastic_window_s,
        load_baseline_n=config.load_baseline_n,
        load_k_sigma=config.load_k_sigma,
        load_abs_floor=config.load_abs_floor,
        load_status=load_status,
    )
    dt = float(resolved["dt_s"])
    t_mech = clock["t_mid_s"] + dt
    sampled = {}
    before = after = None
    for key, column in (("force", "force"), ("stress", "stress"), ("strain", "strain")):
        values, before, after = blank_outside(tensile["t_s"], tensile[key], t_mech)
        sampled[key] = values
    assert before is not None and after is not None
    intact = ~(before | after)
    n_before = int(np.count_nonzero(before))
    n_after = int(np.count_nonzero(after))
    n_intact = int(np.count_nonzero(intact))
    scan_span = float(clock["t_start_s"][-1] + clock["exposure_s"][-1] - clock["t_start_s"][0])
    tensile_span = float(tensile["t_s"][-1] - tensile["t_s"][0])
    duration_delta = float(scan_span - tensile_span)
    duration_mismatch = abs(duration_delta) >= float(config.duration_mismatch_s)
    intervals = np.diff(clock["t_start_s"]) if clock["t_start_s"].size > 1 else np.array([])
    complex_cases = []
    if resolved["onset_unresolved"]:
        complex_cases.append("unresolved_onset")
    if n_before:
        complex_cases.append("pre_record_frames")
    if n_after:
        complex_cases.append("post_record_frames")
    if duration_mismatch:
        complex_cases.append("duration_mismatch")
    if clock["nominal_period_disagrees"]:
        complex_cases.append("nominal_period_disagrees")
    if stats["nonmonotonic"]:
        complex_cases.append("nonmonotonic_frame_clock")

    table = []
    for i in range(clock["n_finite_frames"]):
        interval = None if i == 0 else float(intervals[i - 1])
        row_intact = bool(intact[i])
        table.append(
            {
                "frame": i,
                "source_index": int(clock["source_index"][i]),
                "t_start_s": _round_num(clock["t_start_s"][i]),
                "exposure_s": _round_num(clock["exposure_s"][i]),
                "t_mid_s": _round_num(clock["t_mid_s"][i]),
                "interval_from_prev_s": _round_num(interval),
                "t_mech_s": _round_num(t_mech[i]),
                "force": _round_num(sampled["force"][i]) if row_intact else None,
                "stress": _round_num(sampled["stress"][i]) if row_intact else None,
                "strain": _round_num(sampled["strain"][i]) if row_intact else None,
                "eps_hkl": _round_num(clock["eps_hkl"][i]),
                "valid_intact": 1 if row_intact else 0,
                "before_record": bool(before[i]),
                "after_record": bool(after[i]),
            }
        )
    decision = {
        "profile": profile,
        "method": "elastic_onset_vs_load_rise",
        "wall_clock_lock": False,
        "subsecond_lock": False,
        "clocks_stretched": False,
        "used_nominal_frame_time_as_clock": False,
        "used_frame_index_times_exposure": False,
        "used_gui_offset_as_dt": False,
        "n_input_rows": clock["n_input_rows"],
        "n_finite_frames": clock["n_finite_frames"],
        "n_padded_dropped": clock["n_padded_dropped"],
        "virtual_nframes_ignored": clock["virtual_nframes_ignored"],
        "gui_minus_detector_s_ignored": _round_num(clock["gui_minus_detector_s_ignored"]),
        "detector_readout_time_s_ignored": _round_num(clock["detector_readout_time_s"]),
        "nominal_frame_time_s": _round_num(clock["nominal_frame_time_s"]),
        "measured_interval_median_s": _round_num(stats["median_s"]),
        "measured_interval_min_s": _round_num(stats["min_s"]),
        "measured_interval_max_s": _round_num(stats["max_s"]),
        "nominal_period_disagrees": bool(clock["nominal_period_disagrees"]),
        "frame_interval_tolerance_s": _round_num(frame_interval),
        "load_rise_status": load_status,
        "onset_frame": int(resolved["onset_frame"]) if not resolved["onset_unresolved"] else 0,
        "onset_unresolved": bool(resolved["onset_unresolved"]),
        "unresolved_reason": resolved["unresolved_reason"],
        "dt_s": _round_num(dt),
        "rejected_dt_s": _round_num(resolved["rejected_dt_s"]),
        "t_eps_onset_s": _round_num(resolved["t_eps_onset_s"]),
        "t_load_rise_s": _round_num(resolved["t_load_rise_s"]),
        "subframe_dt_s": _round_num(resolved["subframe_dt_s"]),
        "xcorr_dt_s": _round_num(resolved["xcorr_dt_s"]),
        "xcorr_ok": bool(resolved["xcorr_ok"]),
        "n_before_record": n_before,
        "n_after_record": n_after,
        "n_valid_intact": n_intact,
        "pre_record_frames": n_before > 0,
        "post_record_frames": n_after > 0,
        "scan_span_s": _round_num(scan_span),
        "tensile_span_s": _round_num(tensile_span),
        "duration_delta_s": _round_num(duration_delta),
        "duration_mismatch": bool(duration_mismatch),
        "nonmonotonic_frame_clock": bool(stats["nonmonotonic"]),
        "complex_cases": complex_cases,
    }
    decision["agent_notes"] = _notes(decision)
    return {"profile": profile, "decision": decision, "table": table}


def demo_records(name: str) -> tuple[dict, dict]:
    """Planted inputs for the two station demos. Not an alignment result."""
    if name == "heps_id07_eiger":
        n = 40
        start = np.arange(n, dtype=np.float64) * 1.53
        padded = np.concatenate([start, np.full(5, np.nan)])
        rng = np.random.default_rng(0)
        eps = rng.normal(0.0, 1e-6, size=n)
        eps[12:] += 4e-5 + 8e-6 * np.arange(n - 12)
        t = np.arange(0.0, 30.0 + 1e-9, 0.04)
        force = np.where(t < 10.0, 8.0, 8.0 + 3.0 * (t - 10.0))
        frames = {
            "start_time_s": padded.tolist(),
            "count_time_s": 1.0,
            "frame_time_s": 1.0000001,
            "detector_readout_time_s": 1e-7,
            "gui_minus_detector_s": -276.5,
            "virtual_nframes": 10000000,
            "eps_hkl": eps.tolist(),
        }
        tensile = {
            "t_s": t.tolist(),
            "force": force.tolist(),
            "stress": (10.0 * t).tolist(),
            "strain": (0.01 * t).tolist(),
        }
        return frames, tensile
    if name == "extracted_frame_clock":
        n = 24
        start = np.arange(n, dtype=np.float64) * 2.0
        eps = np.zeros(n, dtype=np.float64)
        eps[12:] = 4e-5 + 8e-6 * np.arange(n - 12)
        t = np.arange(0.0, 60.0 + 1e-9, 0.05)
        force = np.where(t < 30.0, 0.2, 5.0)
        frames = {
            "t_start_s": start.tolist(),
            "exposure_s": 0.5,
            "nominal_frame_time_s": 0.5,
            "eps_hkl": eps.tolist(),
        }
        tensile = {
            "t_s": t.tolist(),
            "force": force.tolist(),
            "stress": (2.0 * t).tolist(),
            "strain": (0.002 * t).tolist(),
        }
        return frames, tensile
    raise AlignError(f"unknown demo {name!r}")


def dumps_result(result: dict) -> str:
    return json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Align diffraction frames to a tensile curve without a GUI."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    align = sub.add_parser("align", help="write the table and decision record as JSON")
    align.add_argument("--profile", choices=sorted(PROFILES), default=None)
    align.add_argument("--demo", choices=sorted(PROFILES), default=None)
    align.add_argument("--frames", default=None, help="JSON frame record")
    align.add_argument("--tensile", default=None, help="JSON tensile record")
    args = parser.parse_args(argv)
    if args.command != "align":
        parser.error("unknown command")
    if args.demo:
        profile = args.demo
        frames, tensile = demo_records(profile)
    else:
        if not args.profile or not args.frames or not args.tensile:
            parser.error("pass --demo or all of --profile, --frames, and --tensile")
        profile = args.profile
        with open(args.frames, encoding="utf-8") as handle:
            frames = json.load(handle)
        with open(args.tensile, encoding="utf-8") as handle:
            tensile = json.load(handle)
    result = align_scan(frames, tensile, profile)
    print(dumps_result(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
