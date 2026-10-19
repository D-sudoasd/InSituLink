"""Station profiles. Clock extraction differs. The alignment core does not."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from strainlink.time_align import measured_interval_stats


@dataclass(frozen=True)
class AlignConfig:
    name: str
    onset_baseline_n: int = 8
    onset_k_sigma: float = 4.0
    onset_min_run: int = 3
    load_baseline_n: int = 25
    load_k_sigma: float = 4.0
    load_min_run: int = 5
    load_abs_floor: float = 15.0
    elastic_window_s: float = 90.0
    duration_mismatch_s: float = 30.0
    nominal_disagree_s: float = 0.05


HEPS_ID07_EIGER = AlignConfig(name="heps_id07_eiger", load_abs_floor=15.0)
EXTRACTED_FRAME_CLOCK = AlignConfig(name="extracted_frame_clock", load_abs_floor=1.0)

PROFILES = {
    HEPS_ID07_EIGER.name: HEPS_ID07_EIGER,
    EXTRACTED_FRAME_CLOCK.name: EXTRACTED_FRAME_CLOCK,
}


class ClockError(ValueError):
    """The frame record cannot build a detector clock."""


def _array(record: dict, key: str) -> np.ndarray:
    if key not in record:
        raise ClockError(f"missing {key}")
    return np.asarray(record[key], dtype=np.float64)


def _exposure_on_finite(exposure, finite: np.ndarray, n_finite: int) -> np.ndarray:
    if np.isscalar(exposure) or np.asarray(exposure).ndim == 0:
        return np.full(n_finite, float(exposure), dtype=np.float64)
    values = np.asarray(exposure, dtype=np.float64)
    if values.shape == finite.shape:
        values = values[finite]
    if values.shape != (n_finite,):
        raise ClockError(
            f"exposure length {values.size} does not match {n_finite} finite starts"
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ClockError("exposure must be finite and non-negative")
    return values


def _eps_on_finite(eps, finite: np.ndarray, n_finite: int) -> np.ndarray:
    values = np.asarray(eps, dtype=np.float64)
    if values.shape == finite.shape:
        values = values[finite]
    if values.shape != (n_finite,):
        raise ClockError(f"eps_hkl length {values.size} does not match {n_finite} finite starts")
    return values


def finite_start_clock(
    start_time_s,
    exposure_s,
    eps_hkl,
    *,
    nominal_frame_time_s: float | None = None,
    detector_readout_time_s: float | None = None,
    gui_minus_detector_s: float | None = None,
    virtual_nframes: int | None = None,
    nominal_disagree_s: float = 0.05,
) -> dict[str, Any]:
    """Build the frame clock from finite starts. Decoy fields are stored and unused."""
    start = np.asarray(start_time_s, dtype=np.float64).reshape(-1)
    if start.size == 0:
        raise ClockError("start_time is empty")
    finite = np.isfinite(start)
    n_finite = int(np.count_nonzero(finite))
    if n_finite == 0:
        raise ClockError("no finite detector start times")
    t_start = start[finite]
    source_index = np.flatnonzero(finite).astype(int)
    exposure = _exposure_on_finite(exposure_s, finite, n_finite)
    eps = _eps_on_finite(eps_hkl, finite, n_finite)
    # The GUI offset is a different clock. It is not added here.
    t_mid = t_start + 0.5 * exposure
    stats = measured_interval_stats(t_start)
    nominal = None if nominal_frame_time_s is None else float(nominal_frame_time_s)
    median = stats["median_s"]
    disagrees = False
    if nominal is not None and median is not None:
        disagrees = abs(nominal - median) > float(nominal_disagree_s)
    return {
        "t_start_s": t_start,
        "exposure_s": exposure,
        "t_mid_s": t_mid,
        "eps_hkl": eps,
        "source_index": source_index,
        "n_input_rows": int(start.size),
        "n_finite_frames": n_finite,
        "n_padded_dropped": int(start.size - n_finite),
        "nominal_frame_time_s": nominal,
        "detector_readout_time_s": (
            None if detector_readout_time_s is None else float(detector_readout_time_s)
        ),
        "gui_minus_detector_s_ignored": (
            None if gui_minus_detector_s is None else float(gui_minus_detector_s)
        ),
        "virtual_nframes_ignored": None if virtual_nframes is None else int(virtual_nframes),
        "interval_stats": stats,
        "nominal_period_disagrees": bool(disagrees),
        "used_nominal_frame_time_as_clock": False,
        "used_frame_index_times_exposure": False,
        "used_gui_offset_as_dt": False,
    }


def clock_from_heps_id07(record: dict) -> dict[str, Any]:
    """HEPS ID07 Eiger clock from already-read metadata arrays. No image pixels."""
    if "start_time_s" not in record:
        raise ClockError("heps_id07_eiger requires start_time_s")
    if "count_time_s" not in record and "exposure_s" not in record:
        raise ClockError("heps_id07_eiger requires count_time_s")
    exposure = record["count_time_s"] if "count_time_s" in record else record["exposure_s"]
    virtual = record.get("virtual_nframes")
    clock = finite_start_clock(
        record["start_time_s"],
        exposure,
        record["eps_hkl"],
        nominal_frame_time_s=record.get("frame_time_s"),
        detector_readout_time_s=record.get("detector_readout_time_s"),
        gui_minus_detector_s=record.get("gui_minus_detector_s"),
        virtual_nframes=virtual,
        nominal_disagree_s=HEPS_ID07_EIGER.nominal_disagree_s,
    )
    clock["profile"] = HEPS_ID07_EIGER.name
    return clock


def clock_from_extracted_table(record: dict) -> dict[str, Any]:
    """A station that already exported per-frame starts. No HEPS HDF5."""
    if "t_start_s" not in record:
        raise ClockError("extracted_frame_clock requires t_start_s")
    if "exposure_s" not in record:
        raise ClockError("extracted_frame_clock requires exposure_s")
    clock = finite_start_clock(
        record["t_start_s"],
        record["exposure_s"],
        record["eps_hkl"],
        nominal_frame_time_s=record.get("nominal_frame_time_s"),
        detector_readout_time_s=record.get("detector_readout_time_s"),
        gui_minus_detector_s=record.get("gui_minus_detector_s"),
        virtual_nframes=record.get("virtual_nframes"),
        nominal_disagree_s=EXTRACTED_FRAME_CLOCK.nominal_disagree_s,
    )
    clock["profile"] = EXTRACTED_FRAME_CLOCK.name
    return clock


CLOCK_READERS = {
    HEPS_ID07_EIGER.name: clock_from_heps_id07,
    EXTRACTED_FRAME_CLOCK.name: clock_from_extracted_table,
}
