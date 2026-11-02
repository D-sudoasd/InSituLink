"""Name strain-reliability flags without changing published series.

Returned names do not replace strain, remove a baseline, or publish a
recomputed stress. Crosshead and eps_hkl are never copied into strain.
"""

from __future__ import annotations

import numpy as np


def _finite_in_order(values) -> np.ndarray:
    """Finite samples in their original order. The input sequence is not written."""
    if isinstance(values, (str, bytes)):
        raise ValueError("channel values must be numeric")
    array = np.asarray(values if values is not None else [], dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError("channel values must be one dimensional")
    return array[np.isfinite(array)]


def _channel_series(channel) -> np.ndarray | None:
    if not isinstance(channel, dict) or "values" not in channel:
        return None
    return _finite_in_order(channel.get("values"))


def _strain_series(channels: dict) -> np.ndarray | None:
    """Use the channel named strain, otherwise a channel whose role is strain.

    Extension, crosshead, and eps_hkl are not strain sources.
    """
    if "strain" in channels:
        channel = channels["strain"]
    else:
        channel = next(
            (
                item
                for item in channels.values()
                if isinstance(item, dict) and item.get("role") == "strain"
            ),
            None,
        )
    if channel is None:
        return None
    if not isinstance(channel, dict):
        raise ValueError("strain channel must be a mapping")
    return _channel_series(channel)


def _span(values: np.ndarray) -> float:
    return float(np.max(values) - np.min(values))


def _noise(values: np.ndarray) -> float:
    differences = np.abs(np.diff(values))
    if differences.size == 0 or not np.any(differences):
        return 0.0
    return float(np.median(differences))


def _drop(values: np.ndarray) -> float:
    """Largest fall from an earlier finite sample to a later one."""
    if values.size < 2:
        return 0.0
    earlier_peak = np.maximum.accumulate(values)[:-1]
    worst = float(np.max(earlier_peak - values[1:]))
    return worst if worst > 0.0 else 0.0


def _units_match(left, right) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    left_unit = left.get("unit")
    right_unit = right.get("unit")
    return isinstance(left_unit, str) and left_unit == right_unit


def strain_reliability(channels: dict) -> dict:
    """Return {"flags": [...], "evaluated": bool}.

    channels maps a name to {"values": sequence, "unit": str, "role": optional}.
    """
    if not isinstance(channels, dict):
        raise ValueError("channels must map names to channel records")
    strain = _strain_series(channels)
    if strain is None or strain.size < 5:
        return {"flags": [], "evaluated": False}

    span = _span(strain)
    noise = _noise(strain)
    count = int(strain.size)
    window = min(count, max(5, count // 10))
    baseline = float(np.median(strain[:window]))
    flags: list[str] = []
    if span <= 3.0 * noise:
        flags.append("stuck")
    elif _drop(strain) >= 0.5 * span:
        flags.append("sign_or_runaway")
    if span > 0.0 and abs(baseline) >= 20.0 * span:
        flags.append("offset_removed")

    extension = channels.get("extension")
    crosshead = channels.get("crosshead")
    if _units_match(extension, crosshead):
        extension_values = _channel_series(extension)
        crosshead_values = _channel_series(crosshead)
        if (
            extension_values is not None
            and crosshead_values is not None
            and extension_values.size
            and crosshead_values.size
            and _span(extension_values) > 3.0 * _span(crosshead_values)
        ):
            flags.append("large_vs_crosshead")
    return {"flags": flags, "evaluated": True}


def _component(values, name: str) -> tuple[np.ndarray, bool] | None:
    """Return a 1-d series and whether a scalar should broadcast.

    A length-1 sequence stays a sequence and is not treated as a scalar.
    """
    if values is None:
        return None
    if isinstance(values, (str, bytes)) or isinstance(values, bool):
        raise ValueError(f"{name} must be numeric")
    if isinstance(values, (int, float, np.integer, np.floating)):
        return np.asarray([float(values)], dtype=float), True
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return array.reshape(1), True
    if array.ndim != 1:
        raise ValueError(f"{name} must be one dimensional")
    return array, False


def stress_identity(stress, force, area) -> str | None:
    """Return "stress_identity_unresolved" or None.

    None means do not check (a series is missing) or the declared stress matches force/area.
    """
    parts = (
        _component(stress, "stress"),
        _component(force, "force"),
        _component(area, "area"),
    )
    if any(part is None for part in parts):
        return None

    sequences = [part[0] for part in parts if not part[1]]
    if sequences:
        length = int(sequences[0].size)
        if any(int(series.size) != length for series in sequences[1:]):
            return "stress_identity_unresolved"
    else:
        length = 1

    expanded = []
    for series, scalar in parts:
        if scalar:
            expanded.append(np.full(length, float(series[0]), dtype=float))
        else:
            expanded.append(series)
    stress_values, force_values, area_values = expanded
    usable = (
        np.isfinite(stress_values)
        & np.isfinite(force_values)
        & np.isfinite(area_values)
        & (area_values != 0.0)
    )
    if not np.any(usable):
        return "stress_identity_unresolved"
    stress_values = stress_values[usable]
    force_values = force_values[usable]
    area_values = area_values[usable]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        quotient = force_values / area_values
    finite_quotient = np.isfinite(quotient)
    if not np.any(finite_quotient):
        return "stress_identity_unresolved"
    stress_values = stress_values[finite_quotient]
    quotient = quotient[finite_quotient]
    scale = np.maximum(np.abs(stress_values), np.abs(quotient))
    largest = float(np.max(scale))
    keep = scale >= 0.05 * largest
    if int(np.count_nonzero(keep)) < 5:
        return "stress_identity_unresolved"
    scale = scale[keep]
    residual = np.abs(stress_values[keep] - quotient[keep])
    relative = np.zeros(scale.shape, dtype=float)
    nonzero = scale > 0.0
    relative[nonzero] = residual[nonzero] / scale[nonzero]
    if float(np.median(relative)) > 0.02:
        return "stress_identity_unresolved"
    return None
