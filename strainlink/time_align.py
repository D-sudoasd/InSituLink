"""Physics-event frame to tensile-curve alignment.

The frame clock is the finite detector start plus half the recorded exposure.
Δt shifts that midpoint onto the tensile clock. It is not a wall-clock lock.
"""

from __future__ import annotations

from typing import Any

import numpy as np

RISING_BEFORE_ABS = 1e-6


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def detect_monotonic_onset(
    y,
    *,
    baseline_n: int = 8,
    k_sigma: float = 4.0,
    min_run: int = 3,
    rising: bool = True,
    min_abs: float = 0.0,
) -> int | None:
    """First index of a sustained departure from the baseline window.

    Returns 0 when the series is already departed at index 0.
    Returns None when no sustained departure is found.
    """
    series = _as_float_array(y)
    if series.size == 0:
        raise ValueError("empty series")
    n_base = max(1, min(int(baseline_n), int(series.size)))
    base = series[:n_base]
    mu = float(np.median(base))
    mad = float(np.median(np.abs(base - mu))) * 1.4826
    noise = mad if mad > 0 else float(np.std(base))
    if noise <= 0:
        noise = max(abs(mu) * 1e-9, 1e-12)
    direction = 1.0 if rising else -1.0
    scored = direction * (series - mu)
    thresh = max(k_sigma * noise, float(min_abs))
    run = 0
    first = None
    for i, score in enumerate(scored):
        if score >= thresh:
            if first is None:
                first = i
            run += 1
            if run >= min_run:
                return int(first)
        else:
            run = 0
            first = None
    if scored[0] >= thresh:
        return 0
    return None


def detect_load_rise(
    force,
    *,
    baseline_n: int = 25,
    k_sigma: float = 4.0,
    min_run: int = 5,
    abs_floor: float = 15.0,
) -> tuple[int, str]:
    """Where tensile force leaves preload.

    Returns ``(index, status)``. Status is ``found`` or ``at_start`` only
    after ``min_run`` consecutive samples clear the floor. A lone prefix
    spike, or any run shorter than that, is ``missing`` and must not become Δt.
    """
    series = _as_float_array(force)
    if series.size == 0:
        raise ValueError("empty force")
    n_base = max(1, min(int(baseline_n), int(series.size)))
    mu = float(np.median(series[:n_base]))
    mad = float(np.median(np.abs(series[:n_base] - mu))) * 1.4826
    floor = max(k_sigma * mad, float(abs_floor))
    scored = series - mu
    run = 0
    first = None
    for i, score in enumerate(scored):
        if score >= floor:
            if first is None:
                first = i
            run += 1
            if run >= min_run:
                status = "at_start" if int(first) == 0 else "found"
                return int(first), status
        else:
            run = 0
            first = None
    return 0, "missing"


def onset_offset_seconds(t_eps, i_eps: int, t_force, i_force: int) -> float:
    """Δt = t_load_rise − t_strain_onset. Query time is t_mid + Δt."""
    t_eps = _as_float_array(t_eps)
    t_force = _as_float_array(t_force)
    return float(t_force[int(i_force)] - t_eps[int(i_eps)])


def interpolate_hold_after(t, y, query, dt: float = 0.0):
    """Linear sample at query + dt.

    Before the record the value is NaN. After the record the last sample is
    held and the after-flag is 1. Callers that publish stress must still
    blank after-record rows.
    """
    t = _as_float_array(t)
    y = _as_float_array(y)
    q = _as_float_array(query) + float(dt)
    after = q > t[-1]
    before = q < t[0]
    values = np.interp(q, t, y).astype(np.float64, copy=True)
    values[after] = y[-1]
    values[before] = np.nan
    return values, after.astype(np.int8)


def interpolative_crossing_time(t, y, i_cross: int, level: float, *, rising: bool = True) -> float:
    """Linear time of a threshold crossing between i_cross-1 and i_cross."""
    t = _as_float_array(t)
    y = _as_float_array(y)
    i = int(i_cross)
    if i <= 0:
        return float(t[0])
    if i >= t.size:
        return float(t[-1])
    y0, y1 = float(y[i - 1]), float(y[i])
    level = float(level)
    if not rising:
        y0, y1 = -y0, -y1
        level = -level
    denom = y1 - y0
    if denom == 0.0:
        return float(t[i])
    frac = float(np.clip((level - y0) / denom, 0.0, 1.0))
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


def _pearson_r(a, b) -> float:
    a = _as_float_array(a)
    b = _as_float_array(b)
    if a.size < 2 or b.size < 2 or a.size != b.size:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if den <= 0.0:
        return 0.0
    return float(np.dot(a, b) / den)


def elastic_xcorr_dt(
    t_mid,
    eps,
    t_force,
    force,
    *,
    elastic_tmax: float = 90.0,
    dt_lo: float = -20.0,
    dt_hi: float = 20.0,
    n_grid: int = 801,
    min_abs_corr: float = 0.5,
    min_points: int = 8,
    min_sharp: float = 0.25,
):
    """Derivative correlation of ε and force. Levels of two slow ramps are not a clock.

    Returns ``(dt, peak_abs_corr, ok)``. ``ok`` is false for a weak or edge peak.
    """
    t_mid = _as_float_array(t_mid)
    eps = _as_float_array(eps)
    t_force = _as_float_array(t_force)
    force = _as_float_array(force)
    if t_mid.size == 0 or t_force.size == 0:
        return 0.0, 0.0, False
    n_el = int(np.searchsorted(t_mid, elastic_tmax, side="right"))
    n_el = max(n_el, min(16, int(eps.size)))
    n_el = min(n_el, int(eps.size), int(t_mid.size))
    t_e = t_mid[:n_el]
    de = np.diff(eps[:n_el])
    n_grid = max(5, int(n_grid))
    dt_grid = np.linspace(float(dt_lo), float(dt_hi), n_grid)
    scores = np.zeros(dt_grid.size, dtype=np.float64)
    n_used = np.zeros(dt_grid.size, dtype=np.int32)
    for i, dt in enumerate(dt_grid):
        f_hat, after = interpolate_hold_after(t_force, force, t_e, float(dt))
        mask = np.isfinite(f_hat) & (after == 0)
        ok = mask[1:] & mask[:-1]
        n_used[i] = int(ok.sum())
        if n_used[i] < int(min_points):
            continue
        scores[i] = _pearson_r(de[ok], np.diff(f_hat)[ok])
    abs_s = np.abs(scores)
    i_peak = int(np.argmax(abs_s))
    peak = float(abs_s[i_peak])
    at_edge = i_peak == 0 or i_peak == scores.size - 1
    p95 = float(np.percentile(abs_s, 95))
    sharp = peak - p95
    if (
        at_edge
        or n_used[i_peak] < int(min_points)
        or peak < float(min_abs_corr)
        or sharp < float(min_sharp)
    ):
        return float(dt_grid[i_peak]), peak, False
    y0, y1, y2 = abs_s[i_peak - 1], abs_s[i_peak], abs_s[i_peak + 1]
    denom = float(y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-18:
        delta = 0.0
    else:
        delta = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
    step = float(dt_grid[1] - dt_grid[0])
    return float(dt_grid[i_peak] + delta * step), peak, True


def interpolative_event_dt(
    t_mid,
    eps,
    t_force,
    force,
    i_eps: int,
    i_force: int,
    *,
    baseline_n: int = 8,
    k_sigma: float = 4.0,
    load_baseline_n: int = 25,
    load_k_sigma: float = 4.0,
    load_abs_floor: float = 15.0,
) -> float:
    """Sub-frame Δt from the same kσ and load-rise thresholds."""
    t_mid = _as_float_array(t_mid)
    eps = _as_float_array(eps)
    t_force = _as_float_array(t_force)
    force = _as_float_array(force)
    n_base = max(1, min(int(baseline_n), int(eps.size)))
    base = eps[:n_base]
    mu_eps = float(np.median(base))
    mad_eps = float(np.median(np.abs(base - mu_eps))) * 1.4826
    noise = mad_eps if mad_eps > 0 else float(np.std(base))
    if noise <= 0:
        noise = max(abs(mu_eps) * 1e-9, 1e-12)
    thresh_eps = mu_eps + max(float(k_sigma) * noise, 0.0)
    n_base_f = max(1, min(int(load_baseline_n), int(force.size)))
    mu_f = float(np.median(force[:n_base_f]))
    mad_f = float(np.median(np.abs(force[:n_base_f] - mu_f))) * 1.4826
    floor_f = mu_f + max(float(load_k_sigma) * mad_f, float(load_abs_floor))
    t_eps = interpolative_crossing_time(t_mid, eps, int(i_eps), thresh_eps, rising=True)
    t_load = interpolative_crossing_time(t_force, force, int(i_force), floor_f, rising=True)
    return float(t_load - t_eps)


def _exceeds_one_frame(delta: float, frame_interval: float) -> bool:
    """True only when the disagreement is more than one measured frame."""
    limit = float(frame_interval)
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.53
    return abs(float(delta)) > limit * (1.0 + 1e-9) + 1e-12


def resolve_event_dt(
    t_mid,
    eps,
    t_force,
    i_force: int,
    *,
    baseline_n: int = 8,
    k_sigma: float = 4.0,
    min_run: int = 3,
    force=None,
    frame_interval_s: float = 1.53,
    elastic_window_s: float = 90.0,
    load_baseline_n: int = 25,
    load_k_sigma: float = 4.0,
    load_abs_floor: float = 15.0,
    load_status: str = "found",
) -> dict[str, Any]:
    """Choose Δt. Unresolved results publish 0 and keep the rejected candidate."""
    t_mid = _as_float_array(t_mid)
    eps = _as_float_array(eps)
    t_force = _as_float_array(t_force)
    result = {
        "dt_s": 0.0,
        "onset_frame": 0,
        "onset_unresolved": True,
        "unresolved_reason": "onset_missing",
        "rejected_dt_s": None,
        "t_eps_onset_s": None,
        "t_load_rise_s": float(t_force[int(i_force)]) if t_force.size else None,
        "subframe_dt_s": None,
        "xcorr_dt_s": None,
        "xcorr_ok": False,
    }
    if load_status == "missing":
        result["unresolved_reason"] = "load_rise_missing"
        return result

    n_el = int(np.searchsorted(t_mid, float(elastic_window_s), side="right"))
    n_el = max(n_el, min(16, int(eps.size)))
    n_el = min(n_el, int(eps.size))
    i_eps = detect_monotonic_onset(
        eps[:n_el],
        baseline_n=baseline_n,
        k_sigma=k_sigma,
        min_run=min_run,
        rising=True,
        min_abs=0.0,
    )
    if i_eps is None:
        return result
    if i_eps == 0:
        result["unresolved_reason"] = "onset_at_first_frame"
        result["t_eps_onset_s"] = float(t_mid[0])
        return result

    dt = onset_offset_seconds(t_mid, i_eps, t_force, i_force)
    result["rejected_dt_s"] = float(dt)
    result["onset_frame"] = int(i_eps)
    result["t_eps_onset_s"] = float(t_mid[int(i_eps)])
    result["t_load_rise_s"] = float(t_force[int(i_force)])
    t_query = t_mid + dt
    pre = t_query < t_force[0]
    if np.any(pre) and float(np.nanmax(np.abs(eps[pre]))) > RISING_BEFORE_ABS:
        result["unresolved_reason"] = "rising_before_record"
        result["onset_frame"] = 0
        return result
    if force is None:
        result["dt_s"] = float(dt)
        result["onset_unresolved"] = False
        result["unresolved_reason"] = ""
        result["rejected_dt_s"] = None
        return result

    dt_interp = interpolative_event_dt(
        t_mid,
        eps,
        t_force,
        force,
        int(i_eps),
        int(i_force),
        baseline_n=baseline_n,
        k_sigma=k_sigma,
        load_baseline_n=load_baseline_n,
        load_k_sigma=load_k_sigma,
        load_abs_floor=load_abs_floor,
    )
    result["subframe_dt_s"] = float(dt_interp)
    if _exceeds_one_frame(dt - dt_interp, frame_interval_s):
        result["unresolved_reason"] = "subframe_disagreement"
        result["onset_frame"] = 0
        return result
    dt_xc, _corr, xc_ok = elastic_xcorr_dt(
        t_mid,
        eps,
        t_force,
        force,
        elastic_tmax=elastic_window_s,
    )
    result["xcorr_dt_s"] = float(dt_xc)
    result["xcorr_ok"] = bool(xc_ok)
    if xc_ok and _exceeds_one_frame(dt - dt_xc, frame_interval_s):
        result["unresolved_reason"] = "xcorr_disagreement"
        result["onset_frame"] = 0
        return result
    result["dt_s"] = float(dt)
    result["onset_unresolved"] = False
    result["unresolved_reason"] = ""
    result["rejected_dt_s"] = None
    return result


def measured_interval_stats(t_start: np.ndarray) -> dict[str, Any]:
    t_start = _as_float_array(t_start)
    if t_start.size < 2:
        return {
            "median_s": None,
            "min_s": None,
            "max_s": None,
            "nonmonotonic": False,
            "nonpositive": False,
        }
    dt = np.diff(t_start)
    finite = dt[np.isfinite(dt)]
    nonmonotonic = bool(np.any(finite <= 0))
    if finite.size == 0:
        median = None
    else:
        median = float(np.median(finite))
    return {
        "median_s": median,
        "min_s": float(np.min(finite)) if finite.size else None,
        "max_s": float(np.max(finite)) if finite.size else None,
        "nonmonotonic": nonmonotonic,
        "nonpositive": nonmonotonic,
    }


def blank_outside(t, y, query):
    """Linear interpolation. Outside the record the published value is empty."""
    t = _as_float_array(t)
    y = _as_float_array(y)
    q = _as_float_array(query)
    before = q < t[0]
    after = q > t[-1]
    values = np.interp(np.clip(q, t[0], t[-1]), t, y).astype(np.float64, copy=True)
    values[before | after] = np.nan
    return values, before, after
