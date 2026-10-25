"""Exposure integration and evidence-bounded clock decisions.

All offsets obey source time = detector time + offset. No function sorts,
extrapolates, or stretches an input clock.
"""

from __future__ import annotations

import numpy as np

from .errors import InputError


def _array(values, name):
    try:
        result = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as exc:
        raise InputError(f"{name} must be numeric", "invalid_numeric", field=name) from exc
    if result.ndim != 1:
        raise InputError(f"{name} must be one dimensional", "invalid_shape", field=name)
    return result


def _clock(values, name="t_s"):
    t = _array(values, name)
    if not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise InputError(f"{name} must be finite and strictly increasing", "invalid_clock", field=name)
    return t


def _finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be finite", "invalid_numeric", field=name) from exc
    if not np.isfinite(number):
        raise InputError(f"{name} must be finite", "invalid_numeric", field=name)
    return number


def _series(t, values):
    y = _array(values, "values")
    if len(y) != len(t):
        raise InputError("Time and value lengths differ", "length_mismatch")
    return y


def _kind(kind):
    if kind not in {"continuous", "step"}:
        raise InputError("kind must be continuous or step", "invalid_kind")


def _gap_limit(t, max_gap_s=None):
    if max_gap_s is None:
        return 3 * float(np.median(np.diff(t))) if len(t) > 1 else float("inf")
    limit = _finite(max_gap_s, "max_gap_s")
    if limit <= 0:
        raise InputError("max_gap_s must be positive", "invalid_gap")
    return limit


def _point(t, y, q, kind, gap):
    if q < t[0] or q > t[-1]:
        return None
    i = int(np.searchsorted(t, q, side="right")) - 1
    if t[i] == q:
        return float(y[i]) if np.isfinite(y[i]) else None
    if i + 1 >= len(t) or t[i + 1] - t[i] > gap or not np.isfinite(y[i]):
        return None
    if kind == "step":
        return float(y[i])
    if not np.isfinite(y[i + 1]):
        return None
    return float(y[i] + (y[i + 1] - y[i]) * ((q - t[i]) / (t[i + 1] - t[i])))


def sample_channel(t, values, starts, ends, offset_s=0.0, kind="continuous", max_gap_s=None):
    """Return exact time-weighted statistics over each detector exposure.

    ``valid`` requires complete exposure support. A midpoint is independently
    sampled, including exact isolated finite samples. A step sample holds on
    [t[i], t[i+1]); a nonfinite sample ends that hold at its own timestamp.
    """
    t = _clock(t)
    y = _series(t, values)
    _kind(kind)
    offset = _finite(offset_s, "offset_s")
    starts, ends = _array(starts, "starts"), _array(ends, "ends")
    if len(starts) != len(ends):
        raise InputError("Exposure lengths differ", "length_mismatch")
    if not np.isfinite(starts).all() or not np.isfinite(ends).all() or np.any(ends < starts):
        raise InputError("Exposure bounds must be finite with end >= start", "invalid_exposure")
    with np.errstate(over="ignore"):
        shifted_starts, shifted_ends = starts + offset, ends + offset
    if not np.isfinite(shifted_starts).all() or not np.isfinite(shifted_ends).all():
        raise InputError("Shifted exposure bounds must be finite", "invalid_exposure")
    gap = _gap_limit(t, max_gap_s)
    rows = []
    for a, b in zip(shifted_starts, shifted_ends):
        mid = _point(t, y, a + (b - a) / 2, kind, gap)
        flags = []
        if a < t[0]:
            flags.append("before_record")
        if b > t[-1]:
            flags.append("after_record")
        if a == b:
            full = mid is not None
            if not full and not flags:
                flags.append("nonfinite" if np.any((t == a) & ~np.isfinite(y)) else "gap")
            rows.append(dict(mid=mid, mean=mid, min=mid, max=mid, coverage=float(full),
                             valid=full, flags=flags))
            continue
        area, covered, extrema = 0.0, 0.0, []
        first = max(0, int(np.searchsorted(t, a, side="right")) - 1)
        last = min(len(t) - 1, int(np.searchsorted(t, b, side="left")))
        for i in range(first, last):
            lo, hi = max(a, t[i]), min(b, t[i + 1])
            if hi <= lo:
                continue
            if t[i + 1] - t[i] > gap:
                if "gap" not in flags:
                    flags.append("gap")
                continue
            if not np.isfinite(y[i]) or (kind == "continuous" and not np.isfinite(y[i + 1])):
                if "nonfinite" not in flags:
                    flags.append("nonfinite")
                continue
            if kind == "step":
                v0 = v1 = float(y[i])
            else:
                slope = (y[i + 1] - y[i]) / (t[i + 1] - t[i])
                v0, v1 = float(y[i] + slope * (lo - t[i])), float(y[i] + slope * (hi - t[i]))
            covered += hi - lo
            area += (hi - lo) * (v0 + v1) / 2
            extrema.extend((v0, v1))
        duration = b - a
        coverage = min(1.0, max(0.0, float(covered / duration)))
        full = bool(extrema) and not flags and abs(covered - duration) <= 1e-12 * duration
        if not full:
            flags.append("incomplete_coverage")
        if mid is None:
            flags.append("midpoint_unavailable")
        rows.append(dict(mid=mid, mean=float(area / duration) if full else None,
                         min=min(extrema) if full else None, max=max(extrema) if full else None,
                         coverage=coverage, valid=full, flags=flags))
    return rows


def _decision(method, reason, *, dt=None, candidate=None, uncertainty=None, evidence=None):
    return dict(status="resolved" if dt is not None else "needs_input", method=method,
                dt_s=dt, candidate_dt_s=candidate, uncertainty_s=uncertainty,
                reason=reason, evidence=evidence or {})


def _declared_uncertainty(clock):
    if "uncertainty_s" not in clock:
        return None
    uncertainty = _finite(clock["uncertainty_s"], "uncertainty_s")
    if uncertainty < 0:
        raise InputError("Clock uncertainty must be nonnegative", "invalid_uncertainty")
    return uncertainty


def _frame_clock_metadata(frames):
    """Summarize available metadata without filling or reordering frame times.

    Only adjacent finite starts measure an actual frame interval. Skipping an
    interior missing frame would measure a multi-frame duration instead.
    """
    def number(value):
        try:
            return float(value)
        except (ValueError, TypeError, OverflowError):
            return float("nan")

    mids = np.array([number(f.get("t_mid_s")) for f in frames])
    starts = np.array([number(f.get("t_start_s", f.get("t_mid_s"))) for f in frames])
    finite_mid, finite_start = np.isfinite(mids), np.isfinite(starts)
    adjacent = finite_start[:-1] & finite_start[1:]
    differences = np.diff(starts)
    intervals = differences[adjacent & (differences > 0)]
    usable = finite_mid & finite_start
    evidence = dict(
        frame_interval_s=float(np.median(intervals)) if len(intervals) else None,
        n_frames=len(frames), n_valid_frames=int(usable.sum()),
        n_invalid_frames=int((~usable).sum()),
        n_invalid_frame_midpoints=int((~finite_mid).sum()),
        n_invalid_frame_starts=int((~finite_start).sum()),
        n_usable_frame_intervals=len(intervals),
        n_nonpositive_frame_intervals=int((adjacent & (differences <= 0)).sum()),
        clocks_stretched=False,
    )
    return mids, starts, evidence


def _onset(t, values, valid, baseline_n, min_run, k_sigma, direction):
    """Detect a sustained departure only after a demonstrably static prefix."""
    y = _series(t, values)
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 1 or len(valid) != len(y):
        raise InputError("Feature validity length differs", "length_mismatch")
    good = valid & np.isfinite(y)
    # The full declared baseline and a later sustained run must be available;
    # an identical initial pair does not establish the remaining baseline.
    n = baseline_n
    if len(y) < n + min_run or not good[:n].all():
        return None, "baseline_missing"
    floor = max(float(np.nanmax(np.abs(y[good]))) * 1e-10, 1e-12) if good.any() else 1e-12
    if np.ptp(y[:n]) > floor:
        if n < 3:
            return None, "baseline_missing"
        diffs = np.diff(y[:n])
        delta = float(np.median(diffs))
        diff_noise = 1.4826 * float(np.median(np.abs(diffs - delta)))
        if abs(delta) > max(k_sigma * diff_noise, floor):
            return None, "initially_changing"
        # Monotonic (including quantized) changes and strong net trends do not
        # establish a static baseline.
        residual = y[:n] - np.linspace(y[0], y[n - 1], n)
        noise = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        monotonic = np.all(diffs >= -floor) or np.all(diffs <= floor)
        if monotonic or (abs(y[n - 1] - y[0]) > max(k_sigma * noise, floor)
                         and np.all(diffs * delta >= 0)):
            return None, "initially_changing"
    base = float(np.median(y[:n]))
    noise = 1.4826 * float(np.median(np.abs(y[:n] - base)))
    # Quantized records can have a zero MAD despite real baseline scatter.
    if noise == 0 and np.any(y[:n] != base):
        noise = float(np.std(y[:n]))
    threshold = max(k_sigma * noise, floor)
    signs = (1,) if direction == "rise" else (-1,) if direction == "fall" else (1, -1)
    candidates = []
    max_gap = _gap_limit(t)
    for sign in signs:
        run = 0
        for i in range(n, len(y)):
            contiguous = i == 0 or t[i] - t[i - 1] <= max_gap
            if good[i] and sign * (y[i] - base) > threshold:
                run = run + 1 if contiguous else 1
                if run >= min_run:
                    first = i - min_run + 1
                    if good[first - 1] and t[first] - t[first - 1] <= max_gap:
                        candidates.append((float(t[first]), float(t[first] - t[first - 1])))
                    break
            else:
                run = 0
    if not candidates:
        return None, "event_missing"
    return min(candidates), "event_found"


def resolve_clock(frames, source, features=None, experiment="generic"):
    """Resolve a declared clock relationship or retain an event candidate.

    Thermal waveform responses never establish a clock relationship. Event
    uncertainty includes both sampled onset brackets; it is not sub-frame sync.
    """
    t = _clock(source.get("t_s", []), "source.t_s")
    mids, starts, evidence = _frame_clock_metadata(frames)
    interval = evidence["frame_interval_s"]
    clock = source.get("clock") or {}
    method = clock.get("method")
    if method in {"offset", "shared"}:
        if clock.get("confirmed") is not True or not isinstance(clock.get("evidence"), str) or not clock["evidence"].strip():
            return _decision(method, "explicit_clock_confirmation_required", evidence=evidence)
        dt = _finite(clock.get("offset_s"), "offset_s") if method == "offset" else 0.0
        evidence["confirmation"] = clock["evidence"]
        return _decision(method, "user_confirmed_clock_relationship", dt=dt,
                         uncertainty=_declared_uncertainty(clock), evidence=evidence)
    if method == "anchors":
        pairs = clock.get("pairs") or []
        if not pairs:
            return _decision(method, "anchors_missing", evidence=evidence)
        offsets = [_finite(p.get("source_time_s"), "source_time_s") -
                   _finite(p.get("frame_time_s"), "frame_time_s") for p in pairs]
        dt = float(np.median(offsets))
        tolerance = clock.get("tolerance_s", interval)
        if tolerance is None:
            tolerance = 0.0
        tolerance = _finite(tolerance, "tolerance_s")
        if tolerance < 0:
            raise InputError("Anchor tolerance must be nonnegative", "invalid_tolerance")
        spread = float(max(abs(o - dt) for o in offsets))
        declared = _declared_uncertainty(clock)
        uncertainty = max(spread, declared) if declared is not None else (
            max(spread, interval) if interval is not None else None)
        evidence.update(anchor_offsets_s=offsets, tolerance_s=tolerance,
                        anchor_residuals_s=[o - dt for o in offsets], anchor_max_residual_s=spread,
                        uncertainty_basis="declared" if declared is not None else (
                            "frame_interval" if interval is not None else "unavailable"))
        if spread > tolerance + 1e-12:
            return _decision(method, "inconsistent_anchors", candidate=dt, uncertainty=uncertainty, evidence=evidence)
        return _decision(method, "consistent_anchors", dt=dt, uncertainty=uncertainty, evidence=evidence)
    config = clock.get("event") or source.get("event") or (clock if method == "event" else None)
    if config is None:
        return _decision(method, "clock_method_required", evidence=evidence)
    if method not in {None, "event"}:
        return _decision(method, "unsupported_clock_method", evidence=evidence)
    method = "event"
    # Event correspondence needs an intact chronological frame clock. Explicit
    # clock declarations and anchors above remain usable for the finite rows.
    _clock(mids, "frames.t_mid_s")
    _clock(starts, "frames.t_start_s")
    channel_name = config.get("channel", "force" if experiment == "tensile" else None)
    channel = source.get("channels", {}).get(channel_name)
    if channel is None:
        return _decision(method, "event_channel_missing", evidence=evidence)
    experiment = channel.get("experiment", experiment)
    try:
        baseline_n = int(config.get("baseline_samples", 8))
        min_run = int(config.get("min_run", 3))
    except (ValueError, TypeError, OverflowError) as exc:
        raise InputError("Event sample counts must be integers", "invalid_event") from exc
    k_sigma = _finite(config.get("k_sigma", 4), "k_sigma")
    direction = config.get("direction")
    if baseline_n < 2 or min_run < 1 or k_sigma <= 0 or direction not in {None, "rise", "fall"}:
        raise InputError("Invalid event detection settings", "invalid_event")
    external, cause = _onset(t, channel["values"], np.isfinite(_series(t, channel["values"])),
                             baseline_n, min_run, k_sigma, direction)
    if external is None:
        return _decision(method, "source_" + cause, evidence=evidence)
    evidence["source_onset_s"] = external[0]
    candidates, uncertainties, failures, signal_events = [], [], [], []
    for signal in (features or {}).get("signals", []):
        signal_t = _clock(signal.get("t_s", []), "features.t_s")
        valid = signal.get("valid", [True] * len(signal_t))
        onset, cause = _onset(signal_t, signal.get("values", []), valid, baseline_n, min_run, k_sigma, None)
        if onset is None:
            failures.append({"id": signal.get("id"), "reason": cause})
            continue
        candidate = external[0] - onset[0]
        candidates.append(candidate)
        uncertainties.append(external[1] + onset[1])
        signal_events.append(dict(id=signal.get("id"), onset_s=onset[0], candidate_dt_s=candidate))
    evidence.update(signal_events=signal_events, unavailable_signals=failures)
    if not candidates:
        return _decision(method, "feature_event_missing_or_baseline_unavailable", evidence=evidence)
    candidate = float(np.median(candidates))
    uncertainty = float(max(uncertainties))
    evidence["sampling_uncertainty_s"] = uncertainty
    evidence["sparse_sampling"] = interval is None or uncertainty > interval
    if interval is None:
        return _decision(method, "frame_interval_unavailable", candidate=candidate, uncertainty=uncertainty, evidence=evidence)
    if max(candidates) - min(candidates) > interval + 1e-12:
        return _decision(method, "conflicting_feature_events", candidate=candidate, uncertainty=uncertainty, evidence=evidence)
    metadata = (features or {}).get("metadata") or {}
    latency_keys = {k: v for k, v in metadata.items() if "latency" in k or "lag" in k}
    if latency_keys:
        evidence["physical_response_latency"] = latency_keys
    if experiment == "thermal":
        evidence["waveform_only"] = True
        evidence["physical_response_lag_possible"] = True
        return _decision(method, "thermal_response_is_not_clock_evidence", candidate=candidate,
                         uncertainty=uncertainty, evidence=evidence)
    if latency_keys:
        return _decision(method, "physical_response_latency_is_not_clock_evidence", candidate=candidate,
                         uncertainty=uncertainty, evidence=evidence)
    common_load = experiment == "tensile" and str(channel_name).lower() in {"force", "load", "stress", "strain"}
    if not common_load and config.get("confirmed_same_event") is not True:
        return _decision(method, "same_event_confirmation_required", candidate=candidate,
                         uncertainty=uncertainty, evidence=evidence)
    return _decision(method, "common_event_onset", dt=candidate, candidate=candidate,
                     uncertainty=uncertainty, evidence=evidence)


def label_phases(t, values, query, experiment="generic", kind="continuous", tolerance=None, max_gap_s=None):
    """Label local chronological phases without merging repeated cycles."""
    t = _clock(t)
    y = _series(t, values)
    _kind(kind)
    query = _array(query, "query")
    gap = _gap_limit(t, max_gap_s)
    dt, dy = np.diff(t), np.diff(y)
    supported = np.isfinite(y[:-1]) & np.isfinite(y[1:]) & (dt <= gap)
    slopes = np.divide(dy, dt)
    if tolerance is None:
        finite = y[np.isfinite(y)]
        scale = max(float(np.ptp(finite)), float(np.max(np.abs(finite))), 1.0) if len(finite) else 1.0
        # Constant ramps have constant slopes. Their successive differences
        # estimate noise without treating ramp magnitude as measurement scatter.
        # Use only adjacent supported intervals, never differences across gaps.
        differences = np.diff(slopes)[supported[:-1] & supported[1:]]
        noise = 0.0
        if len(differences):
            center = float(np.median(differences))
            mad = 1.4826 * float(np.median(np.abs(differences - center)))
            # Piecewise exact cycles can put several large turns in a short
            # record. The lower quartile preserves their noise-free plateaus.
            # 0.318639 is the 25th percentile of an absolute standard normal.
            lower = float(np.quantile(np.abs(differences), 0.25)) / 0.31863936396437514
            noise = min(mad, lower) / np.sqrt(2)
        threshold = max(scale * 1e-9, 4 * noise)
    else:
        threshold = _finite(tolerance, "tolerance")
        if threshold < 0:
            raise InputError("Phase tolerance must be nonnegative", "invalid_tolerance")
    changed = supported & (np.abs(slopes) > threshold)
    first_change = int(np.flatnonzero(changed)[0]) if changed.any() else None
    up, down = {"thermal": ("heating", "cooling"), "tensile": ("loading", "unloading")}.get(
        experiment, ("increasing", "decreasing"))
    result = []
    for q in query:
        if not np.isfinite(q) or _point(t, y, q, kind, gap) is None:
            result.append("unknown")
            continue
        if len(t) == 1:
            result.append("baseline")
            continue
        node = max(0, int(np.searchsorted(t, q, side="right")) - 1)
        i = min(len(dt) - 1, node)
        if kind == "step":
            # Between jumps a previous-sample hold has zero local slope.
            if node > 0 and t[node] == q and changed[node - 1]:
                result.append(up if slopes[node - 1] > 0 else down)
            else:
                result.append("baseline" if first_change is None or q < t[first_change + 1] else "hold")
            continue
        if not supported[i]:
            result.append("unknown")
        elif first_change is None or i < first_change:
            result.append("baseline")
        elif slopes[i] > threshold:
            result.append(up)
        elif slopes[i] < -threshold:
            result.append(down)
        else:
            result.append("hold")
    return result
