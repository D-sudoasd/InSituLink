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


def _point_support(t, y, q, kind, gap):
    """Return value-contributing rows and weights, plus the support interval."""
    if q < t[0] or q > t[-1]:
        return (), (), None, "outside_record"
    i = int(np.searchsorted(t, q, side="right")) - 1
    if t[i] == q:
        return ((i,), (1.,), [i, i], "exact") if np.isfinite(y[i]) else ((), (), [i, i], "nonfinite")
    bracket = [i, i + 1]
    if i + 1 >= len(t) or t[i + 1] - t[i] > gap:
        return (), (), bracket, "gap"
    if not np.isfinite(y[i]):
        return (), (), bracket, "nonfinite"
    if kind == "step":
        return (i,), (1.,), bracket, "held"
    if not np.isfinite(y[i + 1]):
        return (), (), bracket, "nonfinite"
    weight = float((q - t[i]) / (t[i + 1] - t[i]))
    return (i, i + 1), (1 - weight, weight), bracket, "interpolated"


def _point(t, y, q, kind, gap):
    indices, weights, _, _ = _point_support(t, y, q, kind, gap)
    return float(sum(y[i] * w for i, w in zip(indices, weights))) if indices else None


def _append_interval_range(ranges, index):
    if ranges and ranges[-1][1] == index - 1:
        ranges[-1][1] = index
    else:
        ranges.append([index, index])


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
        query = float(a + (b - a) / 2)
        indices, weights, bracket, point_status = _point_support(t, y, query, kind, gap)
        mid = float(sum(y[i] * w for i, w in zip(indices, weights))) if indices else None
        exposure = dict(start_s=float(a), end_s=float(b), overlapping_row_range=None,
                        supported_interval_row_ranges=[], supported_duration_s=0.,
                        statistics_definition="time_weighted_over_exposure")
        provenance = dict(
            model="piecewise_linear" if kind == "continuous" else "previous_sample_hold",
            model_definition=("Linear between adjacent finite samples within max_gap_s."
                              if kind == "continuous" else
                              "Right-continuous value y[i] on [t[i], t[i+1]) within max_gap_s."),
            time_coordinate="source_local_seconds", offset_s=offset,
            max_gap_s=float(gap) if np.isfinite(gap) else None,
            row_index_definition="zero_based_source_data_row; interval ranges include both left-end row indices",
            midpoint=dict(query_s=query, status=point_status, bracket_row_indices=bracket,
                          support=[dict(row_index=i, time_s=float(t[i]), value=float(y[i]), weight=w)
                                   for i, w in zip(indices, weights)]), exposure=exposure)
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
                             valid=full, flags=flags, provenance=provenance))
            continue
        area, covered, extrema = 0.0, 0.0, []
        first = max(0, int(np.searchsorted(t, a, side="right")) - 1)
        last = min(len(t) - 1, int(np.searchsorted(t, b, side="left")))
        for i in range(first, last):
            lo, hi = max(a, t[i]), min(b, t[i + 1])
            if hi <= lo:
                continue
            if exposure["overlapping_row_range"] is None:
                exposure["overlapping_row_range"] = [i, i + 1]
            else:
                exposure["overlapping_row_range"][1] = i + 1
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
            _append_interval_range(exposure["supported_interval_row_ranges"], i)
            area += (hi - lo) * (v0 + v1) / 2
            extrema.extend((v0, v1))
        duration = b - a
        coverage = min(1.0, max(0.0, float(covered / duration)))
        exposure["supported_duration_s"] = float(covered)
        full = bool(extrema) and not flags and abs(covered - duration) <= 1e-12 * duration
        if not full:
            flags.append("incomplete_coverage")
        if mid is None:
            flags.append("midpoint_unavailable")
        rows.append(dict(mid=mid, mean=float(area / duration) if full else None,
                         min=min(extrema) if full else None, max=max(extrema) if full else None,
                          coverage=coverage, valid=full, flags=flags, provenance=provenance))
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


def _baseline_stats(values):
    base = float(np.median(values))
    noise = 1.4826 * float(np.median(np.abs(values - base)))
    if noise == 0 and np.any(values != base):
        noise = float(np.std(values))
    return base, noise


def _onset(t, values, valid, baseline_n, min_run, k_sigma, direction, max_gap_s=None, threshold_scale=1.):
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
    # Quantized records can have a zero MAD despite real baseline scatter.
    base, noise = _baseline_stats(y[:n])
    threshold = max(k_sigma * noise * threshold_scale, floor)
    signs = (1,) if direction == "rise" else (-1,) if direction == "fall" else (1, -1)
    candidates = []
    gap_candidate = False
    max_gap = _gap_limit(t, max_gap_s)
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
                    elif t[first] - t[first - 1] > max_gap:
                        gap_candidate = True
                    break
            else:
                run = 0
    if not candidates:
        return None, "event_crosses_gap" if gap_candidate else "event_missing"
    return min(candidates), "event_found"


def _event_measurement(t, values, valid, baseline_n, min_run, k_sigma, direction, max_gap_s=None):
    """Keep a sampled threshold crossing separate from its timing sensitivity.

    A k-sigma departure is not the physical ramp onset. The threshold divided
    by the observed crossing slope estimates how much its time can move because
    of baseline scatter. A second crossing at half the threshold checks that a
    single steep sample does not hide this sensitivity. These are diagnostics,
    not fitted latency corrections or statistical confidence intervals. A sharp
    step stays bracketed by its adjacent samples; a slow noisy ramp must remain
    an unconfirmed candidate.
    """
    onset, cause = _onset(t, values, valid, baseline_n, min_run, k_sigma, direction, max_gap_s)
    if onset is None:
        return None, cause
    y = _series(t, values)
    index = int(np.searchsorted(t, onset[0]))
    base, noise = _baseline_stats(y[:baseline_n])
    slope = abs(float(y[index] - y[index - 1])) / onset[1]
    slope_timing = k_sigma * noise / slope if slope > 0 else 0.
    lower, lower_cause = _onset(t, y, valid, baseline_n, min_run, k_sigma, direction, max_gap_s, .5)
    sensitivity = abs(onset[0] - lower[0]) if lower is not None else None
    threshold_timing = max(slope_timing, sensitivity or 0.)
    return dict(onset_s=onset[0], row_index=index,
                bracket_s=[float(t[index - 1]), float(t[index])], sampling_interval_s=onset[1],
                baseline_level=base, baseline_noise=noise, k_sigma=k_sigma,
                noise_threshold=k_sigma * noise, crossing_slope_abs=slope,
                threshold_slope_timing_s=slope_timing, threshold_sensitivity_s=sensitivity,
                comparison_threshold_scale=.5,
                comparison_onset_s=lower[0] if lower is not None else None,
                comparison_status=lower_cause,
                threshold_timing_s=threshold_timing,
                threshold_timing_ambiguous=lower is None or threshold_timing > onset[1] * (1 + 1e-12),
                max_gap_s=float(_gap_limit(t, max_gap_s))), cause


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
                        anchor_frame_range_s=[min(float(p["frame_time_s"]) for p in pairs),
                                              max(float(p["frame_time_s"]) for p in pairs)],
                        anchor_source_range_s=[min(float(p["source_time_s"]) for p in pairs),
                                               max(float(p["source_time_s"]) for p in pairs)],
                        mapping_model="constant_offset",
                        anchor_range_definition="observation span; fixed-offset validity outside this span is not measured",
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
    external, cause = _event_measurement(t, channel["values"], np.isfinite(_series(t, channel["values"])),
                                        baseline_n, min_run, k_sigma, direction, channel.get("max_gap_s"))
    if external is None:
        return _decision(method, "source_" + cause, evidence=evidence)
    evidence.update(source_onset_s=external["onset_s"], source_event=external)
    candidates, uncertainties, sampling_widths, failures, signal_events = [], [], [], [], []
    threshold_ambiguous = external["threshold_timing_ambiguous"]
    for signal in (features or {}).get("signals", []):
        signal_t = _clock(signal.get("t_s", []), "features.t_s")
        valid = signal.get("valid", [True] * len(signal_t))
        onset, cause = _event_measurement(signal_t, signal.get("values", []), valid,
                                         baseline_n, min_run, k_sigma, None, signal.get("max_gap_s"))
        if onset is None:
            failures.append({"id": signal.get("id"), "reason": cause})
            continue
        candidate = external["onset_s"] - onset["onset_s"]
        candidates.append(candidate)
        sampling = external["sampling_interval_s"] + onset["sampling_interval_s"]
        sampling_widths.append(sampling)
        uncertainties.append(sampling + external["threshold_timing_s"] + onset["threshold_timing_s"])
        threshold_ambiguous |= onset["threshold_timing_ambiguous"]
        signal_events.append(dict(id=signal.get("id"), candidate_dt_s=candidate, **onset))
    evidence.update(signal_events=signal_events, unavailable_signals=failures)
    if not candidates:
        return _decision(method, "feature_event_missing_or_baseline_unavailable", evidence=evidence)
    candidate = float(np.median(candidates))
    uncertainty = float(max(uncertainties))
    evidence["sampling_uncertainty_s"] = float(max(sampling_widths))
    evidence["uncertainty_basis"] = "sample_brackets_plus_noise_threshold_timing_sensitivity"
    evidence["uncertainty_interpretation"] = "sampling and threshold sensitivity estimate; not a statistical confidence interval"
    evidence["sparse_sampling"] = interval is None or max(sampling_widths) > interval
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
    if threshold_ambiguous:
        return _decision(method, "event_threshold_timing_ambiguous", candidate=candidate,
                         uncertainty=uncertainty, evidence=evidence)
    common_load = experiment == "tensile" and str(channel_name).lower() in {"force", "load", "stress", "strain"}
    if not common_load and config.get("confirmed_same_event") is not True:
        return _decision(method, "same_event_confirmation_required", candidate=candidate,
                         uncertainty=uncertainty, evidence=evidence)
    return _decision(method, "common_event_onset", dt=candidate, candidate=candidate,
                     uncertainty=uncertainty, evidence=evidence)


def _phase_context(t, values, experiment, kind, tolerance, max_gap_s):
    t = _clock(t)
    y = _series(t, values)
    _kind(kind)
    gap = _gap_limit(t, max_gap_s)
    dt, dy = np.diff(t), np.diff(y)
    supported = np.isfinite(y[:-1]) & np.isfinite(y[1:]) & (dt <= gap)
    slopes = np.divide(dy, dt)
    # A numerical rounding threshold must scale with differences, not with the
    # unit's numerical size or its zero (e.g. metres/nanometres, Celsius/Kelvin).
    slope_scale = float(np.max(np.abs(slopes[supported]))) if supported.any() else 0.
    rounding = 8 * np.finfo(float).eps * slope_scale
    if tolerance is None:
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
        threshold = max(rounding, 4 * noise)
    else:
        threshold = _finite(tolerance, "tolerance")
        if threshold < 0:
            raise InputError("Phase tolerance must be nonnegative", "invalid_tolerance")
    changed = supported & (np.abs(slopes) > threshold)
    first_change = int(np.flatnonzero(changed)[0]) if changed.any() else None
    up, down = {"thermal": ("heating", "cooling"), "tensile": ("loading", "unloading")}.get(
        experiment, ("increasing", "decreasing"))
    return dict(t=t, y=y, kind=kind, gap=gap, dt=dt, slopes=slopes, supported=supported,
                changed=changed, first_change=first_change, threshold=threshold, up=up, down=down,
                rounding=rounding, tolerance_source="estimated_noise" if tolerance is None else "declared")


def _phase_labels(context, query):
    t, y, kind, gap = (context[k] for k in ("t", "y", "kind", "gap"))
    dt, slopes, supported, changed = (context[k] for k in ("dt", "slopes", "supported", "changed"))
    first_change, threshold, up, down = (context[k] for k in ("first_change", "threshold", "up", "down"))
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


def label_phases(t, values, query, experiment="generic", kind="continuous", tolerance=None, max_gap_s=None):
    """Label local chronological phases without merging repeated cycles.

    An explicit tolerance has units of channel units/second. Otherwise the
    threshold estimates slope noise; it is not an instrument resolution claim.
    """
    return _phase_labels(_phase_context(t, values, experiment, kind, tolerance, max_gap_s),
                         _array(query, "query"))


def exposure_phases(t, values, starts, ends, offset_s=0.0, experiment="generic",
                    kind="continuous", tolerance=None, max_gap_s=None):
    """Describe midpoint phase and chronological phases within each exposure.

    The positive-duration exposure is [start, end): a step exactly at its right
    boundary does not change its exposure statistics. Unknown portions remain
    explicit and never count as evidence of a physical phase transition.
    """
    context = _phase_context(t, values, experiment, kind, tolerance, max_gap_s)
    t = context["t"]
    starts, ends = _array(starts, "starts"), _array(ends, "ends")
    if len(starts) != len(ends):
        raise InputError("Exposure lengths differ", "length_mismatch")
    if not np.isfinite(starts).all() or not np.isfinite(ends).all() or np.any(ends < starts):
        raise InputError("Exposure bounds must be finite with end >= start", "invalid_exposure")
    offset = _finite(offset_s, "offset_s")
    with np.errstate(over="ignore"):
        shifted_starts, shifted_ends = starts + offset, ends + offset
    if not np.isfinite(shifted_starts).all() or not np.isfinite(shifted_ends).all():
        raise InputError("Shifted exposure bounds must be finite", "invalid_exposure")
    result = []
    for a, b in zip(shifted_starts, shifted_ends):
        midpoint = _phase_labels(context, [a + (b - a) / 2])[0]
        queries = []
        if a == b:
            queries.append(a)
        else:
            if a < t[0]:
                queries.append(a + (min(b, t[0]) - a) / 2)
            first = max(0, int(np.searchsorted(t, a, side="right")) - 1)
            last = min(len(t) - 1, int(np.searchsorted(t, b, side="left")))
            for i in range(first, last):
                lo, hi = max(a, t[i]), min(b, t[i + 1])
                if hi > lo:
                    queries.append(lo + (hi - lo) / 2)
            if kind == "step":
                queries.extend(t[np.searchsorted(t, a):np.searchsorted(t, b)])
            if b > t[-1]:
                queries.append(max(a, t[-1]) + (b - max(a, t[-1])) / 2)
        phases = []
        for phase in _phase_labels(context, sorted(queries)):
            if not phases or phases[-1] != phase:
                phases.append(phase)
        if not phases:
            phases = ["unknown"]
        known = set(phases) - {"unknown"}
        result.append(dict(midpoint=midpoint, phases=phases, crosses_phase=len(known) > 1,
                           unsupported="unknown" in phases, tolerance_slope=context["threshold"],
                           tolerance_source=context["tolerance_source"], roundoff_slope=context["rounding"]))
    return result
