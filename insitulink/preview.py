"""Optional, headless diffraction preanalysis with explicit physical identities.

The fitted features are unnamed peaks, not crystallographic phase assignments.
Image readers and integration dependencies are imported only for image input.
"""
from __future__ import annotations

import bz2
import gzip
from io import BytesIO
from pathlib import Path
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths

from .errors import InputError
from .inputs import canonical_source_file


AXES = {"d_A", "q_A^-1", "two_theta_deg"}
# Preserve existing frame/custom columns in peak rows. Only the new declaration
# and nested provenance records belong exclusively to frame_records/manifest.
_FRAME_PROVENANCE_FIELDS = frozenset({"source_file_declared", "dataset_declared", "declared_source",
                                     "time_provenance", "physical_source", "identity_issues"})


def _positive(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a positive finite number", "preview_invalid_config", field=name)
    if not np.isfinite(number) or number <= 0:
        raise InputError(f"{name} must be a positive finite number", "preview_invalid_config", field=name)
    return number


def _path(value, base_dir, role):
    if not isinstance(value, (str, Path)) or not str(value):
        raise InputError(f"Supply a {role} path", "preview_missing_input", field=role)
    path = Path(value)
    path = path if path.is_absolute() else base_dir / path
    if not path.is_file():
        raise InputError(f"Missing {role}: {path}", "preview_missing_file", path=str(path), role=role)
    return path.resolve()


def _identity(frame):
    return {key: frame.get(key) for key in ("source_file", "dataset", "frame_index", "original_id")}


def _matches(frame, identity):
    return all((os.path.normcase(frame.get(key) or "") == os.path.normcase(value or ""))
               if key == "source_file" else frame.get(key) == value
               for key, value in identity.items())


def _canonical_identities(config, frames, base_dir):
    """Apply the request-relative source rule to copies before comparisons."""
    def canonical(identity):
        return {**identity, "source_file": canonical_source_file(identity["source_file"], base_dir)} if "source_file" in identity else dict(identity)
    frames = [canonical(frame) for frame in frames]
    config = dict(config)
    for key in ("files", "baseline_frames"):
        if isinstance(config.get(key), list):
            config[key] = [canonical(entry) if isinstance(entry, dict) else entry for entry in config[key]]
    return config, frames


def _noise(y):
    # Second differences remove a smooth background and most peak curvature.
    delta = np.diff(y, n=2)
    return max(float(np.median(np.abs(delta - np.median(delta))) / (0.67448975 * np.sqrt(6))),
               float(np.ptp(y)) * 1e-9, np.finfo(float).eps)


def _candidates(x, y, snr_min):
    noise = _noise(y)
    smooth = gaussian_filter1d(y, 1)
    # A line from both outer deciles limits sloping-background artifacts.
    n = max(2, len(x) // 10)
    edge = np.r_[np.arange(n), np.arange(len(x) - n, len(x))]
    line = np.polyfit(x[edge] - x.mean(), smooth[edge], 1)
    corrected = smooth - np.polyval(line, x - x.mean())
    peaks, properties = find_peaks(corrected, prominence=snr_min * noise, distance=3)
    return peaks, properties, noise


def _invalid(reason, **values):
    return {"center": None, "center_sigma": None, "rms": None, "snr": None,
            "status": "invalid", "reason": reason, **values}


def fit_peak(axis, intensity, window, *, snr_min=5.0):
    """Fit one resolved pseudo-Voigt peak and a linear background in a window.

    The uncertainty is a residual-based local covariance estimate, not an
    instrument-calibration uncertainty. Multiple resolved maxima are rejected.
    """
    x = np.asarray(axis, dtype=float)
    y = np.asarray(intensity, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y) & (x >= window["lower"]) & (x <= window["upper"])
    x, y = x[keep], y[keep]
    order = np.argsort(x)
    x, y = x[order], y[order]
    if len(x) < 12 or np.any(np.diff(x) <= 0):
        return _invalid("insufficient_samples")
    peaks, props, noise = _candidates(x, y, snr_min)
    if not len(peaks):
        return _invalid("weak_or_vanished_peak")
    if len(peaks) > 1:
        return _invalid("multiple_resolved_peaks")
    span = float(x[-1] - x[0])
    spacing = float(np.median(np.diff(x)))
    reference = float(x.mean())
    n = max(2, len(x) // 10)
    edge = np.r_[np.arange(n), np.arange(len(x) - n, len(x))]
    slope, background = np.polyfit(x[edge] - reference, y[edge], 1)
    index = peaks[0]
    amplitude = max(float(y[index] - background - slope * (x[index] - reference)), noise)
    width = float(peak_widths(gaussian_filter1d(y, 1), peaks, rel_height=.5)[0][0] * spacing)
    width = np.clip(width, spacing * 2, span * .5)

    def model(p):
        height, center, fwhm, eta, intercept, gradient = p
        z = (x - center) / fwhm
        return height * (eta / (1 + 4 * z*z) + (1-eta) * np.exp(-4*np.log(2)*z*z)) + intercept + gradient*(x-reference)

    initial = [amplitude, x[index], width, .5, background, slope]
    lower = [0, x[0], spacing, 0, -np.inf, -np.inf]
    upper = [np.inf, x[-1], span*.75, 1, np.inf, np.inf]
    result = least_squares(lambda p: model(p)-y, initial, bounds=(lower, upper), x_scale="jac")
    residual = model(result.x) - y
    rms = float(np.sqrt(np.mean(residual**2)))
    height, center, fwhm, eta, intercept, gradient = result.x
    snr = float(height / max(noise, rms))
    diagnostics = {"center": float(center), "rms": rms, "snr": snr,
                   "fwhm": float(fwhm), "eta": float(eta), "n_points": len(x)}
    if not result.success:
        return _invalid("fit_failed", **diagnostics)
    if snr < snr_min:
        return _invalid("weak_or_vanished_peak", **diagnostics)
    if center-x[0] < max(spacing*2, fwhm*.5) or x[-1]-center < max(spacing*2, fwhm*.5):
        return _invalid("peak_at_window_boundary", **diagnostics)
    if fwhm <= spacing*1.01 or fwhm >= span*.74:
        return _invalid("unresolved_width", **diagnostics)
    try:
        if np.linalg.matrix_rank(result.jac) < len(result.x):
            return _invalid("singular_covariance", **diagnostics)
        covariance = np.linalg.inv(result.jac.T @ result.jac) * float(residual @ residual) / (len(x)-len(result.x))
        sigma = float(np.sqrt(covariance[1, 1]))
    except np.linalg.LinAlgError:
        return _invalid("singular_covariance", **diagnostics)
    if not np.isfinite(sigma) or sigma > fwhm*.25:
        return _invalid("uncertain_position", **diagnostics)
    return {**diagnostics, "center_sigma": sigma, "status": "valid", "reason": ""}


def _to_d(center, sigma, axis, wavelength):
    if center <= 0:
        return None, None
    if axis == "d_A":
        return center, sigma
    if axis == "q_A^-1":
        return 2*np.pi/center, 2*np.pi*sigma/center**2
    if center >= 180:
        return None, None
    theta = np.deg2rad(center/2)
    d = wavelength/(2*np.sin(theta))
    return float(d), float(abs(d/np.tan(theta)*np.pi/360*sigma))


def _spectra(config, frames, base_dir, source_bytes=None):
    entries = config.get("files")
    if not isinstance(entries, list) or not entries:
        raise InputError("Supply spectra files with frame_index and path", "preview_missing_spectra")
    if any(not isinstance(e, dict) or type(e.get("frame_index")) is not int or e["frame_index"] < 0 for e in entries):
        raise InputError("Each spectrum needs a nonnegative frame_index", "preview_spectrum_identity")
    columns = config.get("columns", [0, 1])
    if not isinstance(columns, list) or len(columns) != 2 or any(type(c) is not int or c < 0 for c in columns):
        raise InputError("columns must contain two nonnegative column indices", "preview_invalid_config")
    spectra, dependencies = [], []
    for frame in frames:
        matches = [e for e in entries if _matches(frame, {k: e[k] for k in ("frame_index", "source_file", "dataset") if k in e})]
        if len(matches) != 1:
            raise InputError("Each frame needs exactly one unambiguous spectrum", "preview_spectrum_identity", frame=_identity(frame))
        entry = matches[0]
        matched_frames = [f for f in frames if _matches(f, {k: entry[k] for k in ("frame_index", "source_file", "dataset") if k in entry})]
        if len(matched_frames) != 1:
            raise InputError("Spectrum frame_index needs source_file/dataset disambiguation", "preview_spectrum_identity", frame=_identity(frame))
        path = _path(entry.get("path"), base_dir, "spectrum")
        delimiter = entry.get("delimiter", config.get("delimiter", "," if path.suffix.lower() == ".csv" else None))
        skip = entry.get("skiprows", config.get("skiprows", config.get("header_rows", 0)))
        if source_bytes is not None and str(path) not in source_bytes:
            raise InputError("Captured spectrum bytes are missing", "preview_missing_snapshot", path=str(path))
        try:
            content = path
            if source_bytes is not None:
                payload = source_bytes[str(path)]
                # Match loadtxt's path-based support for compressed text files.
                if path.suffix.lower() == ".gz":
                    payload = gzip.decompress(payload)
                elif path.suffix.lower() == ".bz2":
                    payload = bz2.decompress(payload)
                content = BytesIO(payload)
            data = np.loadtxt(content, delimiter=delimiter, skiprows=skip, usecols=columns, ndmin=2)
        except (OSError, ValueError, TypeError) as exc:
            raise InputError(f"Cannot read spectrum {path}: {exc}", "preview_invalid_spectrum", path=str(path)) from exc
        spectra.append((data[:, 0], data[:, 1]))
        dependencies.append({"path": str(path), "role": "spectrum"})
    return spectra, dependencies, {}


def _images(config, frames, base_dir, output_dir):
    for key in ("mask", "correct_solid_angle", "polarization_factor"):
        if key not in config:
            raise InputError(f"Image preview requires explicit {key}", "preview_missing_correction", field=key)
    if type(config["correct_solid_angle"]) is not bool:
        raise InputError("correct_solid_angle must be boolean", "preview_invalid_config")
    polarization = config["polarization_factor"]
    if polarization is not None and (isinstance(polarization, bool) or not isinstance(polarization, (int, float)) or
                                     not np.isfinite(polarization) or abs(polarization) > 1):
        raise InputError("polarization_factor must be null or between -1 and 1", "preview_invalid_config")
    geometry = _path(config.get("geometry"), base_dir, "geometry")
    try:
        import pyFAI
        import fabio
    except ImportError as exc:
        raise InputError("Image preview requires pyFAI and fabio", "preview_missing_dependency") from exc
    try:
        integrator = pyFAI.load(str(geometry))
    except Exception as exc:
        raise InputError(f"Cannot read geometry: {exc}", "preview_invalid_geometry") from exc
    wavelength = config.get("wavelength_A")
    if wavelength is None:
        wavelength = integrator.wavelength * 1e10 if integrator.wavelength is not None else None
    wavelength = _positive(wavelength, "wavelength_A")
    if integrator.wavelength is not None and "wavelength_A" in config and not np.isclose(integrator.wavelength*1e10, wavelength, rtol=1e-5):
        raise InputError("Configured wavelength disagrees with PONI wavelength", "preview_wavelength_mismatch")
    integrator.wavelength = wavelength*1e-10
    dependencies = [{"path": str(geometry), "role": "geometry"}]
    user_mask = None
    if config["mask"] is not None:
        mask_path = _path(config["mask"], base_dir, "mask")
        try:
            with fabio.open(str(mask_path)) as mask_image:
                user_mask = np.asarray(mask_image.data) != 0
        except Exception as exc:
            raise InputError(f"Cannot read mask: {exc}", "preview_invalid_mask", path=str(mask_path)) from exc
        dependencies.append({"path": str(mask_path), "role": "mask"})
    sector = config.get("azimuth_range")
    if sector is not None and (not isinstance(sector, (list, tuple)) or len(sector) != 2 or
                               not all(isinstance(v, (int, float)) and np.isfinite(v) for v in sector) or sector[0] >= sector[1]):
        raise InputError("azimuth_range must be an increasing pair in degrees", "preview_invalid_config")
    npt = config.get("npt", 1500)
    if type(npt) is not int or npt < 12:
        raise InputError("npt must be an integer >= 12", "preview_invalid_config")
    spectra = []
    cache_paths = []
    from .hdf5_sources import discover_dependencies
    storage = discover_dependencies(frames)
    if not storage["complete"]:
        raise InputError("Cannot verify all detector storage dependencies", "preview_unverified_image_sources", issues=storage["issues"])
    dependencies.extend({"path": path, "role": "detector_image"} for path in storage["paths"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for number, frame in enumerate(frames):
        source = _path(frame.get("source_file"), base_dir, "detector_image")
        index = frame["frame_index"]
        if frame.get("dataset"):
            try:
                import h5py
                with h5py.File(source, "r") as handle:
                    dataset = handle[frame["dataset"]]
                    data = np.asarray(dataset[index] if dataset.ndim == 3 else dataset[()] if dataset.ndim == 2 and index == 0 else [])
            except (ImportError, KeyError, OSError, IndexError, ValueError, TypeError, RuntimeError) as exc:
                raise InputError(f"Cannot read detector frame: {exc}", "preview_invalid_image", frame=_identity(frame)) from exc
        else:
            try:
                with fabio.open(str(source)) as image:
                    data = np.asarray(image.data if index == 0 else image.getframe(index).data).copy()
            except Exception as exc:
                raise InputError(f"Cannot read detector frame: {exc}", "preview_invalid_image", frame=_identity(frame)) from exc
        if data.ndim != 2 or min(data.shape) < 2:
            raise InputError("Requested detector frame must be a 2D image", "preview_invalid_image", frame=_identity(frame))
        mask = ~np.isfinite(data)
        for addition in (integrator.detector.mask, user_mask):
            if addition is not None:
                if addition.shape != data.shape:
                    raise InputError("Mask shape differs from detector frame", "preview_mask_shape")
                mask |= addition != 0
        kwargs = {"unit": "q_A^-1", "mask": mask, "correctSolidAngle": config["correct_solid_angle"],
                  "polarization_factor": polarization, "method": ("no", "histogram", "cython")}
        if sector is not None:
            kwargs["azimuth_range"] = tuple(sector)
        try:
            integrated = integrator.integrate1d(np.where(mask, 0, data), npt, **kwargs)
        except Exception as exc:
            raise InputError(f"Cannot integrate detector frame: {exc}", "preview_integration_failed", frame=_identity(frame)) from exc
        q = np.asarray(integrated.radial)
        intensity = np.asarray(integrated.intensity, dtype=float).copy()
        intensity[np.asarray(integrated.count) == 0] = np.nan
        axis = config.get("axis", "q_A^-1")
        x = q if axis == "q_A^-1" else np.divide(2*np.pi, q) if axis == "d_A" else np.rad2deg(2*np.arcsin(q*wavelength/(4*np.pi)))
        spectra.append((x, intensity))
        cache = output_dir / f"integrated_{number:06d}.npz"
        np.savez_compressed(cache, q_A_inv=q, intensity=intensity)
        cache_paths.append(str(cache))
    return spectra, dependencies, {"wavelength_A": wavelength, "integration_axis": "q_A^-1",
                                   "azimuth_range": sector, "integrated_files": cache_paths,
                                   "storage_dependencies": storage}


def _automatic_windows(spectra, max_peaks, snr_min):
    representative = sorted({0, len(spectra)//2, len(spectra)-1})
    candidates = []
    for index in representative:
        x, y = spectra[index]
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        order = np.argsort(x)
        x, y = x[order], y[order]
        if len(x) < 12:
            return []
        peaks, props, _ = _candidates(x, y, snr_min)
        widths = peak_widths(gaussian_filter1d(y, 1), peaks)[0] * np.median(np.diff(x)) if len(peaks) else []
        candidates.append([(float(x[p]), max(float(w)*3, float(np.ptp(x))*.015), float(prom))
                           for p, w, prom in zip(peaks, widths, props["prominences"])])
    common = []
    for center, halfwidth, prominence in candidates[0]:
        matches = [[p for p in values if abs(p[0]-center) < halfwidth*.5] for values in candidates[1:]]
        if all(len(values) == 1 for values in matches):
            centers = [center] + [v[0][0] for v in matches]
            halfwidth = max([halfwidth] + [v[0][1] for v in matches])
            common.append((min(centers)-halfwidth, max(centers)+halfwidth, prominence))
    common.sort(key=lambda p: -p[2])
    chosen = []
    for low, high, _ in common:
        if all(high < w["lower"] or low > w["upper"] for w in chosen):
            chosen.append({"id": f"peak_{len(chosen)+1}", "lower": low, "upper": high})
            if len(chosen) == max_peaks:
                break
    return chosen


def _baseline_indices(requested, frames):
    if requested is None:
        return list(range(min(8, len(frames))))
    if not isinstance(requested, list) or not requested:
        raise InputError("baseline_frames must be a nonempty list of frame identities", "preview_invalid_baseline")
    indices = []
    for identity in requested:
        if type(identity) is int:
            matches = [i for i, f in enumerate(frames) if f["frame_index"] == identity]
        elif isinstance(identity, dict) and identity:
            matches = [i for i, f in enumerate(frames) if _matches(f, identity)]
        else:
            matches = []
        if len(matches) != 1 or matches[0] in indices:
            raise InputError("Baseline identities must identify unique requested frames", "preview_ambiguous_baseline", identity=identity)
        indices.append(matches[0])
    return indices


def extract_features(config: dict, frames: list[dict], base_dir: Path, output_dir: Path,
                     *, source_bytes: dict[str, bytes] | None = None) -> dict:
    """Extract valid peak positions and baseline-relative lattice-spacing shifts.

    For spectra, source_bytes optionally supplies the exact captured contents,
    keyed by canonical absolute path. A supplied mapping must cover every input
    spectrum; it never falls back to reopening an unbound file. Image inputs
    keep their existing readers and do not use this argument.
    """
    if not isinstance(config, dict) or config.get("type") not in ("spectra", "images"):
        raise InputError("Supply preview type spectra or images", "preview_missing_config")
    axis = config.get("axis", "q_A^-1" if config["type"] == "images" else None)
    if axis not in AXES:
        raise InputError("Supply axis d_A, q_A^-1, or two_theta_deg", "preview_invalid_axis")
    if not frames:
        raise InputError("No frames supplied for preview", "preview_missing_frames")
    for frame in frames:
        if not isinstance(frame, dict) or type(frame.get("frame_index")) is not int or frame["frame_index"] < 0 or not isinstance(frame.get("t_mid_s"), (int, float)) or not np.isfinite(frame["t_mid_s"]):
            raise InputError("Every preview frame needs frame_index and finite t_mid_s", "preview_invalid_frame")
        if config["type"] == "images" and not frame.get("source_file"):
            raise InputError("Image preview needs a raw source_file for every frame", "preview_missing_input", field="source_file")
    base_dir, output_dir = Path(base_dir), Path(output_dir)
    config, frames = _canonical_identities(config, frames, base_dir)
    if any(right["t_mid_s"] <= left["t_mid_s"] for left, right in zip(frames, frames[1:])):
        raise InputError("Preview frame times must increase in acquisition order", "preview_nonmonotonic_time")
    baseline_indices = _baseline_indices(config.get("baseline_frames"), frames)
    snr_min = _positive(config.get("snr_min", 5), "snr_min")
    max_peaks = config.get("max_peaks", 3)
    if type(max_peaks) is not int or max_peaks < 1:
        raise InputError("max_peaks must be a positive integer", "preview_invalid_config")
    if config["type"] == "spectra":
        spectra, dependencies, extra = _spectra(config, frames, base_dir, source_bytes)
    else:
        spectra, dependencies, extra = _images(config, frames, base_dir, output_dir)
    wavelength = extra.get("wavelength_A", config.get("wavelength_A"))
    if axis == "two_theta_deg":
        wavelength = _positive(wavelength, "wavelength_A")
    windows = config.get("windows")
    if windows is None:
        windows = _automatic_windows(spectra, max_peaks, snr_min)
    if not isinstance(windows, list):
        raise InputError("windows must be a list", "preview_invalid_windows")
    ids = set()
    for w in windows:
        if not isinstance(w, dict) or not isinstance(w.get("id"), str) or not w["id"] or w["id"] in ids:
            raise InputError("Every window needs a unique peak id", "preview_invalid_windows")
        if not all(isinstance(w.get(k), (int, float)) and np.isfinite(w[k]) for k in ("lower", "upper")) or w["lower"] >= w["upper"]:
            raise InputError("Window bounds must be finite and increasing", "preview_invalid_windows")
        ids.add(w["id"])
    rows, signals, issues, baselines = [], [], [], {}
    if not windows:
        issues.append({"code": "preview_no_common_peaks", "message": "No reliable common peaks found in representative frames"})
    for window in windows:
        overlap = any(other is not window and max(window["lower"], other["lower"]) <= min(window["upper"], other["upper"]) for other in windows)
        peak_rows = []
        previous = None
        for frame, (x, y) in zip(frames, spectra):
            fit = _invalid("overlapping_windows") if overlap else fit_peak(x, y, window, snr_min=snr_min)
            if fit["status"] == "valid":
                if previous is not None and abs(fit["center"]-previous) > .3*(window["upper"]-window["lower"]):
                    fit["status"], fit["reason"] = "invalid", "unstable_tracking"
                else:
                    previous = fit["center"]
            d, sigma = _to_d(fit["center"], fit["center_sigma"], axis, wavelength) if fit["status"] == "valid" else (None, None)
            if fit["status"] == "valid" and d is None:
                fit["status"], fit["reason"] = "invalid", "nonphysical_position"
            peak_frame = {key: value for key, value in frame.items() if key not in _FRAME_PROVENANCE_FIELDS}
            peak_rows.append({**peak_frame, **fit, "peak_id": window["id"], "axis": axis, "d_A": d,
                              "d_sigma_A": sigma, "strain": None})
        baseline = [peak_rows[i] for i in baseline_indices]
        good = [row for row in baseline if row["status"] == "valid"]
        # Invalid selected baseline observations cannot silently change the reference state.
        baseline_ok = len(good) == len(baseline)
        if baseline_ok:
            ds = np.asarray([row["d_A"] for row in good])
            d0 = float(np.median(ds))
            spread = float(np.max(np.abs(ds-d0)))
            baseline_ok = spread <= max(d0*5e-4, 5*max(row["d_sigma_A"] for row in good))
        if not baseline_ok:
            d0 = None
            issues.append({"code": "preview_ambiguous_baseline", "peak_id": window["id"], "message": "Baseline has invalid peaks or inconsistent lattice spacings"})
        baselines[window["id"]] = d0
        for row in peak_rows:
            if row["status"] == "valid":
                if d0 is None:
                    row["status"], row["reason"] = "invalid", "ambiguous_baseline"
                else:
                    row["strain"] = (row["d_A"]-d0)/d0
            if row["status"] != "valid":
                issues.append({"code": "preview_invalid_feature", "peak_id": window["id"], "frame": _identity(row), "reason": row["reason"]})
        rows.extend(peak_rows)
        signals.append({"id": window["id"], "t_s": [f["t_mid_s"] for f in frames],
                        "values": [r["strain"] for r in peak_rows], "valid": [r["status"] == "valid" for r in peak_rows]})
    return {"rows": rows, "signals": signals, "issues": issues,
            "metadata": {"axis": axis, "windows": windows, "baseline_frames": [_identity(frames[i]) for i in baseline_indices],
                         "d0_A": baselines, "dependencies": dependencies, "phase_assignment": "unassigned", **extra}}
