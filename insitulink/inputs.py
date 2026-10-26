"""Read frame clocks and external channels without opening detector pixels."""

from __future__ import annotations

import csv
import io
import json
import math
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path

import numpy as np

from .errors import InputError
from .time_coordinates import decimal_time


_UNITS = {
    "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
    "ms": 1e-3, "millisecond": 1e-3, "milliseconds": 1e-3,
    "us": 1e-6, "µs": 1e-6, "microsecond": 1e-6, "microseconds": 1e-6,
    "ns": 1e-9, "nanosecond": 1e-9, "nanoseconds": 1e-9,
    "min": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hour": 3600.0, "hours": 3600.0,
}
_MAX_METADATA_ROWS = 10_000_000


def _path(value, base_dir):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(base_dir) / path).resolve(strict=False)


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _missing(value):
    return value is None or (isinstance(value, str) and not value.strip())


def canonical_source_file(value, base_dir) -> str | None:
    """Resolve a declared raw source using the frame reader's missing semantics."""
    return None if _missing(value) else str(_path(value, base_dir))


def _number(value, label, *, missing=False):
    if _missing(value) and missing:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputError(f"{label} must be numeric", value=str(value)) from exc
    if not math.isfinite(result):
        if missing:
            return None
        raise InputError(f"{label} must be finite")
    return result


def _factor(unit):
    key = _text(unit).strip().lower()
    if key not in _UNITS:
        raise InputError(f"Unsupported time unit: {unit}", code="invalid_time_unit")
    return _UNITS[key]


def _iso_decimal(value, settings):
    fraction = Decimal(0)
    if isinstance(value, datetime):
        stamp = value
    else:
        text = _text(value).strip().replace("Z", "+00:00")
        # datetime retains only microseconds; preserve the complete ISO fraction.
        part = re.search(r"[Tt ]\d{2}:?\d{2}:?\d{2}([.,]\d+)", text)
        if part:
            fraction = Decimal("0." + part[1][1:])
            text = text[:part.start(1)] + text[part.end(1):]
        try:
            stamp = datetime.fromisoformat(text)
        except (TypeError, ValueError) as exc:
            raise InputError(f"Invalid ISO8601 timestamp: {value}", code="invalid_timestamp") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        zone = settings.get("timezone")
        if zone in ("Z", "UTC", "+00:00"):
            tz = timezone.utc
        elif isinstance(zone, str) and re.fullmatch(r"[+-]\d{2}:\d{2}", zone):
            hours, minutes = map(int, zone[1:].split(":"))
            if hours > 23 or minutes > 59:
                raise InputError(f"Invalid timezone offset: {zone}")
            tz = timezone((1 if zone[0] == "+" else -1) * timedelta(hours=hours, minutes=minutes))
        else:
            raise InputError("ISO8601 timestamps require an encoded timezone or time.timezone", code="missing_timezone")
        stamp = stamp.replace(tzinfo=tz)
    delta = stamp.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    with localcontext() as context:
        context.prec = 50
        return Decimal(delta.days * 86400 + delta.seconds) + Decimal(delta.microseconds) / 1_000_000 + fraction


def _iso(value, settings):
    return float(_iso_decimal(value, settings))


def _time_settings(config, attrs=None):
    attrs = attrs or {}
    result = {}
    sources = {}
    if "units" in attrs:
        units = _text(attrs["units"])
        match = re.match(r"^(.+?)\s+since\s+(.+)$", units, re.I)
        if match:
            result.update(unit=match[1], start=match[2])
            sources.update(unit="timestamp_dataset.units", start="timestamp_dataset.units")
        else:
            result["unit"] = units
            sources["unit"] = "timestamp_dataset.units"
    for name in ("start", "start_time", "scaling_factor", "timezone"):
        if name in attrs:
            result["start" if name == "start_time" else name] = attrs[name]
            sources["start" if name == "start_time" else name] = "timestamp_dataset." + name
    requested = config.get("time") or {}
    if not isinstance(requested, dict):
        raise InputError("time settings must be an object")
    result.update(requested)
    sources.update({name: "request.time." + name for name in requested})
    result.setdefault("unit", "s")
    sources.setdefault("unit", "default_seconds")
    _factor(result["unit"])
    scale = decimal_time(result.get("scaling_factor", 1), "time.scaling_factor")
    if scale <= 0:
        raise InputError("time.scaling_factor must be positive")
    result["scaling_factor"] = float(scale)
    result["scaling_factor_exact"] = str(scale)
    sources.setdefault("scaling_factor", "default_one")
    result["sources"] = sources
    result["raw_attributes"] = {name: _raw_value(attrs[name]) for name in
                                ("units", "start", "start_time", "scaling_factor", "timezone") if name in attrs}
    result["requested"] = {name: _raw_value(value) for name, value in requested.items()}
    return result


def _exact_times(values, settings):
    factor = Decimal(str(_factor(settings["unit"]))) * decimal_time(settings.get("scaling_factor_exact", settings["scaling_factor"]))
    declared_origin = settings.get("start") is not None
    origin = _iso_decimal(settings["start"], settings) if declared_origin else Decimal(0)
    result = []
    numeric_rows, iso_rows = [], []
    for index, value in enumerate(values):
        if _missing(value):
            result.append(None)
            continue
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, datetime):
            result.append(_iso_decimal(value, settings))
            iso_rows.append(index)
            continue
        try:
            number = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            result.append(_iso_decimal(value, settings))
            iso_rows.append(index)
        else:
            if number.is_finite():
                with localcontext() as context:
                    context.prec = 50
                    converted = number * factor + origin
                result.append(converted if math.isfinite(float(converted)) else None)
                numeric_rows.append(index)
            else:
                result.append(None)
    if numeric_rows and iso_rows and not declared_origin:
        raise InputError("Mixed numeric and ISO timestamps require time.start for the numeric clock origin",
                         "mixed_time_basis", numeric_rows=numeric_rows, iso_rows=iso_rows)
    return result, "utc_epoch" if declared_origin or iso_rows else "relative"


def _times(values, settings):
    exact, basis = _exact_times(values, settings)
    return [None if value is None else float(value) for value in exact], basis


def _raw_value(value):
    """Keep original scalar representations JSON-safe, including missing markers."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _time_provenance(settings, origin, field):
    return {"source": origin, "field": field, "unit": _text(settings["unit"]),
            "declared_start": _raw_value(settings.get("start")),
            "timezone": _raw_value(settings.get("timezone")),
            "scaling_factor": str(settings.get("scaling_factor_exact", settings["scaling_factor"])),
            "setting_sources": dict(settings.get("sources", {})),
            "raw_attributes": dict(settings.get("raw_attributes", {})),
            "requested": dict(settings.get("requested", {})),
            "coordinate_origin": "first_finite_timestamp_or_reconstructed_frame",
            "precision_note": "t_*_s/t_s are approximate public coordinates; use time_origin_s with local coordinates for computation."}


def _records(config, base_dir, *, with_row_numbers=False, table_bytes=None):
    row_numbers = None
    if "data" in config:
        rows, origin = config["data"], "inline"
    elif config.get("path"):
        path = _path(config["path"], base_dir)
        origin = str(path)
        try:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                with (io.StringIO(table_bytes.decode("utf-8-sig"), newline="") if table_bytes is not None
                      else path.open("r", encoding="utf-8-sig", newline="")) as stream:
                    reader = csv.reader(stream)
                    header = next(reader, [])
                    empty = [i for i, key in enumerate(header) if _missing(key)]
                    duplicates = list(dict.fromkeys(key for i, key in enumerate(header) if key in header[:i]))
                    if not header or empty or duplicates:
                        raise InputError("CSV requires unique, nonempty column names; "
                                         f"duplicate columns: {duplicates}; empty column positions: {empty}",
                                         code="invalid_table_header", duplicate_columns=duplicates,
                                         empty_column_positions=empty, source_file=str(path))
                    rows, row_numbers = [], []
                    while True:
                        number = reader.line_num + 1
                        values = next(reader, None)
                        if values is None:
                            break
                        if not values:
                            continue
                        row = dict(zip(header, values))
                        row.update({key: None for key in header[len(values):]})
                        if len(values) > len(header):
                            row[None] = values[len(header):]
                        rows.append(row)
                        row_numbers.append(number)
            elif suffix == ".json":
                with (io.StringIO(table_bytes.decode("utf-8-sig")) if table_bytes is not None
                      else path.open("r", encoding="utf-8-sig")) as stream:
                    rows = json.load(stream, parse_float=Decimal)
            elif suffix == ".xlsx":
                try:
                    import openpyxl
                except ImportError as exc:
                    raise InputError("Reading XLSX requires openpyxl", code="missing_dependency") from exc
                book = openpyxl.load_workbook(io.BytesIO(table_bytes) if table_bytes is not None else path,
                                              read_only=True, data_only=True)
                try:
                    sheet = book[config["sheet"]] if "sheet" in config else book.active
                    data = iter(sheet.iter_rows(values_only=True))
                    header = next(data, ())
                    if any(_missing(key) for key in header) or len(set(header)) != len(header):
                        raise InputError("XLSX requires unique, nonempty column names")
                    kept = [(number, row) for number, row in enumerate(data, 2) if any(not _missing(v) for v in row)]
                    rows = [dict(zip(header, row)) for _, row in kept]
                    row_numbers = [number for number, _ in kept]
                finally:
                    book.close()
            else:
                raise InputError(f"Unsupported table extension: {suffix}")
        except (OSError, ValueError, KeyError) as exc:
            if isinstance(exc, InputError):
                raise
            raise InputError(f"Cannot read table {path}: {exc}") from exc
    else:
        raise InputError("A table path or inline data is required", code="missing_input")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise InputError("Table data must be a nonempty list of records")
    if with_row_numbers:
        return rows, origin, row_numbers if row_numbers is not None else list(range(1, len(rows) + 1))
    return rows, origin


def _period(config):
    period = config.get("constant_period")
    if not period:
        return None
    if not isinstance(period, dict):
        raise InputError("constant_period must be an object")
    if period.get("confirmed") is not True:
        return None
    exposure = decimal_time(period.get("exposure_s"), "constant_period.exposure_s")
    interval = decimal_time(period.get("interval_s"), "constant_period.interval_s")
    start = decimal_time(period.get("start_s", 0), "constant_period.start_s")
    if exposure < 0 or interval <= 0 or interval < exposure:
        raise InputError("Confirmed constant period requires interval_s >= exposure_s >= 0 and interval_s > 0")
    return exposure, interval, start


def _optional_decimal(value, label):
    if _missing(value):
        return None
    try:
        result = Decimal(_text(value))
    except InvalidOperation as exc:
        raise InputError(f"{label} must be numeric") from exc
    return result if result.is_finite() else None


def _frame_rows(records, config, base_dir, origin, *, starts=None, exposures=None, settings=None,
                row_numbers=None, raw_exposures=None, exposure_provenance=None):
    columns = {"start": "t_start_s", "exposure": "exposure_s", "frame_index": "frame_index",
               "source_file": "source_file", "dataset": "dataset", "id": "frame_id"}
    columns.update(config.get("columns") or {})
    settings = settings or _time_settings(config)
    raw_base_dir = _path(config["source_root"], base_dir) if config.get("source_root") else base_dir
    raw_starts = starts if starts is not None else [row.get(columns["start"]) for row in records]
    starts, basis = _exact_times(raw_starts, settings)
    row_numbers = row_numbers if row_numbers is not None else list(range(1, len(records) + 1))
    if exposures is None:
        # exposure_s is always seconds; an explicitly mapped exposure column uses time.unit.
        exposure_unit = config.get("exposure_unit", settings["unit"] if columns["exposure"] != "exposure_s" else "s")
        raw_exposures = [row.get(columns["exposure"]) for row in records]
        exposures = [_optional_decimal(value, f"row {i} exposure") for i, value in enumerate(raw_exposures)]
        exposures = [None if value is None else value * Decimal(str(_factor(exposure_unit))) for value in exposures]
        exposure_provenance = {"field": columns["exposure"], "unit": _text(exposure_unit),
                               "unit_source": "request.exposure_unit" if "exposure_unit" in config else
                               "time.unit" if columns["exposure"] != "exposure_s" else "default_seconds",
                               "scaling_factor": "1"}
    period = _period(config)
    if not any(value is not None for value in starts) and period is None:
        raise InputError("Frame timestamps are required; reconstruction requires constant_period.confirmed=true", code="missing_frame_times")
    result, reconstructed, exact_starts, exact_exposures = [], [], [], []
    reconstruction_origin = (_iso_decimal(settings["start"], settings)
                             if settings.get("start") is not None else Decimal(0))
    start_field = (config.get("start_dataset", "/entry/instrument/detector/start_time")
                   if config.get("type") == "hdf5" else columns["start"])
    provenance = _time_provenance(settings, origin, start_field)
    provenance["sheet"] = config.get("sheet")
    if (period is not None and basis == "utc_epoch" and settings.get("start") is None
            and any(value is None for value in starts) and "start_s" not in config["constant_period"]):
        raise InputError("Reconstructing missing ISO frame times requires time.start or an explicit epoch constant_period.start_s",
                         "mixed_time_basis")
    for i, (row, start, exposure) in enumerate(zip(records, starts, exposures)):
        source = canonical_source_file(row.get(columns["source_file"]), raw_base_dir)
        if source is not None and _missing(row.get(columns["frame_index"])):
            raise InputError(f"row {i} referencing a raw source requires an explicit file-local frame_index",
                             code="missing_frame_index", row=i, source_file=source,
                             frame_index_column=columns["frame_index"])
        index = _number(row.get(columns["frame_index"], i), f"row {i} frame_index")
        if index < 0 or index != int(index):
            raise InputError(f"row {i} frame_index must be a nonnegative integer")
        index = int(index)
        was_reconstructed = start is None and period is not None
        raw_time_missing = start is None
        if was_reconstructed:
            with localcontext() as context:
                context.prec = 50
                start = reconstruction_origin + period[2] + index * period[1]
            reconstructed.append(i)
        if exposure is None and period is not None:
            exposure = period[0]
        if exposure is not None and (not exposure.is_finite() or exposure < 0):
            raise InputError(f"row {i} exposure must be finite and nonnegative")
        dataset = row.get(columns["dataset"])
        item = {name: _raw_value(value) for name, value in row.items()}
        # CSV numerical features remain usable by the feature-alignment stage.
        for name, value in row.items():
            if isinstance(value, Decimal) or (isinstance(value, str) and value.strip()):
                try:
                    item[name] = float(value)
                except ValueError:
                    pass
        row_provenance = {**provenance, "raw_value": _raw_value(raw_starts[i]),
                          "source_row_number": row_numbers[i],
                          "reconstructed": was_reconstructed,
                          "exposure": {**(exposure_provenance or {}),
                                       "raw_value": _raw_value(raw_exposures[i]) if raw_exposures is not None else None}}
        if period is not None:
            row_provenance["constant_period"] = {key: _raw_value(value) for key, value in config["constant_period"].items()}
        with localcontext() as context:
            context.prec = 50
            end = None if start is None or exposure is None else start + exposure
            mid = None if start is None or exposure is None else start + exposure / 2
        item.update(source_file=source,
                    source_file_declared=_raw_value(row.get(columns["source_file"])),
                    dataset_declared=_raw_value(dataset),
                    dataset=None if _missing(dataset) else str(dataset), frame_index=index,
                    frame_number=index + 1, original_id=_raw_value(row.get(columns["id"], row.get("original_id"))),
                    t_start_s=None if start is None else float(start),
                    exposure_s=None if exposure is None else float(exposure),
                    t_end_s=None if end is None else float(end),
                    t_mid_s=None if mid is None else float(mid),
                    source_row_number=row_numbers[i], time_provenance=row_provenance,
                    clock_source="confirmed_constant_period" if was_reconstructed else origin,
                    time_missing=start is None, raw_time_missing=raw_time_missing)
        result.append(item)
        exact_starts.append(start)
        exact_exposures.append(exposure)
    coordinate_origin = next((value for value in exact_starts if value is not None), reconstruction_origin)
    with localcontext() as context:
        context.prec = 50
        for row, start, exposure in zip(result, exact_starts, exact_exposures):
            relative = None if start is None else start - coordinate_origin
            row.update(time_origin_s=str(coordinate_origin),
                       t_start_rel_s=None if relative is None else float(relative),
                       t_end_rel_s=None if relative is None or exposure is None else float(relative + exposure),
                       t_mid_rel_s=None if relative is None or exposure is None else float(relative + exposure / 2))
    finite = [(i, value) for i, value in enumerate(exact_starts) if value is not None]
    decreases = [b[0] for a, b in zip(finite, finite[1:]) if b[1] < a[1]]
    seen, duplicates = set(), []
    for i, value in finite:
        if value in seen:
            duplicates.append(i)
        seen.add(value)
    return result, {"source": origin, "time_basis": basis, "n_input_rows": len(records), "n_frames": len(result),
                    "missing_time_indices": [row["frame_index"] for row in result if row["time_missing"]],
                    "nonmonotonic_rows": decreases, "duplicate_time_rows": duplicates,
                    "reconstructed_rows": reconstructed, "reconstructed": bool(reconstructed),
                    "reconstructed_frame_indices": [result[i]["frame_index"] for i in reconstructed],
                    "time_origin_s": str(coordinate_origin), "source_row_numbers": row_numbers,
                    "time_provenance": provenance, "time": dict(settings)}


def _hdf5_frames(config, base_dir):
    try:
        import h5py
    except ImportError as exc:
        raise InputError("Reading HDF5 requires h5py", code="missing_dependency") from exc
    if not config.get("path"):
        raise InputError("HDF5 frames require path")
    path = _path(config["path"], base_dir)
    data_name = config.get("data_dataset", "/entry/data/data")
    start_name = config.get("start_dataset", "/entry/instrument/detector/start_time")
    exposure_name = config.get("exposure_dataset", "/entry/instrument/detector/count_time")
    try:
        with h5py.File(path, "r") as handle:
            data = handle.get(data_name)
            if not isinstance(data, h5py.Dataset) or data.ndim not in (2, 3):
                raise InputError("Detector data must be a 2D image or a 3D frame stack",
                                 source_file=str(path), dataset=data_name)
            data_count = 1 if data.ndim == 2 else int(data.shape[0])
            count_details = {"source_file": str(path), "dataset": data_name,
                             "data_shape": list(data.shape), "data_frame_count": data_count}
            requested_count = config.get("frame_count", data_count)
            if "frame_count" in config:
                requested_count = _number(requested_count, "frame_count")
                if requested_count < 1 or requested_count != int(requested_count):
                    raise InputError("frame_count must be a positive integer")
                requested_count = int(requested_count)
                if requested_count > data_count:
                    raise InputError("frame_count exceeds the physical detector frame count",
                                     code="frame_count_conflict", conflict_source="frame_count",
                                     frame_count=requested_count, **count_details)
            stamp = handle.get(start_name)
            attrs = dict(stamp.attrs) if isinstance(stamp, h5py.Dataset) else {}
            settings = _time_settings(config, attrs)
            if stamp is not None:
                if not isinstance(stamp, h5py.Dataset) or stamp.ndim > 1 or stamp.size > _MAX_METADATA_ROWS:
                    raise InputError("Frame timestamp dataset must be a bounded scalar or vector")
                raw_starts = np.asarray(stamp[()]).reshape(-1).tolist()
                starts, _ = _exact_times(raw_starts, settings)
            else:
                raw_starts, starts = [], []
            period = _period(config)
            finite = [i for i, value in enumerate(starts) if value is not None]
            if finite and finite[-1] >= data_count:
                raise InputError("Frame timestamps reference indices beyond the physical detector frame count",
                                 code="frame_count_conflict", conflict_source="timestamps",
                                 timestamp_count=len(starts), last_finite_timestamp_index=finite[-1],
                                 invalid_timestamp_indices=[i for i in finite if i >= data_count],
                                 **count_details)
            dropped = []
            frame_config = config
            count_source = "frame_count" if "frame_count" in config else "finite_timestamps"
            if (stamp is not None and stamp.ndim == 0 and finite and requested_count != 1
                    and (period is not None or "frame_count" not in config)):
                if period is None:
                    raise InputError("A scalar acquisition start requires confirmed constant-period reconstruction for multiple frames",
                                     code="missing_frame_times")
                count = _number(requested_count, "frame_count")
                if count < 1 or count != int(count) or count > _MAX_METADATA_ROWS:
                    raise InputError("Confirmed reconstruction requires a bounded positive frame_count")
                count = int(count)
                frame_config = dict(config, constant_period=dict(config["constant_period"]))
                origin = _iso_decimal(settings["start"], settings) if settings.get("start") is not None else Decimal(0)
                with localcontext() as context:
                    context.prec = 50
                    frame_config["constant_period"]["start_s"] = str(starts[0] - origin)
                raw_starts = raw_starts + [None] * (count - 1)
                count_source = "frame_count" if "frame_count" in config else "data_dataset"
            elif finite:
                count = requested_count if "frame_count" in config else finite[-1] + 1
                if count > _MAX_METADATA_ROWS:
                    raise InputError("Frame metadata requires a bounded positive frame_count")
                dropped = list(range(count, len(starts)))
                raw_starts = raw_starts[:count] + [None] * max(0, count - len(raw_starts))
            elif period is not None:
                count = requested_count
                count = _number(count, "frame_count")
                if count < 1 or count != int(count) or count > _MAX_METADATA_ROWS:
                    raise InputError("Confirmed reconstruction requires a bounded positive frame_count")
                count = int(count)
                raw_starts = [None] * count
                count_source = "frame_count" if "frame_count" in config else "data_dataset"
            else:
                raise InputError("No finite frame timestamps in HDF5; confirmed reconstruction is required", code="missing_frame_times")
            exposure = handle.get(exposure_name)
            exposure_count = None
            exposure_unit = config.get("exposure_unit", "s")
            exposure_unit_source = "config.exposure_unit" if "exposure_unit" in config else "default_seconds"
            exposure_scale = 1.0
            if exposure is None:
                exposures = [None] * count
                raw_exposures = [None] * count
            else:
                if not isinstance(exposure, h5py.Dataset) or exposure.ndim > 1 or exposure.size > _MAX_METADATA_ROWS:
                    raise InputError("Exposure dataset must be a bounded scalar or vector")
                values = np.asarray(exposure[()]).reshape(-1).tolist()
                exposure_count = len(values)
                if "exposure_unit" not in config and "units" in exposure.attrs:
                    exposure_unit = exposure.attrs["units"]
                    exposure_unit_source = "exposure_dataset.units"
                exposure_scale_exact = decimal_time(exposure.attrs.get("scaling_factor", 1), "exposure scaling_factor")
                exposure_scale = float(exposure_scale_exact)
                factor = Decimal(str(_factor(exposure_unit))) * exposure_scale_exact
                raw_exposures = values * count if len(values) == 1 else (values + [None] * max(0, count - len(values)))[:count]
                values = [_optional_decimal(value, "exposure") for value in values]
                exposures = [None if value is None else value * factor for value in values]
                exposures = exposures * count if len(exposures) == 1 else (exposures + [None] * max(0, count - len(exposures)))[:count]
            records = [{"source_file": str(path), "dataset": data_name, "frame_index": i} for i in range(count)]
            # Retain raw values so timestamp units/origin are applied exactly once.
            rows, metadata = _frame_rows(records, frame_config, base_dir, f"{path}:{start_name}",
                                          starts=raw_starts[:count], exposures=exposures, settings=settings,
                                          raw_exposures=raw_exposures,
                                          exposure_provenance={"field": exposure_name, "unit": _text(exposure_unit),
                                              "unit_source": exposure_unit_source,
                                              "raw_unit": _raw_value(exposure.attrs.get("units")) if exposure is not None else None,
                                              "scaling_factor": str(exposure_scale)})
            mismatches = {}
            for name, value in (("data_dataset", data_count), ("frame_count", config.get("frame_count")), ("exposure_dataset", exposure_count)):
                if value is not None and value != count and not (name == "exposure_dataset" and value == 1):
                    mismatches[name] = {"declared_count": value, "retained_count": count}
            metadata.update(n_input_rows=len(starts) if starts else count, timestamp_count=len(starts),
                            data_dataset=data_name, data_frame_count=data_count,
                            dropped_padded_indices=dropped, count_mismatches=mismatches,
                            frame_count_source=count_source,
                            observed_timestamp_frame_count=finite[-1] + 1 if finite else 0,
                            frame_count_unresolved=count < data_count and "frame_count" not in config,
                            exposure_unit=_text(exposure_unit), exposure_unit_source=exposure_unit_source,
                            exposure_scaling_factor=exposure_scale)
            if metadata["frame_count_unresolved"]:
                metadata["frame_count_unresolved_reason"] = (
                    "Physical detector frame count exceeds the last finite timestamp; "
                    "supply frame_count and clock evidence for missing timestamps")
            return rows, metadata
    except (OSError, KeyError, ValueError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError(f"Cannot read HDF5 metadata {path}: {exc}") from exc


def read_frames(config: dict, base_dir: Path, *, table_bytes=None) -> tuple[list[dict], dict]:
    """Return acquisition-order frames and explicit clock-quality metadata."""
    if config.get("type", "table") == "hdf5":
        return _hdf5_frames(config, base_dir)
    if config.get("type", "table") != "table":
        raise InputError("Frame type must be table or hdf5")
    rows, origin, row_numbers = _records(config, base_dir, with_row_numbers=True, table_bytes=table_bytes)
    return _frame_rows(rows, config, base_dir, origin, row_numbers=row_numbers)


def read_source(config: dict, base_dir: Path, *, table_bytes=None) -> dict:
    """Return a strictly increasing external clock and independently typed channels."""
    rows, origin, row_numbers = _records(config, base_dir, with_row_numbers=True, table_bytes=table_bytes)
    source_id = config.get("id")
    if not isinstance(source_id, str) or not source_id.strip():
        raise InputError("External source requires a nonempty id")
    time_column = config.get("time_column", "time")
    if any(time_column not in row for row in rows):
        raise InputError(f"Missing external time column: {time_column}")
    settings = _time_settings(config)
    raw_times = [row[time_column] for row in rows]
    exact_times, basis = _exact_times(raw_times, settings)
    if any(value is None for value in exact_times) or any(b <= a for a, b in zip(exact_times, exact_times[1:])):
        raise InputError("External timestamps must be finite and strictly increasing", code="invalid_external_time")
    coordinate_origin = exact_times[0]
    with localcontext() as context:
        context.prec = 50
        relative_times = [float(value - coordinate_origin) for value in exact_times]
    provenance = _time_provenance(settings, origin, time_column)
    provenance["sheet"] = config.get("sheet")
    provenance.update(raw_values=[_raw_value(value) for value in raw_times], source_row_numbers=row_numbers)
    definitions = config.get("channels")
    if not isinstance(definitions, list) or not definitions:
        raise InputError("External source requires at least one channel")
    channels = {}
    for channel in definitions:
        if not isinstance(channel, dict):
            raise InputError("Channel definitions must be objects")
        name = channel.get("name")
        if not isinstance(name, str) or not name.strip() or name in channels:
            raise InputError("Channel names must be nonempty and unique")
        unit = channel.get("unit")
        if not isinstance(unit, str) or not unit.strip():
            raise InputError(f"Channel {name} requires a declared unit (use '1' for dimensionless)")
        kind = channel.get("kind", "continuous")
        if kind not in ("continuous", "step"):
            raise InputError(f"Channel {name} kind must be continuous or step")
        column = channel.get("column", name)
        if any(column not in row for row in rows):
            raise InputError(f"Missing channel column: {column}")
        values = []
        for i, row in enumerate(rows):
            value = row[column]
            if _missing(value):
                values.append(np.nan)
            else:
                try:
                    values.append(float(value))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise InputError(f"Channel {name}, row {i} must be numeric") from exc
        result = dict(channel, values=np.asarray(values, dtype=float), unit=unit, kind=kind,
                      required=channel.get("required", True))
        if not isinstance(result["required"], bool):
            raise InputError(f"Channel {name} required must be boolean")
        for option in ("max_gap_s", "phase_tolerance"):
            if option in result and result[option] is not None:
                result[option] = _number(result[option], f"Channel {name} {option}")
                if result[option] < 0 or (option == "max_gap_s" and result[option] == 0):
                    raise InputError(f"Channel {name} {option} must be positive for max_gap_s or nonnegative for phase_tolerance")
        channels[name] = result
    return {"id": source_id, "t_s": np.asarray(exact_times, dtype=float),
            "time_origin_s": str(coordinate_origin), "t_rel_s": np.asarray(relative_times, dtype=float),
            "source_row_numbers": row_numbers, "channels": channels,
            "clock": dict(config.get("clock") or {}),
            "metadata": {"source": origin, "n_rows": len(rows), "time_basis": basis, "time": settings,
                         "time_origin_s": str(coordinate_origin), "time_provenance": provenance}}


def referenced_paths(config: dict, base_dir: Path, *, table_bytes=None) -> list[Path]:
    """List declared inputs without opening detector files or interpreting their pixels."""
    paths = []
    if config.get("path"):
        paths.append(_path(config["path"], base_dir))
    if config.get("type", "table") == "table":
        rows, _ = _records(config, base_dir, table_bytes=table_bytes)
        column = (config.get("columns") or {}).get("source_file", "source_file")
        raw_base_dir = _path(config["source_root"], base_dir) if config.get("source_root") else base_dir
        paths.extend(_path(row[column], raw_base_dir) for row in rows if not _missing(row.get(column)))
    return list(dict.fromkeys(paths))
