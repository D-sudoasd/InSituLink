"""Exact time origins with small floating-point coordinates for computation.

Public timestamps may be large UTC seconds. Subtract their decimal origins
before converting to float so nanosecond acquisition intervals remain usable.
"""

from decimal import Decimal, InvalidOperation, localcontext

from .errors import InputError


def decimal_time(value, label="time") -> Decimal:
    """Read a finite decimal value without first rounding it to binary float."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise InputError(f"{label} must be numeric", "invalid_timestamp") from exc
    if not result.is_finite():
        raise InputError(f"{label} must be finite", "invalid_timestamp")
    return result


def time_relative_s(value_s, origin_s) -> float:
    """Convert a public-coordinate time to seconds relative to an exact origin."""
    with localcontext() as context:
        context.prec = 50
        return float(decimal_time(value_s) - decimal_time(origin_s, "time origin"))


def clock_offset_local_s(dt_s, frame_origin_s, source_origin_s) -> float:
    """Preserve source = frame + dt when both clocks use local coordinates."""
    with localcontext() as context:
        context.prec = 50
        return float(decimal_time(dt_s, "offset") + decimal_time(frame_origin_s)
                     - decimal_time(source_origin_s))


def clock_offset_public_s(local_dt_s, frame_origin_s, source_origin_s) -> float:
    """Return the existing public offset convention from a local-coordinate fit."""
    with localcontext() as context:
        context.prec = 50
        return float(decimal_time(local_dt_s, "offset") - decimal_time(frame_origin_s)
                     + decimal_time(source_origin_s))
