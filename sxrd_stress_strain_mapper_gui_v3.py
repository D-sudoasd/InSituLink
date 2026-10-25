#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SXRD stress--strain mapper GUI
Version 3: added high-contrast plotting, unit warnings, and optional smoothing guides.

Purpose
-------
Use a complete reference stress--strain curve to complete missing stress or strain
values for in-situ synchrotron tensile datasets, where each row normally corresponds
to one spectrum / frame / acquisition point.

Typical use cases
-----------------
1) Beamline table has strain only  -> interpolate reference curve to obtain stress.
2) Beamline table has stress only  -> inverse-interpolate reference curve to obtain strain.
3) Beamline table already has both -> convert units, check and export aligned table.

Recommended interpolation
-------------------------
PCHIP shape-preserving interpolation is used when scipy is available. If scipy is not
installed, the app falls back to linear interpolation automatically.

Dependencies
------------
pip install pandas numpy matplotlib scipy openpyxl

Run
---
python sxrd_stress_strain_mapper_gui.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Microsoft YaHei", "SimHei"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

try:
    from scipy.interpolate import PchipInterpolator
    SCIPY_AVAILABLE = True
except Exception:
    PchipInterpolator = None
    SCIPY_AVAILABLE = False

try:
    from scipy.signal import savgol_filter
    SCIPY_SIGNAL_AVAILABLE = True
except Exception:
    savgol_filter = None
    SCIPY_SIGNAL_AVAILABLE = False


STRAIN_FRACTION = "fraction / 工程应变，例如 0.05"
STRAIN_PERCENT = "percent / 百分数，例如 5"
STRESS_MPA = "MPa"
STRESS_GPA = "GPa"
STRESS_PA = "Pa"

MODE_STRAIN_ONLY = "线站只有应变：由参考曲线插值得到应力"
MODE_STRESS_ONLY = "线站只有应力：由参考曲线反插值得到应变"
MODE_BOTH = "线站已有应力和应变：仅整合、换算、可视化"

METHOD_PCHIP = "PCHIP 形状保持插值（推荐）"
METHOD_LINEAR = "Linear 线性插值"

SMOOTH_NONE = "不平滑：只显示原始映射点"
SMOOTH_ROLLING_MEDIAN = "Rolling median 滚动中位数（抗离群点）"
SMOOTH_ROLLING_MEAN = "Rolling mean 滚动平均"
SMOOTH_SAVGOL = "Savitzky-Golay 平滑（保形，需 scipy）"

WIZARD_NEED_REFERENCE = "待加载参考"
WIZARD_NEED_STATION = "待加载线站"
WIZARD_NEED_CONFIRM = "待确认推荐"
WIZARD_READY = "可运行"
WIZARD_DONE = "已完成"

ALIGNMENT_FACTOR_WARN_MIN = 0.5
ALIGNMENT_FACTOR_WARN_MAX = 2.0

CONTROL_PANEL_WIDTH = 430
CONTROL_WRAP = 320

DEFAULT_PLOT_PRESET = "Publication"
PLOT_EXPORT_PRESETS = {
    "Raw inspection": {
        "description": "软件内快速检查：字号较小，导出 300 dpi。",
        "figsize": (8.6, 5.2),
        "dpi": 300,
        "font_size": 9,
        "title_size": 10,
        "legend_size": 8,
        "line_width": 1.4,
        "marker_size": 3.6,
        "scatter_size": 30,
        "axis_width": 0.8,
        "grid": True,
    },
    "Single-column figure": {
        "description": "论文单栏：约 89 mm 宽，600 dpi。",
        "figsize": (3.5, 3.0),
        "dpi": 600,
        "font_size": 8,
        "title_size": 9,
        "legend_size": 7,
        "line_width": 1.0,
        "marker_size": 2.8,
        "scatter_size": 22,
        "axis_width": 0.7,
        "grid": False,
    },
    "Double-column figure": {
        "description": "论文双栏：约 180 mm 宽，600 dpi。",
        "figsize": (7.1, 4.6),
        "dpi": 600,
        "font_size": 9,
        "title_size": 10,
        "legend_size": 8,
        "line_width": 1.2,
        "marker_size": 3.2,
        "scatter_size": 28,
        "axis_width": 0.8,
        "grid": False,
    },
    "Presentation": {
        "description": "汇报 PPT：字号和线宽更大，300 dpi。",
        "figsize": (10.0, 5.8),
        "dpi": 300,
        "font_size": 12,
        "title_size": 13,
        "legend_size": 10,
        "line_width": 2.0,
        "marker_size": 5.0,
        "scatter_size": 45,
        "axis_width": 1.0,
        "grid": True,
    },
    "Publication": {
        "description": "论文级默认：低饱和度配色、600 dpi、紧凑留白。",
        "figsize": (7.1, 4.8),
        "dpi": 600,
        "font_size": 9,
        "title_size": 10,
        "legend_size": 8,
        "line_width": 1.25,
        "marker_size": 3.2,
        "scatter_size": 30,
        "axis_width": 0.8,
        "grid": False,
    },
}

PLOT_COLORS = {
    "reference": "#1f2937",
    "mapped": "#b45309",
    "stress": "#2563eb",
    "strain": "#047857",
    "smooth": "#0f766e",
    "invalid": "#7c3aed",
    "axis": "#334155",
    "spine": "#475569",
    "grid": "#d7dee8",
}


# -----------------------------
# Basic data utilities
# -----------------------------

def _looks_numeric(value) -> bool:
    try:
        float(str(value).strip())
        return True
    except Exception:
        return False


def _numeric_column_names(columns) -> bool:
    """Detect headerless files accidentally read with the first numeric row as header."""
    if len(columns) == 0:
        return False
    n_num = sum(_looks_numeric(c) for c in columns)
    return n_num >= max(1, int(0.6 * len(columns)))


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop completely empty rows/columns
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    # Normalize column names
    df.columns = [str(c).strip() if str(c).strip() else f"col_{i+1}" for i, c in enumerate(df.columns)]
    return df


def read_table(path: str) -> pd.DataFrame:
    """Read csv/txt/dat/xlsx files with reasonable automatic delimiter/header handling."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(p)
        if _numeric_column_names(df.columns):
            df = pd.read_excel(p, header=None)
            df.columns = [f"col_{i+1}" for i in range(df.shape[1])]
        return _clean_dataframe(df)

    encodings = ["utf-8-sig", "utf-8", "gbk", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(p, sep=None, engine="python", encoding=enc)
            if df.shape[1] == 1:
                # Fallback for whitespace-separated numeric txt/dat files
                df = pd.read_csv(p, sep=r"\s+|,|;|\t", engine="python", encoding=enc)
            if _numeric_column_names(df.columns):
                df = pd.read_csv(p, sep=None, engine="python", encoding=enc, header=None)
                if df.shape[1] == 1:
                    df = pd.read_csv(p, sep=r"\s+|,|;|\t", engine="python", encoding=enc, header=None)
                df.columns = [f"col_{i+1}" for i in range(df.shape[1])]
            return _clean_dataframe(df)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"无法读取文件：{path}\n最后一次错误：{last_error}")


def to_numeric_series(df: pd.DataFrame, col: str, label: str) -> pd.Series:
    if not col or col not in df.columns:
        raise ValueError(f"请选择有效的 {label} 列。")
    s = pd.to_numeric(df[col], errors="coerce")
    return s


def _normalized_name(col) -> str:
    return str(col).strip().lower()


def _is_strain_name(col) -> bool:
    name = _normalized_name(col)
    return any(token in name for token in ["strain", "epsilon", "eps", "应变"])


def _is_stress_name(col) -> bool:
    name = _normalized_name(col)
    return any(token in name for token in ["stress", "sigma", "应力", "mpa", "gpa"])


def _is_id_name(col) -> bool:
    name = _normalized_name(col)
    return any(token in name for token in ["id", "frame", "spectrum", "scan", "index", "谱线", "帧", "编号", "序号"])


def _numeric_values(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.array([], dtype=float)
    arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _numeric_columns(df: pd.DataFrame, min_points: int = 1) -> list:
    cols = []
    for col in df.columns:
        if len(_numeric_values(df, col)) >= min_points:
            cols.append(col)
    return cols


def _abs_max(values) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    return float(np.nanmax(np.abs(finite)))


def recommend_strain_unit(values, column_name: str = "") -> Tuple[str, str]:
    name = _normalized_name(column_name)
    raw_max = _abs_max(values)
    if "%" in name or "percent" in name or "pct" in name:
        return STRAIN_PERCENT, "列名包含百分数信息，推荐 percent。"
    if "fraction" in name or "frac" in name:
        return STRAIN_FRACTION, "列名包含 fraction 信息，推荐 fraction。"
    if np.isfinite(raw_max) and raw_max > 1.0 and raw_max <= 100.0:
        return STRAIN_PERCENT, f"该列最大绝对值约 {raw_max:.4g}，更像百分数应变。"
    return STRAIN_FRACTION, f"该列最大绝对值约 {raw_max:.4g}，更像 fraction 应变。"


def recommend_stress_unit(values, column_name: str = "") -> Tuple[str, str]:
    name = _normalized_name(column_name)
    raw_max = _abs_max(values)
    if "gpa" in name:
        return STRESS_GPA, "列名包含 GPa，推荐 GPa。"
    if "mpa" in name:
        return STRESS_MPA, "列名包含 MPa，推荐 MPa。"
    if "pa" in name and "mpa" not in name and "gpa" not in name:
        return STRESS_PA, "列名包含 Pa，推荐 Pa。"
    if np.isfinite(raw_max) and raw_max > 100000:
        return STRESS_PA, f"该列最大绝对值约 {raw_max:.4g}，更像 Pa。"
    return STRESS_MPA, f"该列最大绝对值约 {raw_max:.4g}，默认按 MPa 处理。"


def _first_matching_numeric(df: pd.DataFrame, predicate, excluded=None) -> str:
    excluded = set(excluded or [])
    for col in _numeric_columns(df):
        if col not in excluded and predicate(col):
            return col
    return ""


def _fallback_reference_columns(df: pd.DataFrame, strain_col: str = "", stress_col: str = "") -> Tuple[str, str, str, str]:
    numeric_cols = _numeric_columns(df)
    strain_reason = ""
    stress_reason = ""
    if not strain_col and numeric_cols:
        scored = []
        for col in numeric_cols:
            max_abs = _abs_max(_numeric_values(df, col))
            if np.isfinite(max_abs):
                scored.append((max_abs, col))
        if scored:
            scored.sort(key=lambda item: item[0])
            strain_col = scored[0][1]
            strain_reason = f"未从列名识别到应变，选择数值范围较小的 {strain_col}。"
    if not stress_col:
        candidates = [col for col in numeric_cols if col != strain_col]
        if candidates:
            scored = [(_abs_max(_numeric_values(df, col)), col) for col in candidates]
            scored = [(score, col) for score, col in scored if np.isfinite(score)]
            if scored:
                scored.sort(key=lambda item: item[0], reverse=True)
                stress_col = scored[0][1]
                stress_reason = f"未从列名识别到应力，选择数值范围较大的 {stress_col}。"
    return strain_col, stress_col, strain_reason, stress_reason


def recommend_reference_setup(df: pd.DataFrame) -> dict:
    strain_col = _first_matching_numeric(df, _is_strain_name)
    stress_col = _first_matching_numeric(df, _is_stress_name, excluded=[strain_col])
    strain_reason = f"根据列名识别 {strain_col} 为参考应变。" if strain_col else ""
    stress_reason = f"根据列名识别 {stress_col} 为参考应力。" if stress_col else ""
    strain_col, stress_col, fallback_strain_reason, fallback_stress_reason = _fallback_reference_columns(df, strain_col, stress_col)
    strain_reason = strain_reason or fallback_strain_reason or "未能可靠推荐参考应变列，请手动确认。"
    stress_reason = stress_reason or fallback_stress_reason or "未能可靠推荐参考应力列，请手动确认。"
    strain_unit, strain_unit_reason = recommend_strain_unit(_numeric_values(df, strain_col), strain_col)
    stress_unit, stress_unit_reason = recommend_stress_unit(_numeric_values(df, stress_col), stress_col)
    return {
        "strain_col": strain_col,
        "stress_col": stress_col,
        "strain_unit": strain_unit,
        "stress_unit": stress_unit,
        "strain_reason": strain_reason,
        "stress_reason": stress_reason,
        "strain_unit_reason": strain_unit_reason,
        "stress_unit_reason": stress_unit_reason,
    }


def recommend_station_setup(df: pd.DataFrame) -> dict:
    id_col = _first_matching_numeric(df, _is_id_name)
    excluded = [id_col] if id_col else []
    strain_col = _first_matching_numeric(df, _is_strain_name, excluded=excluded)
    stress_col = _first_matching_numeric(df, _is_stress_name, excluded=excluded + ([strain_col] if strain_col else []))
    numeric_inputs = [col for col in _numeric_columns(df) if col != id_col]

    if not strain_col and not stress_col and numeric_inputs:
        candidate = numeric_inputs[0]
        max_abs = _abs_max(_numeric_values(df, candidate))
        if np.isfinite(max_abs) and max_abs > 100.0:
            stress_col = candidate
        else:
            strain_col = candidate

    if strain_col and stress_col:
        mode = MODE_BOTH
        mode_reason = "同时识别到应变列和应力列，推荐仅整合换算模式。"
    elif stress_col:
        mode = MODE_STRESS_ONLY
        mode_reason = "识别到应力输入，推荐由参考曲线反查应变。"
    else:
        mode = MODE_STRAIN_ONLY
        mode_reason = "识别到应变输入，推荐由参考曲线插值得到应力。"

    strain_unit, strain_unit_reason = recommend_strain_unit(_numeric_values(df, strain_col), strain_col)
    stress_unit, stress_unit_reason = recommend_stress_unit(_numeric_values(df, stress_col), stress_col)
    return {
        "mode": mode,
        "id_col": id_col,
        "strain_col": strain_col,
        "stress_col": stress_col,
        "strain_unit": strain_unit,
        "stress_unit": stress_unit,
        "mode_reason": mode_reason,
        "id_reason": f"根据列名识别 {id_col} 为谱线/帧编号。" if id_col else "未识别到明确编号列，将按行号生成 spectrum_id。",
        "strain_reason": f"根据列名或数值范围推荐 {strain_col} 为线站应变。" if strain_col else "当前模式不需要线站应变列。",
        "stress_reason": f"根据列名或数值范围推荐 {stress_col} 为线站应力。" if stress_col else "当前模式不需要线站应力列。",
        "strain_unit_reason": strain_unit_reason,
        "stress_unit_reason": stress_unit_reason,
    }


def get_wizard_state(has_reference: bool, has_station: bool, recommendation_confirmed: bool, has_result: bool) -> str:
    if has_result:
        return WIZARD_DONE
    if not has_reference:
        return WIZARD_NEED_REFERENCE
    if not has_station:
        return WIZARD_NEED_STATION
    if not recommendation_confirmed:
        return WIZARD_NEED_CONFIRM
    return WIZARD_READY


def convert_strain_to_fraction(values, unit: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if unit == STRAIN_PERCENT:
        return arr / 100.0
    return arr


def convert_stress_to_mpa(values, unit: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if unit == STRESS_GPA:
        return arr * 1000.0
    if unit == STRESS_PA:
        return arr / 1_000_000.0
    return arr


def _odd_window(window: int, n: int, minimum: int = 3) -> int:
    """Return a safe odd smoothing window not larger than n."""
    try:
        w = int(window)
    except Exception:
        w = minimum
    w = max(minimum, w)
    if w % 2 == 0:
        w += 1
    if n > 0 and w > n:
        w = n if n % 2 == 1 else n - 1
    return max(minimum, w)


def smooth_numeric_series(values, method: str, window: int = 7, polyorder: int = 2) -> np.ndarray:
    """Smooth a numeric sequence for visual guidance while preserving NaN positions.

    This function is intentionally conservative: it does not delete raw points and it
    returns NaN at locations where the original data are NaN/inf.
    """
    arr = np.asarray(values, dtype=float)
    out = arr.astype(float).copy()
    finite_mask = np.isfinite(arr)
    n_finite = int(finite_mask.sum())
    if method == SMOOTH_NONE or n_finite < 3:
        return np.full_like(arr, np.nan, dtype=float)

    # Temporarily interpolate internal NaNs so rolling/Savgol can run, then restore NaNs.
    s = pd.Series(arr, dtype="float64")
    filled = s.interpolate(limit_direction="both")

    if method == SMOOTH_ROLLING_MEDIAN:
        w = max(3, int(window))
        out = filled.rolling(window=w, center=True, min_periods=max(1, w // 2)).median().to_numpy(dtype=float)
    elif method == SMOOTH_ROLLING_MEAN:
        w = max(3, int(window))
        out = filled.rolling(window=w, center=True, min_periods=max(1, w // 2)).mean().to_numpy(dtype=float)
    elif method == SMOOTH_SAVGOL:
        if not SCIPY_SIGNAL_AVAILABLE:
            # Fallback: rolling median is safer than silently doing nothing.
            w = max(3, int(window))
            out = filled.rolling(window=w, center=True, min_periods=max(1, w // 2)).median().to_numpy(dtype=float)
        else:
            w = _odd_window(int(window), len(filled), minimum=3)
            po = max(1, int(polyorder))
            if po >= w:
                po = max(1, w - 1)
            out = savgol_filter(filled.to_numpy(dtype=float), window_length=w, polyorder=po, mode="interp")
    else:
        return np.full_like(arr, np.nan, dtype=float)

    out = np.asarray(out, dtype=float)
    out[~finite_mask] = np.nan
    return out


def finite_xy(x, y) -> pd.DataFrame:
    tmp = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)})
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    return tmp


def _finite_range(values, label: str) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError(f"{label} 没有有限数值，无法判断范围。")
    return float(np.min(finite)), float(np.max(finite))


def within_reference_ranges(eps_fraction, stress_mpa, ref_eps_fraction, ref_stress_mpa) -> np.ndarray:
    """Return True only where strain and stress are finite and inside reference ranges."""
    eps = np.asarray(eps_fraction, dtype=float)
    sig = np.asarray(stress_mpa, dtype=float)
    eps_min, eps_max = _finite_range(ref_eps_fraction, "参考应变")
    sig_min, sig_max = _finite_range(ref_stress_mpa, "参考应力")
    return (
        np.isfinite(eps)
        & np.isfinite(sig)
        & (eps >= eps_min)
        & (eps <= eps_max)
        & (sig >= sig_min)
        & (sig <= sig_max)
    )


def zero_to_first_finite(values) -> Tuple[np.ndarray, float]:
    """Subtract the first finite value while leaving NaN/inf entries unchanged."""
    arr = np.asarray(values, dtype=float)
    zeroed = arr.copy()
    finite_idx = np.flatnonzero(np.isfinite(arr))
    if finite_idx.size == 0:
        return zeroed, np.nan
    offset = float(arr[finite_idx[0]])
    zeroed[finite_idx] = zeroed[finite_idx] - offset
    return zeroed, offset


def _positive_finite_max(values, label: str) -> float:
    arr = np.asarray(values, dtype=float)
    finite_positive = arr[np.isfinite(arr) & (arr > 0)]
    if finite_positive.size == 0:
        raise ValueError(f"{label}缺少有效正应变。")
    return float(np.nanmax(finite_positive))


def _positive_finite_max_for_alignment(values, label: str, quantity: str) -> float:
    arr = np.asarray(values, dtype=float)
    finite_positive = arr[np.isfinite(arr) & (arr > 0)]
    if finite_positive.size == 0:
        raise ValueError(f"{label}缺少有效正{quantity}。")
    return float(np.nanmax(finite_positive))


def compute_strain_alignment_diagnostics(ref_eps_fraction, station_eps_fraction) -> dict:
    """Compute max-strain alignment diagnostics using finite positive strain only."""
    try:
        ref_max = _positive_finite_max(ref_eps_fraction, "参考曲线")
        station_max = _positive_finite_max(station_eps_fraction, "线站数据")
    except ValueError as exc:
        raise ValueError(f"无法计算应变对齐系数：{exc}") from exc

    factor = ref_max / station_max
    warning = ""
    if factor < ALIGNMENT_FACTOR_WARN_MIN or factor > ALIGNMENT_FACTOR_WARN_MAX:
        warning = (
            "对齐系数超出常见范围，请检查单位、列选择和实验阶段是否一致。"
        )
    return {
        "factor": float(factor),
        "reference_max_strain_fraction": ref_max,
        "station_max_strain_fraction": station_max,
        "reference_max_strain_percent": ref_max * 100.0,
        "station_max_strain_percent": station_max * 100.0,
        "warning": warning,
    }


def apply_strain_alignment(station_eps_fraction, diagnostics: dict) -> np.ndarray:
    """Scale station strain by a precomputed max-strain alignment factor."""
    factor = float(diagnostics["factor"])
    return np.asarray(station_eps_fraction, dtype=float) * factor


def compute_stress_alignment_diagnostics(ref_stress_mpa, station_stress_mpa) -> dict:
    """Compute max-stress alignment diagnostics using finite positive stress only."""
    try:
        ref_max = _positive_finite_max_for_alignment(ref_stress_mpa, "参考曲线", "应力")
        station_max = _positive_finite_max_for_alignment(station_stress_mpa, "线站数据", "应力")
    except ValueError as exc:
        raise ValueError(f"无法计算应力对齐系数：{exc}") from exc

    factor = ref_max / station_max
    warning = ""
    if factor < ALIGNMENT_FACTOR_WARN_MIN or factor > ALIGNMENT_FACTOR_WARN_MAX:
        warning = (
            "对齐系数超出常见范围，请检查单位、列选择和实验阶段是否一致。"
        )
    return {
        "factor": float(factor),
        "reference_max_stress_MPa": ref_max,
        "station_max_stress_MPa": station_max,
        "warning": warning,
    }


def apply_stress_alignment(station_stress_mpa, diagnostics: dict) -> np.ndarray:
    """Scale station stress by a precomputed max-stress alignment factor."""
    factor = float(diagnostics["factor"])
    return np.asarray(station_stress_mpa, dtype=float) * factor


def _average_duplicate_x(tmp: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    counts = tmp.groupby("x", dropna=False).size()
    duplicate_counts = counts[counts > 1]
    diagnostics = {
        "duplicate_groups": int(len(duplicate_counts)),
        "duplicate_rows": int(duplicate_counts.sum()),
    }
    averaged = tmp.groupby("x", as_index=False)["y"].mean().sort_values("x")
    return averaged, diagnostics


def make_interpolator_with_diagnostics(
    x,
    y,
    method: str = METHOD_PCHIP,
) -> Tuple[Callable[[np.ndarray], np.ndarray], float, float, str, dict]:
    """Return interpolator plus duplicate-x diagnostics."""
    tmp = finite_xy(x, y).sort_values("x")
    if tmp.empty or len(tmp) < 2:
        raise ValueError("参考曲线有效数据点少于 2 个，无法插值。")

    tmp, diagnostics = _average_duplicate_x(tmp)
    if len(tmp) < 2:
        raise ValueError("参考曲线自变量去重后少于 2 个点，无法插值。")

    x_arr = tmp["x"].to_numpy(dtype=float)
    y_arr = tmp["y"].to_numpy(dtype=float)
    xmin = float(np.nanmin(x_arr))
    xmax = float(np.nanmax(x_arr))

    use_pchip = method == METHOD_PCHIP and SCIPY_AVAILABLE
    if use_pchip:
        interp = PchipInterpolator(x_arr, y_arr, extrapolate=False)

        def f(v):
            return np.asarray(interp(np.asarray(v, dtype=float)), dtype=float)

        return f, xmin, xmax, "PCHIP", diagnostics

    def f(v):
        return np.interp(np.asarray(v, dtype=float), x_arr, y_arr, left=np.nan, right=np.nan)

    method_used = "Linear" if method == METHOD_LINEAR else "Linear（未安装 scipy，自动降级）"
    return f, xmin, xmax, method_used, diagnostics


def make_interpolator(
    x,
    y,
    method: str = METHOD_PCHIP,
) -> Tuple[Callable[[np.ndarray], np.ndarray], float, float, str]:
    """Return f(value), xmin, xmax, method_used. Duplicate x values are averaged."""
    f, xmin, xmax, method_used, _ = make_interpolator_with_diagnostics(x, y, method=method)
    return f, xmin, xmax, method_used


def _prepare_inverse_source_branch(eps_fraction, stress_mpa, use_pre_peak: bool = True) -> pd.DataFrame:
    tmp = finite_xy(eps_fraction, stress_mpa).rename(columns={"x": "eps", "y": "stress"})
    tmp = tmp.sort_values("eps").reset_index(drop=True)
    if tmp.empty or len(tmp) < 2:
        raise ValueError("参考曲线有效数据点少于 2 个，无法建立反插值。")

    if use_pre_peak:
        peak_idx = int(tmp["stress"].idxmax())
        tmp = tmp.iloc[: peak_idx + 1].copy()
    return tmp


def compute_inverse_strain_intervals(
    eps_fraction,
    stress_mpa,
    query_stress_mpa,
    use_pre_peak: bool = True,
) -> pd.DataFrame:
    """Find all piecewise-linear strain solutions for each queried stress."""
    columns = [
        "inverse_strain_min_fraction",
        "inverse_strain_max_fraction",
        "inverse_strain_min_percent",
        "inverse_strain_max_percent",
        "inverse_mapping_is_ambiguous",
        "inverse_ambiguity_width_percent",
    ]
    branch = _prepare_inverse_source_branch(eps_fraction, stress_mpa, use_pre_peak=use_pre_peak)
    eps_arr = branch["eps"].to_numpy(dtype=float)
    sig_arr = branch["stress"].to_numpy(dtype=float)
    queries = np.asarray(query_stress_mpa, dtype=float)
    stress_tol = max(1e-12, 1e-9 * max(1.0, float(np.nanmax(np.abs(sig_arr)))))
    eps_tol = 1e-12
    rows = []

    for query in queries:
        solutions = []
        if np.isfinite(query):
            for i in range(len(branch) - 1):
                eps0, eps1 = eps_arr[i], eps_arr[i + 1]
                sig0, sig1 = sig_arr[i], sig_arr[i + 1]
                dsig = sig1 - sig0

                if abs(dsig) <= stress_tol:
                    if abs(query - sig0) <= stress_tol:
                        solutions.extend([eps0, eps1])
                    continue

                lower = min(sig0, sig1) - stress_tol
                upper = max(sig0, sig1) + stress_tol
                if lower <= query <= upper:
                    frac = (query - sig0) / dsig
                    if -eps_tol <= frac <= 1.0 + eps_tol:
                        frac = min(1.0, max(0.0, frac))
                        solutions.append(eps0 + frac * (eps1 - eps0))

        unique = []
        for value in sorted(float(v) for v in solutions if np.isfinite(v)):
            if not unique or abs(value - unique[-1]) > eps_tol:
                unique.append(value)

        if unique:
            eps_min = float(min(unique))
            eps_max = float(max(unique))
            width_percent = (eps_max - eps_min) * 100.0
            ambiguous = bool(width_percent > eps_tol * 100.0)
        else:
            eps_min = np.nan
            eps_max = np.nan
            width_percent = np.nan
            ambiguous = False

        rows.append(
            {
                "inverse_strain_min_fraction": eps_min,
                "inverse_strain_max_fraction": eps_max,
                "inverse_strain_min_percent": eps_min * 100.0 if np.isfinite(eps_min) else np.nan,
                "inverse_strain_max_percent": eps_max * 100.0 if np.isfinite(eps_max) else np.nan,
                "inverse_mapping_is_ambiguous": ambiguous,
                "inverse_ambiguity_width_percent": width_percent,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def build_inverse_branch(
    eps_fraction,
    stress_mpa,
    use_pre_peak: bool = True,
    monotonicize: bool = True,
) -> pd.DataFrame:
    """Prepare a mostly single-valued stress->strain branch for inverse interpolation.

    Stress->strain mapping is not always unique. By default this function keeps the
    pre-UTS branch and optionally removes stress drops, which is usually the safest
    choice for in-situ tensile datasets before necking/localization.
    """
    tmp, _ = build_inverse_branch_with_diagnostics(
        eps_fraction,
        stress_mpa,
        use_pre_peak=use_pre_peak,
        monotonicize=monotonicize,
    )
    return tmp


def build_inverse_branch_with_diagnostics(
    eps_fraction,
    stress_mpa,
    use_pre_peak: bool = True,
    monotonicize: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Prepare the scalar inverse branch and report duplicate stress aggregation."""
    tmp = _prepare_inverse_source_branch(eps_fraction, stress_mpa, use_pre_peak=use_pre_peak)

    if monotonicize:
        keep_idx = []
        running_max = -np.inf
        # A tiny tolerance avoids deleting nearly equal points because of floating noise.
        tol = max(1e-9, 1e-6 * max(1.0, float(np.nanmax(np.abs(tmp["stress"].to_numpy())))))
        for idx, row in tmp.iterrows():
            sig = float(row["stress"])
            if sig >= running_max - tol:
                keep_idx.append(idx)
                if sig > running_max:
                    running_max = sig
        tmp = tmp.loc[keep_idx].copy()

    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=["eps", "stress"])
    tmp = tmp.sort_values("stress")
    counts = tmp.groupby("stress", dropna=False).size()
    duplicate_counts = counts[counts > 1]
    diagnostics = {
        "duplicate_groups": int(len(duplicate_counts)),
        "duplicate_rows": int(duplicate_counts.sum()),
    }
    # Duplicate stress values are averaged for scalar output; intervals retain the ambiguity.
    tmp = tmp.groupby("stress", as_index=False)["eps"].mean().sort_values("stress")
    if len(tmp) < 2:
        raise ValueError("反插值分支去重后少于 2 个点。请检查参考曲线或关闭/调整反插值选项。")
    return tmp, diagnostics


# -----------------------------
# GUI application
# -----------------------------

class ToolTip:
    """Small ttk-compatible tooltip without adding a GUI dependency."""

    def __init__(self, widget, text: str, wraplength: int = 360):
        self.widget = widget
        self.text = text
        self.wraplength = wraplength
        self.window = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _event=None):
        if self.window is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.window,
            text=self.text,
            justify="left",
            wraplength=self.wraplength,
            background="#111827",
            foreground="#f8fafc",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=5,
            font=("TkDefaultFont", 9),
        )
        label.pack()

    def _hide(self, _event=None):
        if self.window is not None:
            self.window.destroy()
            self.window = None


class StressStrainMapperApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("InSituLink · 参考曲线应力-应变映射工具")
        self.root.geometry("1366x768")
        self.root.minsize(1180, 720)

        self.ref_df: Optional[pd.DataFrame] = None
        self.station_df: Optional[pd.DataFrame] = None
        self.result_df: Optional[pd.DataFrame] = None
        self.ref_path: Optional[str] = None
        self.station_path: Optional[str] = None
        self._mapping_running = False
        self._tooltips = []
        self._last_plot_ref_clean: Optional[pd.DataFrame] = None

        self._build_variables()
        self._build_layout()
        self._log_startup_notes()

    def _build_variables(self):
        self.ref_strain_col = tk.StringVar()
        self.ref_stress_col = tk.StringVar()
        self.ref_strain_unit = tk.StringVar(value=STRAIN_FRACTION)
        self.ref_stress_unit = tk.StringVar(value=STRESS_MPA)

        self.station_mode = tk.StringVar(value=MODE_STRAIN_ONLY)
        self.station_id_col = tk.StringVar()
        self.station_strain_col = tk.StringVar()
        self.station_stress_col = tk.StringVar()
        self.station_strain_unit = tk.StringVar(value=STRAIN_FRACTION)
        self.station_stress_unit = tk.StringVar(value=STRESS_MPA)

        self.interp_method = tk.StringVar(value=METHOD_PCHIP)
        self.zero_reference = tk.BooleanVar(value=False)
        self.inverse_pre_peak = tk.BooleanVar(value=True)
        self.inverse_monotonic = tk.BooleanVar(value=True)

        self.smooth_method = tk.StringVar(value=SMOOTH_NONE)
        self.smooth_window = tk.IntVar(value=7)
        self.smooth_polyorder = tk.IntVar(value=2)
        self.show_raw_points = tk.BooleanVar(value=True)
        self.show_smoothed = tk.BooleanVar(value=False)
        self.align_strain_max_to_reference = tk.BooleanVar(value=False)
        self.align_stress_max_to_reference = tk.BooleanVar(value=False)
        self.plot_preset = tk.StringVar(value=DEFAULT_PLOT_PRESET)

        self.recommendation_confirmed = tk.BooleanVar(value=False)
        self.advanced_visible = tk.BooleanVar(value=False)
        self.wizard_state = tk.StringVar(value=WIZARD_NEED_REFERENCE)
        self.ref_recommendation = tk.StringVar(value="加载参考曲线后，程序会自动推荐应变列、应力列和单位。")
        self.station_recommendation = tk.StringVar(value="加载线站数据后，程序会自动推荐映射方向、输入列和单位。")
        self.strain_alignment_hint = tk.StringVar(value="加载参考曲线和线站应变后，将显示最大塑性应变对齐建议。")
        self.stress_alignment_hint = tk.StringVar(value="加载参考曲线和线站应力后，将显示最大应力对齐建议。")
        self.result_status = tk.StringVar(value="等待加载数据。")
        self.result_summary = tk.StringVar(value="确认推荐并运行后，这里会显示有效行数、超范围行数和导出提示。")

    def _build_layout(self):
        self._configure_styles()
        self.root.geometry("1366x768")
        self.root.minsize(1180, 720)

        shell = ttk.Frame(self.root, style="App.TFrame")
        shell.pack(fill=tk.BOTH, expand=True)
        shell.rowconfigure(1, weight=1)
        shell.columnconfigure(0, weight=1)

        status_bar = ttk.Frame(shell, style="Header.TFrame", padding=(16, 9))
        status_bar.grid(row=0, column=0, sticky="ew")
        status_bar.columnconfigure(0, weight=1)
        status_bar.columnconfigure(1, weight=0)
        ttk.Label(
            status_bar,
            text="InSituLink · 参考曲线应力-应变映射工具",
            style="AppTitle.TLabel",
        ).grid(row=0, column=0, sticky="w")
        self.wizard_state_label = ttk.Label(status_bar, textvariable=self.wizard_state, style="Muted.TLabel")
        self.wizard_state_label.grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.status_bar_status_label = ttk.Label(
            status_bar,
            textvariable=self.result_status,
            style="StatusPill.TLabel",
            anchor="center",
        )
        self.status_bar_status_label.grid(row=0, column=1, rowspan=2, sticky="e", padx=(12, 0))

        main = ttk.PanedWindow(shell, orient=tk.HORIZONTAL)
        main.grid(row=1, column=0, sticky="nsew", padx=10, pady=(8, 10))

        self.left_panel = ttk.Frame(main, width=CONTROL_PANEL_WIDTH, style="Panel.TFrame")
        self.left_panel.grid_propagate(False)
        self.right_panel = ttk.Frame(main, style="Workspace.TFrame")
        main.add(self.left_panel, weight=0)
        main.add(self.right_panel, weight=1)

        self._build_controls(self.left_panel)
        self._build_visual_area(self.right_panel)

    def _configure_styles(self):
        self.root.configure(bg="#eef3f8")
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background="#eef3f8")
        style.configure("Header.TFrame", background="#f8fbff")
        style.configure("Panel.TFrame", background="#f8fafc")
        style.configure("Workspace.TFrame", background="#eef3f8")
        style.configure("AppTitle.TLabel", background="#f8fbff", foreground="#0f172a", font=("TkDefaultFont", 14, "bold"))
        style.configure("Muted.TLabel", background="#f8fbff", foreground="#526173")
        style.configure("StatusPill.TLabel", background="#e6f4ee", foreground="#116149", padding=(12, 5), font=("TkDefaultFont", 10, "bold"))
        style.configure("Section.TLabelframe", background="#ffffff", bordercolor="#d8e1eb", lightcolor="#d8e1eb", darkcolor="#d8e1eb", relief="solid")
        style.configure("Section.TLabelframe.Label", background="#ffffff", foreground="#0f172a", font=("TkDefaultFont", 10, "bold"))
        style.configure("SectionBody.TFrame", background="#ffffff")
        style.configure("Control.TLabel", background="#ffffff", foreground="#1f2937")
        style.configure("Hint.TLabel", background="#ffffff", foreground="#667085")
        style.configure("BlueHint.TLabel", background="#ffffff", foreground="#1f5f85")
        style.configure("Primary.TButton", padding=(11, 8), font=("TkDefaultFont", 10, "bold"))
        style.configure("Secondary.TButton", padding=(8, 6))
        style.configure("Export.TButton", padding=(10, 7), font=("TkDefaultFont", 10, "bold"))
        style.configure("Danger.TButton", padding=(8, 6))
        style.configure("Tool.TCheckbutton", background="#ffffff", foreground="#1f2937")
        style.configure("TCombobox", padding=(4, 3))
        style.configure("TSpinbox", padding=(4, 3))
        style.configure("Workspace.TNotebook", background="#eef3f8", borderwidth=0, tabmargins=(4, 4, 4, 0))
        style.configure("Workspace.TNotebook.Tab", padding=(18, 8), font=("TkDefaultFont", 10, "bold"))
        style.map(
            "Workspace.TNotebook.Tab",
            background=[("selected", "#ffffff"), ("active", "#f8fbff")],
            foreground=[("selected", "#0f172a"), ("active", "#1f5f85")],
        )
        style.configure("Treeview", background="#ffffff", fieldbackground="#ffffff", foreground="#111827", rowheight=26, bordercolor="#d8e1eb")
        style.configure("Treeview.Heading", background="#eef3f8", foreground="#0f172a", font=("TkDefaultFont", 9, "bold"), padding=(6, 5))
        style.map("Treeview", background=[("selected", "#dbeafe")], foreground=[("selected", "#0f172a")])

    def _build_scrollable_controls(self, parent) -> ttk.Frame:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        self.control_canvas = tk.Canvas(parent, background="#f8fafc", highlightthickness=0, borderwidth=0, width=CONTROL_PANEL_WIDTH - 14)
        control_scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.control_canvas.yview)
        self.control_canvas.configure(yscrollcommand=control_scrollbar.set)
        self.control_canvas.grid(row=0, column=0, sticky="nsew")
        control_scrollbar.grid(row=0, column=1, sticky="ns")

        body = ttk.Frame(self.control_canvas, style="Panel.TFrame", padding=(10, 8))
        self.control_window = self.control_canvas.create_window((0, 0), window=body, anchor="nw")

        def update_scrollregion(_event=None):
            self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))

        def sync_width(event):
            self.control_canvas.itemconfigure(self.control_window, width=event.width)

        body.bind("<Configure>", update_scrollregion)
        self.control_canvas.bind("<Configure>", sync_width)
        self.control_canvas.bind_all("<MouseWheel>", lambda event: self.control_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))
        return body

    def _grid_labeled(self, parent, row: int, label: str, widget, pady: int = 2):
        ttk.Label(parent, text=label, style="Control.TLabel", width=10).grid(row=row, column=0, sticky="w", pady=pady, padx=(0, 8))
        widget.grid(row=row, column=1, sticky="ew", pady=pady)

    def _add_tooltip(self, widget, text: str):
        self._tooltips.append(ToolTip(widget, text))
        return widget

    def _current_plot_preset(self) -> dict:
        return PLOT_EXPORT_PRESETS.get(self.plot_preset.get(), PLOT_EXPORT_PRESETS[DEFAULT_PLOT_PRESET])

    def _on_plot_preset_changed(self, *_):
        preset = self._current_plot_preset()
        self.log(f"绘图预设已切换为 {self.plot_preset.get()}：{preset['description']}")
        if self.result_df is not None and self._last_plot_ref_clean is not None:
            self._update_plot(self._last_plot_ref_clean, self.result_df)
            self._set_result_status("绘图预设已修改，预览已刷新。", "#24527a")

    def _apply_axis_style(self, axis, preset: dict, *, grid: Optional[bool] = None):
        axis.set_facecolor("#ffffff")
        axis.tick_params(
            colors=PLOT_COLORS["axis"],
            labelsize=preset["font_size"],
            width=preset["axis_width"],
            length=4,
            direction="out",
        )
        axis.tick_params(which="minor", length=2, width=max(0.5, preset["axis_width"] * 0.7), direction="out")
        axis.xaxis.label.set_color(PLOT_COLORS["axis"])
        axis.yaxis.label.set_color(PLOT_COLORS["axis"])
        axis.xaxis.label.set_size(preset["font_size"])
        axis.yaxis.label.set_size(preset["font_size"])
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
        try:
            axis.xaxis.set_minor_locator(AutoMinorLocator(2))
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))
        except Exception:
            pass
        for spine in axis.spines.values():
            spine.set_color(PLOT_COLORS["spine"])
            spine.set_linewidth(preset["axis_width"])
        use_grid = preset["grid"] if grid is None else grid
        if use_grid:
            axis.grid(True, color=PLOT_COLORS["grid"], linewidth=0.55, alpha=0.75)
        else:
            axis.grid(False)

    def _apply_figure_layout(self, *, for_export: bool = False):
        preset = self._current_plot_preset()
        if for_export:
            self.fig.set_size_inches(*preset["figsize"], forward=True)
        self.fig.patch.set_facecolor("#ffffff")
        self.fig.align_labels()
        self.fig.tight_layout(pad=1.2, h_pad=1.7)

    @staticmethod
    def _savefig_kwargs(path: str, preset: dict) -> dict:
        suffix = Path(path).suffix.lower()
        kwargs = {
            "bbox_inches": "tight",
            "pad_inches": 0.04,
            "facecolor": "white",
            "edgecolor": "none",
        }
        if suffix not in {".pdf", ".svg", ".eps"}:
            kwargs["dpi"] = preset["dpi"]
        return kwargs

    def _create_collapsible_section(self, parent, row: int, title: str, summary: str) -> ttk.Frame:
        container = ttk.Frame(parent, style="SectionBody.TFrame")
        container.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        container.columnconfigure(0, weight=1)
        visible = tk.BooleanVar(value=False)
        header = ttk.Checkbutton(
            container,
            text=f"{title}  ·  {summary}",
            variable=visible,
            command=lambda t=title: self._toggle_advanced_section(t),
            style="Tool.TCheckbutton",
        )
        header.grid(row=0, column=0, sticky="ew", padx=6, pady=(4, 2))
        body = ttk.Frame(container, style="SectionBody.TFrame", padding=(12, 6, 8, 8))
        body.columnconfigure(1, weight=1)
        self.advanced_sections[title] = {"frame": container, "body": body, "visible": visible, "header": header}
        return body

    def _toggle_advanced_section(self, title: str):
        section = self.advanced_sections[title]
        if bool(section["visible"].get()):
            section["body"].grid(row=1, column=0, sticky="ew")
        else:
            section["body"].grid_remove()
        if hasattr(self, "control_canvas"):
            self.root.after_idle(lambda: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all")))

    def _build_controls(self, parent):
        body = self._build_scrollable_controls(parent)
        body.columnconfigure(0, weight=1)

        ref_box = ttk.LabelFrame(body, text="1 参考曲线", style="Section.TLabelframe", padding=(10, 9))
        st_box = ttk.LabelFrame(body, text="2 线站数据", style="Section.TLabelframe", padding=(10, 9))
        result_box = ttk.LabelFrame(body, text="3 映射运行", style="Section.TLabelframe", padding=(10, 9))
        ref_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        st_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        result_box.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        # Step 1: reference file
        ref_box.columnconfigure(1, weight=1)
        ttk.Label(ref_box, text="加载完整参考曲线，确认应变/应力列与单位。", style="Hint.TLabel", wraplength=CONTROL_WRAP).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        self.load_reference_button = ttk.Button(ref_box, text="导入参考曲线 CSV/TXT/XLSX", command=self.load_reference, style="Secondary.TButton")
        self.load_reference_button.grid(row=1, column=0, columnspan=2, sticky="ew")
        self._add_tooltip(self.load_reference_button, "导入完整参考应力-应变曲线。程序只读取表格并推荐列和单位，不会修改原始文件。")
        self.ref_label = ttk.Label(ref_box, text="未加载", style="Hint.TLabel", wraplength=CONTROL_WRAP)
        self.ref_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 8))
        ttk.Label(ref_box, textvariable=self.ref_recommendation, style="BlueHint.TLabel", wraplength=CONTROL_WRAP).grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 8))

        self.ref_strain_combo = ttk.Combobox(ref_box, textvariable=self.ref_strain_col, state="readonly", width=26)
        self.ref_stress_combo = ttk.Combobox(ref_box, textvariable=self.ref_stress_col, state="readonly", width=26)
        self.ref_strain_unit_combo = ttk.Combobox(ref_box, textvariable=self.ref_strain_unit, values=[STRAIN_FRACTION, STRAIN_PERCENT], state="readonly", width=26)
        self.ref_stress_unit_combo = ttk.Combobox(ref_box, textvariable=self.ref_stress_unit, values=[STRESS_MPA, STRESS_GPA, STRESS_PA], state="readonly", width=26)
        self._grid_labeled(ref_box, 4, "应变列", self.ref_strain_combo)
        self._grid_labeled(ref_box, 5, "应力列", self.ref_stress_combo)
        self._grid_labeled(ref_box, 6, "应变单位", self.ref_strain_unit_combo)
        self._grid_labeled(ref_box, 7, "应力单位", self.ref_stress_unit_combo)

        # Step 2: station file and recommendation confirmation
        st_box.columnconfigure(1, weight=1)
        ttk.Label(st_box, text="加载线站数据，检查推荐模式、编号和输入列。", style="Hint.TLabel", wraplength=CONTROL_WRAP).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        self.load_station_button = ttk.Button(st_box, text="导入线站数据 CSV/TXT/XLSX", command=self.load_station, style="Secondary.TButton")
        self.load_station_button.grid(row=1, column=0, columnspan=2, sticky="ew")
        self._add_tooltip(self.load_station_button, "导入每张谱线对应的应变、应力或二者兼有的数据表。程序会按列名和值域给出推荐。")
        self.station_label = ttk.Label(st_box, text="未加载", style="Hint.TLabel", wraplength=CONTROL_WRAP)
        self.station_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 8))
        ttk.Label(st_box, textvariable=self.station_recommendation, style="BlueHint.TLabel", wraplength=CONTROL_WRAP).grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 8))

        mode_combo = ttk.Combobox(st_box, textvariable=self.station_mode,
                                  values=[MODE_STRAIN_ONLY, MODE_STRESS_ONLY, MODE_BOTH],
                                  state="readonly", width=38)
        mode_combo.bind("<<ComboboxSelected>>", self._on_station_mode_changed)

        self.station_id_combo = ttk.Combobox(st_box, textvariable=self.station_id_col, state="readonly", width=26)
        self.station_strain_combo = ttk.Combobox(st_box, textvariable=self.station_strain_col, state="readonly", width=26)
        self.station_stress_combo = ttk.Combobox(st_box, textvariable=self.station_stress_col, state="readonly", width=26)
        self.station_strain_unit_combo = ttk.Combobox(st_box, textvariable=self.station_strain_unit, values=[STRAIN_FRACTION, STRAIN_PERCENT], state="readonly", width=26)
        self.station_stress_unit_combo = ttk.Combobox(st_box, textvariable=self.station_stress_unit, values=[STRESS_MPA, STRESS_GPA, STRESS_PA], state="readonly", width=26)
        self._grid_labeled(st_box, 4, "模式", mode_combo)
        self._grid_labeled(st_box, 5, "编号列", self.station_id_combo)
        self._grid_labeled(st_box, 6, "应变列", self.station_strain_combo)
        self._grid_labeled(st_box, 7, "应力列", self.station_stress_combo)
        self._grid_labeled(st_box, 8, "应变单位", self.station_strain_unit_combo)
        self._grid_labeled(st_box, 9, "应力单位", self.station_stress_unit_combo)

        self.mode_hint = ttk.Label(st_box, text="", style="Hint.TLabel", wraplength=CONTROL_WRAP)
        self.mode_hint.grid(row=10, column=0, columnspan=2, sticky="w", pady=(5, 0))
        self.confirm_button = ttk.Button(st_box, text="确认列与单位", command=self.confirm_recommendations, style="Secondary.TButton")
        self.confirm_button.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self._add_tooltip(self.confirm_button, "确认当前模式、列和单位选择。修改任何列或单位后，需要重新确认再运行。")
        for combo in [
            self.ref_strain_combo,
            self.ref_stress_combo,
            self.ref_strain_unit_combo,
            self.ref_stress_unit_combo,
            self.station_id_combo,
            self.station_strain_combo,
            self.station_stress_combo,
            self.station_strain_unit_combo,
            self.station_stress_unit_combo,
        ]:
            combo.bind("<<ComboboxSelected>>", self._mark_recommendation_dirty)
        self._update_station_mode_hint()

        # Step 3: run, quality summary, export
        result_box.columnconfigure(0, weight=0)
        result_box.columnconfigure(1, weight=1)
        self.result_status_label = ttk.Label(result_box, textvariable=self.result_status, style="BlueHint.TLabel", wraplength=CONTROL_WRAP)
        self.result_status_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        ttk.Label(result_box, textvariable=self.result_summary, style="Hint.TLabel", wraplength=CONTROL_WRAP).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))

        self.run_button = ttk.Button(result_box, text="开始映射 / 更新图", command=self.run_mapping, style="Primary.TButton")
        self.run_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(2, 5))
        self._add_tooltip(self.run_button, "按当前列、单位和高级设置执行映射，并刷新曲线图、结果表和日志。")
        self.export_button = ttk.Button(result_box, text="导出结果 CSV/XLSX", command=self.export_result, style="Secondary.TButton")
        self.export_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=2)
        self._add_tooltip(self.export_button, "导出当前映射结果表。导出的列会优先放置谱线编号、应变和应力结果，原始列仍会保留。")
        self.plot_preset_combo = ttk.Combobox(
            result_box,
            textvariable=self.plot_preset,
            values=list(PLOT_EXPORT_PRESETS.keys()),
            state="readonly",
            width=22,
        )
        self._grid_labeled(result_box, 4, "出图预设", self.plot_preset_combo)
        self.plot_preset_combo.bind("<<ComboboxSelected>>", self._on_plot_preset_changed)
        self._add_tooltip(self.plot_preset_combo, "控制图像尺寸、dpi、字号、线宽、marker、图例和是否显示网格。Publication 为论文级默认。")
        self.save_plot_button = ttk.Button(result_box, text="导出图像 PNG/TIFF/PDF/SVG/EPS", command=self.save_plot, style="Export.TButton")
        self.save_plot_button.grid(row=5, column=0, columnspan=2, sticky="ew", pady=2)
        self._add_tooltip(self.save_plot_button, "按当前出图预设导出高分辨率位图或矢量图；使用 tight bounding box 避免标签和图例被裁切。")

        ttk.Checkbutton(
            result_box,
            text="显示高级设置",
            variable=self.advanced_visible,
            command=self._toggle_advanced_settings,
            style="Tool.TCheckbutton",
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 4))

        self.advanced_frame = ttk.LabelFrame(body, text="高级设置", style="Section.TLabelframe", padding=(8, 7))
        self.advanced_frame.columnconfigure(0, weight=1)
        opt_box = self.advanced_frame
        self.advanced_sections = {}

        interp_body = self._create_collapsible_section(opt_box, 0, "插值与反插值", "PCHIP；峰值前；去除下降点")
        interp_body.columnconfigure(1, weight=1)
        self._grid_labeled(
            interp_body,
            0,
            "插值方法",
            ttk.Combobox(interp_body, textvariable=self.interp_method, values=[METHOD_PCHIP, METHOD_LINEAR], state="readonly", width=30),
        )

        ttk.Checkbutton(interp_body, text="应力→应变时只使用峰值应力前的加载分支", variable=self.inverse_pre_peak, style="Tool.TCheckbutton").grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(interp_body, text="应力→应变时自动删除非单调应力下降点", variable=self.inverse_monotonic, style="Tool.TCheckbutton").grid(row=2, column=0, columnspan=2, sticky="w", pady=2)

        method_note = "PCHIP 可用" if SCIPY_AVAILABLE else "未检测到 scipy：PCHIP 会自动降级为线性插值"
        ttk.Label(interp_body, text=method_note, style="Hint.TLabel", wraplength=CONTROL_WRAP - 10).grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 0))

        view_box = self._create_collapsible_section(opt_box, 1, "平滑显示", "原始点；可选 smooth_* 辅助列")
        view_box.columnconfigure(1, weight=1)

        ttk.Checkbutton(view_box, text="显示原始映射点", variable=self.show_raw_points, style="Tool.TCheckbutton").grid(row=0, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(view_box, text="显示/导出平滑辅助列", variable=self.show_smoothed, style="Tool.TCheckbutton").grid(row=1, column=0, columnspan=2, sticky="w", pady=2)

        self._grid_labeled(view_box, 2, "平滑方法", ttk.Combobox(
            view_box,
            textvariable=self.smooth_method,
            values=[SMOOTH_NONE, SMOOTH_ROLLING_MEDIAN, SMOOTH_ROLLING_MEAN, SMOOTH_SAVGOL],
            state="readonly",
            width=34,
        ))

        self._grid_labeled(view_box, 3, "窗口点数", ttk.Spinbox(view_box, from_=3, to=101, increment=2, textvariable=self.smooth_window, width=8))
        view_box.grid_slaves(row=3, column=1)[0].grid(sticky="w")

        self._grid_labeled(view_box, 4, "SG 阶数", ttk.Spinbox(view_box, from_=1, to=5, increment=1, textvariable=self.smooth_polyorder, width=8))
        view_box.grid_slaves(row=4, column=1)[0].grid(sticky="w")

        ttk.Label(
            view_box,
            text="说明：平滑默认只作为视觉辅助，并新增 smooth_* 列；不会覆盖原始 mapped_* 数据。",
            style="Hint.TLabel",
            wraplength=CONTROL_WRAP - 10,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(5, 0))

        align_box = self._create_collapsible_section(opt_box, 2, "应变 / 应力对齐", "最大值比例缩放；保留审计列")
        align_box.columnconfigure(0, weight=1)
        self.align_strain_check = ttk.Checkbutton(
            align_box,
            text="将线站最大塑性应变缩放到参考曲线最大塑性应变",
            variable=self.align_strain_max_to_reference,
            command=self._on_alignment_option_changed,
            style="Tool.TCheckbutton",
        )
        self.align_strain_check.grid(row=0, column=0, sticky="w", pady=2)
        ttk.Label(
            align_box,
            textvariable=self.strain_alignment_hint,
            style="Hint.TLabel",
            wraplength=CONTROL_WRAP - 10,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.align_stress_check = ttk.Checkbutton(
            align_box,
            text="将线站最大应力缩放到参考曲线最大应力",
            variable=self.align_stress_max_to_reference,
            command=self._on_alignment_option_changed,
            style="Tool.TCheckbutton",
        )
        self.align_stress_check.grid(row=2, column=0, sticky="w", pady=(8, 2))
        ttk.Label(
            align_box,
            textvariable=self.stress_alignment_hint,
            style="Hint.TLabel",
            wraplength=CONTROL_WRAP - 10,
        ).grid(row=3, column=0, sticky="w", pady=(4, 0))

        ref_option_box = self._create_collapsible_section(opt_box, 3, "起点归零 / 基线校正", "参考和线站都减去首个有效点")
        ttk.Checkbutton(
            ref_option_box,
            text="参考曲线和线站数据首个有效点归零（仅在存在零点/预载偏移时勾选）",
            variable=self.zero_reference,
            command=self._on_alignment_option_changed,
            style="Tool.TCheckbutton",
        ).grid(row=0, column=0, sticky="w", pady=2)

        self._toggle_advanced_settings()
        self._update_wizard_state_display()

    def _build_visual_area(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        self.detail_notebook = ttk.Notebook(parent, style="Workspace.TNotebook")
        self.detail_notebook.grid(row=0, column=0, sticky="nsew")

        plot_tab = ttk.Frame(self.detail_notebook, style="Workspace.TFrame", padding=(8, 8, 8, 8))
        plot_tab.rowconfigure(0, weight=1)
        plot_tab.columnconfigure(0, weight=1)
        self.detail_notebook.add(plot_tab, text="曲线视图")

        self.plot_frame = ttk.LabelFrame(plot_tab, text="参考曲线与映射结果", style="Section.TLabelframe", padding=8)
        self.plot_frame.grid(row=0, column=0, sticky="nsew")
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(8.6, 5.2), dpi=100, facecolor="#ffffff")
        self.ax_curve = self.fig.add_subplot(211)
        self.ax_series = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=2.2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")

        table_box = ttk.Frame(self.detail_notebook, style="Workspace.TFrame", padding=(8, 8, 8, 8))
        table_box.rowconfigure(0, weight=1)
        table_box.columnconfigure(0, weight=1)
        self.detail_notebook.add(table_box, text="结果表")

        self.tree = ttk.Treeview(table_box, show="headings")
        yscroll = ttk.Scrollbar(table_box, orient="vertical", command=self.tree.yview)
        xscroll = ttk.Scrollbar(table_box, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")

        log_box = ttk.Frame(self.detail_notebook, style="Workspace.TFrame", padding=(8, 8, 8, 8))
        log_box.rowconfigure(0, weight=1)
        log_box.columnconfigure(0, weight=1)
        self.detail_notebook.add(log_box, text="问题日志")
        self.log_text = tk.Text(
            log_box,
            height=8,
            wrap="word",
            relief="flat",
            background="#ffffff",
            foreground="#0f172a",
            padx=10,
            pady=8,
            spacing1=2,
            spacing3=4,
        )
        log_scroll = ttk.Scrollbar(log_box, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll.grid(row=0, column=1, sticky="ns")

    def _log_startup_notes(self):
        self.log("按 1→2→3 完成映射：先加载参考曲线，再加载线站数据，确认推荐后运行。")
        self.log("如果单位或列选择不确定，先看蓝色推荐理由；高级设置默认不需要修改。")

    def log(self, text: str):
        self.log_text.insert(tk.END, str(text) + "\n")
        self.log_text.see(tk.END)

    def _log_exception(self, context: str, exc: Exception):
        self.log(f"{context}：{exc}")
        self.log("请检查文件路径、表头、所选列、单位和输出路径是否有效。下面是技术详情，便于追踪问题：")
        self.log(traceback.format_exc())

    def _toggle_advanced_settings(self):
        if bool(self.advanced_visible.get()):
            self.advanced_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
            if hasattr(self, "advanced_sections") and not any(bool(section["visible"].get()) for section in self.advanced_sections.values()):
                first_key = next(iter(self.advanced_sections), None)
                if first_key is not None:
                    self.advanced_sections[first_key]["visible"].set(True)
                    self._toggle_advanced_section(first_key)
        else:
            self.advanced_frame.grid_remove()
        if hasattr(self, "control_canvas"):
            self.root.after_idle(lambda: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all")))

    def _set_result_status(self, text: str, color: str = "#555555"):
        self.result_status.set(text)
        if hasattr(self, "result_status_label"):
            self.result_status_label.config(foreground=color)
        if hasattr(self, "status_bar_status_label"):
            self.status_bar_status_label.config(foreground=color)

    def _mark_recommendation_dirty(self, *_):
        if self.ref_df is not None and self.station_df is not None:
            self.recommendation_confirmed.set(False)
            self.result_df = None
            self._last_plot_ref_clean = None
            self._set_result_status("推荐已修改，请重新确认后再运行。", "#8a6d1d")
            self.result_summary.set("列、单位或模式已改变；旧结果已失效。")
            self._update_strain_alignment_hint()
            self._update_stress_alignment_hint()
            self._update_wizard_state_display()

    def _on_alignment_option_changed(self):
        self._update_strain_alignment_hint()
        self._update_stress_alignment_hint()
        if self.result_df is not None:
            self._clear_result_preview()
            self._set_result_status("高级设置已修改，请重新运行映射。", "#8a6d1d")
            self.result_summary.set("对齐或起点归零设置已改变；旧结果已失效。")
            self._update_wizard_state_display()

    def _reference_strain_for_alignment(self) -> np.ndarray:
        if self.ref_df is None:
            raise ValueError("请先加载参考曲线。")
        eps_raw = to_numeric_series(self.ref_df, self.ref_strain_col.get(), "参考应变")
        eps = convert_strain_to_fraction(eps_raw, self.ref_strain_unit.get())
        tmp = pd.Series(eps).replace([np.inf, -np.inf], np.nan).dropna().sort_values()
        if self.zero_reference.get() and len(tmp) > 0:
            tmp, _ = zero_to_first_finite(tmp.to_numpy(dtype=float))
            tmp = pd.Series(tmp).replace([np.inf, -np.inf], np.nan).dropna().sort_values()
        return tmp.to_numpy(dtype=float)

    def _station_strain_for_alignment(self) -> np.ndarray:
        if self.station_df is None:
            raise ValueError("请先加载线站数据。")
        eps_raw = to_numeric_series(self.station_df, self.station_strain_col.get(), "线站应变")
        eps = convert_strain_to_fraction(eps_raw, self.station_strain_unit.get())
        if self.zero_reference.get():
            eps, _ = zero_to_first_finite(eps)
        return eps

    def _reference_stress_for_alignment(self) -> np.ndarray:
        if self.ref_df is None:
            raise ValueError("请先加载参考曲线。")
        sig_raw = to_numeric_series(self.ref_df, self.ref_stress_col.get(), "参考应力")
        sig = convert_stress_to_mpa(sig_raw, self.ref_stress_unit.get())
        tmp = pd.Series(sig).replace([np.inf, -np.inf], np.nan).dropna()
        if self.zero_reference.get() and len(tmp) > 0:
            tmp, _ = zero_to_first_finite(tmp.to_numpy(dtype=float))
            tmp = pd.Series(tmp).replace([np.inf, -np.inf], np.nan).dropna().sort_values()
        else:
            tmp = tmp.sort_values()
        return tmp.to_numpy(dtype=float)

    def _station_stress_for_alignment(self) -> np.ndarray:
        if self.station_df is None:
            raise ValueError("请先加载线站数据。")
        sig_raw = to_numeric_series(self.station_df, self.station_stress_col.get(), "线站应力")
        sig = convert_stress_to_mpa(sig_raw, self.station_stress_unit.get())
        if self.zero_reference.get():
            sig, _ = zero_to_first_finite(sig)
        return sig

    def _update_strain_alignment_hint(self):
        if not hasattr(self, "strain_alignment_hint"):
            return
        if self.station_mode.get() == MODE_STRESS_ONLY:
            self.strain_alignment_hint.set("当前为应力→应变反查模式，没有线站应变列；最大应变对齐不会应用。")
            if hasattr(self, "align_strain_check"):
                self.align_strain_check.config(state="disabled")
            return
        if hasattr(self, "align_strain_check"):
            self.align_strain_check.config(state="normal")
        if self.ref_df is None or self.station_df is None:
            self.strain_alignment_hint.set("加载参考曲线和线站应变后，将显示最大塑性应变对齐建议。")
            return
        try:
            diag = compute_strain_alignment_diagnostics(
                self._reference_strain_for_alignment(),
                self._station_strain_for_alignment(),
            )
            hint = (
                f"建议系数 = {diag['factor']:.6g}；"
                f"参考最大 {diag['reference_max_strain_percent']:.4g}%，"
                f"线站最大 {diag['station_max_strain_percent']:.4g}%。"
            )
            if diag["warning"]:
                hint += " " + diag["warning"]
            self.strain_alignment_hint.set(hint)
        except Exception as exc:
            self.strain_alignment_hint.set(f"暂不能计算对齐建议：{exc}")

    def _update_stress_alignment_hint(self):
        if not hasattr(self, "stress_alignment_hint"):
            return
        if self.station_mode.get() == MODE_STRAIN_ONLY:
            self.stress_alignment_hint.set("当前为应变→应力插值模式，没有线站应力列；最大应力对齐不会应用。")
            if hasattr(self, "align_stress_check"):
                self.align_stress_check.config(state="disabled")
            return
        if hasattr(self, "align_stress_check"):
            self.align_stress_check.config(state="normal")
        if self.ref_df is None or self.station_df is None:
            self.stress_alignment_hint.set("加载参考曲线和线站应力后，将显示最大应力对齐建议。")
            return
        try:
            diag = compute_stress_alignment_diagnostics(
                self._reference_stress_for_alignment(),
                self._station_stress_for_alignment(),
            )
            hint = (
                f"建议系数 = {diag['factor']:.6g}；"
                f"参考最大 {diag['reference_max_stress_MPa']:.4g} MPa，"
                f"线站最大 {diag['station_max_stress_MPa']:.4g} MPa。"
            )
            if diag["warning"]:
                hint += " " + diag["warning"]
            self.stress_alignment_hint.set(hint)
        except Exception as exc:
            self.stress_alignment_hint.set(f"暂不能计算对齐建议：{exc}")

    def _update_wizard_state_display(self):
        state = get_wizard_state(
            self.ref_df is not None,
            self.station_df is not None,
            bool(self.recommendation_confirmed.get()),
            self.result_df is not None,
        )
        self.wizard_state.set(f"当前状态：{state}")
        can_confirm = self.ref_df is not None and self.station_df is not None
        can_run = can_confirm and bool(self.recommendation_confirmed.get())
        has_result = self.result_df is not None
        if hasattr(self, "confirm_button"):
            self.confirm_button.config(state="normal" if can_confirm else "disabled")
        if hasattr(self, "run_button"):
            self.run_button.config(state="normal" if can_run else "disabled")
        if hasattr(self, "export_button"):
            self.export_button.config(state="normal" if has_result else "disabled")
        if hasattr(self, "save_plot_button"):
            self.save_plot_button.config(state="normal" if has_result else "disabled")

    def _format_reference_recommendation(self, rec: dict, df: pd.DataFrame) -> str:
        parts = [
            f"推荐：应变列 = {rec['strain_col']}，单位 = {rec['strain_unit']}",
            f"推荐：应力列 = {rec['stress_col']}，单位 = {rec['stress_unit']}",
            f"理由：{rec['strain_reason']} {rec['stress_reason']}",
        ]
        try:
            eps = convert_strain_to_fraction(pd.to_numeric(df[rec["strain_col"]], errors="coerce"), rec["strain_unit"])
            sig = convert_stress_to_mpa(pd.to_numeric(df[rec["stress_col"]], errors="coerce"), rec["stress_unit"])
            valid = finite_xy(eps, sig)
            parts.append(
                f"检查：有效点 {len(valid)} 个；应变范围 {valid['x'].min()*100:.4g}% 到 {valid['x'].max()*100:.4g}%；"
                f"应力范围 {valid['y'].min():.4g} 到 {valid['y'].max():.4g} MPa。"
            )
            _, diag = _average_duplicate_x(valid)
            if diag["duplicate_groups"] > 0:
                parts.append(f"提醒：参考应变有 {diag['duplicate_groups']} 组重复点，运行时会按均值聚合并导出诊断列。")
        except Exception:
            parts.append("检查：推荐列暂不能形成有效曲线，请手动确认列和单位。")
        return "\n".join(parts)

    def _format_station_recommendation(self, rec: dict, df: pd.DataFrame) -> str:
        parts = [
            f"推荐模式：{rec['mode']}",
            rec["mode_reason"],
            f"编号列：{rec['id_col'] or '无，按行号生成'}。{rec['id_reason']}",
        ]
        if rec["strain_col"]:
            parts.append(f"应变列：{rec['strain_col']}，单位 = {rec['strain_unit']}。{rec['strain_unit_reason']}")
        if rec["stress_col"]:
            parts.append(f"应力列：{rec['stress_col']}，单位 = {rec['stress_unit']}。{rec['stress_unit_reason']}")
        return "\n".join(parts)

    def _on_station_mode_changed(self, *_):
        self._update_station_mode_hint()
        self._mark_recommendation_dirty()

    def confirm_recommendations(self):
        if self.ref_df is None or self.station_df is None:
            messagebox.showwarning("还不能确认", "请先加载参考曲线和线站数据。")
            return
        self.recommendation_confirmed.set(True)
        self.result_df = None
        self._last_plot_ref_clean = None
        self._set_result_status("推荐已确认，可以运行映射。", "#2e7d32")
        self.result_summary.set("点击“开始映射 / 更新图”后，程序会显示有效行数、超范围行数和导出按钮。")
        self._update_wizard_state_display()
        if hasattr(self, "wizard"):
            self.wizard.select(2)

    def _update_station_mode_hint(self):
        mode = self.station_mode.get()
        if mode == MODE_STRAIN_ONLY:
            hint = "当前模式：线站数据提供应变，程序用参考曲线 ε→σ 得到每张谱线的应力；应力列会被忽略。"
            strain_state, stress_state = "readonly", "disabled"
        elif mode == MODE_STRESS_ONLY:
            hint = "当前模式：线站数据提供应力，程序用参考曲线 σ→ε 反查应变；应变列会被忽略。"
            strain_state, stress_state = "disabled", "readonly"
        else:
            hint = "当前模式：线站文件同时已有应变和应力。程序只做单位换算、对齐、可视化和导出。"
            strain_state, stress_state = "readonly", "readonly"
        if hasattr(self, "mode_hint"):
            self.mode_hint.config(text=hint)
        # Disable irrelevant controls to reduce accidental selection of the wrong mode/column.
        if hasattr(self, "station_strain_combo"):
            self.station_strain_combo.config(state=strain_state)
        if hasattr(self, "station_stress_combo"):
            self.station_stress_combo.config(state=stress_state)
        if hasattr(self, "station_strain_unit_combo"):
            self.station_strain_unit_combo.config(state=strain_state)
        if hasattr(self, "station_stress_unit_combo"):
            self.station_stress_unit_combo.config(state=stress_state)
        self._update_strain_alignment_hint()
        self._update_stress_alignment_hint()

    def load_reference(self):
        path = filedialog.askopenfilename(
            title="选择参考应力-应变曲线文件",
            filetypes=[("Data files", "*.csv *.txt *.dat *.xlsx *.xls"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            df = read_table(path)
            self.ref_df = df
            self.ref_path = path
            self.result_df = None
            self._last_plot_ref_clean = None
            self.recommendation_confirmed.set(False)
            cols = list(df.columns)
            self.ref_strain_combo["values"] = cols
            self.ref_stress_combo["values"] = cols
            rec = recommend_reference_setup(df)
            self.ref_strain_col.set(rec["strain_col"])
            self.ref_stress_col.set(rec["stress_col"])
            self.ref_strain_unit.set(rec["strain_unit"])
            self.ref_stress_unit.set(rec["stress_unit"])
            self.ref_label.config(text=f"{os.path.basename(path)}  ({len(df)} rows, {len(cols)} cols)")
            self.ref_recommendation.set(self._format_reference_recommendation(rec, df))
            self._set_result_status("参考曲线已加载。下一步加载线站数据。", "#24527a")
            self.result_summary.set("参考曲线推荐已填入；如蓝色说明不合理，请手动修正后继续。")
            self.log(f"已加载参考曲线：{path}")
            self.log(f"参考推荐：应变={rec['strain_col']} ({rec['strain_unit']})；应力={rec['stress_col']} ({rec['stress_unit']})")
            self._update_strain_alignment_hint()
            self._update_stress_alignment_hint()
            self._update_wizard_state_display()
            if hasattr(self, "wizard"):
                self.wizard.select(1)
        except Exception as exc:
            messagebox.showerror("读取参考曲线失败", str(exc))
            self._log_exception("读取参考曲线失败", exc)

    def load_station(self):
        path = filedialog.askopenfilename(
            title="选择线站侧谱线对应数据文件",
            filetypes=[("Data files", "*.csv *.txt *.dat *.xlsx *.xls"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            df = read_table(path)
            self.station_df = df
            self.station_path = path
            self.result_df = None
            self._last_plot_ref_clean = None
            self.recommendation_confirmed.set(False)
            cols = list(df.columns)
            display_cols = [""] + cols
            self.station_id_combo["values"] = display_cols
            self.station_strain_combo["values"] = display_cols
            self.station_stress_combo["values"] = display_cols
            rec = recommend_station_setup(df)
            self.station_mode.set(rec["mode"])
            self.station_id_col.set(rec["id_col"])
            self.station_strain_col.set(rec["strain_col"])
            self.station_stress_col.set(rec["stress_col"])
            self.station_strain_unit.set(rec["strain_unit"])
            self.station_stress_unit.set(rec["stress_unit"])
            self.station_label.config(text=f"{os.path.basename(path)}  ({len(df)} rows, {len(cols)} cols)")
            self.station_recommendation.set(self._format_station_recommendation(rec, df))
            self._update_station_mode_hint()
            self._set_result_status("线站数据已加载。请检查蓝色推荐并点击确认。", "#24527a")
            self.result_summary.set("确认推荐后才能运行；这是为了避免列和单位误判。")
            self.log(f"已加载线站数据：{path}")
            self.log(f"线站推荐：{rec['mode']}；编号={rec['id_col'] or '行号'}；应变={rec['strain_col'] or '无'}；应力={rec['stress_col'] or '无'}")
            self._update_strain_alignment_hint()
            self._update_stress_alignment_hint()
            self._update_wizard_state_display()
        except Exception as exc:
            messagebox.showerror("读取线站数据失败", str(exc))
            self._log_exception("读取线站数据失败", exc)

    def _prepare_reference(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
        if self.ref_df is None:
            raise ValueError("请先加载参考完整应力-应变曲线。")

        eps_raw = to_numeric_series(self.ref_df, self.ref_strain_col.get(), "参考应变")
        sig_raw = to_numeric_series(self.ref_df, self.ref_stress_col.get(), "参考应力")

        eps = convert_strain_to_fraction(eps_raw, self.ref_strain_unit.get())
        sig = convert_stress_to_mpa(sig_raw, self.ref_stress_unit.get())
        self._warn_strain_unit_selection("参考曲线", eps_raw, self.ref_strain_unit.get(), eps)

        tmp = pd.DataFrame({"strain_fraction": eps, "stress_MPa": sig})
        tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna().sort_values("strain_fraction")
        zero_audit = {
            "start_zero_applied": bool(self.zero_reference.get()),
            "reference_strain_start_offset_fraction": 0.0,
            "reference_strain_start_offset_percent": 0.0,
            "reference_stress_start_offset_MPa": 0.0,
        }
        if self.zero_reference.get() and len(tmp) > 0:
            tmp["strain_fraction"], eps_offset = zero_to_first_finite(tmp["strain_fraction"].to_numpy(dtype=float))
            tmp["stress_MPa"], sig_offset = zero_to_first_finite(tmp["stress_MPa"].to_numpy(dtype=float))
            zero_audit["reference_strain_start_offset_fraction"] = eps_offset
            zero_audit["reference_strain_start_offset_percent"] = eps_offset * 100.0 if np.isfinite(eps_offset) else np.nan
            zero_audit["reference_stress_start_offset_MPa"] = sig_offset

        if len(tmp) < 2:
            raise ValueError("参考曲线有效点数不足。")

        eps_arr = tmp["strain_fraction"].to_numpy(dtype=float)
        sig_arr = tmp["stress_MPa"].to_numpy(dtype=float)
        return eps_arr, sig_arr, tmp, zero_audit

    def _get_spectrum_ids(self) -> pd.Series:
        assert self.station_df is not None
        id_col = self.station_id_col.get()
        if id_col and id_col in self.station_df.columns:
            return self.station_df[id_col]
        return pd.Series(np.arange(1, len(self.station_df) + 1), name="spectrum_index")

    @staticmethod
    def _unique_column_name(columns, base: str) -> str:
        names = set(columns)
        if base not in names:
            return base
        i = 2
        while f"{base}_{i}" in names:
            i += 1
        return f"{base}_{i}"

    def _insert_spectrum_id_column(self, result: pd.DataFrame) -> pd.DataFrame:
        ids = self._get_spectrum_ids().reset_index(drop=True)
        result = result.copy()
        if "spectrum_id" not in result.columns:
            result.insert(0, "spectrum_id", ids.to_numpy())
            return result

        existing = result.pop("spectrum_id").reset_index(drop=True)
        same_ids = (
            len(existing) == len(ids)
            and existing.astype("string").fillna("<NA>").equals(ids.astype("string").fillna("<NA>"))
        )
        if same_ids:
            result.insert(0, "spectrum_id", existing.to_numpy())
        else:
            source_col = self._unique_column_name(result.columns, "source_spectrum_id")
            result.insert(0, source_col, existing.to_numpy())
            result.insert(0, "spectrum_id", ids.to_numpy())
        return result

    def _clear_result_preview(self):
        self.result_df = None
        self._last_plot_ref_clean = None
        if hasattr(self, "tree"):
            for item in self.tree.get_children():
                self.tree.delete(item)
            self.tree["columns"] = []

    def _log_reference_diagnostics(self, eps_ref: np.ndarray, sig_ref: np.ndarray):
        """Log common unit mistakes without stopping the calculation."""
        if len(eps_ref) == 0 or len(sig_ref) == 0:
            return
        eps_min = float(np.nanmin(eps_ref))
        eps_max = float(np.nanmax(eps_ref))
        sig_max = float(np.nanmax(sig_ref))
        self.log(f"参考曲线检查：应变范围 {eps_min*100:.4g}% 到 {eps_max*100:.4g}%；应力最大值约 {sig_max:.4g} MPa。")
        if eps_max > 1.0:
            self.log("⚠️ 参考曲线最大应变超过 100%。如果你的参考文件横坐标是 0–18 这种百分数，请把‘参考应变单位’改成 percent/百分数，而不是 fraction。")
        elif eps_max < 0.01:
            self.log("⚠️ 参考曲线最大应变小于 1%。如果你的参考文件横坐标是 0–18 这种百分数却选成 fraction/percent 错误，请检查单位。")

    def _warn_strain_unit_selection(self, label: str, raw_values, selected_unit: str, converted_fraction) -> None:
        """Warn about common percent/fraction mistakes without changing the user's choice."""
        try:
            raw = pd.to_numeric(pd.Series(raw_values), errors="coerce").to_numpy(dtype=float)
            conv = np.asarray(converted_fraction, dtype=float)
            raw_abs_max = float(np.nanmax(np.abs(raw)))
            conv_max_percent = float(np.nanmax(np.abs(conv)) * 100.0)
        except Exception:
            return

        if not np.isfinite(raw_abs_max) or not np.isfinite(conv_max_percent):
            return

        if selected_unit == STRAIN_FRACTION and raw_abs_max > 1.0 and raw_abs_max <= 100.0:
            self.log(f"⚠️ {label}：你选择了 fraction，但原始应变最大值约 {raw_abs_max:.4g}。如果文件里的 18 表示 18%，请改选 percent；否则图上会显示成 1800%。")
        if conv_max_percent > 100.0:
            self.log(f"⚠️ {label}：换算后最大应变约 {conv_max_percent:.4g}%，对金属拉伸通常不合理。请重点检查 percent/fraction 单位。")
        if selected_unit == STRAIN_PERCENT and raw_abs_max < 0.5:
            self.log(f"ℹ️ {label}：你选择了 percent，原始最大值约 {raw_abs_max:.4g}，会被解释为 {raw_abs_max:.4g}%。如果 0.05 实际表示 5%，应改选 fraction。")


    def _warn_if_no_valid_mapping(self, result: pd.DataFrame, reason_hint: str = ""):
        if "mapping_is_valid" not in result:
            return
        n = len(result)
        n_valid = int(result["mapping_is_valid"].sum())
        n_out = int((~result["within_reference_range"].astype(bool)).sum()) if "within_reference_range" in result else 0
        if n > 0 and n_valid == 0:
            msg = "所有谱线点都没有成功映射：请优先检查模式、列选择和单位。"
            if reason_hint:
                msg += "\n\n" + reason_hint
            self.log("❌ " + msg.replace("\n", " "))
            messagebox.showwarning("映射结果全无效", msg)
        elif n > 0 and n_out > 0.5 * n:
            self.log(f"⚠️ 超过一半的数据点超出参考曲线范围：{n_out}/{n}。请检查参考曲线单位和线站输入单位。")

    def _build_strain_alignment_payload(
        self,
        eps_ref: np.ndarray,
        raw_station_eps: np.ndarray,
        zeroed_station_eps: np.ndarray,
        enabled: bool,
    ) -> Tuple[np.ndarray, dict]:
        raw_station_eps = np.asarray(raw_station_eps, dtype=float)
        zeroed_station_eps = np.asarray(zeroed_station_eps, dtype=float)
        try:
            diag = compute_strain_alignment_diagnostics(eps_ref, zeroed_station_eps)
        except ValueError:
            if enabled:
                raise
            diag = {
                "factor": np.nan,
                "reference_max_strain_fraction": np.nan,
                "station_max_strain_fraction": np.nan,
                "reference_max_strain_percent": np.nan,
                "station_max_strain_percent": np.nan,
                "warning": "",
            }

        if enabled:
            aligned = apply_strain_alignment(zeroed_station_eps, diag)
            applied_factor = float(diag["factor"])
            self.log(
                "已启用塑性应变最大值对齐："
                f"参考最大 {diag['reference_max_strain_percent']:.4g}%，"
                f"线站最大 {diag['station_max_strain_percent']:.4g}%，"
                f"系数 {applied_factor:.6g}。"
            )
            if diag["warning"]:
                self.log("⚠️ 应变对齐：" + diag["warning"])
        else:
            aligned = zeroed_station_eps
            applied_factor = 1.0

        audit = {
            "raw_station_strain_fraction": raw_station_eps,
            "raw_station_strain_percent": raw_station_eps * 100.0,
            "zeroed_station_strain_fraction": zeroed_station_eps,
            "zeroed_station_strain_percent": zeroed_station_eps * 100.0,
            "aligned_station_strain_fraction": aligned,
            "aligned_station_strain_percent": aligned * 100.0,
            "strain_alignment_factor": applied_factor,
            "strain_alignment_applied": bool(enabled),
            "reference_max_strain_percent": diag["reference_max_strain_percent"],
            "station_max_strain_percent": diag["station_max_strain_percent"],
        }
        return aligned, audit

    @staticmethod
    def _add_strain_alignment_audit_columns(result: pd.DataFrame, audit: dict) -> None:
        for col, value in audit.items():
            result[col] = value

    def _build_stress_alignment_payload(
        self,
        sig_ref: np.ndarray,
        raw_station_sig: np.ndarray,
        zeroed_station_sig: np.ndarray,
        enabled: bool,
    ) -> Tuple[np.ndarray, dict]:
        raw_station_sig = np.asarray(raw_station_sig, dtype=float)
        zeroed_station_sig = np.asarray(zeroed_station_sig, dtype=float)
        try:
            diag = compute_stress_alignment_diagnostics(sig_ref, zeroed_station_sig)
        except ValueError:
            if enabled:
                raise
            diag = {
                "factor": np.nan,
                "reference_max_stress_MPa": np.nan,
                "station_max_stress_MPa": np.nan,
                "warning": "",
            }

        if enabled:
            aligned = apply_stress_alignment(zeroed_station_sig, diag)
            applied_factor = float(diag["factor"])
            self.log(
                "已启用应力最大值对齐："
                f"参考最大 {diag['reference_max_stress_MPa']:.4g} MPa，"
                f"线站最大 {diag['station_max_stress_MPa']:.4g} MPa，"
                f"系数 {applied_factor:.6g}。"
            )
            if diag["warning"]:
                self.log("⚠️ 应力对齐：" + diag["warning"])
        else:
            aligned = zeroed_station_sig
            applied_factor = 1.0

        audit = {
            "raw_station_stress_MPa": raw_station_sig,
            "zeroed_station_stress_MPa": zeroed_station_sig,
            "aligned_station_stress_MPa": aligned,
            "stress_alignment_factor": applied_factor,
            "stress_alignment_applied": bool(enabled),
            "reference_max_stress_MPa": diag["reference_max_stress_MPa"],
            "station_max_stress_MPa": diag["station_max_stress_MPa"],
        }
        return aligned, audit

    @staticmethod
    def _add_stress_alignment_audit_columns(result: pd.DataFrame, audit: dict) -> None:
        for col, value in audit.items():
            result[col] = value

    def _apply_start_zeroing(self, values) -> Tuple[np.ndarray, float]:
        arr = np.asarray(values, dtype=float)
        if self.zero_reference.get():
            return zero_to_first_finite(arr)
        return arr.copy(), 0.0

    @staticmethod
    def _add_start_zero_audit_columns(
        result: pd.DataFrame,
        reference_zero_audit: dict,
        station_strain_offset: float = np.nan,
        station_stress_offset: float = np.nan,
    ) -> None:
        for col, value in reference_zero_audit.items():
            result[col] = value
        result["station_strain_start_offset_fraction"] = station_strain_offset
        result["station_strain_start_offset_percent"] = (
            station_strain_offset * 100.0 if np.isfinite(station_strain_offset) else np.nan
        )
        result["station_stress_start_offset_MPa"] = station_stress_offset

    def run_mapping(self):
        if self._mapping_running:
            return
        started = False
        try:
            if self.station_df is None:
                raise ValueError("请先加载线站侧谱线对应数据。")
            if self.ref_df is None:
                raise ValueError("请先加载参考完整应力-应变曲线。")
            if not bool(self.recommendation_confirmed.get()):
                messagebox.showwarning("请先确认推荐", "请先在第 2 步检查并确认推荐的模式、列和单位。")
                if hasattr(self, "wizard"):
                    self.wizard.select(1)
                self._update_wizard_state_display()
                return

            self._mapping_running = True
            started = True
            self._clear_result_preview()
            self._set_result_status("正在运行映射，请稍候。", "#24527a")
            self._update_wizard_state_display()
            if hasattr(self, "run_button"):
                self.run_button.config(state="disabled")
            self.root.config(cursor="watch")
            self.root.update_idletasks()

            eps_ref, sig_ref, ref_clean, reference_zero_audit = self._prepare_reference()
            self._log_reference_diagnostics(eps_ref, sig_ref)
            if reference_zero_audit["start_zero_applied"]:
                self.log(
                    "已启用起点归零："
                    f"参考应变偏移 {reference_zero_audit['reference_strain_start_offset_percent']:.4g}%，"
                    f"参考应力偏移 {reference_zero_audit['reference_stress_start_offset_MPa']:.4g} MPa。"
                )
            mode = self.station_mode.get()
            method = self.interp_method.get()

            result = self._insert_spectrum_id_column(self.station_df)

            if mode == MODE_STRAIN_ONLY:
                strain_col = self.station_strain_col.get()
                eps_station_raw = to_numeric_series(self.station_df, strain_col, "线站应变")
                eps_station = convert_strain_to_fraction(eps_station_raw, self.station_strain_unit.get())
                self._warn_strain_unit_selection("线站应变", eps_station_raw, self.station_strain_unit.get(), eps_station)
                if np.isfinite(eps_station).any():
                    self.log(f"线站应变检查：{strain_col} 范围 {np.nanmin(eps_station)*100:.4g}% 到 {np.nanmax(eps_station)*100:.4g}%。")
                eps_station_zeroed, station_strain_offset = self._apply_start_zeroing(eps_station)
                if reference_zero_audit["start_zero_applied"]:
                    self.log(
                        f"起点归零：线站应变偏移 {station_strain_offset * 100.0:.4g}%，"
                        f"归零后范围 {np.nanmin(eps_station_zeroed)*100:.4g}% 到 {np.nanmax(eps_station_zeroed)*100:.4g}%。"
                    )

                eps_for_mapping, alignment_audit = self._build_strain_alignment_payload(
                    eps_ref,
                    eps_station,
                    eps_station_zeroed,
                    enabled=bool(self.align_strain_max_to_reference.get()),
                )
                f, xmin, xmax, method_used, interp_diag = make_interpolator_with_diagnostics(eps_ref, sig_ref, method=method)
                sig_mapped = f(eps_for_mapping)
                in_range = (eps_for_mapping >= xmin) & (eps_for_mapping <= xmax)

                self._add_start_zero_audit_columns(
                    result,
                    reference_zero_audit,
                    station_strain_offset=station_strain_offset,
                )
                self._add_strain_alignment_audit_columns(result, alignment_audit)
                result["mapped_strain_fraction"] = eps_for_mapping
                result["mapped_strain_percent"] = eps_for_mapping * 100.0
                result["mapped_stress_MPa"] = sig_mapped
                result["reference_duplicate_strain_groups"] = interp_diag["duplicate_groups"]
                result["reference_duplicate_strain_rows"] = interp_diag["duplicate_rows"]
                result["input_type"] = "strain"
                result["input_column"] = strain_col
                result["interpolation"] = f"epsilon_to_sigma / {method_used}"
                result["within_reference_range"] = in_range
                self.log(f"完成 ε→σ 映射：{method_used}；参考应变范围 {xmin*100:.4g}% 到 {xmax*100:.4g}%。")
                if interp_diag["duplicate_groups"] > 0:
                    self.log(
                        "⚠️ 参考曲线存在重复应变点："
                        f"{interp_diag['duplicate_groups']} 组、{interp_diag['duplicate_rows']} 行已按应力均值聚合后插值。"
                    )

            elif mode == MODE_STRESS_ONLY:
                stress_col = self.station_stress_col.get()
                sig_station_raw = to_numeric_series(self.station_df, stress_col, "线站应力")
                sig_station = convert_stress_to_mpa(sig_station_raw, self.station_stress_unit.get())
                if np.isfinite(sig_station).any():
                    self.log(f"线站应力检查：{stress_col} 范围 {np.nanmin(sig_station):.4g} 到 {np.nanmax(sig_station):.4g} MPa。")
                    if np.nanmax(np.abs(sig_ref)) > 100 and np.nanmax(np.abs(sig_station)) < 10:
                        self.log("⚠️ 线站应力最大值小于 10 MPa，而参考曲线是几百/上千 MPa。强烈怀疑你把应变列当作应力列，或模式选错。")
                sig_station_zeroed, station_stress_offset = self._apply_start_zeroing(sig_station)
                if reference_zero_audit["start_zero_applied"]:
                    self.log(
                        f"起点归零：线站应力偏移 {station_stress_offset:.4g} MPa，"
                        f"归零后范围 {np.nanmin(sig_station_zeroed):.4g} 到 {np.nanmax(sig_station_zeroed):.4g} MPa。"
                    )

                sig_for_mapping, stress_alignment_audit = self._build_stress_alignment_payload(
                    sig_ref,
                    sig_station,
                    sig_station_zeroed,
                    enabled=bool(self.align_stress_max_to_reference.get()),
                )
                inv_branch, inverse_diag = build_inverse_branch_with_diagnostics(
                    eps_ref,
                    sig_ref,
                    use_pre_peak=self.inverse_pre_peak.get(),
                    monotonicize=self.inverse_monotonic.get(),
                )
                f, xmin, xmax, method_used = make_interpolator(inv_branch["stress"], inv_branch["eps"], method=method)
                eps_mapped = f(sig_for_mapping)
                in_range = (sig_for_mapping >= xmin) & (sig_for_mapping <= xmax)
                intervals = compute_inverse_strain_intervals(
                    eps_ref,
                    sig_ref,
                    sig_for_mapping,
                    use_pre_peak=self.inverse_pre_peak.get(),
                )

                self._add_start_zero_audit_columns(
                    result,
                    reference_zero_audit,
                    station_stress_offset=station_stress_offset,
                )
                self._add_stress_alignment_audit_columns(result, stress_alignment_audit)
                result["mapped_stress_MPa"] = sig_for_mapping
                result["mapped_strain_fraction"] = eps_mapped
                result["mapped_strain_percent"] = eps_mapped * 100.0
                for col in intervals.columns:
                    result[col] = intervals[col].to_numpy()
                result["inverse_duplicate_stress_groups"] = inverse_diag["duplicate_groups"]
                result["inverse_duplicate_stress_rows"] = inverse_diag["duplicate_rows"]
                result["input_type"] = "stress"
                result["input_column"] = stress_col
                result["interpolation"] = f"sigma_to_epsilon / {method_used}"
                result["within_reference_range"] = in_range
                self.log(f"完成 σ→ε 反映射：{method_used}；使用反插值分支应力范围 {xmin:.4g} 到 {xmax:.4g} MPa。")
                self.log("提醒：应力→应变反查在屈服平台、锯齿流变、颈缩后可能不是唯一解，结果应视为等效宏观应变。")
                if inverse_diag["duplicate_groups"] > 0:
                    self.log(
                        "⚠️ 反插值分支存在重复应力点："
                        f"{inverse_diag['duplicate_groups']} 组、{inverse_diag['duplicate_rows']} 行已按应变均值生成标量映射。"
                    )
                n_ambiguous = int(intervals["inverse_mapping_is_ambiguous"].sum())
                if n_ambiguous > 0:
                    self.log(f"⚠️ {n_ambiguous} 行应力反查存在多个可能应变；已导出 inverse_strain_min/max 供审计。")

            elif mode == MODE_BOTH:
                strain_col = self.station_strain_col.get()
                stress_col = self.station_stress_col.get()
                eps_station_raw = to_numeric_series(self.station_df, strain_col, "线站应变")
                sig_station_raw = to_numeric_series(self.station_df, stress_col, "线站应力")
                eps_station = convert_strain_to_fraction(eps_station_raw, self.station_strain_unit.get())
                sig_station = convert_stress_to_mpa(sig_station_raw, self.station_stress_unit.get())
                self._warn_strain_unit_selection("线站应变", eps_station_raw, self.station_strain_unit.get(), eps_station)
                eps_station_zeroed, station_strain_offset = self._apply_start_zeroing(eps_station)
                sig_station_zeroed, station_stress_offset = self._apply_start_zeroing(sig_station)
                if reference_zero_audit["start_zero_applied"]:
                    self.log(
                        f"起点归零：线站应变偏移 {station_strain_offset * 100.0:.4g}%，"
                        f"线站应力偏移 {station_stress_offset:.4g} MPa。"
                    )

                eps_for_mapping, alignment_audit = self._build_strain_alignment_payload(
                    eps_ref,
                    eps_station,
                    eps_station_zeroed,
                    enabled=bool(self.align_strain_max_to_reference.get()),
                )
                sig_for_mapping, stress_alignment_audit = self._build_stress_alignment_payload(
                    sig_ref,
                    sig_station,
                    sig_station_zeroed,
                    enabled=bool(self.align_stress_max_to_reference.get()),
                )
                self._add_start_zero_audit_columns(
                    result,
                    reference_zero_audit,
                    station_strain_offset=station_strain_offset,
                    station_stress_offset=station_stress_offset,
                )
                self._add_strain_alignment_audit_columns(result, alignment_audit)
                self._add_stress_alignment_audit_columns(result, stress_alignment_audit)
                result["mapped_strain_fraction"] = eps_for_mapping
                result["mapped_strain_percent"] = eps_for_mapping * 100.0
                result["mapped_stress_MPa"] = sig_for_mapping
                result["input_type"] = "both"
                result["input_column"] = f"strain={strain_col}; stress={stress_col}"
                result["interpolation"] = "none / unit conversion only"
                result["within_reference_range"] = within_reference_ranges(eps_for_mapping, sig_for_mapping, eps_ref, sig_ref)
                self.log("线站数据已有应力和应变：已完成单位换算、整合和可视化。")
            else:
                raise ValueError("未知模式。")

            # Mark NaN mapping explicitly.
            result["mapping_is_valid"] = np.isfinite(result["mapped_strain_fraction"]) & np.isfinite(result["mapped_stress_MPa"])
            self._apply_smoothing_to_result(result)
            self.result_df = result
            self._last_plot_ref_clean = ref_clean.copy()
            self._update_plot(ref_clean, result)
            self._update_table(result)

            n_bad = int((~result["mapping_is_valid"]).sum())
            n_out = int((~result["within_reference_range"].astype(bool)).sum()) if "within_reference_range" in result else 0
            n_valid = len(result) - n_bad
            n_ambiguous = int(result["inverse_mapping_is_ambiguous"].sum()) if "inverse_mapping_is_ambiguous" in result else 0
            self.log(f"映射完成：共 {len(result)} 行；无效/NaN {n_bad} 行；超出参考范围 {n_out} 行。")
            if n_valid == 0:
                self._set_result_status("映射失败：没有有效结果。", "#b00020")
            elif n_out > 0 or n_ambiguous > 0:
                self._set_result_status("映射完成，但存在需要检查的风险提示。", "#8a6d1d")
            else:
                self._set_result_status("映射完成，可以导出结果。", "#2e7d32")
            self.result_summary.set(
                f"总行数 {len(result)}；有效 {n_valid}；无效/NaN {n_bad}；超出参考范围 {n_out}；"
                f"反插值歧义 {n_ambiguous}。"
            )
            self._update_wizard_state_display()
            if hasattr(self, "wizard"):
                self.wizard.select(2)
            hint = ""
            if mode == MODE_STRESS_ONLY:
                hint = "你当前使用的是 σ→ε 反插值模式。若线站文件第二列是 0.00002、0.0015、0.18 这类小数工程应变，请改用‘线站只有应变：由参考曲线插值得到应力’。"
            elif mode == MODE_STRAIN_ONLY:
                hint = "你当前使用的是 ε→σ 插值模式。请确认线站应变单位：0.05 表示 5% 时选 fraction；5 表示 5% 时选 percent。"
            self._warn_if_no_valid_mapping(result, hint)
        except Exception as exc:
            self._set_result_status("映射失败，请检查文件、列和单位。", "#b00020")
            self.result_summary.set(str(exc))
            self._update_wizard_state_display()
            messagebox.showerror("映射失败", str(exc))
            self._log_exception("映射失败", exc)
        finally:
            if started:
                self._mapping_running = False
                self.root.config(cursor="")
                self._update_wizard_state_display()

    def _apply_smoothing_to_result(self, result: pd.DataFrame) -> None:
        method = self.smooth_method.get()
        if method == SMOOTH_NONE or not self.show_smoothed.get():
            return
        try:
            window = int(self.smooth_window.get())
            poly = int(self.smooth_polyorder.get())
            result["smooth_mapped_stress_MPa"] = smooth_numeric_series(result["mapped_stress_MPa"], method, window=window, polyorder=poly)
            result["smooth_mapped_strain_percent"] = smooth_numeric_series(result["mapped_strain_percent"], method, window=window, polyorder=poly)
            result["smoothing_method"] = method
            result["smoothing_window"] = window
            self.log(f"已生成平滑辅助列：{method}，窗口 {window} 点。注意：原始 mapped_* 列未被覆盖。")
            if method == SMOOTH_SAVGOL and not SCIPY_SIGNAL_AVAILABLE:
                self.log("⚠️ 未检测到 scipy.signal，Savitzky-Golay 已自动降级为滚动中位数。")
        except Exception as exc:
            self.log(f"⚠️ 平滑失败，已跳过平滑：{exc}")


    def _update_plot(self, ref_clean: pd.DataFrame, result: pd.DataFrame):
        preset = self._current_plot_preset()
        self.fig.clear()
        self.fig.patch.set_facecolor("#ffffff")
        self.ax_curve = self.fig.add_subplot(211)
        self.ax_series = self.fig.add_subplot(212)

        valid = result["mapping_is_valid"].to_numpy(dtype=bool)
        invalid = ~valid
        use_raw = bool(self.show_raw_points.get())
        use_smooth = bool(self.show_smoothed.get()) and "smooth_mapped_stress_MPa" in result.columns

        # Top: reference curve and mapped points in stress-strain space.
        # Use high-contrast colors and hollow markers so the mapped points remain visible on top of the reference curve.
        self.ax_curve.plot(
            ref_clean["strain_fraction"] * 100.0,
            ref_clean["stress_MPa"],
            color=PLOT_COLORS["reference"],
            linewidth=preset["line_width"] * 1.15,
            alpha=0.92,
            label="Reference curve",
            zorder=1,
        )
        if use_raw:
            self.ax_curve.scatter(
                result.loc[valid, "mapped_strain_percent"],
                result.loc[valid, "mapped_stress_MPa"],
                s=preset["scatter_size"],
                marker="o",
                facecolors="none",
                edgecolors=PLOT_COLORS["mapped"],
                linewidths=preset["axis_width"] * 1.35,
                label="Mapped raw points",
                zorder=3,
            )
        if use_smooth:
            smooth_ok = np.isfinite(result["smooth_mapped_strain_percent"]) & np.isfinite(result["smooth_mapped_stress_MPa"])
            self.ax_curve.plot(
                result.loc[smooth_ok, "smooth_mapped_strain_percent"],
                result.loc[smooth_ok, "smooth_mapped_stress_MPa"],
                color=PLOT_COLORS["smooth"],
                linewidth=preset["line_width"] * 1.1,
                label="Smoothed guide",
                zorder=2,
            )
        if invalid.any():
            self.ax_curve.scatter(
                result.loc[invalid, "mapped_strain_percent"],
                result.loc[invalid, "mapped_stress_MPa"],
                s=preset["scatter_size"] * 1.15,
                marker="x",
                color=PLOT_COLORS["invalid"],
                linewidths=preset["axis_width"] * 1.2,
                label="Invalid / out of range",
                zorder=4,
            )

        self.ax_curve.set_xlabel("Engineering strain / %")
        self.ax_curve.set_ylabel("Engineering stress / MPa")
        self.ax_curve.set_title("Reference curve + mapped in-situ points", fontsize=preset["title_size"], color="#0f172a")
        self.ax_curve.legend(loc="best", frameon=False, fontsize=preset["legend_size"], handlelength=2.0)

        # Unit sanity annotation: useful when percent/fraction is selected incorrectly.
        try:
            xmax = float(np.nanmax(ref_clean["strain_fraction"].to_numpy(dtype=float) * 100.0))
            if xmax > 100:
                self.ax_curve.text(
                    0.01,
                    0.96,
                    "Unit warning: strain range > 100%; check percent/fraction",
                    transform=self.ax_curve.transAxes,
                    va="top",
                    ha="left",
                    fontsize=preset["font_size"],
                    color="#b22222",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#b22222", alpha=0.85),
                )
        except Exception:
            pass

        # Bottom: mapped stress/strain vs spectrum order.
        x = np.arange(1, len(result) + 1)
        try:
            sid_numeric = pd.to_numeric(result["spectrum_id"], errors="coerce")
            if sid_numeric.notna().sum() == len(result):
                x = sid_numeric.to_numpy(dtype=float)
                xlabel = "Spectrum / frame ID"
            else:
                xlabel = "Row order"
        except Exception:
            xlabel = "Row order"

        if use_raw:
            self.ax_series.plot(
                x,
                result["mapped_stress_MPa"],
                marker="o",
                markersize=preset["marker_size"],
                linewidth=preset["line_width"],
                color=PLOT_COLORS["stress"],
                alpha=0.85,
                label="Stress raw / MPa",
            )
        if use_smooth:
            self.ax_series.plot(
                x,
                result["smooth_mapped_stress_MPa"],
                linewidth=preset["line_width"] * 1.1,
                color=PLOT_COLORS["mapped"],
                label="Stress smoothed / MPa",
            )
        self.ax_series.set_xlabel(xlabel)
        self.ax_series.set_ylabel("Stress / MPa")

        ax2 = self.ax_series.twinx()
        if use_raw:
            ax2.plot(
                x,
                result["mapped_strain_percent"],
                marker="s",
                markersize=max(2.4, preset["marker_size"] * 0.9),
                linewidth=preset["line_width"],
                linestyle="--",
                color=PLOT_COLORS["strain"],
                alpha=0.78,
                label="Strain raw / %",
            )
        if use_smooth:
            ax2.plot(
                x,
                result["smooth_mapped_strain_percent"],
                linewidth=preset["line_width"] * 1.05,
                linestyle="--",
                color=PLOT_COLORS["smooth"],
                label="Strain smoothed / %",
            )
        ax2.set_ylabel("Strain / %")

        lines1, labels1 = self.ax_series.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        self.ax_series.legend(lines1 + lines2, labels1 + labels2, loc="best", frameon=False, fontsize=preset["legend_size"], handlelength=2.0)
        self.ax_series.set_title("Mapped stress and strain vs spectrum/frame order", fontsize=preset["title_size"], color="#0f172a")

        self._apply_axis_style(self.ax_curve, preset)
        self._apply_axis_style(self.ax_series, preset)
        self._apply_axis_style(ax2, preset, grid=False)

        self._apply_figure_layout()
        self.canvas.draw()

    def _update_table(self, df: pd.DataFrame):
        # Select important columns first, then append a few original columns if space allows.
        priority = [
            "spectrum_id",
            "mapped_strain_fraction",
            "mapped_strain_percent",
            "mapped_stress_MPa",
            "start_zero_applied",
            "reference_strain_start_offset_fraction",
            "reference_strain_start_offset_percent",
            "reference_stress_start_offset_MPa",
            "station_strain_start_offset_fraction",
            "station_strain_start_offset_percent",
            "station_stress_start_offset_MPa",
            "raw_station_strain_fraction",
            "raw_station_strain_percent",
            "zeroed_station_strain_fraction",
            "zeroed_station_strain_percent",
            "aligned_station_strain_fraction",
            "aligned_station_strain_percent",
            "raw_station_stress_MPa",
            "zeroed_station_stress_MPa",
            "aligned_station_stress_MPa",
            "strain_alignment_factor",
            "strain_alignment_applied",
            "reference_max_strain_percent",
            "station_max_strain_percent",
            "stress_alignment_factor",
            "stress_alignment_applied",
            "reference_max_stress_MPa",
            "station_max_stress_MPa",
            "smooth_mapped_stress_MPa",
            "smooth_mapped_strain_percent",
            "within_reference_range",
            "mapping_is_valid",
            "inverse_mapping_is_ambiguous",
            "inverse_strain_min_fraction",
            "inverse_strain_max_fraction",
            "inverse_strain_min_percent",
            "inverse_strain_max_percent",
            "inverse_ambiguity_width_percent",
            "inverse_duplicate_stress_groups",
            "inverse_duplicate_stress_rows",
            "reference_duplicate_strain_groups",
            "reference_duplicate_strain_rows",
            "input_type",
            "input_column",
            "interpolation",
        ]
        remaining = [c for c in df.columns if c not in priority]
        cols = [c for c in priority if c in df.columns] + remaining[:6]
        preview = df[cols].head(200).copy()

        for item in self.tree.get_children():
            self.tree.delete(item)
        self.tree["columns"] = cols
        for col in cols:
            self.tree.heading(col, text=col)
            width = 120
            if col in {"spectrum_id", "input_type", "interpolation"}:
                width = 110
            elif col.startswith("mapped_") or col.startswith("smooth_"):
                width = 150
            elif "offset" in col or "alignment" in col or "duplicate" in col or "inverse" in col:
                width = 180
            self.tree.column(col, width=width, minwidth=90, anchor="center", stretch=True)
        self.tree.tag_configure("odd", background="#f8fafc")
        self.tree.tag_configure("even", background="#ffffff")

        for row_index, (_, row) in enumerate(preview.iterrows()):
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, (float, np.floating)):
                    if np.isfinite(val):
                        vals.append(f"{val:.8g}")
                    else:
                        vals.append("NaN")
                else:
                    vals.append(str(val))
            tag = "odd" if row_index % 2 else "even"
            self.tree.insert("", tk.END, values=vals, tags=(tag,))

    @staticmethod
    def _build_export_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        priority = ["spectrum_id", "mapped_strain_percent", "mapped_stress_MPa"]
        cols = [col for col in priority if col in df.columns]
        cols.extend(col for col in df.columns if col not in cols)
        return df.loc[:, cols].copy()

    def export_result(self):
        if self.result_df is None:
            messagebox.showwarning("没有结果", "请先运行映射。")
            return
        path = filedialog.asksaveasfilename(
            title="导出映射结果",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx")],
        )
        if not path:
            return
        try:
            export_df = self._build_export_dataframe(self.result_df)
            if path.lower().endswith(".xlsx"):
                export_df.to_excel(path, index=False)
            else:
                export_df.to_csv(path, index=False, encoding="utf-8-sig")
            self.log(f"已导出结果：{path}")
            messagebox.showinfo("导出完成", f"已导出：\n{path}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))
            self._log_exception("导出结果失败", exc)

    def save_plot(self):
        if self.result_df is None:
            messagebox.showwarning("没有图", "请先运行映射。")
            return
        preset = self._current_plot_preset()
        path = filedialog.asksaveasfilename(
            title=f"保存当前图 - {self.plot_preset.get()}",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("TIFF", "*.tif *.tiff"),
                ("PDF", "*.pdf"),
                ("SVG", "*.svg"),
                ("EPS", "*.eps"),
            ],
        )
        if not path:
            return
        original_size = self.fig.get_size_inches().copy()
        try:
            self._apply_figure_layout(for_export=True)
            self.fig.savefig(path, **self._savefig_kwargs(path, preset))
            self.log(f"已保存图：{path}")
            self.log(
                f"导出预设：{self.plot_preset.get()}；尺寸 {preset['figsize'][0]} x {preset['figsize'][1]} inch；"
                f"dpi {preset['dpi']}（矢量格式忽略 dpi）。"
            )
            messagebox.showinfo("保存完成", f"已保存：\n{path}\n\n预设：{self.plot_preset.get()}")
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))
            self._log_exception("保存图像失败", exc)
        finally:
            self.fig.set_size_inches(*original_size, forward=True)
            self._apply_figure_layout()
            self.canvas.draw_idle()


def main():
    root = tk.Tk()
    app = StressStrainMapperApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
