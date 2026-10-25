"""Self-contained, offline alignment evidence; no GUI backend is imported."""

from __future__ import annotations

import base64
from html import escape
from io import BytesIO
import json

import numpy as np

from .storage import json_value


def make_report(state, frames, table, sources, features):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    sections = []
    times = np.asarray([row.get("t_mid_s") for row in frames], dtype=float)
    zero = float(times[np.isfinite(times)][0]) if np.isfinite(times).any() else 0.0
    signals = (features or {}).get("signals", [])
    for source in sources:
        decision = state.get("decisions", {}).get(source["id"], {})
        dt = decision.get("dt_s")
        candidate = decision.get("candidate_dt_s")
        for name, channel in source["channels"].items():
            figure = Figure(figsize=(9, 3.4), layout="constrained")
            FigureCanvasAgg(figure)
            axis = figure.subplots()
            t = np.asarray(source["t_s"], dtype=float)
            y = np.asarray(channel["values"], dtype=float)
            axis.plot(t - zero, y, color="0.6", label="External record before offset")
            proposed = dt if dt is not None else candidate
            if proposed is not None:
                label = "Accepted time mapping" if dt is not None else "Candidate only; confirmation required"
                axis.plot(t - proposed - zero, y, "-" if dt is not None else "--", color="#156b89", label=label)
            key = f"{source['id']}.{name}"
            sampled = [row.get(f"{key}.mid") for row in table]
            if len(sampled) == len(times):
                axis.scatter(times - zero, np.asarray(sampled, dtype=float), s=9, color="#be5918", label="Frame midpoint")
            axis.set_xlabel("Time relative to first detector midpoint (s)")
            axis.set_ylabel(f"{name} ({channel['unit']})")
            axis.set_title(f"{source['id']} / {name}: {decision.get('status', 'unavailable')}")
            if signals:
                second = axis.twinx()
                for signal in signals[:3]:
                    st = np.asarray(signal["t_s"], dtype=float)
                    sy = np.asarray(signal["values"], dtype=float)
                    valid = np.asarray(signal.get("valid", np.isfinite(sy)), dtype=bool)
                    second.plot(st[valid] - zero, sy[valid], alpha=.55, linewidth=1, label=signal["id"])
                second.set_ylabel("Relative diffraction feature (fraction)")
            axis.legend(loc="best", fontsize=8)
            buffer = BytesIO()
            figure.savefig(buffer, format="png", dpi=120)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            sections.append(f'<img alt="{escape(key)} alignment evidence" src="data:image/png;base64,{encoded}">')
            figure.clear()
    summary = {key: state.get(key) for key in ("status", "experiment", "n_frames", "n_selected", "n_valid_analysis", "issues", "decisions", "frame_metadata", "stages")}
    details = escape(json.dumps(json_value(summary), ensure_ascii=False, indent=2, allow_nan=False))
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>InSituLink alignment report</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 20px;color:#17313c}}img{{width:100%;height:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f7;padding:16px}}a{{color:#126384}}</style>
<h1>InSituLink</h1><p>状态：<strong>{escape(str(state['status']))}</strong>。
保留 {state.get('n_frames', 0)} 帧，其中可用于所选分析的帧为 {state.get('n_valid_analysis', 0)}。</p>
<p>每个原始帧独立对应外部参数；未解决的时间关系保留为待确认。图中虚线仅为候选结果。
本报告记录软件计算和输入依据，不证明实验标定或材料机制。合成数据验证不代表真实线站精度。</p>
<p><a href="frames.csv">逐帧参数表</a> · <a href="frame_manifest.json">原始帧清单</a> · <a href="decisions.json">时间偏移依据</a> · <a href="handoff.json">DIDASFlow 交接包</a></p>
{''.join(sections)}<h2>计算与判断记录</h2><pre>{details}</pre></html>'''
