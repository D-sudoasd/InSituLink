"""Self-contained, offline frame records and alignment evidence."""
from __future__ import annotations

import base64
from collections import Counter
from html import escape
from io import BytesIO
import json
import numpy as np

from .numerics import _gap_limit
from .report_assets import REPORT_CSS, REPORT_SCRIPT
from .storage import json_value


def _json(value):
    return json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False)


def _embedded_json(value):
    """Keep source text inert inside an HTML script-data element."""
    return (_json(value).replace("&", "\\u0026").replace("<", "\\u003c")
            .replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def _text(value):
    if value is None or value == "":
        return "未提供"
    return escape(_json(value) if isinstance(value, (dict, list, tuple)) else str(value))


def _plot_coordinates(times, values, max_gap_s=None, valid=None):
    """Keep NaNs; break unsupported intervals before steps-post/linear plotting."""
    t, y = np.asarray(times, dtype=float), np.asarray(values, dtype=float).copy()
    if t.ndim != 1 or y.ndim != 1 or len(t) != len(y):
        return np.asarray([]), np.asarray([])
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != y.shape:
            return np.asarray([]), np.asarray([])
        y[~mask] = np.nan
    finite_times = t[np.isfinite(t)]
    increasing = finite_times[np.r_[True, np.diff(finite_times) > 0]] if len(finite_times) else finite_times
    limit = _gap_limit(increasing, max_gap_s)
    x_plot, y_plot = [], []
    for i, (x, value) in enumerate(zip(t, y)):
        if i and (not np.isfinite(t[i - 1]) or not np.isfinite(x)
                  or x <= t[i - 1] or x - t[i - 1] > limit):
            x_plot.append(np.nan)
            y_plot.append(np.nan)
        x_plot.append(x)
        y_plot.append(value if np.isfinite(value) else np.nan)
    return np.asarray(x_plot), np.asarray(y_plot)


def _plots(state, frames, table, sources, features):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    sections = []
    local_times = any("t_mid_rel_s" in row for row in frames)
    times = np.asarray([row.get("t_mid_rel_s" if local_times else "t_mid_s") for row in frames], dtype=float)
    zero = 0.0 if local_times else float(times[np.isfinite(times)][0]) if np.isfinite(times).any() else 0.0
    frame_origin = next((row.get("time_origin_s") for row in frames if row.get("time_origin_s") is not None), "0")
    signals = (features or {}).get("signals", [])
    for source in sources:
        decision = state.get("decisions", {}).get(source["id"], {})
        dt, candidate = decision.get("dt_s"), decision.get("candidate_dt_s")
        source_times = source.get("t_rel_s", source["t_s"]) if local_times else source["t_s"]
        if local_times:
            from .time_coordinates import clock_offset_local_s

            source_origin = source.get("time_origin_s", "0")
            dt = (decision["local_dt_s"] if "local_dt_s" in decision else
                  clock_offset_local_s(dt, frame_origin, source_origin) if dt is not None else None)
            candidate = (decision["local_candidate_dt_s"] if "local_candidate_dt_s" in decision else
                         clock_offset_local_s(candidate, frame_origin, source_origin) if candidate is not None else None)
        for name, channel in source["channels"].items():
            figure = Figure(figsize=(9, 3.4), layout="constrained")
            FigureCanvasAgg(figure)
            axis = figure.subplots()
            t, y = _plot_coordinates(source_times, channel["values"], channel.get("max_gap_s"))
            style = "steps-post" if channel.get("kind") == "step" else "default"
            axis.plot(t - zero, y, color="0.6", drawstyle=style,
                      label="External local clock (unmapped)" if local_times else "External record before offset")
            proposed = dt if dt is not None else candidate
            if proposed is not None:
                label = "Accepted time mapping" if dt is not None else "Candidate only; confirmation required"
                axis.plot(t - proposed - zero, y, "-" if dt is not None else "--",
                          drawstyle=style, color="#156b89", label=label)
            key = f"{source['id']}.{name}"
            sampled = np.asarray([row.get(f"{key}.mid") for row in table], dtype=float)
            if len(sampled) == len(times):
                supported = np.asarray([row.get(f"{key}.valid") is True for row in table], dtype=bool)
                axis.scatter(times[supported] - zero, sampled[supported], s=9,
                             color="#be5918", label="Frame midpoint; full exposure supported")
                partial = ~supported & np.isfinite(sampled)
                if partial.any():
                    axis.scatter(times[partial] - zero, sampled[partial], s=22, marker="x",
                                 color="#984b79", label="Midpoint only; incomplete exposure")
            axis.set_xlabel("Local time (s); gray: external clock, mapped: detector clock" if local_times
                            else "Time relative to first detector midpoint (s)")
            axis.set_ylabel(f"{name} ({channel.get('unit', '')})")
            axis.set_title(f"{source['id']} / {name}: {decision.get('status', 'unavailable')}")
            if signals:
                second = axis.twinx()
                for signal in signals[:3]:
                    st, sy = _plot_coordinates(signal.get("t_s", []), signal.get("values", []),
                                              signal.get("max_gap_s"), signal.get("valid"))
                    second.plot(st - zero, sy, alpha=.55, linewidth=1, label=signal.get("id", "feature"))
                second.set_ylabel("Relative diffraction feature (fraction)")
                handles, labels = axis.get_legend_handles_labels()
                others, other_labels = second.get_legend_handles_labels()
                axis.legend(handles + others, labels + other_labels, loc="best", fontsize=7)
            else:
                axis.legend(loc="best", fontsize=7)
            buffer = BytesIO()
            figure.savefig(buffer, format="png", dpi=120)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            origins = (f' 探测器起点：{_text(frame_origin)} s；外部记录起点：{_text(source.get("time_origin_s", "0"))} s。'
                       '灰线使用外部设备自身的局部时钟，不能据此认定同步。' if local_times else '')
            sections.append(f'<figure><img alt="{escape(key, quote=True)} alignment evidence" '
                            f'src="data:image/png;base64,{encoded}"><figcaption>{_text(key)} · '
                            f'{"右连续前值保持" if style == "steps-post" else "连续记录"}；缺值和超过允许间隔的区间断线。{origins}</figcaption></figure>')
            figure.clear()
    return "".join(sections)


def _source_descriptions(sources, decisions):
    descriptions = []
    for source in sources:
        channels = []
        for name, channel in source["channels"].items():
            descriptor = {key: channel.get(key) for key in
                          ("unit", "kind", "max_gap_s", "phase_tolerance", "experiment", "column")}
            descriptor.update(name=name, id=f"{source['id']}.{name}", required=channel.get("required", True))
            channels.append(descriptor)
        descriptions.append({"id": source["id"], "metadata": source.get("metadata", {}),
                             "time_origin_s": source.get("time_origin_s"),
                             "clock": source.get("clock", {}), "decision": decisions.get(source["id"], {}),
                             "channels": channels})
    return descriptions


def _source_overview(sources):
    if not sources:
        return '<p>尚无可读取的外部记录；请先处理输入问题。</p>'
    rows = []
    for source in sources:
        decision = source["decision"]
        channels = "、".join(f"{c['name']} ({c.get('unit') or '未声明单位'}，{'必需' if c['required'] else '可选'})"
                            for c in source["channels"])
        evidence = {"metadata": source["metadata"], "time_origin_s": source.get("time_origin_s"),
                    "clock": source["clock"], "decision": decision}
        offset = decision.get("dt_s")
        offset_label = format(offset, ".12g") if offset is not None else None
        rows.append(f'<tr><th scope="row">{_text(source["id"])}</th><td>{_text(channels)}</td>'
                    f'<td>{_text(decision.get("status"))}</td><td>{_text(offset_label)}</td>'
                    f'<td><details><summary>时间来源与依据</summary><pre>{escape(_json(evidence))}</pre></details></td></tr>')
    return ('<div class="table-scroll" tabindex="0" role="region" aria-label="设备时间关系，可横向滚动">'
            '<table><caption>外部记录时间 = 探测器时间 + Δt；候选偏移不作为已确认映射。</caption>'
            '<thead><tr><th scope="col">设备</th><th scope="col">通道</th><th scope="col">时钟状态</th>'
            '<th scope="col">Δt (s)</th><th scope="col">依据</th></tr></thead><tbody>'
            + "".join(rows) + '</tbody></table></div>')


def make_report(state, frames, table, sources, features):
    """Render a portable report. Readiness is supplied by workflow, never inferred."""
    rows = table or frames or []
    descriptions = _source_descriptions(sources, state.get("decisions", {}))
    summary_keys = ("status", "experiment", "experiment_metadata", "run_id", "version", "code_version",
                    "started_at", "finished_at", "request_path", "request_sha256", "n_frames", "n_selected",
                    "n_valid_analysis", "ready_for_integration", "raw_source_issues", "identity_validation",
                    "raw_identity_verified", "issues", "decisions", "frame_metadata", "stages", "next_action")
    summary = {key: state.get(key) for key in summary_keys if key in state}
    payload = {"state": summary, "sources": descriptions, "rows": rows}
    ready = state.get("ready_for_integration")
    ready_label = "可交付原始帧引用" if ready is True else "尚不能交付真实积分" if ready is False else "未评估"
    raw_issues = state.get("raw_source_issues") or []
    reason_counts = Counter(str(item.get("reason", item.get("code", "未说明"))) for item in raw_issues if isinstance(item, dict))
    reasons = "；".join(f"{reason}：{count} 帧" for reason, count in sorted(reason_counts.items()))
    if ready is False and not reasons:
        reasons = "请检查时钟依据、所选帧有效性和原始帧身份核验结果。"
    if ready is True and state.get("raw_identity_verified") is not True:
        reasons = (reasons + "；" if reasons else "") + "原始寻址尚待消费者预检。"
    identity = state.get("identity_validation")
    identity_html = (f'<details><summary>原始帧身份核验结果</summary><pre>{escape(_json(identity))}</pre></details>'
                     if identity is not None else '<p class="muted">原始帧身份：本次结果未提供核验信息。</p>')
    issues = state.get("issues") or []
    issue_html = "".join(f'<li><pre>{escape(_json(item))}</pre></li>' for item in issues)
    issues_html = (f'<section aria-labelledby="issues-heading"><h2 id="issues-heading">需要处理的输入或依据</h2>'
                   f'<ul class="issues">{issue_html}</ul></section>' if issues else '')
    metadata = [("实验类型", state.get("experiment")), ("实验信息", state.get("experiment_metadata")),
                ("运行 ID", state.get("run_id")), ("软件版本", state.get("version")),
                ("代码版本", state.get("code_version")), ("开始时间", state.get("started_at")),
                ("完成时间", state.get("finished_at")), ("请求文件", state.get("request_path")),
                ("请求 SHA-256", state.get("request_sha256"))]
    metadata_html = "".join(f'<dt>{label}</dt><dd>{_text(value)}</dd>' for label, value in metadata if value is not None)
    devices = "".join(f'<option value="{escape(s["id"], quote=True)}">{_text(s["id"])}</option>' for s in descriptions)
    n_selected = state.get("n_selected", sum(row.get("selected") is True for row in rows))
    n_valid = state.get("n_valid_analysis", sum(row.get("valid_analysis") is True for row in rows))
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>InSituLink · 逐帧身份与时间对应</title>
<style>{REPORT_CSS}</style></head><body>
<a class="skip-link" href="#frame-browser">跳到逐帧记录</a>
<header><p class="eyebrow">InSituLink · 离线结果</p><h1>逐帧身份与时间对应</h1>
<div class="summary-grid"><div><span>对齐状态</span><strong>{_text(state.get('status'))}</strong></div>
<div><span>保留 / 选择 / 分析有效</span><strong>{len(rows)} / {_text(n_selected)} / {_text(n_valid)}</strong></div>
<div><span>原始帧交付状态</span><strong>{ready_label}</strong></div></div>
<p>{_text(reasons) if reasons else '每个原始帧独立保留；参数重复不会合并帧。'}</p>
<p class="muted">DIDASFlow 交接为文件格式约定，尚未自动接入。时钟确认、参数有效性和原始帧核验分别记录；软件结果不替代实验标定。</p>
<details><summary>运行与实验信息</summary><dl class="metadata">{metadata_html}</dl></details>{identity_html}</header>
<nav aria-label="结果文件"><a href="frames.csv">完整逐帧参数表</a><a href="frame_manifest.json">原始帧清单</a>
<a href="decisions.json">时间偏移依据</a><a href="handoff.json">交接包</a><a href="status.json">运行状态</a></nav>
{issues_html}
<section aria-labelledby="devices-heading"><h2 id="devices-heading">设备与时钟依据</h2>{_source_overview(descriptions)}</section>
<section id="frame-browser" aria-labelledby="frames-heading"><h2 id="frames-heading">逐帧记录</h2>
<p id="filter-help" class="muted">默认保留全部帧。全部设备按所选分析的必需通道判断；指定设备后按该设备全部通道判断。未选择单独标记，待确认指时间关系或输入依据尚未解决。</p>
<div class="filters"><label class="search">查找物理帧<input id="frame-search" type="search" placeholder="UID、原始 ID、源路径或数据集" aria-describedby="filter-help"></label>
<label>状态<select id="frame-status"><option value="all">全部</option><option value="valid">有效</option><option value="invalid">无效</option><option value="pending">待确认</option><option value="unselected">未选择</option></select></label>
<label>设备<select id="frame-device"><option value="">全部设备</option>{devices}</select></label>
<label>开始 (s)<input id="time-min" type="number" step="any" placeholder="不限" aria-describedby="time-help"></label>
<label>结束 (s)<input id="time-max" type="number" step="any" placeholder="不限" aria-describedby="time-help"></label>
<button id="reset-filters" type="button">重置筛选</button></div>
<p id="time-help" class="muted">按曝光区间与所填范围相交筛选；时间坐标见下方说明。缺失时间不会作为 0。</p><p id="time-basis" class="muted"></p>
<div class="toolbar"><span id="frame-count" role="status" aria-live="polite"></span><label>每页<select id="page-size"><option>25</option><option selected>50</option><option>100</option></select></label>
<button id="copy-csv" type="button">复制筛选表 (CSV)</button><button id="copy-json" type="button">复制筛选 JSON</button></div>
<div class="table-scroll" tabindex="0" role="region" aria-label="逐帧身份表，可横向滚动"><table id="frame-table"><caption>点击物理帧 UID 查看曝光、时钟依据、参数和来源。</caption>
<thead><tr><th scope="col">采集顺序</th><th scope="col">物理帧 UID</th><th scope="col">原始 ID</th><th scope="col">原始定位</th><th scope="col">曝光区间 (s)</th><th scope="col">状态</th><th scope="col">参数中点值</th><th scope="col">原因</th></tr></thead><tbody id="frame-rows"></tbody></table></div>
<div class="pagination"><button id="previous-page" type="button">上一页</button><span id="page-position"></span><button id="next-page" type="button">下一页</button></div>
<noscript><p>逐帧筛选需要浏览器 JavaScript。所有原始表格和 JSON 仍可通过上方链接离线读取。</p></noscript>
<section id="frame-detail" class="frame-detail" aria-labelledby="detail-heading" hidden><div class="toolbar"><h3 id="detail-heading" tabindex="-1">帧详情</h3><button id="close-detail" type="button">返回表格</button><button id="copy-detail" type="button">复制该帧 JSON</button></div><div id="detail-content"></div></section>
<details id="copy-fallback" hidden><summary>复制内容</summary><p>若浏览器未允许直接复制，请在下框按 Ctrl+C（macOS 按 ⌘C）。</p><textarea id="copy-text" rows="8" readonly aria-label="可复制的筛选结果"></textarea></details><p id="copy-status" role="status" aria-live="polite"></p></section>
<section aria-labelledby="plots-heading"><h2 id="plots-heading">时间对应与采样</h2><p class="muted">虚线仅为候选偏移。记录缺口和失效特征保持断线；只有中点可用的曝光以叉号表示。</p>{_plots(state, frames, rows, sources, features)}</section>
<details><summary>完整计算与判断记录</summary><pre>{escape(_json(summary))}</pre></details>
<script id="report-data" type="application/json">{_embedded_json(payload)}</script><script>{REPORT_SCRIPT}</script>
</body></html>'''
