"""Small embedded browser assets for the offline report; no external runtime."""

REPORT_CSS = r"""
:root{color-scheme:light;--ink:#17313c;--muted:#4e6670;--line:#ccd9de;--blue:#126384;--wash:#f2f6f7}

*{box-sizing:border-box}
body{font:16px/1.55 system-ui,sans-serif;color:var(--ink);background:#fff;max-width:1280px;margin:24px auto;padding:0 24px}
h1{font-size:clamp(1.65rem,4vw,2.2rem);line-height:1.2}
h2{font-size:1.35rem;margin-top:2rem}
h3{font-size:1.1rem}
p{margin:.7rem 0}
a{color:var(--blue)}
button,input,select,textarea{font:inherit}
button,input,select{min-height:42px;border:1px solid #9aafb9;border-radius:6px;background:#fff;color:var(--ink);padding:7px 10px}
button{cursor:pointer}
button:hover{background:#eaf3f7}
button:disabled{cursor:default;color:#687b84;background:#f1f4f5}
button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible,[tabindex]:focus-visible,textarea:focus-visible{outline:3px solid #a24b00;outline-offset:3px}
button.link{border:0;color:var(--blue);text-decoration:underline;text-align:left;padding:3px 0;min-height:36px;overflow-wrap:anywhere}
input,select{max-width:100%;min-width:0}
label{display:flex;flex-direction:column;gap:4px;font-size:.9rem;font-weight:600}
.eyebrow{font-size:.85rem;letter-spacing:.08em;color:var(--muted)}
.muted{font-size:.9rem;color:var(--muted)}
.summary-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.summary-grid>div{padding:14px;background:var(--wash);border-left:3px solid var(--blue)}
.summary-grid span{display:block;font-size:.85rem;color:var(--muted)}
.summary-grid strong{display:block;margin-top:5px;overflow-wrap:anywhere}
nav{display:flex;gap:10px 22px;flex-wrap:wrap;margin:22px 0}
details{margin:12px 0}
summary{cursor:pointer;font-weight:600;min-height:32px;padding:3px 0}
pre{font:13px/1.5 ui-monospace,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:var(--wash);padding:14px;border-radius:6px;max-height:480px;overflow:auto}
.metadata{display:grid;grid-template-columns:minmax(110px,180px) minmax(0,1fr);gap:8px 14px;margin:12px 0}
.metadata dt{font-weight:600}
.metadata dd{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}
.table-scroll{width:100%;overflow:auto;border:1px solid var(--line);border-radius:8px;overscroll-behavior-x:contain}
table{border-collapse:collapse;width:100%;font-size:.88rem}
caption{padding:10px;text-align:left;color:var(--muted)}
th,td{text-align:left;vertical-align:top;padding:10px 12px;border-bottom:1px solid var(--line);overflow-wrap:anywhere}
th{background:var(--wash)}
tr:last-child td,tr:last-child th{border-bottom:0}
#frame-table{min-width:1080px;table-layout:fixed}
#frame-table th:nth-child(1){width:75px}
#frame-table th:nth-child(2){width:190px}
#frame-table th:nth-child(3){width:120px}
#frame-table th:nth-child(4){width:230px}
#frame-table th:nth-child(5){width:135px}
#frame-table th:nth-child(6){width:80px}
#frame-table th:nth-child(7){width:180px}
#frame-table th:nth-child(8){width:200px}
.filters{display:flex;flex-wrap:wrap;gap:12px;align-items:end;background:var(--wash);padding:16px;border-radius:8px}
.filters .search{flex:1 1 250px}
.filters input[type=number]{width:140px}
.toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin:14px 0}
.toolbar>span:first-child{flex:1 1 200px}
.toolbar label{flex-direction:row;align-items:center}
.pagination{display:flex;justify-content:center;align-items:center;gap:18px;margin:18px 0}
.badge{font-size:.82rem;font-weight:600;display:inline-block;padding:3px 7px;border-radius:4px;background:#e8eef1;color:#3e5360}
.badge.valid{background:#e1f1e7;color:#216341}
.badge.invalid{background:#fbe7e2;color:#8f3228}
.badge.pending{background:#fff1cc;color:#765700}
.frame-detail{background:#fbfcfd;border:2px solid var(--blue);padding:18px;border-radius:8px;scroll-margin-top:20px}
.frame-detail .toolbar h3{flex:1 1 180px;margin:0}
.frame-detail h4{margin:1.2rem 0 .5rem}
.frame-detail .channel{padding:12px 0;border-top:1px solid var(--line)}
figure{margin:18px 0}
figure img{width:100%;height:auto}
figcaption{font-size:.85rem;color:var(--muted)}
.issues{padding-left:22px}
.issues pre{margin:6px 0}
.skip-link{position:absolute;top:-100px;left:12px;background:#fff;padding:10px;z-index:10}
.skip-link:focus{top:8px}
textarea{width:100%;resize:vertical;border:1px solid var(--line);padding:10px;white-space:pre}
code{overflow-wrap:anywhere}
[hidden]{display:none!important}
@media(max-width:600px){body{padding:0 12px;margin:16px auto}
.summary-grid{grid-template-columns:1fr}
.filters{padding:12px}
.filters label{flex:1 1 130px}
.filters .search{flex-basis:100%}
.filters input[type=number]{width:100%}
.metadata{grid-template-columns:1fr;gap:3px}
.metadata dd{margin-bottom:10px}
.frame-detail{padding:12px}
.toolbar button{flex:1 1 170px}
.toolbar label{flex:0 1 auto}
th,td{padding:8px}
.pagination{gap:10px}
nav{gap:10px 16px}
}

"""

REPORT_SCRIPT = r"""
(() => {
  "use strict";
  const labels = {valid: "有效", invalid: "无效", pending: "待确认", unselected: "未选择"};
  const pendingFlags = new Set(["clock_unresolved", "incomplete_run", "source_input_unavailable", "input_changed", "frame_count_unresolved"]);
  const flagLabels = {clock_unresolved:"时钟关系待确认", missing_frame_time_or_exposure:"缺少帧时间或曝光", nonmonotonic_frame_clock:"帧时间重复或回退", nonfinite:"记录包含缺值", gap:"记录间隔超限", incomplete_coverage:"曝光覆盖不足", midpoint_unavailable:"中点不可用", before_record:"早于记录", after_record:"晚于记录", incomplete_run:"输入处理未完成", source_input_unavailable:"外部记录不可用", input_changed:"运行期间输入改变"};
  function numeric(value) { return value === null || value === undefined || value === "" || typeof value === "boolean" ? null : Number.isFinite(Number(value)) ? Number(value) : null; }
  function flags(value) { return Array.isArray(value) ? value : value ? [String(value)] : []; }
  function reasonLabel(value) { return flagLabels[value] ? `${flagLabels[value]} (${value})` : String(value); }
  function channelsFor(sources, device) { return sources.filter(s => !device || s.id === device).flatMap(s => s.channels || []); }
  function statusFor(row, sources, device) {
    if (row.selected === false) return "unselected";
    const channels = channelsFor(sources, device).filter(c => device || c.required !== false);
    const valid = device ? channels.length > 0 && channels.every(c => row[`${c.id}.valid`] === true) : row.valid_analysis === true;
    if (valid) return "valid";
    const unresolved = flags(row.frame_flags).some(f => pendingFlags.has(f)) || channels.some(c => flags(row[`${c.id}.flags`]).some(f => pendingFlags.has(f)));
    const missingClock = sources.some(s => (!device || s.id === device) && (s.channels || []).some(c => (device || c.required !== false) && row[`${c.id}.valid`] !== true) && s.decision && s.decision.status && s.decision.status !== "resolved");
    return unresolved || missingClock ? "pending" : "invalid";
  }
  function reasonsFor(row, sources, device) {
    const reasons = flags(row.frame_flags).map(reasonLabel);
    for (const channel of channelsFor(sources, device).filter(c => device || c.required !== false)) {
      if (row[`${channel.id}.valid`] === false) {
        const issue = flags(row[`${channel.id}.flags`]);
        reasons.push(`${channel.id}: ${issue.length ? issue.map(reasonLabel).join("、") : "参数无效"}`);
      }
    }
    if (row.selected === false) reasons.unshift("未纳入本次分析；参数记录仍保留");
    return reasons;
  }
  function prepareRecords(rows, sources) {
    return rows.map((row, index) => ({row, index,
      search: [row.frame_uid, row.legacy_frame_uid, row.original_id, row.frame_id, row.source_file, row.dataset, row.scan_id].filter(v => v !== null && v !== undefined).join(" ").toLocaleLowerCase(),
      start: numeric(row.t_start_rel_s === undefined ? row.t_start_s : row.t_start_rel_s),
      end: numeric(row.t_end_rel_s === undefined ? row.t_end_s : row.t_end_rel_s),
      status: statusFor(row, sources, ""), deviceStates: Object.fromEntries(sources.map(s => [s.id, statusFor(row, sources, s.id)]))
    }));
  }
  function filterRecords(records, filters) {
    const query = (filters.query || "").trim().toLocaleLowerCase();
    const minimum = numeric(filters.minimum), maximum = numeric(filters.maximum);
    if (minimum !== null && maximum !== null && minimum > maximum) return [];
    return records.filter(record => {
      if (query && !record.search.includes(query)) return false;
      const status = filters.device ? record.deviceStates[filters.device] : record.status;
      if (!status || filters.status && filters.status !== "all" && filters.status !== status) return false;
      if ((minimum !== null || maximum !== null) && (record.start === null || record.end === null)) return false;
      if (minimum !== null && record.end < minimum || maximum !== null && record.start > maximum) return false;
      return true;
    });
  }
  function csv(rows) {
    const keys = [...new Set(rows.flatMap(row => Object.keys(row)))];
    const cell = value => { const text = value === null || value === undefined ? "" : typeof value === "object" ? JSON.stringify(value) : String(value); return `"${text.replace(/"/g, '""')}"`; };
    return [keys.map(cell).join(","), ...rows.map(row => keys.map(k => cell(row[k])).join(","))].join("\r\n");
  }
  if (typeof document === "undefined") {
    if (typeof module !== "undefined") module.exports = {numeric, statusFor, reasonsFor, prepareRecords, filterRecords, csv};
    return;
  }
  const data = JSON.parse(document.getElementById("report-data").textContent);
  const sources = data.sources || [], records = prepareRecords(data.rows || [], sources);
  const byId = id => document.getElementById(id);
  const fields = {query:byId("frame-search"), status:byId("frame-status"), device:byId("frame-device"), minimum:byId("time-min"), maximum:byId("time-max")};
  let filtered = records, page = 0, selectedRecord = null, lastFrameButton = null;
  function node(tag, text, className) { const element = document.createElement(tag); if (text !== undefined) element.textContent = text; if (className) element.className = className; return element; }
  function display(value) { if (value === null || value === undefined || value === "") return "—"; if (typeof value === "boolean") return value ? "是" : "否"; if (typeof value === "number") return Number.isFinite(value) ? String(value) : "—"; return typeof value === "object" ? JSON.stringify(value) : String(value); }
  function definitionList(entries) { const list = node("dl", undefined, "metadata"); for (const [key,value] of entries) { if (value === undefined) continue; list.append(node("dt",key),node("dd",display(value))); } return list; }
  function jsonDetail(title, value, open=false) { const section = node("details"); section.open=open; section.append(node("summary",title),node("pre",JSON.stringify(value,null,2))); return section; }
  function badge(status) { return node("span",labels[status] || status,`badge ${status}`); }
  const relative = records.some(r => r.row.t_start_rel_s !== undefined);
  byId("time-basis").textContent = relative ? "筛选坐标：探测器相对时间 (s)。时间起点见帧详情；每帧同时保留原始时间。" : "筛选坐标：输入归一化后的探测器时间 (s)，与逐帧表 t_start_s / t_end_s 一致。";
  function parameterSummary(row, device) {
    return channelsFor(sources,device).map(c => `${c.id}: ${display(row[`${c.id}.mid`])}${c.unit ? " " + c.unit : ""}${row[`${c.id}.valid`] === false && numeric(row[`${c.id}.mid`]) !== null ? "（仅中点）" : ""}`).join("\n");
  }
  function renderTable() {
    const size = Number(byId("page-size").value), pages = Math.max(1,Math.ceil(filtered.length/size));
    page=Math.min(Math.max(0,page),pages-1);
    const body=byId("frame-rows"), fragment=document.createDocumentFragment(), device=fields.device.value;
    for (const record of filtered.slice(page*size,(page+1)*size)) {
      const row=record.row, tr=node("tr"), status=device ? record.deviceStates[device] : record.status;
      tr.append(node("td",display(row.sequence_index === undefined ? record.index : row.sequence_index)));
      const idCell=node("td"), button=node("button",display(row.frame_uid || row.original_id || `记录 ${record.index + 1}`),"link");
      button.type="button"; button.setAttribute("aria-label",`查看帧 ${display(row.original_id ?? row.frame_uid ?? record.index+1)} 详情`);
      button.addEventListener("click",() => showDetail(record,button)); idCell.append(button); tr.append(idCell);
      tr.append(node("td",display(row.original_id === undefined ? row.frame_id : row.original_id)));
      const locator=node("td"); locator.append(node("div",display(row.source_file)),node("div",`${display(row.dataset)} · index ${display(row.frame_index)}`));tr.append(locator);
      tr.append(node("td",`${display(record.start)} → ${display(record.end)}`));const statusCell=node("td");statusCell.append(badge(status));tr.append(statusCell);
      const parameters=node("td",parameterSummary(row,device));parameters.style.whiteSpace="pre-line";tr.append(parameters);
      tr.append(node("td",reasonsFor(row,sources,device).join("；") || "—"));fragment.append(tr);
    }
    if (!filtered.length) { const tr=node("tr"), td=node("td","没有符合条件的帧。原始记录未被删除。");td.colSpan=8;tr.append(td);fragment.append(tr); }
    body.replaceChildren(fragment);
    const invalidRange=numeric(fields.minimum.value)!==null && numeric(fields.maximum.value)!==null && Number(fields.minimum.value)>Number(fields.maximum.value);
    byId("frame-count").textContent=invalidRange ? "时间范围无效：开始不能晚于结束。" : `匹配 ${filtered.length} / ${records.length} 帧；本页 ${filtered.length ? page*size+1 : 0}–${Math.min((page+1)*size,filtered.length)}`;
    byId("page-position").textContent=`第 ${page+1} / ${pages} 页`;
    byId("previous-page").disabled=page===0;byId("next-page").disabled=page>=pages-1;
  }
  function updateFilters() { page=0;filtered=filterRecords(records,Object.fromEntries(Object.entries(fields).map(([key,field])=>[key,field.value])));renderTable(); }
  function addChannel(parent,row,channel) {
    const prefix=channel.id, section=node("section",undefined,"channel");section.append(node("h4",`${prefix} · ${channel.required === false ? "可选" : "必需"}`));
    section.append(definitionList([["单位",row[`${prefix}.unit`] || channel.unit],["插值模型",channel.kind],["阶段",row[`${prefix}.phase`]],["曝光内阶段",row[`${prefix}.phase_exposure`]],
      ["中点值",row[`${prefix}.mid`]],["曝光均值",row[`${prefix}.mean`]],["曝光最小值",row[`${prefix}.min`]],["曝光最大值",row[`${prefix}.max`]],["曝光覆盖率",row[`${prefix}.coverage`]],["参数有效",row[`${prefix}.valid`]],["具体标记",flags(row[`${prefix}.flags`]).map(reasonLabel).join("、") || "无"]]));
    const provenance=row[`${prefix}.provenance`];
    if (provenance !== undefined && provenance !== null) section.append(jsonDetail("采样来源、支持行与映射",provenance));
    else section.append(node("p","本次结果未提供逐点采样来源；请核对设备记录与时钟依据。","muted"));
    parent.append(section);
  }
  function showDetail(record,button) {
    selectedRecord=record;lastFrameButton=button;const row=record.row, content=byId("detail-content");content.replaceChildren();
    byId("detail-heading").textContent=`帧 ${display(row.original_id ?? row.frame_uid ?? record.index+1)}`;
    content.append(badge(record.status));
    const reasons=reasonsFor(row,sources,"");if (reasons.length) content.append(node("p",reasons.join("；")));
    content.append(node("h4","物理定位与曝光"),definitionList([["物理帧 UID",row.frame_uid],["旧 UID",row.legacy_frame_uid],["原始 ID",row.original_id],["采集顺序（从 0 开始）",row.sequence_index],["源文件",row.source_file],["数据集",row.dataset],["文件内索引（从 0 开始）",row.frame_index],["物理来源",row.physical_source],["身份状态",row.identity_status],["身份范围",row.identity_scope],["身份问题",row.identity_issues],["已选择",row.selected],["分析有效",row.valid_analysis],
      ["曝光开始 (s)",row.t_start_s],["曝光结束 (s)",row.t_end_s],["曝光中点 (s)",row.t_mid_s],["曝光时长 (s)",row.exposure_s],["时间起点 (s)",row.time_origin_s],["相对开始 (s)",row.t_start_rel_s],["相对结束 (s)",row.t_end_rel_s],["相对中点 (s)",row.t_mid_rel_s],["时间来源",row.clock_source],["原始时间缺失",row.raw_time_missing]]));
    for (const source of sources) {
      const section=node("section");section.append(node("h4",`${source.id} · 时钟与参数`));
      const decision=source.decision || {};
      section.append(definitionList([["时钟状态",decision.status],["确认偏移 Δt (s)",decision.dt_s],["候选偏移（未确认） (s)",decision.dt_s === null || decision.dt_s === undefined ? decision.candidate_dt_s : undefined],["偏移不确定度 (s)",decision.uncertainty_s],["方法",decision.method],["判断依据",decision.reason]]));
      section.append(jsonDetail("设备时间来源与时钟证据",{metadata:source.metadata,time_origin_s:source.time_origin_s,clock:source.clock,decision,mapping:(row.clock_mappings || {})[source.id]}));
      for (const channel of source.channels || []) addChannel(section,row,channel);
      content.append(section);
    }
    content.append(jsonDetail("该帧完整 JSON",row));byId("frame-detail").hidden=false;byId("detail-heading").focus();byId("frame-detail").scrollIntoView({block:"start"});
  }
  async function copyText(text) {
    byId("copy-status").textContent="";
    try { if (!navigator.clipboard) throw new Error("clipboard unavailable");await navigator.clipboard.writeText(text);byId("copy-status").textContent="已复制。"; }
    catch (_) { const fallback=byId("copy-fallback");fallback.hidden=false;fallback.open=true;byId("copy-text").value=text;byId("copy-text").focus();byId("copy-text").select();byId("copy-status").textContent="内容已准备，请按 Ctrl+C（macOS 按 ⌘C）复制。"; }
  }
  let searchTimer;
  fields.query.addEventListener("input",()=>{clearTimeout(searchTimer);searchTimer=setTimeout(updateFilters,120);});
  for (const [key,field] of Object.entries(fields)) if (key!=="query") field.addEventListener("input",updateFilters);
  byId("reset-filters").addEventListener("click",()=>{clearTimeout(searchTimer);for (const [key,field] of Object.entries(fields)) field.value=key==="status" ? "all" : "";updateFilters();});
  byId("page-size").addEventListener("change",()=>{page=0;renderTable();});
  byId("previous-page").addEventListener("click",()=>{page--;renderTable();});byId("next-page").addEventListener("click",()=>{page++;renderTable();});
  byId("close-detail").addEventListener("click",()=>{byId("frame-detail").hidden=true;if(lastFrameButton && lastFrameButton.isConnected) lastFrameButton.focus();else fields.query.focus();});
  byId("copy-csv").addEventListener("click",()=>{clearTimeout(searchTimer);updateFilters();copyText(csv(filtered.map(r=>r.row)));});
  byId("copy-json").addEventListener("click",()=>{clearTimeout(searchTimer);updateFilters();copyText(JSON.stringify(filtered.map(r=>r.row),null,2));});
  byId("copy-detail").addEventListener("click",()=>{if(selectedRecord)copyText(JSON.stringify(selectedRecord.row,null,2));});
  renderTable();
})();
"""
