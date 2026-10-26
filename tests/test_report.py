"""Report evidence, inert source text and offline filtering semantics."""

import json
from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from insitulink.report import _embedded_json, _plot_coordinates, make_report
from insitulink.report_assets import REPORT_SCRIPT


class ReportDataParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_data = False
        self.data = []
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            attributes = dict(attrs)
            self.scripts.append(attributes)
            self.in_data = attributes.get("id") == "report-data"

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_data = False

    def handle_data(self, data):
        if self.in_data:
            self.data.append(data)


class ReportTest(unittest.TestCase):
    def test_plot_breaks_gaps_and_preserves_missing_samples(self):
        x, y = _plot_coordinates([0, 1, 2, 10, 11], [0, 1, None, 3, 4], max_gap_s=2)
        np.testing.assert_equal(x, [0, 1, 2, np.nan, 10, 11])
        np.testing.assert_equal(y, [0, 1, np.nan, np.nan, 3, 4])
        x, y = _plot_coordinates([0, 1, 2, 9], [0, 1, 2, 3])
        np.testing.assert_equal(x, [0, 1, 2, np.nan, 9])
        np.testing.assert_equal(y, [0, 1, 2, np.nan, 3])

    def test_invalid_feature_remains_a_break_at_original_time(self):
        x, y = _plot_coordinates([0, 1, 2], [1, 9, 2], valid=[True, False, True])
        np.testing.assert_equal(x, [0, 1, 2])
        np.testing.assert_equal(y, [1, np.nan, 2])

    def test_render_uses_step_holds_gap_breaks_and_feature_mask(self):
        from matplotlib.axes import Axes

        source = {"id": "pressure", "t_s": [0, 1, 2, 10],
                  "channels": {"P": {"unit": "MPa", "kind": "step", "max_gap_s": 2,
                                      "values": [0, 1, None, 3]}}}
        frames = [{"frame_uid": "f", "t_mid_s": .5}]
        table = [{**frames[0], "pressure.P.mid": 0, "pressure.P.valid": False}]
        captured = []
        original = Axes.plot

        def capture(axis, *args, **kwargs):
            lines = original(axis, *args, **kwargs)
            captured.extend((line.get_label(), line.get_drawstyle(), line.get_xydata(),
                             line.get_path().vertices.copy()) for line in lines)
            return lines

        with patch.object(Axes, "plot", capture):
            html = make_report({"status": "completed", "decisions": {"pressure": {"status": "resolved", "dt_s": 0}}},
                               frames, table, [source], {"signals": [{"id": "eps", "t_s": [0, 1, 2],
                                                                        "values": [0, 9, 2], "valid": [True, False, True]}]})
        accepted = next(item for item in captured if item[0] == "Accepted time mapping")
        self.assertEqual(accepted[1], "steps-post")
        np.testing.assert_equal(accepted[2][:, 0], [-.5, .5, 1.5, np.nan, 9.5])
        # The step path holds 0 until the next sample, rather than ramping to 1.
        np.testing.assert_equal(accepted[3][:3], [[-.5, 0], [.5, 0], [.5, 1]])
        feature = next(item for item in captured if item[0] == "eps")
        np.testing.assert_equal(feature[2][:, 1], [0, np.nan, 2])
        self.assertIn("data:image/png;base64,", html)
        self.assertIn("只有中点可用", html)

    def test_report_separates_alignment_readiness_and_identity_verification(self):
        state = {"status": "completed", "ready_for_integration": False, "n_selected": 1,
                 "n_valid_analysis": 1, "version": "1.0.test", "code_version": "build-abc",
                 "run_id": "run-123", "started_at": "2026-10-26T00:00:00Z", "request_sha256": "abc123",
                 "raw_source_issues": [{"frame_uid": "f", "reason": "raw_source_not_referenced"}]}
        html = make_report(state, [], [{"frame_uid": "f", "selected": True, "valid_analysis": True}], [], {})
        for value in ("completed", "尚不能交付真实积分", "raw_source_not_referenced", "1.0.test", "build-abc", "run-123", "abc123"):
            self.assertIn(value, html)
        state.update(ready_for_integration=True, raw_identity_verified=False, raw_source_issues=[])
        html = make_report(state, [], [], [], {})
        self.assertIn("原始寻址尚待消费者预检", html)
        self.assertIn("未评估", make_report({"status": "completed"}, [], [], [], {}))

    def test_device_table_formats_offsets_without_changing_json_values(self):
        offsets = {"mechanics": 6.250000000000001, "furnace": -4.750000000000004}
        state = {"status": "completed", "decisions": {
            name: {"status": "resolved", "dt_s": value} for name, value in offsets.items()}}
        sources = [{"id": name, "t_s": [], "channels": {}} for name in offsets]
        html = make_report(state, [], [], sources, {})
        self.assertIn('<td>resolved</td><td>6.25</td>', html)
        self.assertIn('<td>resolved</td><td>-4.75</td>', html)
        parser = ReportDataParser()
        parser.feed(html)
        payload = json.loads("".join(parser.data))
        for source in payload["sources"]:
            self.assertEqual(source["decision"]["dt_s"], offsets[source["id"]])
        self.assertEqual(payload["state"]["decisions"], state["decisions"])

    def test_plot_uses_nanosecond_local_coordinates_and_local_offsets(self):
        from matplotlib.axes import Axes

        frames = [{"frame_uid": "f", "t_mid_s": 1800000000., "t_mid_rel_s": 5e-10,
                   "time_origin_s": "1800000000.000000000"}]
        source = {"id": "sensor", "time_origin_s": "1900000000.000000000",
                  "t_s": [1900000000., 1900000000., 1900000000.], "t_rel_s": [0, 1e-9, 2e-9],
                  "channels": {"T": {"unit": "K", "kind": "continuous", "values": [1, 2, 3]}}}
        decision = {"status": "resolved", "dt_s": 1e8, "local_dt_s": 0.0}
        captured = []
        original = Axes.plot

        def capture(axis, *args, **kwargs):
            lines = original(axis, *args, **kwargs)
            captured.extend((line.get_label(), line.get_xydata()) for line in lines)
            return lines

        with patch.object(Axes, "plot", capture):
            html = make_report({"status": "completed", "decisions": {"sensor": decision}}, frames,
                               [{**frames[0], "sensor.T.mid": 1.5, "sensor.T.valid": True}], [source],
                               {"metadata": {"time_coordinate": "frame_local_seconds"},
                                "signals": [{"id": "eps", "t_s": [5e-10, 1.5e-9], "values": [0, 1]}]})
        mapped = next(points for label, points in captured if label == "Accepted time mapping")
        feature = next(points for label, points in captured if label == "eps")
        np.testing.assert_equal(mapped[:, 0], [0, 1e-9, 2e-9])
        np.testing.assert_equal(feature[:, 0], [5e-10, 1.5e-9])
        self.assertIn("1800000000.000000000", html)
        self.assertIn("1900000000.000000000", html)
        self.assertIn("灰线使用外部设备自身的局部时钟", html)

    def test_untrusted_fields_are_inert_and_round_trip_without_loss(self):
        hostile = '</script><script>alert("source")</script><img src=x onerror=alert(1)>&\u2028\u2029'
        row = {"frame_uid": hostile, "original_id": hostile, "source_file": hostile,
               "dataset": hostile, "selected": False, "valid_analysis": False, "t_start_s": None,
               "sensor.T.provenance": {"source": hostile}, "sensor.T.mid": 0}
        state = {"status": "needs_input", "experiment_metadata": {"sample": hostile},
                 "issues": [{"message": hostile}], "decisions": {}}
        html = make_report(state, [], [row], [], {})
        parser = ReportDataParser()
        parser.feed(html)
        self.assertEqual(len(parser.scripts), 2)
        self.assertTrue(all("src" not in script for script in parser.scripts))
        payload = json.loads("".join(parser.data))
        self.assertEqual(payload["rows"][0], row)
        self.assertNotIn(hostile, html)
        self.assertIn("\\u003c/script\\u003e", html)
        self.assertNotIn("innerHTML", REPORT_SCRIPT)
        self.assertEqual(json.loads(_embedded_json({"value": hostile}))["value"], hostile)

    def test_report_keeps_rows_in_data_without_rendering_all_dom_rows(self):
        rows = [{"frame_uid": f"frame-{i}", "selected": True, "valid_analysis": True} for i in range(1000)]
        html = make_report({"status": "completed"}, [], rows, [], {})
        parser = ReportDataParser()
        parser.feed(html)
        self.assertEqual(len(json.loads("".join(parser.data))["rows"]), 1000)
        self.assertIn('<tbody id="frame-rows"></tbody>', html)
        self.assertIn('aria-live="polite"', html)
        self.assertIn('id="copy-json"', html)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for offline filter execution")
    def test_real_javascript_filters_selection_devices_null_time_and_copy(self):
        script = REPORT_SCRIPT + r'''
const assert = require("node:assert/strict");
const {prepareRecords,filterRecords,reasonsFor,csv,numeric} = module.exports;
const sources = [
 {id:"furnace",decision:{status:"resolved"},channels:[{id:"furnace.T",required:true}]},
 {id:"pressure",decision:{status:"resolved"},channels:[{id:"pressure.P",required:false}]}
];
const rows = [
 {frame_uid:"A",original_id:'sample,"A"',source_file:"C:/raw/scan.h5",selected:true,valid_analysis:true,t_start_s:0,t_end_s:1,"furnace.T.valid":true,"furnace.T.mid":0,"pressure.P.valid":false,"pressure.P.flags":["after_record"]},
 {frame_uid:"B",selected:true,valid_analysis:false,t_start_s:2,t_end_s:3,"furnace.T.valid":false,"furnace.T.flags":["nonfinite"],"pressure.P.valid":true},
 {frame_uid:"C",selected:true,valid_analysis:false,t_start_s:null,t_end_s:null,"furnace.T.valid":false,"furnace.T.flags":["clock_unresolved"]},
 {frame_uid:"D",selected:false,valid_analysis:false,t_start_s:4,t_end_s:5,"furnace.T.valid":true,"pressure.P.valid":true},
 {frame_uid:"E",selected:true,valid_analysis:true,t_start_s:1800000000,t_end_s:1800000001,t_start_rel_s:0.0000001,t_end_rel_s:0.0000002,"furnace.T.valid":true,"pressure.P.valid":true}
];
const records=prepareRecords(rows,sources);
const ids=filter=>filterRecords(records,filter).map(r=>r.row.frame_uid);
assert.deepEqual(ids({status:"all",minimum:"",maximum:""}),["A","B","C","D","E"]);
assert.deepEqual(ids({status:"valid"}),["A","E"]);
assert.deepEqual(ids({status:"invalid"}),["B"]);
assert.deepEqual(ids({status:"pending"}),["C"]);
assert.deepEqual(ids({status:"unselected"}),["D"]);
assert.deepEqual(ids({device:"pressure",status:"invalid"}),["A","C"]);
assert.deepEqual(ids({device:"pressure",status:"valid"}),["B","E"]);
assert.deepEqual(ids({query:"RAW/SCAN"}),["A"]);
assert.deepEqual(ids({query:'sample,"a"'}),["A"]);
assert.deepEqual(ids({minimum:0,maximum:0}),["A"]);
assert.deepEqual(ids({minimum:0.00000015,maximum:0.00000016}),["A","E"]);
assert.deepEqual(ids({minimum:3,maximum:2}),[]);
assert.equal(numeric(null),null);assert.equal(numeric(""),null);assert.equal(numeric(0),0);
assert(reasonsFor(rows[1],sources,"").join().includes("nonfinite"));
assert(!reasonsFor(rows[0],sources,"").join().includes("after_record"));
assert(reasonsFor(rows[0],sources,"pressure").join().includes("after_record"));
assert(csv([rows[0]]).includes('"sample,""A"""'));
assert(csv([{value:null,zero:0,nested:{row_index:2}}]).includes('"","0","{""row_index"":2}"'));
console.log("offline filter and copy assertions passed");
'''
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report-filter.cjs"
            path.write_text(script, encoding="utf-8")
            result = subprocess.run([shutil.which("node"), str(path)], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("assertions passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
