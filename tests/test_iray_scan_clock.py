"""Synthetic scan-span, sector, continuation and strain-flag contracts.

Numbers are planted. This file does not read beamtime arrays.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from insitulink.numerics import resolve_clock
from insitulink.workflow import run
from strainlink.time_align import resolve_event_dt


def event_fixture(onsets=(12,), offset=4, roles=None):
    """40-frame force ramp. Detected positive onsets are one sample after the kink."""
    mids = np.arange(40, dtype=float)
    source_t = np.arange(80, dtype=float) - 20
    frames = [dict(index=i, t_mid_s=float(t), t_start_s=float(t - 0.1), exposure_s=0.2)
              for i, t in enumerate(mids)]
    source_y = np.maximum(source_t - (12 + offset), 0)
    roles = list(roles or ["sector"] * len(onsets))
    signals = [dict(id=f"sector-{i}", role=roles[i], t_s=mids.tolist(),
                    values=np.maximum(mids - onset, 0).tolist(), valid=[True] * len(mids))
               for i, onset in enumerate(onsets)]
    source = dict(t_s=source_t.tolist(),
                  channels={"force": {"values": source_y.tolist(), "unit": "N"}},
                  clock={"method": "event", "channel": "force", "min_elastic_strain": 1e-3})
    return frames, source, {"signals": signals, "metadata": {}}, mids


def _scan_frames():
    step = 185 / 159
    t_start = [i * step for i in range(160)]
    t_mid = [t + 0.15 for t in t_start]
    frames = [dict(index=i, t_mid_s=t_mid[i], t_start_s=t_start[i], exposure_s=0.3) for i in range(160)]
    return frames, t_mid


def _force_on_5397():
    """Eight-sample baseline, then a sample exactly at 5.397 that leaves baseline."""
    pre = np.arange(-40.0, 5.397, 0.5)
    pre = pre[pre < 5.397 - 1e-9]
    times = np.concatenate([pre, [5.397], np.arange(5.397 + 0.5, 220.0, 0.5)])
    values = np.where(times < 5.397 - 1e-12, 0.0, times - 5.397 + 1.0)
    return times, values


def scan_span_case(*, nominal_period=True):
    frames, t_mid = _scan_frames()
    times, force = _force_on_5397()
    # Wall dt maps frame 27 onto the force sample at 5.397. Value there is already 1.
    signal = [0.0 if i < 27 else (t_mid[i] - t_mid[27] + 1.0) for i in range(160)]
    clock = dict(method="scan_span", start_s=0, end_s=185, exposure_s=0.3, n=160,
                 channel="force", baseline_samples=8, min_elastic_strain=1e-3)
    if nominal_period:
        clock["nominal_period_s"] = 1
    source = dict(t_s=times.tolist(), channels={"force": {"values": force.tolist(), "unit": "N"}}, clock=clock)
    features = {"signals": [dict(id="sector-0", role="sector", t_s=t_mid, values=signal, valid=[True] * 160)],
                "metadata": {}}
    return frames, source, features


def _continuation_parts(force_at_zero, *, link=True, reason="onset_missing"):
    clock = {"method": "event", "channel": "force", "wall_start_s": 1016}
    if link:
        clock["continuation_of"] = "parent-source"
        clock["parent_wall_span"] = {
            "source_id": "parent-source", "local_dt_s": 20.0, "uncertainty_s": 1.89,
            "wall_start_s": 1000, "preload": 1,
        }
    source = {
        "id": "child-scan" if link else "sampleA-1",
        "path": "scans/sample_0002",
        "t_s": [0.0, 1.0, 2.0, 3.0],
        "time_origin_s": "0",
        "channels": {"force": {"values": [float(force_at_zero), 6.0, 7.0, 8.0], "unit": "N"}},
        "clock": clock,
        "metadata": {"source": "scans/sample_0002"},
    }
    decision = {"status": "needs_input", "method": "event", "reason": reason, "dt_s": None,
                "candidate_dt_s": None, "uncertainty_s": None, "evidence": {}}
    frames = [dict(time_origin_s="0", index=0, t_mid_s=0.0, t_start_s=0.0, exposure_s=0.2)]
    return decision, source, frames


class SectorClockTests(unittest.TestCase):
    def test_conflicting_positive_sectors_stay_unresolved(self):
        frames, source, features, _ = event_fixture(onsets=(12, 18), roles=("sector", "sector"))
        result = resolve_clock(frames, source, features, "tensile")
        self.assertEqual(result["status"], "needs_input")
        self.assertIsNone(result["dt_s"])
        self.assertEqual(result["reason"], "conflicting_feature_events")
        self.assertEqual(len(result["evidence"]["signal_events"]), 2)

    def test_reversing_sector_does_not_veto_the_load_tracker(self):
        frames, source, features, mids = event_fixture(onsets=(12,))
        reversing = [0.0] * len(mids)
        # Above 1e-3 for a few samples, then back to baseline through the rest of the 40 s record.
        reversing[20:24] = [0.004, 0.008, 0.006, 0.003]
        features["signals"].append(dict(id="reversing", role="sector", t_s=mids.tolist(),
                                         values=reversing, valid=[True] * len(mids)))
        result = resolve_clock(frames, source, features, "tensile")
        self.assertEqual(result["status"], "resolved")
        self.assertIsNotNone(result["dt_s"])
        self.assertEqual(result["dt_s"], 4)
        self.assertNotEqual(result["dt_s"], -3)
        self.assertNotEqual(result["reason"], "conflicting_feature_events")

    def test_poisson_control_does_not_veto(self):
        frames, source, features, mids = event_fixture(onsets=(12,))
        negative = (-np.maximum(mids - 18, 0)).tolist()
        features["signals"].append(dict(id="poisson", role="control", control="contraction",
                                         t_s=mids.tolist(), values=negative, valid=[True] * len(mids)))
        result = resolve_clock(frames, source, features, "tensile")
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["dt_s"], 4)
        self.assertNotEqual(result["reason"], "conflicting_feature_events")
        self.assertIsNotNone(result["dt_s"])


class ScanSpanTests(unittest.TestCase):
    def test_scan_span_publishes_wall_clock_and_period_uncertainty(self):
        frames, source, features = scan_span_case()
        result = resolve_clock(frames, source, features, "tensile")
        self.assertEqual(result["reason"], "scan_span_wall_clock")
        self.assertEqual(result["status"], "resolved")
        self.assertAlmostEqual(result["dt_s"], -26.16809433962264, places=9)
        self.assertAlmostEqual(result["uncertainty_s"], 4.415094339622642, places=9)
        self.assertIs(result["evidence"]["clocks_stretched"], False)
        self.assertIs(result["evidence"]["used_nominal_period_as_clock"], False)
        self.assertEqual(result["evidence"]["published_clock"], "wall_span")

    def test_scan_span_missing_nominal_period_is_unresolved(self):
        frames, source, features = scan_span_case(nominal_period=False)
        result = resolve_clock(frames, source, features, "tensile")
        self.assertIsNone(result["dt_s"])
        self.assertEqual(result["reason"], "scan_span_fields_missing")
        self.assertNotEqual(result["status"], "resolved")


class ContinuationTests(unittest.TestCase):
    def test_continuation_requires_an_explicit_field(self):
        from insitulink.workflow_inputs import apply_continuation

        for reason in ("source_onset_missing", "onset_missing"):
            with self.subTest(reason=reason):
                decision, source, frames = _continuation_parts(5.0, link=False, reason=reason)
                self.assertNotIn("continuation_of", source["clock"])
                self.assertNotIn("continuation_of", decision)
                self.assertTrue(str(source["path"]).endswith("_0002"))
                self.assertEqual(source["id"], "sampleA-1")
                result = apply_continuation(decision, source, frames)
                self.assertEqual(result["status"], "needs_input")
                self.assertEqual(result["reason"], reason)
                self.assertEqual(result.get("method"), "event")
                self.assertNotEqual(result.get("method"), "continuation")
                self.assertIsNone(result["dt_s"])
                self.assertIsNone(result.get("candidate_dt_s"))

    def test_continuation_adds_the_wall_gap_once(self):
        from insitulink.workflow_inputs import apply_continuation

        decision, source, frames = _continuation_parts(5.0, link=True, reason="onset_missing")
        self.assertEqual(source["clock"]["continuation_of"], source["clock"]["parent_wall_span"]["source_id"])
        self.assertGreater(source["channels"]["force"]["values"][0], source["clock"]["parent_wall_span"]["preload"])
        result = apply_continuation(decision, source, frames)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["method"], "continuation")
        self.assertEqual(result["reason"], "continuation_of_published_wall_span")
        self.assertEqual(result["local_dt_s"], 36.0)
        self.assertEqual(result["dt_s"], 36.0)
        self.assertNotEqual(result["dt_s"], 20.0)
        self.assertNotEqual(result["dt_s"], 52.0)
        self.assertAlmostEqual(result["uncertainty_s"], 2.89, places=9)
        self.assertIs(result["evidence"]["filename_used"], False)
        self.assertIs(result["evidence"]["frame_indices_merged"], False)

    def test_continuation_rejected_when_join_force_is_not_above_preload(self):
        from insitulink.workflow_inputs import apply_continuation

        decision, source, frames = _continuation_parts(1.0, link=True, reason="onset_missing")
        self.assertEqual(source["channels"]["force"]["values"][0], source["clock"]["parent_wall_span"]["preload"])
        result = apply_continuation(decision, source, frames)
        self.assertIsNone(result["dt_s"])
        self.assertIsNone(result["candidate_dt_s"])
        self.assertNotEqual(result.get("method"), "continuation")
        self.assertNotEqual(result["status"], "resolved")


class LegacyAndStrainTests(unittest.TestCase):
    def test_legacy_unresolved_dt_stays_zero(self):
        t_mid = np.linspace(0.0, 30.0, 40)
        eps = np.zeros_like(t_mid)
        t_force = np.linspace(0.0, 40.0, 80)
        force = np.maximum(t_force - 12.0, 0.0)
        result = resolve_event_dt(t_mid, eps, t_force, 20, force=force)
        self.assertEqual(result["dt_s"], 0.0)
        self.assertTrue(result["onset_unresolved"])
        self.assertIsNotNone(result["dt_s"])

    def test_strain_flags_do_not_replace_values(self):
        from insitulink.strain_reliability import strain_reliability, stress_identity

        stuck = [0.200, 0.200, 0.201, 0.200, 0.200, 0.201]
        original = list(stuck)
        stuck_flags = strain_reliability({"strain": {"values": stuck, "unit": "1"}})["flags"]
        self.assertIn("stuck", stuck_flags)
        self.assertEqual(stuck, original)

        signed = [0, 0.2, 0.4, -0.2, -1.0, -1.6]
        signed_flags = strain_reliability({"strain": {"values": signed, "unit": "1"}})["flags"]
        self.assertIn("sign_or_runaway", signed_flags)
        self.assertEqual(signed, [0, 0.2, 0.4, -0.2, -1.0, -1.6])

        offset = [24.8, 24.9, 25.0, 25.0, 25.1, 25.2]
        self.assertAlmostEqual(max(offset) - min(offset), 0.4)
        offset_flags = strain_reliability({"strain": {"values": offset, "unit": "1"}})["flags"]
        self.assertIn("offset_removed", offset_flags)
        self.assertEqual(offset, [24.8, 24.9, 25.0, 25.0, 25.1, 25.2])

        extension = [0.0, 2.5, 5.0, 7.5, 10.0]
        crosshead = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.assertAlmostEqual(max(extension) - min(extension), 10)
        self.assertAlmostEqual(max(crosshead) - min(crosshead), 1)
        compared = strain_reliability({
            "strain": {"values": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05], "unit": "1"},
            "extension": {"values": extension, "unit": "mm"},
            "crosshead": {"values": crosshead, "unit": "mm"},
        })["flags"]
        self.assertIn("large_vs_crosshead", compared)
        self.assertEqual(extension, [0.0, 2.5, 5.0, 7.5, 10.0])
        self.assertEqual(crosshead, [0.0, 0.25, 0.5, 0.75, 1.0])

        force = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        area = 2.0
        matching = [value / area for value in force]
        self.assertIsNone(stress_identity(matching, force, area))
        self.assertEqual(stress_identity([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], force, area),
                         "stress_identity_unresolved")


class ArchivedAndOutsideTests(unittest.TestCase):
    def test_archived_scan_does_not_require_the_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            missing = root / "archived" / "frames_missing.csv"
            request = {
                "schema_version": 1,
                "experiment": "tensile",
                "frames": {"type": "table", "archived": True, "path": str(missing)},
                "sources": [{
                    "id": "mechanics",
                    "time_column": "time",
                    "time": {"unit": "s"},
                    "data": [{"time": 0.0, "force": 0.0}, {"time": 1.0, "force": 1.0}],
                    "channels": [{"name": "force", "column": "force", "unit": "N", "kind": "continuous"}],
                    "clock": {"method": "shared", "confirmed": False, "evidence": ""},
                }],
            }
            path = root / "experiment.json"
            path.write_text(json.dumps(request), encoding="utf-8")
            self.assertFalse(missing.exists())
            status = run(path, root / "out")
            codes = [issue.get("code") for issue in status.get("issues", [])]
            self.assertEqual(status["status"], "needs_input", status.get("issues"))
            self.assertIn("archived_scan_skipped", codes)
            for code in codes:
                self.assertNotIn(code, {"unreadable_input", "missing_input", "missing_frames", "preview_missing_file"})
            self.assertFalse(missing.exists())

    def test_outside_record_stress_stays_empty(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = {
                "schema_version": 1,
                "experiment": "tensile",
                "frames": {
                    "time": {"unit": "s"},
                    "data": [
                        {"frame_index": 0, "t_start_s": 0.5, "exposure_s": 0.4},
                        {"frame_index": 1, "t_start_s": 1.5, "exposure_s": 0.4},
                        {"frame_index": 2, "t_start_s": 4.5, "exposure_s": 0.4},
                    ],
                },
                "sources": [{
                    "id": "mechanics",
                    "time_column": "time",
                    "time": {"unit": "s"},
                    "data": [
                        {"time": 0.0, "force": 0.0, "stress": 0.0},
                        {"time": 1.0, "force": 10.0, "stress": 10.0},
                        {"time": 2.0, "force": 20.0, "stress": 20.0},
                        {"time": 3.0, "force": 30.0, "stress": 30.0},
                        {"time": 4.0, "force": 40.0, "stress": 40.0},
                    ],
                    "channels": [
                        {"name": "force", "unit": "N", "kind": "continuous"},
                        {"name": "stress", "unit": "MPa", "kind": "continuous"},
                    ],
                    "clock": {
                        "method": "shared",
                        "confirmed": True,
                        "evidence": "Verified common clock and common origin",
                    },
                }],
            }
            path = root / "experiment.json"
            path.write_text(json.dumps(request), encoding="utf-8")
            output = root / "out"
            status = run(path, output)
            self.assertEqual(status["decisions"]["mechanics"]["status"], "resolved", status.get("issues"))
            records = json.loads((output / "frame_records.json").read_text(encoding="utf-8"))
            last = records["frames"][-1]
            self.assertEqual(last["frame_index"], 2)
            stress = last["devices"]["mechanics"]["parameters"]["stress"]
            for key in ("mid", "mean", "min", "max"):
                self.assertIsNone(stress[key])
                self.assertNotEqual(stress[key], 40.0)
            early = records["frames"][0]["devices"]["mechanics"]["parameters"]["stress"]["mid"]
            self.assertIsInstance(early, float)
