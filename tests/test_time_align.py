"""Fixture tests for the shipped frame–tensile alignment entry."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strainlink.agent import align_scan, demo_records, dumps_result  # noqa: E402
from strainlink.time_align import detect_monotonic_onset  # noqa: E402


def _run_cli(*args: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-m", "strainlink", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr or completed.stdout)
    return completed.stdout


class TimeAlignFixtureTest(unittest.TestCase):
    def test_resolved_heps_clock_uses_midpoint_plus_dt(self):
        frames, tensile = demo_records("heps_id07_eiger")
        result = align_scan(frames, tensile, "heps_id07_eiger")
        decision = result["decision"]
        table = result["table"]

        self.assertFalse(decision["onset_unresolved"])
        self.assertEqual(decision["onset_frame"], 12)
        self.assertEqual(decision["unresolved_reason"], "")
        t_mid = 12 * 1.53 + 0.5
        self.assertAlmostEqual(decision["dt_s"], 15.0 - t_mid)
        self.assertAlmostEqual(table[12]["t_mech_s"], 15.0)
        self.assertAlmostEqual(table[12]["t_mid_s"], t_mid)
        self.assertAlmostEqual(table[12]["stress"], 150.0)
        self.assertAlmostEqual(table[12]["strain"], 0.15)
        self.assertEqual(table[12]["valid_intact"], 1)
        self.assertFalse(decision["wall_clock_lock"])
        self.assertFalse(decision["subsecond_lock"])
        self.assertFalse(decision["clocks_stretched"])
        self.assertFalse(decision["used_nominal_frame_time_as_clock"])
        self.assertFalse(decision["used_gui_offset_as_dt"])
        self.assertFalse(decision["used_frame_index_times_exposure"])

    def test_padded_starts_ignore_nominal_period_and_gui_offset(self):
        frames, tensile = demo_records("heps_id07_eiger")
        result = align_scan(frames, tensile, "heps_id07_eiger")
        decision = result["decision"]
        table = result["table"]

        self.assertEqual(decision["n_finite_frames"], 40)
        self.assertEqual(decision["n_input_rows"], 45)
        self.assertEqual(decision["n_padded_dropped"], 5)
        self.assertEqual(decision["virtual_nframes_ignored"], 10000000)
        self.assertNotEqual(len(table), 10000000)
        self.assertAlmostEqual(decision["gui_minus_detector_s_ignored"], -276.5)
        self.assertAlmostEqual(table[0]["t_start_s"], 0.0)
        self.assertAlmostEqual(table[1]["interval_from_prev_s"], 1.53)
        self.assertNotAlmostEqual(table[1]["interval_from_prev_s"], 1.0)
        self.assertAlmostEqual(decision["nominal_frame_time_s"], 1.0000001)
        self.assertTrue(decision["nominal_period_disagrees"])
        self.assertIn("nominal_period_disagrees", decision["complex_cases"])
        self.assertAlmostEqual(decision["measured_interval_median_s"], 1.53)
        self.assertAlmostEqual(decision["detector_readout_time_s_ignored"], 1e-7)

    def test_before_and_after_record_leave_stress_and_strain_empty(self):
        frames, tensile = demo_records("heps_id07_eiger")
        result = align_scan(frames, tensile, "heps_id07_eiger")
        decision = result["decision"]
        table = result["table"]

        self.assertTrue(decision["pre_record_frames"])
        self.assertTrue(decision["post_record_frames"])
        self.assertGreater(decision["n_before_record"], 0)
        self.assertGreater(decision["n_after_record"], 0)
        self.assertIn("pre_record_frames", decision["complex_cases"])
        self.assertIn("post_record_frames", decision["complex_cases"])
        early = table[0]
        self.assertTrue(early["before_record"])
        self.assertLess(early["t_mech_s"], 0.0)
        self.assertIsNone(early["stress"])
        self.assertIsNone(early["strain"])
        self.assertIsNone(early["force"])
        self.assertEqual(early["valid_intact"], 0)
        self.assertIsNotNone(early["eps_hkl"])
        late = table[-1]
        self.assertTrue(late["after_record"])
        self.assertGreater(late["t_mech_s"], 30.0)
        self.assertIsNone(late["stress"])
        self.assertIsNone(late["strain"])
        self.assertEqual(late["valid_intact"], 0)

    def test_duration_mismatch_does_not_stretch_either_clock(self):
        frames, tensile = demo_records("heps_id07_eiger")
        result = align_scan(frames, tensile, "heps_id07_eiger")
        decision = result["decision"]
        row = result["table"][-1]
        self.assertTrue(decision["duration_mismatch"])
        self.assertFalse(decision["clocks_stretched"])
        self.assertGreater(abs(decision["duration_delta_s"]), 30.0)
        self.assertAlmostEqual(row["t_mech_s"], row["t_mid_s"] + decision["dt_s"])
        self.assertTrue(row["after_record"])
        self.assertIsNone(row["stress"])
        self.assertIsNone(row["strain"])

    def test_flat_eps_is_unresolved_with_zero_dt(self):
        frames, tensile = demo_records("heps_id07_eiger")
        frames["eps_hkl"] = [0.0] * len(frames["eps_hkl"])
        result = align_scan(frames, tensile, "heps_id07_eiger")
        decision = result["decision"]
        self.assertTrue(decision["onset_unresolved"])
        self.assertEqual(decision["unresolved_reason"], "onset_missing")
        self.assertEqual(decision["dt_s"], 0.0)
        self.assertIn("unresolved_onset", decision["complex_cases"])
        self.assertIsNone(decision["rejected_dt_s"])

    def test_onset_already_at_first_frame_publishes_zero_dt(self):
        eps = np.array([1e-3, 0.0, 1e-6, -1e-6, 5e-7, -5e-7, 2e-7, -2e-7], dtype=np.float64)
        self.assertEqual(detect_monotonic_onset(eps), 0)
        frames, tensile = demo_records("heps_id07_eiger")
        series = np.zeros(len(frames["eps_hkl"]), dtype=np.float64)
        series[:8] = eps
        frames["eps_hkl"] = series.tolist()
        result = align_scan(frames, tensile, "heps_id07_eiger")
        decision = result["decision"]
        self.assertTrue(decision["onset_unresolved"])
        self.assertEqual(decision["unresolved_reason"], "onset_at_first_frame")
        self.assertEqual(decision["dt_s"], 0.0)

    def test_rising_strain_before_the_record_does_not_invent_a_lag(self):
        frames, tensile = demo_records("heps_id07_eiger")
        frames["eps_hkl"] = np.linspace(0.0, 8.0e-5, len(frames["eps_hkl"])).tolist()
        result = align_scan(frames, tensile, "heps_id07_eiger")
        decision = result["decision"]
        self.assertTrue(decision["onset_unresolved"])
        self.assertEqual(decision["unresolved_reason"], "rising_before_record")
        self.assertEqual(decision["dt_s"], 0.0)
        self.assertIsNotNone(decision["rejected_dt_s"])
        self.assertNotEqual(decision["rejected_dt_s"], 0.0)

    def test_subframe_disagreement_keeps_dt_at_zero(self):
        n = 20
        start = (np.arange(n) * 1.53).tolist()
        eps = np.zeros(n)
        eps[8:] = 1e-4
        t_pre = np.linspace(0.0, 5.0, 40)
        t_post = np.array([12.0, 12.1, 12.2, 12.3, 12.4, 13.0])
        t = np.concatenate([t_pre, t_post])
        force = np.concatenate([np.full(t_pre.shape, 8.0), np.full(t_post.shape, 100.0)])
        frames = {
            "start_time_s": start,
            "count_time_s": 1.0,
            "frame_time_s": 1.53,
            "eps_hkl": eps.tolist(),
        }
        tensile = {
            "t_s": t.tolist(),
            "force": force.tolist(),
            "stress": (2.0 * t).tolist(),
            "strain": (0.01 * t).tolist(),
        }
        result = align_scan(frames, tensile, "heps_id07_eiger")
        decision = result["decision"]
        self.assertTrue(decision["onset_unresolved"])
        self.assertEqual(decision["unresolved_reason"], "subframe_disagreement")
        self.assertEqual(decision["dt_s"], 0.0)
        self.assertGreater(abs(decision["rejected_dt_s"] - decision["subframe_dt_s"]), 1.53)

    def test_missing_load_rise_does_not_invent_a_lag(self):
        frames, tensile = demo_records("heps_id07_eiger")
        tensile["force"] = [8.0] * len(tensile["t_s"])
        result = align_scan(frames, tensile, "heps_id07_eiger")
        decision = result["decision"]
        self.assertEqual(decision["load_rise_status"], "missing")
        self.assertTrue(decision["onset_unresolved"])
        self.assertEqual(decision["unresolved_reason"], "load_rise_missing")
        self.assertEqual(decision["dt_s"], 0.0)

    def test_extracted_clock_profile_uses_the_same_core(self):
        frames, tensile = demo_records("extracted_frame_clock")
        result = align_scan(frames, tensile, "extracted_frame_clock")
        decision = result["decision"]
        table = result["table"]
        self.assertEqual(decision["profile"], "extracted_frame_clock")
        self.assertFalse(decision["onset_unresolved"])
        self.assertEqual(decision["onset_frame"], 12)
        self.assertAlmostEqual(decision["dt_s"], 30.0 - (12 * 2.0 + 0.25))
        self.assertAlmostEqual(table[12]["t_mid_s"], 24.25)
        self.assertAlmostEqual(table[12]["t_mech_s"], 30.0)
        self.assertAlmostEqual(table[12]["stress"], 60.0)
        self.assertAlmostEqual(table[12]["exposure_s"], 0.5)
        self.assertAlmostEqual(table[1]["interval_from_prev_s"], 2.0)
        self.assertTrue(decision["nominal_period_disagrees"])
        self.assertFalse(decision["duration_mismatch"])
        self.assertEqual(decision["n_valid_intact"], 24)
        self.assertEqual(decision["virtual_nframes_ignored"], None)
        self.assertIsNone(decision["gui_minus_detector_s_ignored"])

    def test_nonmonotonic_starts_are_flagged_and_not_reordered(self):
        frames = {
            "t_start_s": [0.0, 3.0, 1.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
            "exposure_s": 0.5,
            "nominal_frame_time_s": 2.0,
            "eps_hkl": [0.0, 0.0, 0.0, 0.0, 1e-4, 1.1e-4, 1.2e-4, 1.3e-4, 1.4e-4, 1.5e-4],
        }
        t = np.arange(0.0, 40.0, 0.05)
        tensile = {
            "t_s": t.tolist(),
            "force": np.where(t < 12.0, 0.2, 5.0).tolist(),
            "stress": (2.0 * t).tolist(),
            "strain": (0.002 * t).tolist(),
        }
        result = align_scan(frames, tensile, "extracted_frame_clock")
        self.assertTrue(result["decision"]["nonmonotonic_frame_clock"])
        self.assertIn("nonmonotonic_frame_clock", result["decision"]["complex_cases"])
        self.assertEqual(
            [row["t_start_s"] for row in result["table"][:3]],
            [0.0, 3.0, 1.0],
        )

    def test_agent_entry_is_deterministic_for_both_profiles(self):
        for profile in ("heps_id07_eiger", "extracted_frame_clock"):
            first = _run_cli("align", "--demo", profile)
            second = _run_cli("align", "--demo", profile)
            self.assertEqual(first, second)
            payload = json.loads(first)
            self.assertIn("decision", payload)
            self.assertIn("table", payload)
            self.assertGreater(payload["decision"]["n_valid_intact"], 0)
            intact = [row for row in payload["table"] if row["valid_intact"] == 1]
            self.assertTrue(all(row["stress"] is not None and row["strain"] is not None for row in intact))
            self.assertTrue(payload["decision"]["complex_cases"])
            again = dumps_result(align_scan(*demo_records(profile), profile))
            self.assertEqual(first, again)


if __name__ == "__main__":
    unittest.main()
