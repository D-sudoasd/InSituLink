"""Numerical regressions for the 2026-10-26 alignment audit."""

import unittest

import numpy as np

from insitulink import numerics


def noisy_ramp():
    t = np.arange(100, dtype=float)
    frames = [dict(t_start_s=float(x), t_mid_s=float(x), exposure_s=0) for x in t]
    force = np.maximum(t - 20, 0) * .1
    force[:8] = [-1, .3, -.4, .7, -.8, .5, -.2, .1]
    source = dict(t_s=t, channels={"force": dict(values=force, unit="N")},
                  clock=dict(method="event", channel="force"))
    features = {"signals": [dict(id=f"peak_{i}", t_s=t,
                                 values=np.maximum(t - 20, 0) * scale)
                             for i, scale in enumerate((1e-5, 2e-5))]}
    return frames, source, features


class EventAuditTests(unittest.TestCase):
    def test_noise_threshold_delay_cannot_be_confirmed_as_clock_offset(self):
        frames, source, features = noisy_ramp()
        result = numerics.resolve_clock(frames, source, features, "tensile")
        self.assertEqual(result["status"], "needs_input")
        self.assertIsNone(result["dt_s"])
        self.assertEqual(result["candidate_dt_s"], 26)
        self.assertGreaterEqual(result["uncertainty_s"], 26)
        self.assertEqual(result["reason"], "event_threshold_timing_ambiguous")
        self.assertGreater(result["evidence"]["source_event"]["threshold_timing_s"], 20)

    def test_clear_step_with_noisy_baseline_remains_resolved(self):
        frames, source, features = noisy_ramp()
        source["channels"]["force"]["values"][20:] = 10
        for signal in features["signals"]:
            signal["values"][20:] = 1e-4
        result = numerics.resolve_clock(frames, source, features, "tensile")
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["dt_s"], 0)

    def test_one_steep_sample_cannot_hide_threshold_sensitivity(self):
        frames, source, features = noisy_ramp()
        source["channels"]["force"]["values"][47] = 10
        result = numerics.resolve_clock(frames, source, features, "tensile")
        self.assertIsNone(result["dt_s"])
        self.assertEqual(result["reason"], "event_threshold_timing_ambiguous")
        self.assertGreater(result["evidence"]["source_event"]["threshold_sensitivity_s"], 1)

    def test_event_cannot_cross_explicitly_unsupported_gap(self):
        frames, source, features = noisy_ramp()
        source["t_s"] = np.array(list(range(9)) + list(range(11, 40)))
        source["channels"]["force"].update(
            values=(source["t_s"] >= 9).astype(float), max_gap_s=1.1)
        for signal in features["signals"]:
            signal["values"] = (signal["t_s"] >= 9).astype(float)
        result = numerics.resolve_clock(frames, source, features, "tensile")
        self.assertEqual(result["status"], "needs_input")
        self.assertIsNone(result["dt_s"])
        self.assertEqual(result["reason"], "source_event_crosses_gap")

    def test_explicit_supported_event_gap_can_exceed_default(self):
        frames, source, features = noisy_ramp()
        source["t_s"] = np.array(list(range(9)) + list(range(20, 40)))
        source["channels"]["force"].update(
            values=(source["t_s"] >= 20).astype(float), max_gap_s=12)
        for signal in features["signals"]:
            signal["values"] = (signal["t_s"] >= 20).astype(float)
        self.assertEqual(numerics.resolve_clock(frames, source, features, "tensile")["dt_s"], 0)

    def test_anchors_record_observation_span_without_claiming_drift_correction(self):
        frames, source, _ = noisy_ramp()
        source["clock"] = dict(method="anchors", pairs=[
            dict(frame_time_s=10, source_time_s=13),
            dict(frame_time_s=80, source_time_s=83)])
        result = numerics.resolve_clock(frames, source)
        self.assertEqual(result["dt_s"], 3)
        self.assertEqual(result["evidence"]["anchor_frame_range_s"], [10, 80])
        self.assertEqual(result["evidence"]["anchor_source_range_s"], [13, 83])
        self.assertFalse(result["evidence"]["clocks_stretched"])


class SamplingProvenanceTests(unittest.TestCase):
    def test_linear_midpoint_can_be_reconstructed_from_recorded_support(self):
        row = numerics.sample_channel([0, 1, 3], [0, 2, 6], [.5], [2.5])[0]
        provenance = row["provenance"]
        self.assertEqual(provenance["model"], "piecewise_linear")
        support = provenance["midpoint"]["support"]
        self.assertEqual([p["row_index"] for p in support], [1, 2])
        self.assertEqual([p["time_s"] for p in support], [1, 3])
        self.assertEqual([p["weight"] for p in support], [.75, .25])
        self.assertEqual(sum(p["weight"] * p["value"] for p in support), row["mid"])
        self.assertEqual(provenance["exposure"]["overlapping_row_range"], [0, 2])
        self.assertEqual(provenance["exposure"]["supported_interval_row_ranges"], [[0, 1]])
        self.assertEqual(provenance["exposure"]["supported_duration_s"], 2)

    def test_held_midpoint_records_value_row_and_support_boundary(self):
        row = numerics.sample_channel([0, 1, 3], [2, 6, 9], [.5], [2.5], kind="step")[0]
        provenance = row["provenance"]
        self.assertEqual(provenance["model"], "previous_sample_hold")
        self.assertEqual(provenance["midpoint"]["bracket_row_indices"], [1, 2])
        self.assertEqual(provenance["midpoint"]["support"], [
            dict(row_index=1, time_s=1, value=6, weight=1)])

    def test_gap_provenance_keeps_separate_supported_ranges(self):
        row = numerics.sample_channel([0, 1, 5, 6], [0, 1, 5, 6], [0], [6], max_gap_s=1)[0]
        self.assertFalse(row["valid"])
        provenance = row["provenance"]
        self.assertEqual(provenance["max_gap_s"], 1)
        self.assertEqual(provenance["midpoint"]["status"], "gap")
        self.assertEqual(provenance["midpoint"]["support"], [])
        self.assertEqual(provenance["exposure"]["supported_interval_row_ranges"], [[0, 0], [2, 2]])
        self.assertEqual(provenance["exposure"]["supported_duration_s"], 2)

    def test_zero_exposure_uses_exact_sample_and_has_no_integration_intervals(self):
        row = numerics.sample_channel([1], [5], [1], [1])[0]
        provenance = row["provenance"]
        self.assertEqual(provenance["midpoint"]["status"], "exact")
        self.assertEqual(provenance["exposure"]["supported_interval_row_ranges"], [])
        self.assertEqual(provenance["exposure"]["supported_duration_s"], 0)


class PhaseAuditTests(unittest.TestCase):
    def test_default_phases_are_invariant_under_unit_scale(self):
        t, q = [0, 1, 2, 3], [.5, 1.5, 2.5]
        for scale in (1, 1e-9, 1e9):
            labels = numerics.label_phases(t, np.array([0, .1, .2, .3]) * scale, q, "tensile")
            self.assertEqual(labels, ["loading"] * 3)

    def test_default_phases_are_invariant_under_temperature_zero(self):
        t, q = [0, 1, 2, 3], [.5, 1.5, 2.5]
        for zero in (20, 293.15):
            labels = numerics.label_phases(t, [zero + i * 1e-7 for i in t], q, "thermal")
            self.assertEqual(labels, ["heating"] * 3)

    def test_exposure_crossing_turn_keeps_midpoint_and_both_phases(self):
        result = numerics.exposure_phases([0, 1, 2], [0, 10, 0], [0], [2], experiment="tensile")[0]
        self.assertEqual(result["midpoint"], "unloading")
        self.assertEqual(result["phases"], ["loading", "unloading"])
        self.assertTrue(result["crosses_phase"])
        self.assertFalse(result["unsupported"])

    def test_step_exposure_excludes_jump_at_right_boundary(self):
        result = numerics.exposure_phases([0, 1, 2, 3], [0, 1, 1, 0], [0], [3], kind="step")[0]
        self.assertEqual(result["phases"], ["baseline", "increasing", "hold"])
        self.assertNotIn("decreasing", result["phases"])

    def test_exposure_reports_unsupported_portion_and_declared_tolerance(self):
        result = numerics.exposure_phases([0, 1, 5, 6], [0, 1, 5, 6], [0], [6],
                                          tolerance=.01, max_gap_s=1)[0]
        self.assertTrue(result["unsupported"])
        self.assertEqual(result["tolerance_slope"], .01)
        self.assertEqual(result["tolerance_source"], "declared")


if __name__ == "__main__":
    unittest.main()
