"""Analytic exposure and planted-event tests for the new numerical core."""

import unittest

import numpy as np

from insitulink.errors import InputError
from insitulink.numerics import label_phases, resolve_clock, sample_channel


class ExposureTests(unittest.TestCase):
    def test_analytic_ramp_irregular_exposures(self):
        t = [0, 1, 2.5, 4, 6]
        rows = sample_channel(t, [2 * x + 3 for x in t], [0.2, 1.1, 3], [2.2, 4.7, 6])
        for row, a, b in zip(rows, [0.2, 1.1, 3], [2.2, 4.7, 6]):
            self.assertTrue(row["valid"])
            self.assertAlmostEqual(row["mean"], a + b + 3)
            self.assertAlmostEqual(row["mid"], row["mean"])
            self.assertAlmostEqual(row["min"], 2 * a + 3)
            self.assertAlmostEqual(row["max"], 2 * b + 3)
            self.assertEqual(row["coverage"], 1)

    def test_continuous_internal_extrema(self):
        row = sample_channel([0, 1, 2], [0, 4, 0], [0], [2])[0]
        self.assertEqual((row["mean"], row["min"], row["max"]), (2, 0, 4))

    def test_step_exact_integral_and_internal_jump_extrema(self):
        row = sample_channel([0, 1, 3, 4], [2, 6, -2, 9], [0.5], [3.5], kind="step")[0]
        self.assertAlmostEqual(row["mean"], 4)
        self.assertEqual((row["mid"], row["min"], row["max"]), (6, -2, 6))
        boundary = sample_channel([0, 1, 2], [2, 6, 9], [0], [1], kind="step")[0]
        self.assertEqual((boundary["mean"], boundary["min"], boundary["max"]), (2, 2, 2))
        point = sample_channel([0, 1, 2], [2, 6, 9], [1], [1], kind="step")[0]
        self.assertEqual(point["mean"], 6)

    def test_offset_sign_both_directions(self):
        for offset in [-2, 2]:
            row = sample_channel([0, 1, 2], [0, 10, 20], [-offset], [2 - offset], offset_s=offset)[0]
            self.assertEqual(row["mean"], 10)
            self.assertEqual(row["mid"], 10)

    def test_partial_bounds_do_not_publish_exposure_statistics(self):
        row = sample_channel([0, 1, 2], [0, 10, 20], [-1], [1])[0]
        self.assertEqual(row["mid"], 0)
        self.assertEqual(row["coverage"], 0.5)
        self.assertFalse(row["valid"])
        self.assertTrue(all(row[k] is None for k in ("mean", "min", "max")))
        self.assertIn("before_record", row["flags"])
        row = sample_channel([0, 1, 2], [0, 10, 20], [1], [3])[0]
        self.assertIn("after_record", row["flags"])
        self.assertEqual(row["mid"], 20)

    def test_large_gap_is_unsupported_but_exact_midpoint_is_available(self):
        row = sample_channel([0, 1, 2, 10, 11, 12], [0, 1, 2, 10, 11, 12], [0], [12])[0]
        self.assertAlmostEqual(row["coverage"], 1 / 3)
        self.assertIsNone(row["mid"])
        self.assertIn("gap", row["flags"])
        point = sample_channel([0, 10], [2, 3], [9], [11], max_gap_s=1)[0]
        self.assertEqual(point["mid"], 3)
        self.assertFalse(point["valid"])

    def test_nan_splits_continuous_support(self):
        row = sample_channel([0, 1, 2, 3, 4], [0, 1, np.nan, 3, 4], [0], [4])[0]
        self.assertEqual(row["coverage"], 0.5)
        self.assertIsNone(row["mean"])
        self.assertIn("nonfinite", row["flags"])
        isolated = sample_channel([0, 1, 2], [np.nan, 7, np.nan], [0], [2])[0]
        self.assertEqual(isolated["mid"], 7)
        self.assertEqual(isolated["coverage"], 0)
        self.assertIsNone(isolated["mean"])

    def test_nan_ends_previous_step_hold(self):
        rows = sample_channel([0, 1, 2, 3], [2, np.nan, 6, 9], [0, 1, 2], [1, 2, 3], kind="step")
        self.assertEqual(rows[0]["mean"], 2)
        self.assertIsNone(rows[1]["mean"])
        self.assertEqual(rows[2]["mean"], 6)

    def test_single_point_and_zero_exposure(self):
        rows = sample_channel([1], [5], [1, 0, 0], [1, 0, 2])
        self.assertTrue(rows[0]["valid"])
        self.assertEqual(rows[0]["mean"], 5)
        self.assertFalse(rows[1]["valid"])
        self.assertFalse(rows[2]["valid"])
        self.assertEqual(rows[2]["mid"], 5)

    def test_explicit_gap_limit_and_queries_preserve_input_order(self):
        rows = sample_channel([0, 1, 3], [0, 1, 3], [1, 0], [3, 1], max_gap_s=1.5)
        self.assertFalse(rows[0]["valid"])
        self.assertTrue(rows[1]["valid"])
        self.assertEqual(rows[1]["mean"], 0.5)

    def test_clocks_and_exposures_are_validated_without_reordering(self):
        for t in ([0, 0, 1], [0, 2, 1], [0, float("inf"), 2], []):
            with self.subTest(t=t), self.assertRaises(InputError):
                sample_channel(t, [0] * len(t), [0], [1])
        for kwargs in ({"kind": "cubic"}, {"max_gap_s": 0}, {"offset_s": float("nan")}):
            with self.assertRaises(InputError):
                sample_channel([0, 1], [0, 1], [0], [1], **kwargs)
        for starts, ends in (([1], [0]), ([np.nan], [1]), ([0, 1], [1])):
            with self.assertRaises(InputError):
                sample_channel([0, 1], [0, 1], starts, ends)


def fixture(offset=4, onsets=(12,), source_step=1, frame_step=1, baseline=12):
    """Ground truth offset is confined to fixtures, never runtime decisions."""
    mids = np.arange(40, dtype=float) * frame_step
    source_t = np.arange(80, dtype=float) * source_step - 20
    frames = [dict(index=i, t_mid_s=float(t), t_start_s=float(t - 0.1), exposure_s=0.2) for i, t in enumerate(mids)]
    source_y = np.maximum(source_t - (baseline + offset), 0)
    signals = [dict(id=f"peak-{i}", t_s=mids.tolist(), values=np.maximum(mids - onset, 0).tolist(),
                    valid=[True] * len(mids)) for i, onset in enumerate(onsets)]
    source = dict(t_s=source_t.tolist(), channels={"force": {"values": source_y.tolist(), "unit": "N"}},
                  clock={"method": "event", "channel": "force"})
    return frames, source, {"signals": signals, "metadata": {}}


class ClockTests(unittest.TestCase):
    def test_planted_positive_and_negative_offsets(self):
        for offset in [-7, 5]:
            frames, source, features = fixture(offset)
            result = resolve_clock(frames, source, features, "tensile")
            self.assertEqual(result["status"], "resolved")
            self.assertEqual(result["dt_s"], offset)
            self.assertGreaterEqual(result["uncertainty_s"], 1)
            self.assertFalse(result["evidence"]["clocks_stretched"])

    def test_falling_event_can_align_increasing_feature(self):
        frames, source, features = fixture(-3)
        source["channels"]["force"]["values"] = [-v for v in source["channels"]["force"]["values"]]
        source["clock"]["direction"] = "fall"
        self.assertEqual(resolve_clock(frames, source, features, "tensile")["dt_s"], -3)

    def test_generic_requires_same_event_confirmation(self):
        frames, source, features = fixture()
        result = resolve_clock(frames, source, features)
        self.assertEqual(result["status"], "needs_input")
        self.assertIsNone(result["dt_s"])
        self.assertEqual(result["candidate_dt_s"], 4)
        source["clock"]["confirmed_same_event"] = True
        self.assertEqual(resolve_clock(frames, source, features)["dt_s"], 4)

    def test_thermal_lag_remains_candidate_despite_latency_claim(self):
        frames, source, features = fixture(4, onsets=(17,))
        features["metadata"]["known_response_latency_s"] = 5
        source["clock"]["confirmed_same_event"] = True
        result = resolve_clock(frames, source, features, "thermal")
        self.assertEqual(result["status"], "needs_input")
        self.assertIsNone(result["dt_s"])
        self.assertEqual(result["candidate_dt_s"], -1)
        self.assertTrue(result["evidence"]["physical_response_lag_possible"])

    def test_known_response_latency_not_clock_sync_in_tensile(self):
        frames, source, features = fixture()
        features["metadata"]["response_latency_s"] = 4
        result = resolve_clock(frames, source, features, "tensile")
        self.assertIsNone(result["dt_s"])
        self.assertIn("latency", result["reason"])

    def test_flat_source_and_features_are_unresolved(self):
        for which in ("source", "features"):
            frames, source, features = fixture()
            if which == "source":
                source["channels"]["force"]["values"] = [0] * len(source["t_s"])
            else:
                features["signals"][0]["values"] = [0] * len(frames)
            result = resolve_clock(frames, source, features, "tensile")
            self.assertIsNone(result["dt_s"])
            self.assertIsNone(result["candidate_dt_s"])

    def test_initially_changing_feature_has_no_static_baseline(self):
        frames, source, features = fixture(onsets=(-4,))
        result = resolve_clock(frames, source, features, "tensile")
        self.assertIsNone(result["dt_s"])
        self.assertEqual(result["evidence"]["unavailable_signals"][0]["reason"], "initially_changing")

    def test_two_static_frames_suffice_for_early_onset(self):
        frames, source, features = fixture(4, onsets=(1,), baseline=1)
        self.assertEqual(resolve_clock(frames, source, features, "tensile")["dt_s"], 4)

    def test_conflicting_independent_peak_events_are_unresolved(self):
        frames, source, features = fixture(onsets=(12, 18))
        result = resolve_clock(frames, source, features, "tensile")
        self.assertEqual(result["reason"], "conflicting_feature_events")
        self.assertIsNone(result["dt_s"])
        self.assertEqual(len(result["evidence"]["signal_events"]), 2)

    def test_agreement_uses_actual_frame_interval(self):
        frames, source, features = fixture(onsets=(12, 14), frame_step=2)
        result = resolve_clock(frames, source, features, "tensile")
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["evidence"]["frame_interval_s"], 2)
        frames, source, features = fixture(onsets=(12, 14))
        self.assertEqual(resolve_clock(frames, source, features, "tensile")["status"], "needs_input")

    def test_sparse_sampling_reports_uncertainty(self):
        frames, source, features = fixture(source_step=4)
        result = resolve_clock(frames, source, features, "tensile")
        self.assertGreaterEqual(result["uncertainty_s"], 4)
        self.assertTrue(result["evidence"]["sparse_sampling"])

    def test_interval_uses_starts_with_variable_exposure(self):
        frames, source, features = fixture()
        for i, frame in enumerate(frames):
            frame["t_start_s"] = float(i * 2)
        result = resolve_clock(frames, source, features, "tensile")
        self.assertEqual(result["evidence"]["frame_interval_s"], 2)

    def test_unsustained_spike_is_not_an_event(self):
        frames, source, features = fixture()
        features["signals"][0]["values"] = [0] * len(frames)
        features["signals"][0]["values"][15] = 10
        self.assertIsNone(resolve_clock(frames, source, features, "tensile")["candidate_dt_s"])

    def test_invalid_event_settings_raise_input_error(self):
        for value in ("wrong", None, float("inf")):
            frames, source, features = fixture()
            source["clock"]["min_run"] = value
            with self.assertRaises(InputError):
                resolve_clock(frames, source, features, "tensile")

    def test_tensile_nonmechanical_channel_requires_confirmation(self):
        frames, source, features = fixture()
        source["channels"]["temperature"] = source["channels"].pop("force")
        source["clock"]["channel"] = "temperature"
        self.assertIsNone(resolve_clock(frames, source, features, "tensile")["dt_s"])

    def test_invalid_feature_samples_cannot_create_sustained_event(self):
        frames, source, features = fixture()
        features["signals"][0]["valid"][13:] = [False] * 27
        self.assertIsNone(resolve_clock(frames, source, features, "tensile")["dt_s"])

    def test_missing_method_never_implies_zero(self):
        frames, source, features = fixture()
        source["clock"] = {}
        self.assertIsNone(resolve_clock(frames, source, features)["dt_s"])
        source["clock"]["event"] = {"channel": "force", "confirmed_same_event": True}
        self.assertEqual(resolve_clock(frames, source, features)["dt_s"], 4)

    def test_shared_and_offset_require_explicit_confirmation(self):
        frames, source, features = fixture()
        for method in ("offset", "shared"):
            source["clock"] = {"method": method, "offset_s": -2}
            self.assertIsNone(resolve_clock(frames, source, features)["dt_s"])
            source["clock"].update(confirmed=True, evidence="Reference axes verified by user")
            result = resolve_clock(frames, source, features, "thermal")
            self.assertEqual(result["dt_s"], -2 if method == "offset" else 0)

    def test_anchors_and_inconsistent_offsets(self):
        frames, source, features = fixture()
        source["clock"] = {"method": "anchors", "pairs": [
            {"frame_time_s": 2, "source_time_s": 7}, {"frame_time_s": 20, "source_time_s": 25}]}
        result = resolve_clock(frames, source, features, "thermal")
        self.assertEqual(result["dt_s"], 5)
        self.assertEqual(result["uncertainty_s"], 1)
        self.assertEqual(result["evidence"]["anchor_max_residual_s"], 0)
        self.assertEqual(result["evidence"]["anchor_residuals_s"], [0, 0])
        source["clock"]["pairs"][1]["source_time_s"] = 35
        result = resolve_clock(frames, source, features)
        self.assertEqual(result["reason"], "inconsistent_anchors")
        self.assertIsNone(result["dt_s"])

    def test_anchor_uncertainty_preserves_residuals_and_explicit_precision(self):
        frames, source, features = fixture()
        source["clock"] = {"method": "anchors", "pairs": [
            {"frame_time_s": 2, "source_time_s": 7}, {"frame_time_s": 20, "source_time_s": 25.4}]}
        result = resolve_clock(frames, source, features)
        self.assertEqual(result["uncertainty_s"], 1)
        self.assertAlmostEqual(result["evidence"]["anchor_max_residual_s"], 0.2)
        source["clock"]["uncertainty_s"] = 0.5
        self.assertEqual(resolve_clock(frames, source, features)["uncertainty_s"], 0.5)
        source["clock"]["pairs"] = source["clock"]["pairs"][:1]
        source["clock"]["uncertainty_s"] = 0
        self.assertEqual(resolve_clock(frames, source, features)["uncertainty_s"], 0)
        del source["clock"]["uncertainty_s"]
        result = resolve_clock(frames[:1], source, features)
        self.assertIsNone(result["uncertainty_s"])
        self.assertEqual(result["evidence"]["anchor_max_residual_s"], 0)

    def test_negative_or_nonfinite_declared_uncertainty_is_invalid(self):
        for method in ("anchors", "shared", "offset"):
            for uncertainty in (-1, float("nan")):
                frames, source, features = fixture()
                source["clock"] = dict(method=method, confirmed=True, evidence="Confirmed",
                                       offset_s=4, uncertainty_s=uncertainty,
                                       pairs=[dict(frame_time_s=2, source_time_s=7)])
                with self.subTest(method=method, uncertainty=uncertainty), self.assertRaises(InputError):
                    resolve_clock(frames, source, features)

    def test_explicit_clock_methods_allow_interior_missing_frame(self):
        for method, expected in (("offset", -2), ("shared", 0), ("anchors", 5)):
            with self.subTest(method=method):
                frames, source, features = fixture()
                frames[12] = {"index": 12, "t_mid_s": None, "t_start_s": None, "exposure_s": None}
                original = [f.copy() for f in frames]
                source["clock"] = dict(method=method, offset_s=-2, confirmed=True,
                                       evidence="Reference axes verified", pairs=[
                                           {"frame_time_s": 2, "source_time_s": 7}])
                result = resolve_clock(frames, source, features)
                self.assertEqual(result["dt_s"], expected)
                self.assertEqual(result["evidence"]["frame_interval_s"], 1)
                self.assertEqual(result["evidence"]["n_invalid_frames"], 1)
                self.assertEqual(result["evidence"]["n_valid_frames"], 39)
                self.assertEqual(result["evidence"]["n_usable_frame_intervals"], 37)
                self.assertEqual(frames, original)

    def test_explicit_clock_methods_allow_single_available_frame(self):
        for method, expected in (("offset", 7), ("shared", 0), ("anchors", -3)):
            with self.subTest(method=method):
                frames, source, features = fixture()
                frames = [dict(t_mid_s=None, t_start_s=None), frames[12], {}]
                source["clock"] = dict(method=method, offset_s=7, confirmed=True,
                                       evidence="Reference axes verified", pairs=[
                                           {"frame_time_s": 2, "source_time_s": -1}])
                result = resolve_clock(frames, source, features)
                self.assertEqual(result["dt_s"], expected)
                self.assertIsNone(result["evidence"]["frame_interval_s"])
                self.assertEqual(result["evidence"]["n_invalid_frames"], 2)
                self.assertEqual(result["evidence"]["n_valid_frames"], 1)
                # A genuinely one-frame input is also sufficient for a known
                # relationship; missing neighbors are not required.
                result = resolve_clock([frames[1]], source, features)
                self.assertEqual(result["dt_s"], expected)
                self.assertEqual(result["evidence"]["n_invalid_frames"], 0)

    def test_missing_frame_does_not_replace_explicit_confirmation(self):
        frames, source, features = fixture()
        frames[12]["t_mid_s"] = None
        for method in ("offset", "shared"):
            source["clock"] = dict(method=method, offset_s=4)
            result = resolve_clock(frames, source, features)
            self.assertIsNone(result["dt_s"])
            self.assertEqual(result["reason"], "explicit_clock_confirmation_required")

    def test_missing_frames_are_not_bridged_to_measure_frame_interval(self):
        frames, source, features = fixture()
        source["clock"] = dict(method="offset", offset_s=4, confirmed=True, evidence="Verified")
        frames = [frames[0], {}, frames[2]]
        result = resolve_clock(frames, source, features)
        self.assertIsNone(result["evidence"]["frame_interval_s"])
        self.assertEqual(result["evidence"]["n_usable_frame_intervals"], 0)

    def test_event_alignment_rejects_missing_or_nonmonotonic_frame_clock(self):
        for field in ("t_mid_s", "t_start_s"):
            for value in (None, -1):
                with self.subTest(field=field, value=value):
                    frames, source, features = fixture()
                    frames[12][field] = value
                    with self.assertRaises(InputError):
                        resolve_clock(frames, source, features, "tensile")

    def test_event_alignment_rejects_missing_or_nonmonotonic_feature_clock(self):
        for value in (None, -1):
            frames, source, features = fixture()
            features["signals"][0]["t_s"][12] = value
            with self.assertRaises(InputError):
                resolve_clock(frames, source, features, "tensile")

    def test_external_duplicate_and_reversed_times_rejected(self):
        for times in ([0, 0, 1], [2, 1, 0]):
            frames, source, features = fixture()
            source["t_s"] = times
            with self.assertRaises(InputError):
                resolve_clock(frames, source, features)


class PhaseTests(unittest.TestCase):
    def test_thermal_cycles_stay_chronological(self):
        t = list(range(11))
        values = [20, 20, 20, 30, 40, 40, 30, 20, 20, 30, 40]
        self.assertEqual(label_phases(t, values, [0.5, 2.5, 4.5, 5.5, 7.5, 8.5], "thermal"),
                         ["baseline", "heating", "hold", "cooling", "hold", "heating"])

    def test_tensile_and_generic_labels(self):
        for experiment, names in (("tensile", ["baseline", "loading", "hold", "unloading"]),
                                  ("generic", ["baseline", "increasing", "hold", "decreasing"])):
            self.assertEqual(label_phases([0, 1, 2, 3, 4], [0, 0, 1, 1, 0], [0.5, 1.5, 2.5, 3.5], experiment), names)

    def test_missing_outside_gap_and_nonfinite_are_unknown(self):
        self.assertEqual(label_phases([0, 1, 2, 10, 11], [0, 0, 1, np.nan, 0], [-1, 1.5, 5, 10, 12]),
                         ["unknown", "increasing", "unknown", "unknown", "unknown"])

    def test_explicit_noise_tolerance(self):
        self.assertEqual(label_phases([0, 1, 2, 3], [0, 0.001, 0, 1], [0.5, 1.5, 2.5], tolerance=0.01),
                         ["baseline", "baseline", "increasing"])

    def test_default_tolerance_suppresses_noisy_baseline_and_hold(self):
        t = np.arange(161, dtype=float)
        values = np.interp(t, [0, 40, 80, 120, 160], [0, 0, 80, 80, 0])
        values += np.random.default_rng(782).normal(0, 0.02, len(t))
        query = t[:-1] + 0.5
        labels = label_phases(t, values, query, "tensile")
        self.assertEqual(labels[:40], ["baseline"] * 40)
        self.assertEqual(labels[40:80], ["loading"] * 40)
        self.assertEqual(labels[80:120], ["hold"] * 40)
        self.assertEqual(labels[120:], ["unloading"] * 40)

    def test_default_tolerance_preserves_ramp_from_first_sample(self):
        t = np.arange(50, dtype=float)
        values = 2 * t + np.random.default_rng(3).normal(0, 0.01, len(t))
        self.assertEqual(label_phases(t, values, t[:-1] + 0.5, "tensile"), ["loading"] * 49)

    def test_explicit_zero_tolerance_overrides_default_noise_suppression(self):
        t = np.arange(50, dtype=float)
        values = np.random.default_rng(3).normal(0, 0.01, len(t))
        labels = label_phases(t, values, t[:-1] + 0.5, tolerance=0)
        self.assertEqual(labels, ["increasing" if v > 0 else "decreasing" for v in np.diff(values)])

    def test_step_phases_label_jumps_without_anticipating_them(self):
        self.assertEqual(label_phases([0, 1, 2, 3, 4], [0, 0, 2, 2, 0],
                                     [0.5, 1.5, 2, 2.5, 3.5, 4], kind="step"),
                         ["baseline", "baseline", "increasing", "hold", "hold", "decreasing"])


if __name__ == "__main__":
    unittest.main()
