"""Full-run regressions for baseline, selected event channel and gap support."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from insitulink.workflow import run


class NumericalPipelineRegressions(unittest.TestCase):
    def run_request(self, request):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        request_path = root / "experiment.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        output = root / "results"
        status = run(request_path, output)
        with (output / "frames.csv").open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        return status, rows

    def event_request(self, *, experiment="tensile", onset=20):
        return {
            "schema_version": 1,
            "experiment": experiment,
            "frames": {"type": "table", "time": {"unit": "s"}, "data": [
                {"frame_index": i, "t_start_s": i, "exposure_s": 0,
                 "eps_hkl": 0 if i < onset else 1e-4 + (i - onset) * 1e-5}
                for i in range(50)
            ]},
            "sources": [{
                "id": "device", "time": {"unit": "s"}, "time_column": "time",
                "data": [{"time": i, "force": 0 if i < 20 else 10 + 2 * (i - 20)} for i in range(70)],
                "channels": [{"name": "force", "column": "force", "unit": "N", "kind": "continuous"}],
                "clock": {"method": "event", "channel": "force", "baseline_samples": 8, "min_run": 3, "k_sigma": 4},
            }],
        }

    def test_declared_noisy_baseline_does_not_publish_false_offset(self):
        for second in (0, 0.01):
            with self.subTest(second=second):
                request = self.event_request()
                for row, value in zip(request["sources"][0]["data"], [0, second, 1, 1, 1, 0, -1, 0]):
                    row["force"] = value
                status, rows = self.run_request(request)
                self.assertEqual(status["status"], "completed", status["issues"])
                self.assertEqual(status["decisions"]["device"]["dt_s"], 0)
                self.assertEqual(status["decisions"]["device"]["evidence"]["source_onset_s"], 20)
                self.assertEqual(float(rows[20]["device.force.mid"]), 10)
                self.assertEqual(rows[20]["device.force.valid"], "True")

    def test_quantized_baseline_scatter_does_not_create_early_event(self):
        request = self.event_request()
        source = request["sources"][0]
        for row, value in zip(source["data"], [0, 0, 0, 0, 0, 1, -1, 0]):
            row["force"] = value
        for row in source["data"][10:13]:
            row["force"] = 1
        status, _ = self.run_request(request)
        self.assertEqual(status["status"], "completed", status["issues"])
        self.assertEqual(status["decisions"]["device"]["dt_s"], 0)

    def test_thermal_selected_channel_keeps_clock_and_response_delay_ambiguous(self):
        # The source clock is +4 s; the sample response follows heating by 5 s.
        for experiment in ("generic", "tensile"):
            with self.subTest(experiment=experiment):
                request = self.event_request(experiment=experiment, onset=25)
                source = request["sources"][0]
                source["channels"][0]["experiment"] = "thermal"
                source["clock"]["confirmed_same_event"] = True
                for row in source["data"]:
                    row["force"] = 20 if row["time"] < 24 else 30 + row["time"] - 24
                status, rows = self.run_request(request)
                self.assertEqual(status["status"], "needs_input", status["issues"])
                decision = status["decisions"]["device"]
                self.assertIsNone(decision["dt_s"])
                self.assertEqual(decision["candidate_dt_s"], -1)
                self.assertEqual(decision["reason"], "thermal_response_is_not_clock_evidence")
                self.assertTrue(decision["evidence"]["physical_response_lag_possible"])
                self.assertTrue(all(row["device.force.valid"] == "False" for row in rows))

    def test_unselected_thermal_channel_does_not_change_force_event(self):
        request = self.event_request(experiment="generic")
        source = request["sources"][0]
        source["channels"][0]["experiment"] = "tensile"
        source["channels"].append({"name": "temperature", "column": "temperature", "unit": "degC", "experiment": "thermal"})
        for row in source["data"]:
            row["temperature"] = 20 + row["time"]
        status, _ = self.run_request(request)
        self.assertEqual(status["status"], "completed", status["issues"])
        self.assertEqual(status["decisions"]["device"]["dt_s"], 0)

    def test_explicit_gap_limit_supports_statistics_and_thermal_phase(self):
        request = {
            "schema_version": 1, "experiment": "thermal",
            "frames": {"type": "table", "data": [{"frame_index": 0, "t_start_s": 5, "exposure_s": 1}]},
            "sources": [{
                "id": "furnace", "time_column": "time", "time": {"unit": "s"},
                "data": [{"time": t, "temperature": 20 + t} for t in [0, 1, 2, 10, 11, 12]],
                "channels": [{"name": "temperature", "column": "temperature", "unit": "degC", "max_gap_s": 10}],
                "clock": {"method": "shared", "confirmed": True, "evidence": "Synthetic shared clock and origin"},
            }],
        }
        status, rows = self.run_request(request)
        self.assertEqual(status["status"], "completed", status["issues"])
        self.assertEqual(rows[0]["furnace.temperature.valid"], "True")
        self.assertEqual(float(rows[0]["furnace.temperature.mean"]), 25.5)
        self.assertEqual(rows[0]["furnace.temperature.phase"], "heating")


if __name__ == "__main__":
    unittest.main()
