"""Validate independent synthetic inputs without feeding reference truth back."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from insitulink.demo import create_demo
from insitulink.inputs import read_frames, read_source, referenced_paths
from insitulink.numerics import resolve_clock, sample_channel


class DemoTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)

    def load(self, case):
        path = create_demo(case, self.base / case)
        request = json.loads(path.read_text(encoding="utf-8"))
        frames, _ = read_frames(request["frames"], path.parent)
        sources = [read_source(config, path.parent) for config in request["sources"]]
        features = {"signals": [{"id": "eps_hkl", "t_s": [row["t_mid_s"] for row in frames],
                                  "values": [row["eps_hkl"] for row in frames]}]}
        return path, request, frames, sources, features

    def test_byte_determinism_and_truth_is_not_an_input(self):
        for case in ("tensile", "thermal", "multi"):
            with self.subTest(case=case):
                first = create_demo(case, self.base / (case + "1")).parent
                second = create_demo(case, self.base / (case + "2")).parent
                self.assertEqual({p.name: p.read_bytes() for p in first.iterdir()},
                                 {p.name: p.read_bytes() for p in second.iterdir()})
                request = json.loads((first / "experiment.json").read_text())
                self.assertNotIn("truth", json.dumps(request).lower())
                self.assertTrue(request["metadata"]["synthetic"])
                inputs = referenced_paths(request["frames"], first)
                inputs.extend(first / source["path"] for source in request["sources"])
                self.assertNotIn(first / "truth.json", inputs)
                self.assertTrue(all(p.is_file() for p in inputs))

    def test_frame_identity_and_nonuniform_acquisition_preserved(self):
        for case in ("tensile", "thermal", "multi"):
            with self.subTest(case=case):
                _, _, frames, _, _ = self.load(case)
                self.assertEqual([f["frame_index"] for f in frames], list(range(len(frames))))
                self.assertEqual([f["frame_number"] for f in frames], list(range(1, len(frames) + 1)))
                self.assertEqual(len({f["original_id"] for f in frames}), len(frames))
                self.assertTrue(all(f["original_id"] != str(f["frame_index"]) for f in frames))
                self.assertTrue(all(f["source_file"] is None for f in frames))
                self.assertGreater(np.ptp(np.diff([f["t_start_s"] for f in frames])), 0.2)
                self.assertGreater(np.ptp([f["exposure_s"] for f in frames]), 0.3)
                self.assertTrue(all(abs(f["eps_hkl"]) < 1e-6 for f in frames[:8]))

    def test_tensile_single_channel_resolves_load_onset(self):
        path, request, frames, sources, features = self.load("tensile")
        self.assertEqual(list(sources[0]["channels"]), ["force"])
        result = resolve_clock(frames, sources[0], features, request["experiment"])
        reference = json.loads((path.parent / "truth.json").read_text())
        self.assertEqual(result["status"], "resolved")
        interval = float(np.median(np.diff([f["t_mid_s"] for f in frames])))
        self.assertLess(abs(result["dt_s"] - reference["offsets_s"]["mechanics"]), interval)
        self.assertEqual(result["reason"], "common_event_onset")

    def test_thermal_delay_remains_unresolved_with_a_candidate(self):
        path, request, frames, sources, features = self.load("thermal")
        result = resolve_clock(frames, sources[0], features, request["experiment"])
        reference = json.loads((path.parent / "truth.json").read_text())
        self.assertEqual(result["status"], "needs_input")
        self.assertIsNone(result["dt_s"])
        self.assertIsNotNone(result["candidate_dt_s"])
        self.assertEqual(result["reason"], reference["expected_warning"])
        self.assertTrue(result["evidence"]["physical_response_lag_possible"])
        self.assertGreater(abs(result["candidate_dt_s"] - reference["offsets_s"]["furnace"]), 2)
        temperature = sources[0]["channels"]["temperature"]["values"]
        self.assertGreater(np.count_nonzero(temperature == 20), 20)
        self.assertGreater(np.count_nonzero(temperature == 200), 20)
        self.assertGreater(np.count_nonzero(np.diff(temperature) > 0), 20)
        self.assertGreater(np.count_nonzero(np.diff(temperature) < 0), 20)

    def test_multi_anchor_observations_and_channel_partial_validity(self):
        path, request, frames, sources, features = self.load("multi")
        reference = json.loads((path.parent / "truth.json").read_text())
        with (path.parent / "sync_markers.csv").open(newline="", encoding="utf-8") as stream:
            markers = list(csv.DictReader(stream))
        starts, ends = [f["t_start_s"] for f in frames], [f["t_end_s"] for f in frames]
        sampled = {}
        for source in sources:
            result = resolve_clock(frames, source, features, request["experiment"])
            self.assertEqual(result["status"], "resolved")
            self.assertAlmostEqual(result["dt_s"], reference["offsets_s"][source["id"]])
            for pair in source["clock"]["pairs"]:
                self.assertTrue(any(row["source_id"] == source["id"] and row["marker_id"] == pair["marker_id"]
                    and float(row["frame_time_s"]) == pair["frame_time_s"]
                    and float(row["source_time_s"]) == pair["source_time_s"] for row in markers))
            for name, channel in source["channels"].items():
                sampled[name] = sample_channel(source["t_s"], channel["values"], starts, ends,
                    result["dt_s"], kind=channel["kind"], max_gap_s=channel["max_gap_s"])
        self.assertTrue(all(row["valid"] for row in sampled["force"]))
        self.assertTrue(any(force["valid"] and not stress["valid"]
                            for force, stress in zip(sampled["force"], sampled["stress"])))
        self.assertTrue(any("gap" in row["flags"] for row in sampled["temperature"]))
        self.assertTrue(any("after_record" in row["flags"] for row in sampled["pressure"]))
        self.assertFalse(sources[-1]["channels"]["pressure"]["required"])
        self.assertEqual(sources[-1]["channels"]["pressure"]["kind"], "step")
        self.assertEqual(len({round(float(np.median(np.diff(s["t_s"]))), 5) for s in sources}), 3)

    def test_unknown_case_and_existing_files_are_preserved(self):
        directory = self.base / "output"
        with self.assertRaises(ValueError):
            create_demo("unknown", directory)
        self.assertFalse(directory.exists())
        directory.mkdir()
        (directory / "experiment.json").write_text("existing user data", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            create_demo("tensile", directory)
        self.assertEqual((directory / "experiment.json").read_text(), "existing user data")
        self.assertEqual([p.name for p in directory.iterdir()], ["experiment.json"])


if __name__ == "__main__":
    unittest.main()
