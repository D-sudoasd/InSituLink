"""Storage-closure integration gates, independent of peak fitting accuracy."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from insitulink.errors import InputError
from insitulink.hdf5_sources import discover_dependencies
from insitulink.workflow import resume, run


class StorageWorkflowRegressionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def experiment(self, *, storage_under_output=False):
        output = self.root / "results"
        if storage_under_output:
            output.mkdir()
        data = (output if storage_under_output else self.root) / "data.h5"
        with h5py.File(data, "w") as handle:
            handle.create_dataset("pixels", shape=(16, 2, 2), dtype="u2")
        with h5py.File(self.root / "master.h5", "w") as handle:
            handle["images"] = h5py.ExternalLink(str(data), "/pixels")
            handle["starts"] = np.arange(16, dtype=float)
            handle["exposure"] = .2
        # The extraction callback below isolates dependency handling, not pyFAI.
        (self.root / "geometry.poni").write_text("synthetic test calibration marker")
        config = {"schema_version": 1, "experiment": "tensile",
                  "frames": {"type": "hdf5", "path": "master.h5", "data_dataset": "images",
                             "start_dataset": "starts", "exposure_dataset": "exposure"},
                  "sources": [{"id": "load", "data": [{"time": i, "force": 0. if i < 10 else 1.} for i in range(40)],
                               "channels": [{"name": "force", "unit": "N"}],
                               "clock": {"method": "event", "channel": "force"}}],
                  "features": {"type": "images", "geometry": "geometry.poni", "mask": None,
                               "correct_solid_angle": False, "polarization_factor": None}}
        request = self.root / "experiment.json"
        request.write_text(json.dumps(config), encoding="utf-8")
        return request, output, data

    @staticmethod
    def extracted(config, frames, base, output):
        return {"rows": [], "signals": [{"id": "synthetic", "t_s": [r["t_mid_s"] for r in frames],
                                         "values": [0. if r["frame_index"] < 10 else 1e-4 for r in frames],
                                         "valid": [True] * len(frames)}], "metadata": {}, "issues": []}

    def test_external_storage_inside_output_is_protected_before_writes(self):
        request, output, data = self.experiment(storage_under_output=True)
        before = hashlib.sha256(data.read_bytes()).hexdigest()
        with self.assertRaises(InputError) as caught:
            run(request, output)
        self.assertEqual(caught.exception.code, "unsafe_output")
        self.assertEqual(hashlib.sha256(data.read_bytes()).hexdigest(), before)
        self.assertFalse((output / "status.json").exists())

    def test_actual_external_pixels_changed_during_run_block_delivery(self):
        request, output, data = self.experiment()

        def changing(config, frames, base, preview_output):
            result = self.extracted(config, frames, base, preview_output)
            with h5py.File(data, "r+") as handle:
                handle["pixels"][10, 0, 0] = 100
            return result

        with patch("insitulink.preview.extract_features", side_effect=changing):
            state = run(request, output)
        self.assertEqual(state["status"], "needs_input", state["issues"])
        self.assertEqual(state["n_valid_analysis"], 0)
        self.assertIn("input_changed", str(state["issues"]))
        self.assertFalse(json.loads((output / "handoff.json").read_text())["ready_for_integration"])

    def test_unverified_storage_never_reuses_or_publishes_a_ready_result(self):
        request, output, _ = self.experiment()

        def incomplete(frames):
            result = discover_dependencies(frames)
            result["complete"] = False
            result["issues"].append({"code": "unresolved_test_mapping", "message": "Synthetic unresolved storage mapping"})
            return result

        with patch("insitulink.hdf5_sources.discover_dependencies", side_effect=incomplete), \
                patch("insitulink.preview.extract_features", side_effect=self.extracted) as extract:
            first = run(request, output)
            second = resume(output)
        self.assertEqual(extract.call_count, 2)
        for state in (first, second):
            self.assertEqual(state["status"], "needs_input", state["issues"])
            self.assertEqual(state["stages"]["features"]["status"], "computed")
            self.assertFalse(state["stages"]["features"]["cacheable"])
            self.assertIn("unverified_feature_dependencies", str(state["issues"]))
        self.assertFalse(json.loads((output / "handoff.json").read_text())["ready_for_integration"])


if __name__ == "__main__":
    unittest.main()
