"""End-to-end input identity and metadata regressions using temporary experiments."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from insitulink.workflow import run

try:
    import h5py
except ImportError:
    h5py = None
try:
    from PIL import Image
except ImportError:
    Image = None


class InputPipelineRegressionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def request(self, frames, source=None):
        source = source or {
            "id": "load", "data": [{"time": i, "force": i * 2} for i in range(5)],
            "channels": [{"name": "force", "unit": "N"}],
            "clock": {"method": "shared", "confirmed": True,
                      "evidence": "Synthetic fixture uses the same clock and origin"}}
        request = self.base / "experiment.json"
        request.write_text(json.dumps({"schema_version": 1, "frames": frames, "sources": [source]}),
                           encoding="utf-8")
        return request

    def manifest(self, output):
        return json.loads((output / "frame_manifest.json").read_text(encoding="utf-8"))

    def hdf5(self, starts, *, count=3, exposure=.5):
        with h5py.File(self.base / "raw.h5", "w") as handle:
            handle.create_dataset("/entry/data/data", shape=(count, 2, 2), dtype="u1")
            handle.create_dataset("/entry/instrument/detector/start_time", data=starts)
            handle.create_dataset("/entry/instrument/detector/count_time", data=exposure)
        return {"type": "hdf5", "path": "raw.h5"}

    @unittest.skipIf(h5py is None, "h5py unavailable")
    def test_declared_hdf5_tail_frames_reconstructed_and_delivered(self):
        frames = self.hdf5([0, np.nan, np.nan])
        frames.update(frame_count=3, constant_period={"confirmed": True, "interval_s": 1, "exposure_s": .5})
        output = self.base / "results"
        status = run(self.request(frames), output)
        self.assertEqual(status["status"], "completed", status["issues"])
        manifest = self.manifest(output)
        self.assertEqual([r["frame_index"] for r in manifest["frames"]], [0, 1, 2])
        self.assertEqual([r["t_start_s"] for r in manifest["frames"]], [0, 1, 2])
        self.assertEqual(manifest["metadata"]["reconstructed_frame_indices"], [1, 2])
        self.assertTrue(manifest["metadata"]["reconstructed"])

    @unittest.skipIf(h5py is None, "h5py unavailable")
    def test_unknown_hdf5_tail_count_requires_input(self):
        output = self.base / "results"
        status = run(self.request(self.hdf5([0, np.nan, np.nan])), output)
        self.assertEqual(status["status"], "needs_input", status["issues"])
        self.assertTrue(status["frame_metadata"]["frame_count_unresolved"])
        self.assertFalse(json.loads((output / "handoff.json").read_text())["ready_for_integration"])

    @unittest.skipIf(Image is None, "Pillow unavailable")
    def test_single_frame_tiffs_require_explicit_local_index(self):
        for name in ("one.tiff", "two.tiff"):
            Image.fromarray(np.zeros((2, 2), dtype=np.uint16)).save(self.base / name)
        rows = [{"source_file": name, "t_start_s": i, "exposure_s": .5}
                for i, name in enumerate(("one.tiff", "two.tiff"))]
        status = run(self.request({"data": rows}), self.base / "missing-index")
        self.assertEqual(status["status"], "needs_input", status["issues"])
        self.assertTrue(any(issue["code"] == "missing_frame_index" for issue in status["issues"]))
        self.assertEqual(status["issues"][0]["row"], 0)
        for row in rows:
            row["frame_index"] = 0
        output = self.base / "explicit-index"
        status = run(self.request({"data": rows}), output)
        self.assertEqual(status["status"], "completed", status["issues"])
        manifest = self.manifest(output)["frames"]
        self.assertEqual([r["frame_index"] for r in manifest], [0, 0])
        self.assertEqual([r["sequence_index"] for r in manifest], [0, 1])
        self.assertNotEqual(manifest[0]["frame_uid"], manifest[1]["frame_uid"])
        self.assertEqual([r["source_file"] for r in manifest],
                         [str(self.base / name) for name in ("one.tiff", "two.tiff")])

    @unittest.skipIf(h5py is None, "h5py unavailable")
    def test_explicit_seconds_override_hdf5_millisecond_attribute(self):
        frames = self.hdf5([0], count=1)
        frames["exposure_unit"] = "s"
        with h5py.File(self.base / "raw.h5", "r+") as handle:
            handle["/entry/instrument/detector/count_time"].attrs["units"] = "ms"
        output = self.base / "results"
        status = run(self.request(frames), output)
        self.assertEqual(status["status"], "completed", status["issues"])
        self.assertEqual(self.manifest(output)["frames"][0]["exposure_s"], .5)
        self.assertEqual(status["frame_metadata"]["exposure_unit_source"], "config.exposure_unit")
        with (output / "frames.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(float(next(csv.DictReader(stream))["load.force.mean"]), .5)

    def test_duplicate_csv_headers_rejected_for_frames_and_external_sources(self):
        frames = {"data": [{"t_start_s": 0, "exposure_s": .5}]}
        for index, header in enumerate(("time,force,force", "time,time,force", "time,,force")):
            with self.subTest(header=header):
                (self.base / "log.csv").write_text(header + "\n0,1,2\n1,2,3\n", encoding="utf-8")
                source = {"id": "load", "path": "log.csv", "channels": [{"name": "force", "unit": "N"}]}
                status = run(self.request(frames, source), self.base / f"source-{index}")
                self.assertEqual(status["status"], "needs_input", status["issues"])
                self.assertTrue(any(i["code"] == "invalid_table_header" for i in status["issues"]))
        (self.base / "frames.csv").write_text("frame_id,frame_id,t_start_s,exposure_s\na,b,0,.5\n", encoding="utf-8")
        status = run(self.request({"path": "frames.csv"}), self.base / "frames-header")
        self.assertEqual(status["status"], "needs_input", status["issues"])
        self.assertEqual(status["issues"][0]["duplicate_columns"], ["frame_id"])


if __name__ == "__main__":
    unittest.main()
