"""Regression gates for input protection, precise clock views and durable delivery."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py

from insitulink import workflow
from insitulink.errors import InputError
from insitulink.storage import FileDigests


class AuditWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "results"
        self.request = self.root / "experiment.json"
        self.source = self.root / "source.csv"
        self.source.write_text("time,value\n0,1\n1,1\n2,1\n", encoding="utf-8")
        with h5py.File(self.root / "raw.h5", "w") as handle:
            handle.create_dataset("images", shape=(1, 2, 2), dtype="u2")
        self.config = {"schema_version": 1,
                       "frames": {"data": [{"frame_index": 0, "t_start_s": 0, "exposure_s": .5,
                                             "source_file": "raw.h5", "dataset": "images"}]},
                       "sources": [{"id": "sensor", "path": "source.csv", "channels": [{"name": "value", "unit": "1"}],
                                    "clock": {"method": "shared", "confirmed": True, "evidence": "Synthetic common clock and origin"}}]}
        self.save()

    def save(self):
        self.request.write_text(json.dumps(self.config), encoding="utf-8")

    def load(self, filename):
        return json.loads((self.output / filename).read_text(encoding="utf-8"))

    def row(self):
        with (self.output / "frames.csv").open(encoding="utf-8-sig", newline="") as stream:
            return next(csv.DictReader(stream))

    def external_storage(self, *, table=False, invalid_time=False):
        self.output.mkdir()
        raw = self.output / "frames.csv"
        with h5py.File(self.root / "master.h5", "w") as handle:
            pixels = handle.create_dataset("images", shape=(1, 2, 2), dtype="u2",
                                           external=[(str(raw), 0, h5py.h5f.UNLIMITED)])
            pixels[...] = 7
            handle["starts"] = [0.]
            handle["exposures"] = .5
        if table:
            self.config["frames"]["data"][0]["source_file"] = "master.h5"
            if invalid_time:
                self.config["frames"]["time"] = {"unit": "not-a-time-unit"}
        else:
            self.config["frames"] = {"type": "hdf5", "path": "master.h5", "data_dataset": "images",
                                     "start_dataset": "starts", "exposure_dataset": "exposures"}
        self.save()
        return raw

    def test_metadata_only_indirect_storage_is_protected_before_any_output(self):
        raw = self.external_storage()
        before = raw.read_bytes()
        with self.assertRaises(InputError) as caught:
            workflow.run(self.request, self.output)
        self.assertEqual(caught.exception.code, "unsafe_output")
        self.assertEqual(raw.read_bytes(), before)
        self.assertFalse((self.output / "status.json").exists())

    def test_bad_table_time_does_not_bypass_indirect_storage_protection(self):
        raw = self.external_storage(table=True, invalid_time=True)
        before = raw.read_bytes()
        with self.assertRaises(InputError) as caught:
            workflow.run(self.request, self.output)
        self.assertEqual(caught.exception.code, "unsafe_output")
        self.assertEqual(raw.read_bytes(), before)
        self.assertFalse((self.output / "status.json").exists())

    def test_same_length_restored_timestamp_change_blocks_delivery(self):
        before = self.source.stat()
        real_sample = workflow.sample_channel

        def change_after_sampling(*args, **kwargs):
            result = real_sample(*args, **kwargs)
            self.source.write_bytes(self.source.read_bytes().replace(b",1", b",9"))
            os.utime(self.source, ns=(before.st_atime_ns, before.st_mtime_ns))
            return result

        with patch("insitulink.workflow.sample_channel", side_effect=change_after_sampling):
            state = workflow.run(self.request, self.output)
        self.assertEqual(state["status"], "needs_input", state["issues"])
        self.assertEqual(state["n_valid_analysis"], 0)
        self.assertIn("input_changed", str(state["issues"]))
        self.assertFalse(self.load("handoff.json")["ready_for_integration"])

    def test_source_parser_consumes_the_exact_hashed_snapshot(self):
        original = self.source.read_bytes()
        real_read = workflow.read_source

        def changing(config, base, *, table_bytes=None):
            self.assertEqual(table_bytes, original)
            self.source.write_bytes(original.replace(b",1", b",9"))
            try:
                return real_read(config, base, table_bytes=table_bytes)
            finally:
                self.source.write_bytes(original)

        with patch("insitulink.workflow.read_source", side_effect=changing):
            state = workflow.run(self.request, self.output)
        self.assertEqual(state["status"], "completed", state["issues"])
        self.assertEqual(self.row()["sensor.value.mid"], "1.0")
        bound = next(record for record in state["input_files"] if record["path"] == str(self.source))
        self.assertEqual(bound["sha256"], hashlib.sha256(original).hexdigest())

    def test_changes_during_report_generation_are_checked_before_publication(self):
        real_report = workflow.make_report
        changed = False

        def changing(*args):
            nonlocal changed
            report = real_report(*args)
            if not changed:
                self.source.write_bytes(self.source.read_bytes().replace(b",1", b",9"))
                changed = True
            return report

        with patch("insitulink.workflow.make_report", side_effect=changing):
            state = workflow.run(self.request, self.output)
        self.assertEqual(state["status"], "needs_input", state["issues"])
        self.assertFalse(self.load("handoff.json")["ready_for_integration"])
        self.assertFalse(state["ready_for_integration"])

    def test_nested_status_damage_is_json_and_resume_repairs_it(self):
        state = workflow.run(self.request, self.output)
        state["artifacts"]["frames.csv"] = {}
        (self.output / "status.json").write_text(json.dumps(state), encoding="utf-8")
        checked = workflow.inspect(self.output)
        self.assertEqual(checked["status"], "failed")
        self.assertIn("frames.csv", checked["artifact_verification"]["changed_or_missing"])
        completed = subprocess.run([sys.executable, "-m", "insitulink", "inspect", "--output", str(self.output)],
                                   text=True, capture_output=True, encoding="utf-8")
        self.assertEqual(json.loads(completed.stdout)["status"], "failed")
        self.assertNotIn("Traceback", completed.stderr)
        self.assertEqual(workflow.resume(self.output)["status"], "completed")

    def test_unhashable_saved_status_is_a_structured_issue(self):
        state = workflow.run(self.request, self.output)
        state["status"] = []
        (self.output / "status.json").write_text(json.dumps(state), encoding="utf-8")
        with self.assertRaises(InputError) as caught:
            workflow.inspect(self.output)
        self.assertEqual(caught.exception.code, "invalid_status")

    def test_selection_error_keeps_original_issue_and_delivers_diagnostics(self):
        self.config["analysis"] = {"frame_indices": [999]}
        self.save()
        state = workflow.run(self.request, self.output)
        self.assertEqual(state["status"], "needs_input", state["issues"])
        self.assertNotIn("delivery_failed", str(state["issues"]))
        self.assertTrue((self.output / "report.html").is_file())
        self.assertEqual(self.load("frame_records.json")["frames"][0]["selected"], False)

    def test_invalid_request_does_not_reference_a_previous_request_snapshot(self):
        workflow.run(self.request, self.output)
        self.request.write_text("{invalid JSON", encoding="utf-8")
        state = workflow.run(self.request, self.output)
        self.assertEqual(state["status"], "needs_input", state["issues"])
        self.assertNotIn("delivery_failed", str(state["issues"]))
        self.assertIsNone(self.load("frame_records.json")["request"])
        self.assertEqual(state["request_sha256"], hashlib.sha256(self.request.read_bytes()).hexdigest())
        self.assertNotIn("request.snapshot.json", state["artifacts"])

    def test_local_origins_preserve_shared_offset_and_anchor_meanings(self):
        self.source.write_text("time,value\n98,0\n99,1\n100,2\n101,3\n102,4\n", encoding="utf-8")
        self.config["frames"]["data"][0].update(t_start_s=100, exposure_s=1)
        clocks = [({"method": "shared", "confirmed": True, "evidence": "Same origin"}, 0., 2.5),
                  ({"method": "offset", "offset_s": -1, "confirmed": True, "evidence": "Known marker"}, -1., 1.5),
                  ({"method": "anchors", "pairs": [{"frame_time_s": 100, "source_time_s": 100}]}, 0., 2.5)]
        for clock, offset, value in clocks:
            with self.subTest(method=clock["method"]):
                self.config["sources"][0]["clock"] = clock
                self.save()
                state = workflow.run(self.request, self.output)
                self.assertEqual(state["status"], "completed", state["issues"])
                self.assertEqual(state["decisions"]["sensor"]["dt_s"], offset)
                self.assertEqual(float(self.row()["sensor.value.mid"]), value)

    def test_epoch_nanosecond_alignment_and_provenance_use_local_times(self):
        self.config["frames"] = {"data": [{"ticks": 1, "exposure_s": 1e-9, "frame_index": 0}],
                                 "columns": {"start": "ticks"}, "time": {"unit": "ns", "start": "2026-10-25T00:00:00Z"}}
        self.config["sources"][0].pop("path")
        self.config["sources"][0].update(data=[{"time": i, "value": i} for i in range(4)],
                                          time={"unit": "ns", "start": "2026-10-25T00:00:00Z"})
        self.save()
        state = workflow.run(self.request, self.output)
        self.assertEqual(state["status"], "completed", state["issues"])
        row = self.row()
        self.assertAlmostEqual(float(row["sensor.value.mid"]), 1.5)
        provenance = json.loads(row["sensor.value.provenance"])
        self.assertEqual(provenance["time_coordinate"], "source_local_seconds")
        self.assertEqual(provenance["mapping"]["published_offset_s"], 0.)
        self.assertAlmostEqual(provenance["midpoint"]["query_s"], 1.5e-9, places=18)
        self.assertEqual(len(provenance["midpoint"]["support"]), 2)

    def test_frame_record_and_legacy_selection_preserve_identity_and_traceability(self):
        initial = workflow.run(self.request, self.output)
        manifest = self.load("frame_manifest.json")["frames"][0]
        self.config["analysis"] = {"frame_uids": [manifest["legacy_frame_uid"]]}
        self.save()
        state = workflow.resume(self.output)
        self.assertEqual(state["n_selected"], 1)
        self.assertNotEqual(initial["run_id"], state["run_id"])
        records = self.load("frame_records.json")
        record = records["frames"][0]
        self.assertEqual(record["frame_uid"], manifest["frame_uid"])
        self.assertEqual(record["physical_source"]["frame_index"], 0)
        self.assertEqual(record["identity_status"], "verified")
        sample = record["devices"]["sensor"]["parameters"]["value"]
        self.assertEqual(sample["provenance"]["midpoint"]["support"][0]["source_row_number"], 2)
        self.assertEqual(records["run"]["request_sha256"], state["request_sha256"])
        self.assertTrue(state["raw_identity_verified"])
        (self.output / "frame_records.json").write_text("{}", encoding="utf-8")
        self.assertEqual(workflow.inspect(self.output)["status"], "failed")

    def test_large_public_offset_does_not_round_away_local_nanoseconds(self):
        self.config["frames"] = {"data": [{"t_start_s": 0, "exposure_s": 1e-9, "frame_index": 0}]}
        source = self.config["sources"][0]
        source.pop("path")
        source.update(data=[{"time": i, "value": i} for i in range(4)],
                      time={"unit": "ns", "start": "2026-10-25T00:00:00Z"},
                      clock={"method": "offset", "offset_s": "1792886400.000000001", "confirmed": True,
                             "evidence": "Synthetic independent common marker"})
        self.save()
        state = workflow.run(self.request, self.output)
        self.assertEqual(state["status"], "completed", state["issues"])
        self.assertAlmostEqual(float(self.row()["sensor.value.mid"]), 1.5)
        self.assertEqual(state["decisions"]["sensor"]["dt_s_decimal"], "1792886400.000000001")
        self.assertEqual(state["decisions"]["sensor"]["local_dt_s"], 1e-9)

    def test_metadata_only_alignment_never_hashes_detector_pixels(self):
        real_file = FileDigests.file

        def guarded(instance, path, **kwargs):
            self.assertNotEqual(Path(path).resolve(), (self.root / "raw.h5").resolve())
            return real_file(instance, path, **kwargs)

        with patch.object(FileDigests, "file", guarded):
            state = workflow.run(self.request, self.output)
        self.assertEqual(state["status"], "completed", state["issues"])


if __name__ == "__main__":
    unittest.main()
