"""Import drafts preserve observed inputs and become executable after editing."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from insitulink.errors import InputError
from insitulink import importing
from insitulink.importing import prepare


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frames = self.root / "frames.csv"
        self.frames.write_text("frame_id,t_start_s,exposure_s\nf0,0,1\nf1,1,1\n", encoding="utf-8")
        self.source = self.root / "load.csv"
        self.source.write_text("time,force\n0,0\n1,2\n2,4\n", encoding="utf-8")

    def read(self, path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def profile(self, **changes):
        value = {"schema_version": 1, "experiment": "tensile", "frames": {},
                 "sources": {"load": {"time": {"unit": "s"},
                             "channels": [{"name": "force", "column": "force", "unit": "N"}],
                             "clock": {"method": "shared", "confirmed": True,
                                       "evidence": "Independent trigger record for this test"}}}}
        value.update(changes)
        path = self.root / "profile-input.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_draft_has_specific_actions_no_invented_clock_or_unit(self):
        original = self.frames.read_bytes(), self.source.read_bytes()
        result = prepare(self.frames, {"load": self.source}, self.root / "draft")
        self.assertEqual(result["status"], "needs_input")
        request = self.read(result["artifacts"]["request"])
        source = request["sources"][0]
        self.assertIsNone(source["time"]["unit"])
        self.assertIsNone(source["channels"][0]["unit"])
        self.assertIsNone(source["clock"]["offset_s"])
        self.assertIs(source["clock"]["confirmed"], False)
        self.assertEqual(request["frames"]["time"]["unit"], "s")
        fields = {issue["field"] for issue in result["issues"]}
        self.assertTrue({"sources[0].time.unit", "sources[0].channels[0].unit", "sources[0].clock"} <= fields)
        preview = self.read(result["artifacts"]["preview"])
        self.assertEqual(preview["sources"]["load"]["columns"], ["time", "force"])
        self.assertEqual(preview["sources"]["load"]["row_count"], 3)
        self.assertEqual(original, (self.frames.read_bytes(), self.source.read_bytes()))

    def test_editing_draft_produces_runnable_request_without_second_confirmation(self):
        from insitulink.workflow import run
        result = prepare(self.frames, {"load": self.source}, self.root / "draft")
        path = result["artifacts"]["request"]
        request = self.read(path)
        source = request["sources"][0]
        source["time"]["unit"] = "s"
        source["channels"][0]["unit"] = "N"
        source["clock"] = {"method": "shared", "confirmed": True, "evidence": "Recorded shared trigger and common origin"}
        Path(path).write_text(json.dumps(request), encoding="utf-8")
        status = run(path, self.root / "results")
        self.assertEqual(status["status"], "completed", status.get("issues"))
        with (self.root / "results" / "frames.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([float(row["load.force.mean"]) for row in rows], [1, 3])
        self.assertEqual([row["original_id"] for row in rows], ["f0", "f1"])
        imported = self.read(result["artifacts"]["frame_manifest"])["frames"]
        self.assertEqual([row["frame_uid"] for row in imported], [row["frame_uid"] for row in rows])

    def test_profile_reuses_identity_registration_and_column_settings(self):
        first = prepare(self.frames, {"load": self.source}, self.root / "first", profile_path=self.profile())
        self.assertEqual(first["status"], "completed", first["issues"])
        second = prepare(self.frames, {"load": self.source}, self.root / "second", profile_path=first["artifacts"]["profile"])
        self.assertEqual(second["status"], "completed", second["issues"])
        before = self.read(first["artifacts"]["request"])
        after = self.read(second["artifacts"]["request"])
        for key in ("experiment_id", "acquisition_id", "detector_id"):
            self.assertEqual(before["metadata"][key], after["metadata"][key])
        self.assertEqual(before["metadata"]["import_registration"]["generated_fields"],
                         after["metadata"]["import_registration"]["generated_fields"])
        one = self.read(first["artifacts"]["frame_manifest"])["frames"]
        two = self.read(second["artifacts"]["frame_manifest"])["frames"]
        self.assertEqual([row["frame_uid"] for row in one], [row["frame_uid"] for row in two])

    def test_relative_raw_paths_keep_original_table_base(self):
        raw = self.root / "raw.bin"
        raw.write_bytes(b"untouched raw bytes")
        self.frames.write_text("frame_id,frame_index,t_start_s,exposure_s,source_file\nf0,0,0,1,raw.bin\n", encoding="utf-8")
        result = prepare(self.frames, {"load": self.source}, self.root / "nested" / "draft", profile_path=self.profile())
        request = self.read(result["artifacts"]["request"])
        self.assertEqual(request["frames"]["path"], str(self.frames))
        self.assertEqual(request["frames"]["source_root"], str(self.root))
        manifest = self.read(result["artifacts"]["frame_manifest"])
        self.assertEqual(manifest["frames"][0]["source_file"], str(raw))
        self.assertIn(str(raw), request["frames"]["assets"])

    def test_existing_output_is_never_overwritten(self):
        output = self.root / "draft"
        output.mkdir()
        request = output / "experiment.json"
        request.write_text("existing request", encoding="utf-8")
        with self.assertRaises(InputError) as raised:
            prepare(self.frames, {"load": self.source}, output)
        self.assertEqual(raised.exception.code, "output_exists")
        self.assertEqual(request.read_text(), "existing request")

    def test_output_must_not_contain_referenced_raw_file_even_if_missing(self):
        output = self.root / "new-output"
        self.frames.write_text("frame_index,t_start_s,exposure_s,source_file\n0,0,1,new-output/raw.bin\n", encoding="utf-8")
        with self.assertRaises(InputError) as raised:
            prepare(self.frames, {"load": self.source}, output)
        self.assertEqual(raised.exception.code, "output_contains_input")
        self.assertFalse(output.exists())

    def test_noncanonical_columns_and_units_come_from_profile(self):
        self.frames.write_text("label,begin,open\na,0,500\nb,1000,500\n", encoding="utf-8")
        profile = self.profile(frames={"columns": {"id": "label", "start": "begin", "exposure": "open"},
                                      "time": {"unit": "ms"}, "exposure_unit": "ms"})
        result = prepare(self.frames, {"load": self.source}, self.root / "draft", profile_path=profile)
        self.assertEqual(result["status"], "completed", result["issues"])
        rows = self.read(result["artifacts"]["frame_manifest"])["frames"]
        self.assertEqual([row["exposure_s"] for row in rows], [.5, .5])
        self.assertEqual([row["t_start_s"] for row in rows], [0, 1])

    def test_json_and_xlsx_tables(self):
        import openpyxl
        frames = self.root / "frames.json"
        frames.write_text(json.dumps([{"frame_id": "a", "t_start_s": 0, "exposure_s": 1}]), encoding="utf-8")
        source = self.root / "source.xlsx"
        book = openpyxl.Workbook()
        sheet = book.active
        sheet.append(["time", "force"])
        sheet.append([0, 0])
        sheet.append([1, 2])
        book.save(source)
        book.close()
        result = prepare(frames, {"load": source}, self.root / "draft", profile_path=self.profile())
        self.assertEqual(result["status"], "completed", result["issues"])
        preview = self.read(result["artifacts"]["preview"])
        self.assertEqual(preview["sources"]["load"]["sample_rows"][1]["force"], 2)

    def test_decimal_json_helper_preserves_exact_observations(self):
        frames = self.root / "frames.json"
        frames.write_text('[{"frame_id":"a","t_start_s":1792972800.000000123456789,"exposure_s":0.000000100000000001}]', encoding="utf-8")
        source = self.root / "source.json"
        source.write_text('[{"time":0.0,"force":1.2345678901234567890123456789},{"time":1.0,"force":2.5}]', encoding="utf-8")
        before = frames.read_bytes(), source.read_bytes()
        result = prepare(frames, {"load": source}, self.root / "decimal-helper")
        self.assertEqual(result["status"], "needs_input")
        json.dumps(result, allow_nan=False)
        preview = self.read(result["artifacts"]["preview"])
        row = preview["frames"]["sample_rows"][0]
        self.assertEqual(row["t_start_s"], "1792972800.000000123456789")
        self.assertEqual(row["exposure_s"], "1.00000000001E-7")
        source_row = preview["sources"]["load"]["sample_rows"][0]
        self.assertEqual(source_row["time"], "0.0")
        self.assertEqual(source_row["force"], "1.2345678901234567890123456789")
        for path in result["artifacts"].values():
            self.read(path)
        self.assertEqual(before, (frames.read_bytes(), source.read_bytes()))

    def test_decimal_json_cli_writes_complete_import_package(self):
        frames = self.root / "frames.json"
        frames.write_text('[{"t_start_s":0.0,"exposure_s":0.5}]', encoding="utf-8")
        output = self.root / "decimal-cli"
        command = [sys.executable, "-B", "-m", "insitulink", "prepare", "--frames", str(frames),
                   "--source", f"load={self.source}", "--output", str(output)]
        process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(process.returncode, 2, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result["status"], "needs_input")
        self.assertEqual(set(result["artifacts"]), {"request", "profile", "preview", "actions", "frame_manifest"})
        preview = self.read(result["artifacts"]["preview"])
        self.assertEqual(preview["frames"]["sample_rows"][0], {"t_start_s": "0.0", "exposure_s": "0.5"})
        for path in result["artifacts"].values():
            self.read(path)

    def test_serialization_failure_does_not_claim_output_directory(self):
        original = importing._table_preview
        def unsupported_reader_metadata(config):
            result = original(config)
            result["reader_metadata"] = object()
            return result
        output = self.root / "retryable"
        with patch.object(importing, "_table_preview", side_effect=unsupported_reader_metadata):
            with self.assertRaises(InputError) as raised:
                prepare(self.frames, {"load": self.source}, output)
        self.assertEqual(raised.exception.code, "invalid_import_output")
        self.assertEqual(raised.exception.details["artifact"], "preview")
        self.assertFalse(output.exists())
        result = prepare(self.frames, {"load": self.source}, output)
        self.assertEqual(result["status"], "needs_input")
        self.assertTrue(Path(result["artifacts"]["request"]).is_file())

    def test_missing_frame_columns_are_actionable(self):
        self.frames.write_text("label,actual_start,actual_duration\na,0,1\n", encoding="utf-8")
        result = prepare(self.frames, {"load": self.source}, self.root / "draft")
        codes = {item["code"] for item in result["issues"]}
        self.assertTrue({"missing_frame_start_column", "missing_exposure_column", "missing_time_unit"} <= codes)
        self.assertNotIn("frame_manifest", result["artifacts"])

    def test_invalid_table_header_retains_draft_and_specific_source(self):
        self.source.write_text("time,force,force\n0,1,2\n", encoding="utf-8")
        result = prepare(self.frames, {"load": self.source}, self.root / "draft")
        issue = next(item for item in result["issues"] if item["code"] == "source_preview_unavailable")
        self.assertIn("duplicate", issue["message"])
        self.assertEqual(issue["source"], str(self.source))

    def test_explicit_time_overrides_never_confirm_synchronization(self):
        result = prepare(self.frames, {"load": self.source}, self.root / "draft", frame_time_unit="ms",
                         frame_time_start="2026-10-26T00:00:00Z", source_time_units={"load": "ms"},
                         source_time_starts={"load": "2026-10-26T00:00:00Z"})
        request = self.read(result["artifacts"]["request"])
        self.assertEqual(request["sources"][0]["time"]["unit"], "ms")
        self.assertFalse(request["sources"][0]["clock"]["confirmed"])
        self.assertEqual(result["status"], "needs_input")

    def test_prepare_and_run_share_clock_basis_validation(self):
        profile = self.profile(frames={"time": {"unit": "s", "start": "2026-10-26T00:00:00Z"}})
        result = prepare(self.frames, {"load": self.source}, self.root / "draft", profile_path=profile)
        self.assertEqual(result["status"], "needs_input")
        self.assertIn("incompatible_time_basis", [issue["code"] for issue in result["issues"]])

    def test_hdf_metadata_import_does_not_read_detector_pixels(self):
        import h5py
        frames = self.root / "frames.h5"
        with h5py.File(frames, "w") as handle:
            handle.create_dataset("/entry/data/data", shape=(2, 3, 4), dtype="u2")
            times = handle.create_dataset("/entry/instrument/detector/start_time", data=[0., 1.])
            times.attrs["units"] = "s"
            exposure = handle.create_dataset("/entry/instrument/detector/count_time", data=[.5])
            exposure.attrs["units"] = "s"
        original = h5py.Dataset.__getitem__
        def checked(dataset, key):
            if dataset.ndim >= 2:
                raise AssertionError("Detector pixels must not be read during prepare")
            return original(dataset, key)
        with patch.object(h5py.Dataset, "__getitem__", checked):
            result = prepare(frames, {"load": self.source}, self.root / "draft", profile_path=self.profile())
        self.assertEqual(result["status"], "completed", result["issues"])
        self.assertEqual(result["n_frames"], 2)
        request = self.read(result["artifacts"]["request"])
        self.assertEqual(request["frames"]["time"]["unit"], "s")

    def test_hdf_dependency_under_output_is_protected_before_units_are_known(self):
        import h5py
        frames = self.root / "master.h5"
        with h5py.File(frames, "w") as handle:
            handle["/entry/data/data"] = h5py.ExternalLink("new-output/pixels.h5", "/images")
            handle.create_dataset("/entry/instrument/detector/start_time", data=[0.])
            handle.create_dataset("/entry/instrument/detector/count_time", data=[.5])
        with self.assertRaises(InputError) as raised:
            prepare(frames, {"load": self.source}, self.root / "new-output")
        self.assertEqual(raised.exception.code, "output_contains_input")
        self.assertFalse((self.root / "new-output").exists())

    def test_invalid_physical_identity_is_reported_in_actions(self):
        self.frames.write_text("frame_index,t_start_s,exposure_s,source_file\n0,0,1,missing.bin\n", encoding="utf-8")
        result = prepare(self.frames, {"load": self.source}, self.root / "draft", profile_path=self.profile())
        self.assertEqual(result["status"], "needs_input")
        self.assertIn("raw_source_missing", [issue["code"] for issue in result["issues"]])
        manifest = self.read(result["artifacts"]["frame_manifest"])
        self.assertEqual(manifest["frames"][0]["identity_status"], "invalid")

    def test_cli_json_protocol_duplicate_ids_and_help(self):
        base = [sys.executable, "-B", "-m", "insitulink"]
        result = subprocess.run(base + ["prepare", "--frames", str(self.frames), "--source", f"load={self.source}",
                                       "--output", str(self.root / "cli")], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(json.loads(result.stdout)["schema"], "insitulink.prepare.v1")
        duplicate = subprocess.run(base + ["prepare", "--frames", str(self.frames), "--source", f"load={self.source}",
                                          "--source", f"load={self.source}", "--output", str(self.root / "duplicate")],
                                   capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(duplicate.returncode, 2)
        self.assertEqual(json.loads(duplicate.stdout)["issues"][0]["code"], "invalid_assignment")
        self.assertFalse((self.root / "duplicate").exists())
        help_result = subprocess.run(base + ["prepare", "--help"], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("--profile", help_result.stdout)


if __name__ == "__main__":
    unittest.main()
