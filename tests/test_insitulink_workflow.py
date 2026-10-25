"""Delivery, provenance, resume and public command contract tests."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from insitulink.errors import InputError
from insitulink.demo import create_demo
from insitulink.workflow import inspect, resume, run


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def example(self, case):
        request = create_demo(case, self.root / case)
        return request, request.parent / "results"

    def load(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def add_raw_stack(self, request):
        """Provide real synthetic HDF5 frames for integration-readiness checks."""
        import h5py
        path = request.parent / "frames.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        with h5py.File(request.parent / "raw.h5", "w") as handle:
            handle.create_dataset("images", shape=(len(rows), 2, 2), dtype="u2")
        for row in rows:
            row.update(source_file="raw.h5", dataset="images")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_tensile_delivery_identity_and_frame_precision(self):
        request, output = self.example("tensile")
        status = run(request, output)
        self.assertEqual(status["status"], "completed", status["issues"])
        truth = self.load(request.parent / "truth.json")
        decision = status["decisions"]["mechanics"]
        self.assertLess(abs(decision["dt_s"] - truth["offsets_s"]["mechanics"]), 1.0)
        manifest = self.load(output / "frame_manifest.json")
        with (request.parent / "frames.csv").open(newline="", encoding="utf-8") as stream:
            originals = list(csv.DictReader(stream))
        self.assertEqual(len(manifest["frames"]), len(originals))
        for source, row in zip(originals, manifest["frames"]):
            self.assertEqual(row["frame_index"], int(source["frame_index"]))
            self.assertEqual(row["frame_number"], row["frame_index"] + 1)
            self.assertEqual(str(row["original_id"]), source["frame_id"])
        handoff = self.load(output / "handoff.json")
        self.assertTrue(handoff["alignment_complete"])
        self.assertFalse(handoff["ready_for_integration"])
        self.assertEqual(handoff["raw_source_issues"][0]["reason"], "raw_source_not_referenced")
        self.assertFalse(handoff["raw_frames_copied"])
        self.assertEqual(handoff["consumer_integration"], "file_contract_only")
        self.assertTrue(inspect(output)["artifact_verification"]["ok"])
        html = (output / "report.html").read_text(encoding="utf-8")
        self.assertIn("data:image/png;base64,", html)
        self.assertNotIn('<script src="http', html)

    def test_thermal_candidate_is_not_published_as_clock_zero(self):
        request, output = self.example("thermal")
        status = run(request, output)
        self.assertEqual(status["status"], "needs_input", status["issues"])
        self.assertIsNone(status["decisions"]["furnace"]["dt_s"])
        self.assertIsNotNone(status["decisions"]["furnace"]["candidate_dt_s"])
        self.assertEqual(status["n_valid_analysis"], 0)
        self.assertFalse(self.load(output / "handoff.json")["ready_for_integration"])

    def test_multi_source_missing_channels_do_not_erase_good_values(self):
        request, output = self.example("multi")
        status = run(request, output)
        self.assertEqual(status["status"], "completed", status["issues"])
        self.assertGreater(status["n_valid_analysis"], 0)
        self.assertLess(status["n_valid_analysis"], status["n_frames"])
        with (output / "frames.csv").open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        gaps = [r for r in rows if r["mechanics.stress.valid"] == "False"]
        self.assertTrue(any(r["mechanics.force.mid"] for r in gaps))
        self.assertTrue(any(r["pressure.pressure.valid"] == "False" and r["valid_analysis"] == "True" for r in rows))

    def test_resume_reuses_stages_then_invalidates_only_changed_source(self):
        request, output = self.example("multi")
        first = run(request, output)
        second = resume(output)
        self.assertTrue(all(v["status"] == "reused" for k, v in second["stages"].items() if k != "features"))
        source = request.parent / "mechanics.csv"
        source.write_text(source.read_text().replace(",8.", ",9.", 1), encoding="utf-8")
        # Always change actual bytes, independent of the particular demo baseline.
        source.write_text(source.read_text() + "\n", encoding="utf-8")
        third = resume(output)
        self.assertEqual(third["stages"]["frames"]["status"], "reused")
        self.assertEqual(third["stages"]["source-mechanics"]["status"], "computed")
        self.assertEqual(third["stages"]["source-furnace"]["status"], "reused")
        self.assertEqual(first["n_frames"], third["n_frames"])

    def test_confirmation_resolves_thermal_without_reusing_old_decision(self):
        request, output = self.example("thermal")
        run(request, output)
        config = self.load(request)
        config["sources"][0]["clock"] = {"method": "offset", "offset_s": -4.6, "confirmed": True, "evidence": "Manual independent marker confirmation for this synthetic example"}
        request.write_text(json.dumps(config), encoding="utf-8")
        status = resume(output)
        self.assertEqual(status["status"], "completed", status["issues"])
        self.assertEqual(status["decisions"]["furnace"]["dt_s"], -4.6)
        self.assertEqual(status["stages"]["features"]["status"], "not_needed")
        self.assertEqual(status["stages"]["clock-furnace"]["status"], "computed")

    def test_changed_artifact_is_detected_and_resume_repairs_it(self):
        request, output = self.example("tensile")
        run(request, output)
        (output / "frames.csv").write_text("corrupted", encoding="utf-8")
        self.assertEqual(inspect(output)["status"], "failed")
        self.assertEqual(resume(output)["status"], "completed")
        self.assertTrue(inspect(output)["artifact_verification"]["ok"])

    def test_running_snapshot_does_not_claim_a_failed_delivery(self):
        output = self.root / "running"
        output.mkdir()
        (output / "status.json").write_text(json.dumps({"schema": "insitulink.run.v1", "status": "running", "issues": [], "artifacts": {}}))
        state = inspect(output)
        self.assertEqual(state["status"], "running")
        self.assertFalse(state["artifact_verification"]["ok"])

    def test_malformed_status_inventory_is_a_structured_input_issue(self):
        output = self.root / "invalid-status"
        output.mkdir()
        (output / "status.json").write_text(json.dumps({"schema": "insitulink.run.v1", "status": "completed", "issues": [], "artifacts": "invalid"}))
        with self.assertRaises(InputError) as caught:
            inspect(output)
        self.assertEqual(caught.exception.code, "invalid_status")

    def test_unneeded_preview_does_not_import_or_read_images(self):
        request, output = self.example("multi")
        config = self.load(request)
        config["features"] = {"type": "images", "geometry": "missing.poni", "mask": "missing.edf"}
        request.write_text(json.dumps(config), encoding="utf-8")
        self.assertEqual(run(request, output)["status"], "completed")
        self.assertEqual(inspect(output)["stages"]["features"]["status"], "not_needed")

    def test_bad_external_source_keeps_other_decisions_but_blocks_delivery(self):
        request, output = self.example("multi")
        config = self.load(request)
        config["sources"][1]["path"] = "missing.csv"
        request.write_text(json.dumps(config), encoding="utf-8")
        status = run(request, output)
        self.assertEqual(status["status"], "needs_input")
        self.assertEqual(status["decisions"]["mechanics"]["status"], "resolved")
        self.assertFalse(self.load(output / "handoff.json")["ready_for_integration"])
        self.assertEqual(status["n_valid_analysis"], 0)

    def test_cli_json_and_exit_codes(self):
        for case, code in (("tensile", 0), ("thermal", 2), ("multi", 0)):
            completed = subprocess.run([sys.executable, "-m", "insitulink", "demo", "--case", case, "--output", str(self.root / case)],
                                       text=True, capture_output=True, encoding="utf-8")
            payload = json.loads(completed.stdout)
            self.assertEqual(completed.returncode, code, completed.stderr + str(payload.get("issues")))
            self.assertEqual(payload["status"], "needs_input" if code else "completed")
            self.assertNotIn("truth", json.dumps(payload["stages"]))

    def test_output_cannot_contain_any_input_even_when_request_is_elsewhere(self):
        request, output = self.example("multi")
        output.mkdir()
        original = (request.parent / "mechanics.csv").read_bytes()
        protected = output / "frames.csv"
        protected.write_bytes(original)
        config = self.load(request)
        config["sources"][0]["path"] = "results/frames.csv"
        request.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(InputError, "contains experiment inputs"):
            run(request, output)
        self.assertEqual(protected.read_bytes(), original)
        self.assertFalse((output / "status.json").exists())

    def test_failure_revokes_previously_ready_handoff_and_empty_inventory_is_not_ok(self):
        request, output = self.example("multi")
        self.add_raw_stack(request)
        self.assertEqual(run(request, output)["status"], "completed")
        self.assertTrue(self.load(output / "handoff.json")["ready_for_integration"])
        config = self.load(request)
        config["sources"] = [None]
        request.write_text(json.dumps(config), encoding="utf-8")
        status = resume(output)
        self.assertEqual(status["status"], "needs_input")
        self.assertFalse(self.load(output / "handoff.json")["ready_for_integration"])
        # An empty artifact inventory cannot verify a delivery successfully.
        status["artifacts"] = {}
        (output / "status.json").write_text(json.dumps(status), encoding="utf-8")
        self.assertFalse(inspect(output)["artifact_verification"]["ok"])

    def test_report_failure_cannot_leave_integration_ready(self):
        request, output = self.example("multi")
        self.add_raw_stack(request)
        run(request, output)
        self.assertTrue(self.load(output / "handoff.json")["ready_for_integration"])
        with patch("insitulink.workflow.make_report", side_effect=RuntimeError("report failure")):
            status = resume(output)
        self.assertEqual(status["status"], "failed")
        self.assertFalse(self.load(output / "handoff.json")["ready_for_integration"])
        self.assertFalse(inspect(output)["artifact_verification"]["ok"])

    def test_missing_one_frame_time_does_not_erase_other_confirmed_values(self):
        request, output = self.example("multi")
        path = request.parent / "frames.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        rows[30]["t_start_s"] = ""
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)
        status = run(request, output)
        self.assertEqual(status["status"], "completed", status["issues"])
        self.assertEqual(status["decisions"]["mechanics"]["status"], "resolved")
        self.assertGreater(status["n_valid_analysis"], 0)
        with (output / "frames.csv").open(encoding="utf-8-sig", newline="") as stream:
            result = list(csv.DictReader(stream))
        self.assertEqual(result[30]["valid_analysis"], "False")
        self.assertEqual(result[30]["mechanics.force.mid"], "")
        self.assertNotEqual(result[29]["mechanics.force.mid"], "")

    def test_frame_selection_uses_stable_identity_and_keeps_baseline(self):
        request, output = self.example("multi")
        first = run(request, output)
        rows = self.load(output / "frame_manifest.json")["frames"]
        chosen = [r["frame_uid"] for r in rows[1:4]]
        config = self.load(request)
        config["analysis"] = {"frame_uids": chosen}
        request.write_text(json.dumps(config), encoding="utf-8")
        second = resume(output)
        self.assertEqual(second["n_selected"], 3)
        self.assertEqual(second["n_frames"], first["n_frames"])
        self.assertEqual(self.load(output / "handoff.json")["selected_frame_uids"], chosen)

    def test_hdf5_timestamps_cannot_manufacture_nonexistent_frames(self):
        import h5py
        import numpy as np
        request, output = self.example("multi")
        with h5py.File(request.parent / "detector.h5", "w") as handle:
            handle.create_dataset("pixels", shape=(1, 2, 2), dtype="u2")
            handle["starts"] = [0., 1., 2.]
            handle["exposure"] = 1.
        config = self.load(request)
        config["frames"] = {"type": "hdf5", "path": "detector.h5", "data_dataset": "pixels", "start_dataset": "starts", "exposure_dataset": "exposure"}
        request.write_text(json.dumps(config), encoding="utf-8")
        status = run(request, output)
        self.assertEqual(status["status"], "needs_input")
        self.assertEqual(status["n_valid_analysis"], 0)
        self.assertFalse(self.load(output / "handoff.json")["ready_for_integration"])
        self.assertIn("frame_count_conflict", str(status["issues"]))

    def test_input_changed_during_sampling_blocks_ready_delivery(self):
        from insitulink.numerics import sample_channel
        request, output = self.example("multi")
        source = request.parent / "mechanics.csv"
        mutated = False

        def change_once(*args, **kwargs):
            nonlocal mutated
            result = sample_channel(*args, **kwargs)
            if not mutated:
                source.write_bytes(source.read_bytes() + b"\n")
                mutated = True
            return result

        with patch("insitulink.workflow.sample_channel", side_effect=change_once):
            status = run(request, output)
        self.assertEqual(status["status"], "needs_input")
        self.assertEqual(status["n_valid_analysis"], 0)
        self.assertFalse(self.load(output / "handoff.json")["ready_for_integration"])
        self.assertIn("input_changed", str(status["issues"]))

    def test_shared_clock_does_not_assume_epoch_and_relative_times_match(self):
        request, output = self.example("multi")
        config = self.load(request)
        config["sources"][0]["time"]["start"] = "2026-10-25T00:00:00Z"
        config["sources"][0]["clock"] = {"method": "shared", "confirmed": True, "evidence": "Incorrect shared origin declaration"}
        request.write_text(json.dumps(config), encoding="utf-8")
        status = run(request, output)
        self.assertEqual(status["status"], "needs_input")
        self.assertIn("incompatible_time_basis", str(status["decisions"]["mechanics"]))

    def test_spectral_preanalysis_connects_to_clock_and_invalidates_on_spectrum_change(self):
        import numpy as np
        request, output = self.example("tensile")
        path = request.parent / "frames.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        x = np.linspace(1.95, 2.05, 401)
        rng = np.random.default_rng(124)
        spectra = []
        for row in rows:
            strain = float(row.pop("eps_hkl"))
            center = 2 * (1 + strain)
            z = (x - center) / .002
            y = 8 + 100 * (.35 / (1 + 4*z*z) + .65 * np.exp(-4*np.log(2)*z*z)) + rng.normal(0, .002, len(x))
            filename = f"spectrum-{row['frame_index']}.csv"
            np.savetxt(request.parent / filename, np.c_[x, y], delimiter=",")
            spectra.append({"frame_index": int(row["frame_index"]), "path": filename})
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, [f for f in fields if f != "eps_hkl"])
            writer.writeheader()
            writer.writerows(rows)
        config = self.load(request)
        config["features"] = {"type": "spectra", "axis": "d_A", "files": spectra,
                              "windows": [{"id": "peak_1", "lower": 1.97, "upper": 2.03}], "baseline_frames": list(range(8))}
        request.write_text(json.dumps(config), encoding="utf-8")
        first = run(request, output)
        self.assertEqual(first["status"], "completed", first["issues"])
        self.assertEqual(first["stages"]["features"]["status"], "computed")
        self.assertGreater((output / "peaks.csv").stat().st_size, 100)
        second = resume(output)
        self.assertEqual(second["stages"]["features"]["status"], "reused")
        # Change one spectrum while preserving the surrounding request and tables.
        source = request.parent / spectra[-1]["path"]
        source.write_bytes(source.read_bytes() + b"\n")
        third = resume(output)
        self.assertEqual(third["stages"]["features"]["status"], "computed")
        self.assertEqual(third["stages"]["frames"]["status"], "reused")

    def test_raw_hdf5_preanalysis_reaches_delivery_without_inventing_an_event(self):
        import h5py
        import numpy as np
        from pyFAI.detectors import Detector
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator
        base = self.root / "images"
        base.mkdir()
        detector = Detector(pixel1=1e-4, pixel2=1e-4, max_shape=(192, 192))
        ai = AzimuthalIntegrator(dist=.08, poni1=.0096, poni2=.0096, detector=detector, wavelength=1e-10)
        ai.save(str(base / "geometry.poni"))
        q = ai.array_from_unit(shape=(192, 192), unit="q_A^-1")
        rng = np.random.default_rng(912)
        data = np.stack([12 + 300*np.exp(-4*np.log(2)*((q-c)/.014)**2) + rng.normal(0, .4, q.shape)
                         for c in (.42, .422, .424)]).astype("f4")
        with h5py.File(base / "raw.h5", "w") as handle:
            handle["images"] = data
            handle["starts"] = [0., 3., 6.]
            handle["exposures"] = 1.
        config = {"schema_version": 1, "experiment": "tensile",
                  "frames": {"type": "hdf5", "path": "raw.h5", "data_dataset": "images", "start_dataset": "starts", "exposure_dataset": "exposures"},
                  "sources": [{"id": "mechanics", "data": [{"time": i, "force": 0.} for i in range(15)],
                               "channels": [{"name": "force", "unit": "N"}], "clock": {"method": "event", "channel": "force"}}],
                  "features": {"type": "images", "geometry": "geometry.poni", "mask": None, "correct_solid_angle": False,
                               "polarization_factor": None, "npt": 400, "baseline_frames": [0],
                               "windows": [{"id": "peak_1", "lower": .37, "upper": .47}]}}
        request = base / "experiment.json"
        request.write_text(json.dumps(config), encoding="utf-8")
        output = base / "results"
        status = run(request, output)
        self.assertEqual(status["status"], "needs_input", status["issues"])
        self.assertEqual(status["stages"]["features"]["status"], "computed")
        self.assertEqual(status["features"]["n_signals"], 1)
        self.assertIn("source_", status["decisions"]["mechanics"]["reason"])
        self.assertEqual([r["frame_index"] for r in self.load(output / "frame_manifest.json")["frames"]], [0, 1, 2])
        self.assertFalse(self.load(output / "handoff.json")["ready_for_integration"])
        # Confirmed clocks bypass preanalysis; actual raw frames can then be handed off.
        config["sources"][0]["clock"] = {"method": "shared", "confirmed": True, "evidence": "Synthetic shared trigger and common origin"}
        request.write_text(json.dumps(config), encoding="utf-8")
        confirmed = resume(output)
        self.assertEqual(confirmed["status"], "completed", confirmed["issues"])
        self.assertTrue(self.load(output / "handoff.json")["ready_for_integration"])


if __name__ == "__main__":
    unittest.main()
