"""Selected HDF storage provenance and request-relative preview regressions."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from insitulink.errors import InputError
from insitulink.hdf5_sources import discover_dependencies
from insitulink.preview import extract_features
from insitulink.workflow import run, resume
from tests.test_insitulink_preview import peak


HDF_AVAILABLE = importlib.util.find_spec("h5py") is not None
IMAGES_AVAILABLE = all(importlib.util.find_spec(name) for name in ("h5py", "pyFAI", "fabio"))


@unittest.skipUnless(HDF_AVAILABLE, "optional h5py unavailable")
class StorageDiscoveryTests(unittest.TestCase):
    def setUp(self):
        import h5py
        self.h5py = h5py
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def discover(self, path, dataset="/images"):
        return discover_dependencies([{"source_file": str(path), "dataset": dataset, "frame_index": 0}])

    def test_nested_external_and_soft_links_are_metadata_only(self):
        h5py = self.h5py
        nested = self.base / "nested"
        nested.mkdir()
        data = nested / "data.h5"
        middle = nested / "middle.h5"
        master = self.base / "master.h5"
        with h5py.File(data, "w") as handle:
            group = handle.create_group("detector")
            group.create_dataset("pixels", shape=(2, 4, 4), dtype="u2")
            group["alias"] = h5py.SoftLink("pixels")
            handle["unrelated_missing"] = h5py.ExternalLink("absent.h5", "/images")
        with h5py.File(middle, "w") as handle:
            handle["target"] = h5py.ExternalLink("data.h5", "/detector")
        with h5py.File(master, "w") as handle:
            handle["images"] = h5py.ExternalLink("nested/middle.h5", "/target/alias")
        with patch.object(h5py.Dataset, "__getitem__", side_effect=AssertionError("pixel read")):
            result = self.discover(master)
        self.assertTrue(result["complete"], result)
        self.assertEqual(result["paths"], sorted(str(p.resolve()) for p in (master, middle, data)))
        self.assertEqual(result["datasets"][0]["shape"], [2, 4, 4])

    def virtual(self, filename="data.h5", dataset="/pixels"):
        h5py = self.h5py
        layout = h5py.VirtualLayout(shape=(2, 4, 4), dtype="u2")
        layout[:] = h5py.VirtualSource(filename, dataset, shape=(2, 4, 4))
        path = self.base / "virtual.h5"
        with h5py.File(path, "w", libver="latest") as handle:
            handle.create_virtual_dataset("images", layout, fillvalue=0)
        return path

    def test_virtual_storage_sources_and_nested_links(self):
        h5py = self.h5py
        data = self.base / "data.h5"
        middle = self.base / "middle.h5"
        with h5py.File(data, "w") as handle:
            handle.create_dataset("pixels", shape=(2, 4, 4), dtype="u2")
        with h5py.File(middle, "w") as handle:
            handle["pixels"] = h5py.ExternalLink("data.h5", "/pixels")
        virtual = self.virtual("middle.h5")
        with patch.object(h5py.Dataset, "__getitem__", side_effect=AssertionError("pixel read")):
            result = self.discover(virtual)
        self.assertTrue(result["complete"], result)
        self.assertEqual(result["paths"], sorted(str(p) for p in (virtual, middle, data)))

    def test_missing_virtual_source_never_counts_fill_value_as_verified(self):
        virtual = self.virtual()
        result = self.discover(virtual)
        self.assertFalse(result["complete"])
        self.assertIn(str(self.base / "data.h5"), result["paths"])
        self.assertEqual(result["issues"][0]["code"], "hdf5_missing_storage")

    def test_unavailable_virtual_enumeration_is_explicitly_incomplete(self):
        virtual = self.virtual()
        with patch.object(self.h5py.Dataset, "virtual_sources", side_effect=RuntimeError("metadata unavailable")):
            result = self.discover(virtual)
        self.assertFalse(result["complete"])
        self.assertEqual(result["issues"][0]["code"], "hdf5_dependency_unavailable")
        self.assertIn(str(virtual), result["paths"])

    def test_dynamic_virtual_source_names_are_not_guessed(self):
        h5py = self.h5py
        layout = h5py.VirtualLayout(shape=(2, 4, 4), maxshape=(None, 4, 4), dtype="u2")
        source = h5py.VirtualSource("data_%b.h5", "/pixels", shape=(1, 4, 4))
        layout[0:h5py.h5s.UNLIMITED, :, :] = source
        path = self.base / "dynamic.h5"
        with h5py.File(path, "w", libver="latest") as handle:
            handle.create_virtual_dataset("images", layout)
        result = self.discover(path)
        self.assertFalse(result["complete"], result)
        self.assertTrue(any(i["code"] == "hdf5_dynamic_source" for i in result["issues"]))

    def test_external_raw_storage_obeys_origin_prefix(self):
        path = self.base / "raw.h5"
        raw = self.base / "pixels.raw"
        raw.write_bytes(bytes(64))
        with self.h5py.File(path, "w") as handle:
            handle.create_dataset("images", shape=(2, 4, 4), dtype="u2", external=[("pixels.raw", 0, 64)])
        with patch.dict(os.environ, {"HDF5_EXTFILE_PREFIX": "${ORIGIN}"}):
            result = self.discover(path)
        self.assertTrue(result["complete"], result)
        self.assertEqual(result["paths"], sorted([str(path), str(raw)]))
        raw.unlink()
        with patch.dict(os.environ, {"HDF5_EXTFILE_PREFIX": "${ORIGIN}"}):
            self.assertFalse(self.discover(path)["complete"])

    def test_external_raw_default_is_process_cwd(self):
        path = self.base / "raw.h5"
        raw_name = "only_in_owner_directory.raw"
        (self.base / raw_name).write_bytes(bytes(64))
        with self.h5py.File(path, "w") as handle:
            handle.create_dataset("images", shape=(2, 4, 4), dtype="u2", external=[(raw_name, 0, 64)])
        with patch.dict(os.environ, {"HDF5_EXTFILE_PREFIX": ""}):
            result = self.discover(path)
        self.assertFalse(result["complete"], result)
        self.assertIn(str(Path.cwd() / raw_name), result["paths"])

    def test_link_cycles_are_bounded_and_keep_known_files(self):
        h5py = self.h5py
        one, two = self.base / "one.h5", self.base / "two.h5"
        with h5py.File(one, "w") as handle:
            handle["soft"] = h5py.SoftLink("/soft")
            handle["images"] = h5py.ExternalLink("two.h5", "/images")
        with h5py.File(two, "w") as handle:
            handle["images"] = h5py.ExternalLink("one.h5", "/images")
        for dataset, paths in (("/soft", [one]), ("/images", [one, two])):
            with self.subTest(dataset=dataset):
                result = self.discover(one, dataset)
                self.assertFalse(result["complete"])
                self.assertEqual(result["issues"][0]["code"], "hdf5_link_cycle")
                self.assertEqual(result["paths"], sorted(str(p) for p in paths))

    def test_non_hdf_sources_need_no_image_read(self):
        source = self.base / "image.tif"
        source.write_bytes(b"not opened as an image")
        with patch.object(self.h5py, "File", side_effect=AssertionError("HDF open")):
            result = discover_dependencies([{"source_file": str(source), "dataset": None}, {"source_file": None}])
        self.assertTrue(result["complete"])
        self.assertEqual(result["paths"], [str(source)])


@unittest.skipUnless(IMAGES_AVAILABLE, "optional image dependencies unavailable")
class PreviewPipelineRegressions(unittest.TestCase):
    def setUp(self):
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator
        from pyFAI.detectors import Detector
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        detector = Detector(pixel1=1e-4, pixel2=1e-4, max_shape=(192, 192))
        self.ai = AzimuthalIntegrator(dist=.08, poni1=.0096, poni2=.0096, detector=detector, wavelength=1e-10)
        self.ai.save(str(self.base / "geometry.poni"))
        self.q = self.ai.array_from_unit(shape=(192, 192), unit="q_A^-1")

    def images(self, onset):
        # Identical images in the initial static period establish a full baseline.
        return np.stack([12 + peak(self.q, .42 if i < onset else .422, width=.014, height=300)
                         for i in range(24)]).astype(np.float32)

    def request(self, frames, features):
        return {"schema_version": 1, "experiment": "tensile", "frames": {"type": "table", "data": frames, "time": {"unit": "s"}},
                "sources": [{"id": "load", "data": [{"time": i, "force": 0 if i < 10 else 1} for i in range(30)],
                             "time_column": "time", "time": {"unit": "s"},
                             "channels": [{"name": "force", "column": "force", "unit": "N"}],
                             "clock": {"method": "event", "channel": "force", "baseline_samples": 8, "min_run": 3}}],
                "features": features}

    def image_config(self):
        return {"type": "images", "geometry": "geometry.poni", "mask": None, "correct_solid_angle": False,
                "polarization_factor": None, "npt": 400, "windows": [{"id": "peak_1", "lower": .37, "upper": .47}]}

    def write_request(self, request, name="experiment.json"):
        path = self.base / name
        path.write_text(json.dumps(request), encoding="utf-8")
        return path

    def test_external_pixels_changed_resume_recomputes_and_matches_fresh(self):
        import h5py
        master, data = self.base / "master.h5", self.base / "data.h5"
        with h5py.File(data, "w") as handle:
            handle.create_dataset("images", data=self.images(10))
        with h5py.File(master, "w") as handle:
            handle["images"] = h5py.ExternalLink("data.h5", "/images")
        frames = [{"frame_index": i, "t_start_s": i, "exposure_s": 1, "source_file": "master.h5", "dataset": "/images"} for i in range(24)]
        request = self.write_request(self.request(frames, self.image_config()))
        output = self.base / "results"
        first = run(request, output)
        self.assertEqual(first["status"], "completed", first["issues"])
        self.assertAlmostEqual(first["decisions"]["load"]["dt_s"], -.5)
        self.assertEqual(resume(output)["stages"]["features"]["status"], "reused")
        master_bytes = master.read_bytes()
        with h5py.File(data, "r+") as handle:
            handle["images"][:] = self.images(14)
        self.assertEqual(master.read_bytes(), master_bytes)
        changed = resume(output)
        fresh = run(request, self.base / "fresh")
        self.assertEqual(changed["status"], "completed", changed["issues"])
        self.assertEqual(changed["stages"]["features"]["status"], "computed")
        self.assertAlmostEqual(changed["decisions"]["load"]["dt_s"], -4.5)
        self.assertEqual(changed["decisions"], fresh["decisions"])
        self.assertEqual((output / "peaks.csv").read_bytes(), (self.base / "fresh" / "peaks.csv").read_bytes())
        paths = {item["path"] for item in changed["features"]["metadata"]["dependencies"]}
        self.assertTrue({str(master), str(data)} <= paths)

    def test_relative_spectrum_and_baseline_identities_match_absolute_pipeline(self):
        import fabio
        frames, files = [], []
        x = np.linspace(.37, .47, 601)
        for i, image in enumerate(self.images(10)):
            raw = self.base / f"frame{i}.tif"
            fabio.tifimage.TifImage(data=image).write(str(raw))
            spectrum = self.base / f"spectrum{i}.csv"
            np.savetxt(spectrum, np.c_[x, 12 + peak(x, .42 if i < 10 else .422, width=.014)], delimiter=",")
            frames.append({"frame_index": 0, "t_start_s": i, "exposure_s": 1, "source_file": str(raw)})
            files.append({"frame_index": 0, "source_file": str(raw), "path": spectrum.name})
        features = {"type": "spectra", "axis": "q_A^-1", "files": files,
                    "baseline_frames": [{"frame_index": 0, "source_file": frames[i]["source_file"]} for i in range(8)],
                    "windows": [{"id": "peak_1", "lower": .37, "upper": .47}]}
        original = self.request(frames, features)
        absolute = run(self.write_request(original), self.base / "absolute")
        self.assertEqual(absolute["status"], "completed", absolute["issues"])
        for label, keys in (("spectrum", ["files"]), ("baseline", ["baseline_frames"]), ("both", ["files", "baseline_frames"]), ("frames", [])):
            relative = copy.deepcopy(original)
            for key in keys:
                for item in relative["features"][key]:
                    item["source_file"] = Path(item["source_file"]).name
            if label == "frames":
                for item in relative["frames"]["data"]:
                    item["source_file"] = Path(item["source_file"]).name
            with self.subTest(label=label):
                result = run(self.write_request(relative, label + ".json"), self.base / label)
                self.assertEqual(result["status"], "completed", result["issues"])
                self.assertEqual(result["decisions"], absolute["decisions"])
                self.assertEqual((self.base / label / "peaks.csv").read_bytes(), (self.base / "absolute" / "peaks.csv").read_bytes())
        # The public low-level API also accepts request-relative frame sources,
        # and normalization must not mutate any supplied identity dictionary.
        relative_frames = copy.deepcopy(frames)
        for frame in relative_frames:
            frame["source_file"] = Path(frame["source_file"]).name
            frame["t_mid_s"] = frame["t_start_s"] + .5
        saved_frames, saved_features = copy.deepcopy(relative_frames), copy.deepcopy(features)
        result = extract_features(features, relative_frames, self.base, self.base / "lowlevel")
        self.assertTrue(all(result["signals"][0]["valid"]))
        self.assertEqual(relative_frames, saved_frames)
        self.assertEqual(features, saved_features)
        mismatch = copy.deepcopy(features)
        mismatch["files"][0]["source_file"] = "wrong.tif"
        with self.assertRaises(InputError) as caught:
            extract_features(mismatch, relative_frames, self.base, self.base / "wrong")
        self.assertEqual(caught.exception.code, "preview_spectrum_identity")
        mismatch = copy.deepcopy(features)
        mismatch["baseline_frames"][0]["source_file"] = "wrong.tif"
        with self.assertRaises(InputError) as caught:
            extract_features(mismatch, relative_frames, self.base, self.base / "wrong")
        self.assertEqual(caught.exception.code, "preview_ambiguous_baseline")


if __name__ == "__main__":
    unittest.main()
