"""Deterministic engineering checks for optional diffraction preanalysis."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from insitulink.errors import InputError
from insitulink.preview import extract_features, fit_peak


def peak(x, center, width=.018, height=100, eta=.35):
    z = (x-center)/width
    return height*(eta/(1+4*z*z)+(1-eta)*np.exp(-4*np.log(2)*z*z))


class SpectraPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.x = np.linspace(1.8, 2.2, 801)
        self.rng = np.random.default_rng(421)

    def series(self, centers, *, axis="d_A", height=100):
        frames, files = [], []
        for i, center in enumerate(centers):
            y = 8+3*(self.x-2)+peak(self.x, center, height=height)+self.rng.normal(0, .3, len(self.x))
            path = self.base / f"spectrum{i}.csv"
            np.savetxt(path, np.c_[self.x, y], delimiter=",")
            frames.append({"source_file": "raw.h5", "dataset": "/images", "frame_index": i,
                           "frame_number": i+1, "original_id": f"raw-{i}", "t_mid_s": 20+i*2.5})
            files.append({"frame_index": i, "path": path.name})
        return {"type": "spectra", "axis": axis, "files": files, "baseline_frames": [0],
                "windows": [{"id": "peak_1", "lower": 1.9, "upper": 2.1}]}, frames

    def extract(self, config, frames):
        return extract_features(config, frames, self.base, self.base / "out")

    def test_known_shift_uncertainty_chronology_and_identity(self):
        config, frames = self.series([2., 2.001, 2.002])
        result = self.extract(config, frames)
        self.assertEqual(result["signals"][0]["t_s"], [20, 22.5, 25])
        self.assertEqual(result["signals"][0]["valid"], [True]*3)
        for i, row in enumerate(result["rows"]):
            self.assertAlmostEqual(row["center"], 2+i*.001, delta=6e-5)
            self.assertGreater(row["center_sigma"], 0)
            self.assertLess(row["center_sigma"], 6e-5)
            self.assertAlmostEqual(row["strain"], i*.0005, delta=3e-5)
            self.assertEqual(row["original_id"], f"raw-{i}")
            self.assertEqual(row["frame_number"], i+1)
        self.assertEqual(result["metadata"]["phase_assignment"], "unassigned")

    def test_acquisition_order_is_not_silently_sorted(self):
        config, frames = self.series([2., 2.001])
        with self.assertRaises(InputError) as raised:
            self.extract(config, frames[::-1])
        self.assertEqual(raised.exception.code, "preview_nonmonotonic_time")

    def test_spectra_need_no_raw_source(self):
        config, frames = self.series([2., 2.001])
        for frame in frames:
            frame["source_file"] = None
            frame["dataset"] = None
        result = self.extract(config, frames)
        self.assertEqual(result["signals"][0]["valid"], [True, True])
        self.assertIsNone(result["rows"][0]["source_file"])

    def test_weak_disappearing_and_multiple_peaks(self):
        w = {"lower": 1.9, "upper": 2.1}
        vanished = 10+self.rng.normal(0, 1, len(self.x))
        weak = vanished+peak(self.x, 2, height=1)
        double = 10+peak(self.x, 1.97)+peak(self.x, 2.03)
        self.assertEqual(fit_peak(self.x, vanished, w)["status"], "invalid")
        self.assertEqual(fit_peak(self.x, weak, w)["status"], "invalid")
        self.assertEqual(fit_peak(self.x, double, w)["reason"], "multiple_resolved_peaks")

    def test_invalid_observations_remain_gaps_in_signals(self):
        config, frames = self.series([2., 2.001, 2.07])
        np.savetxt(self.base / "spectrum1.csv", np.c_[self.x, self.rng.normal(10, .3, len(self.x))], delimiter=",")
        result = self.extract(config, frames)
        self.assertEqual(result["signals"][0]["valid"], [True, False, False])
        self.assertEqual(result["signals"][0]["values"][1:], [None, None])
        self.assertEqual(result["rows"][2]["reason"], "unstable_tracking")

    def test_overlapping_windows_invalid(self):
        config, frames = self.series([2., 2.001])
        config["windows"].append({"id": "peak_2", "lower": 2., "upper": 2.15})
        result = self.extract(config, frames)
        self.assertTrue(all(r["reason"] == "overlapping_windows" for r in result["rows"]))
        self.assertTrue(all(r["strain"] is None for r in result["rows"]))

    def test_default_baseline_and_ambiguous_reference(self):
        config, frames = self.series([2., 2., 2., 2., 2., 2., 2., 2., 2.002])
        del config["baseline_frames"]
        result = self.extract(config, frames)
        self.assertEqual(len(result["metadata"]["baseline_frames"]), 8)
        self.assertTrue(result["signals"][0]["valid"][-1])
        self.assertAlmostEqual(result["signals"][0]["values"][-1], .001, delta=3e-5)
        config, frames = self.series([2., 2.01])
        config["baseline_frames"] = [0, 1]
        result = self.extract(config, frames)
        self.assertEqual(result["signals"][0]["valid"], [False, False])
        self.assertIsNone(result["metadata"]["d0_A"]["peak_1"])

    def test_q_conversion_and_two_theta_conversion(self):
        config, frames = self.series([2., 2.002], axis="q_A^-1")
        result = self.extract(config, frames)
        self.assertAlmostEqual(result["rows"][0]["d_A"], np.pi, delta=1e-4)
        self.assertAlmostEqual(result["rows"][1]["strain"], 2/2.002-1, delta=3e-5)
        self.x = np.linspace(19, 21, 1001)
        config, frames = self.series([20., 20.01], axis="two_theta_deg")
        config["windows"] = [{"id": "peak_1", "lower": 19.5, "upper": 20.5}]
        config["wavelength_A"] = 1.
        result = self.extract(config, frames)
        expected = np.sin(np.deg2rad(10))/np.sin(np.deg2rad(10.005))-1
        self.assertAlmostEqual(result["rows"][1]["strain"], expected, delta=1e-5)
        self.assertGreater(result["rows"][0]["d_sigma_A"], 0)

    def test_automatic_common_peaks_not_forced_three(self):
        config, frames = self.series([2., 2.001, 2.002])
        del config["windows"]
        result = self.extract(config, frames)
        self.assertEqual(len(result["signals"]), 1)
        self.assertTrue(all(result["signals"][0]["valid"]))

    def test_no_peak_and_baseline_identity_errors(self):
        config, frames = self.series([2.], height=0)
        del config["windows"]
        result = self.extract(config, frames)
        self.assertEqual(result["signals"], [])
        self.assertEqual(result["issues"][0]["code"], "preview_no_common_peaks")
        config, frames = self.series([2., 2.])
        frames[1]["frame_index"] = 0
        with self.assertRaises(InputError) as raised:
            self.extract(config, frames)
        self.assertEqual(raised.exception.code, "preview_ambiguous_baseline")

    def test_specific_missing_configuration_and_disambiguation(self):
        with self.assertRaises(InputError) as raised:
            self.extract({}, [])
        self.assertEqual(raised.exception.code, "preview_missing_config")
        config, frames = self.series([2., 2.001])
        config["baseline_frames"] = [{"source_file": "raw.h5", "frame_index": 0}]
        frames[1]["frame_index"] = 0
        frames[1]["source_file"] = "other.h5"
        config["files"][0]["source_file"] = "raw.h5"
        config["files"][1].update(frame_index=0, source_file="other.h5")
        self.assertTrue(all(self.extract(config, frames)["signals"][0]["valid"]))
        del config["files"][0]["source_file"]
        with self.assertRaises(InputError) as raised:
            self.extract(config, frames)
        self.assertEqual(raised.exception.code, "preview_spectrum_identity")
        # One unqualified entry cannot silently serve two physical frames.
        config["files"] = [config["files"][0]]
        with self.assertRaises(InputError) as raised:
            self.extract(config, frames)
        self.assertEqual(raised.exception.code, "preview_spectrum_identity")


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("pyFAI", "fabio", "h5py")), "optional image dependencies unavailable")
class ImagePreviewTests(unittest.TestCase):
    def test_raw_frame_selection_mask_and_calibrated_shift(self):
        import fabio
        import h5py
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator
        from pyFAI.detectors import Detector
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            detector = Detector(pixel1=1e-4, pixel2=1e-4, max_shape=(192, 192))
            ai = AzimuthalIntegrator(dist=.08, poni1=.0096, poni2=.0096, detector=detector, wavelength=1e-10)
            ai.save(str(base / "geometry.poni"))
            q = ai.array_from_unit(shape=(192, 192), unit="q_A^-1")
            rng = np.random.default_rng(918)
            data = np.stack([12+peak(q, center, width=.014, height=300)+rng.normal(0, .4, q.shape)
                             for center in (.42, .422, .424)]).astype(np.float32)
            # A masked hot pixel and a nonfinite pixel must not contaminate the fit.
            data[:, 90, 135] = 1e9
            data[:, 80, 130] = np.nan
            with h5py.File(base / "raw.h5", "w") as handle:
                handle.create_dataset("images", data=data)
            mask = np.zeros((192, 192), dtype=np.uint8)
            mask[90, 135] = 1
            mask[(q > .10) & (q < .15)] = 1
            fabio.edfimage.EdfImage(data=mask).write(str(base / "mask.edf"))
            frames = [{"source_file": "raw.h5", "dataset": "images", "frame_index": i,
                       "frame_number": i+1, "original_id": f"det-{i}", "t_mid_s": float(i*3+1)} for i in (0, 2)]
            config = {"type": "images", "geometry": "geometry.poni", "mask": "mask.edf",
                      "correct_solid_angle": False, "polarization_factor": None, "npt": 400,
                      "windows": [{"id": "peak_1", "lower": .37, "upper": .47}], "baseline_frames": [0]}
            result = extract_features(config, frames, base, base / "out")
            self.assertEqual(result["signals"][0]["valid"], [True, True], result["rows"])
            self.assertAlmostEqual(result["rows"][0]["center"], .42, delta=4e-4)
            self.assertAlmostEqual(result["rows"][1]["center"], .424, delta=4e-4)
            self.assertEqual(result["rows"][1]["frame_index"], 2)
            self.assertEqual(result["signals"][0]["t_s"], [1., 7.])
            self.assertAlmostEqual(result["rows"][1]["strain"], .42/.424-1, delta=.001)
            self.assertEqual(result["metadata"]["integration_axis"], "q_A^-1")
            self.assertEqual({d["role"] for d in result["metadata"]["dependencies"]}, {"geometry", "mask", "detector_image"})
            for path in result["metadata"]["integrated_files"]:
                with np.load(path) as integrated:
                    self.assertEqual(integrated["intensity"].ndim, 1)
                    self.assertTrue(np.any(np.isnan(integrated["intensity"])))
            del config["mask"]
            with self.assertRaises(InputError) as raised:
                extract_features(config, frames, base, base / "out")
            self.assertEqual(raised.exception.code, "preview_missing_correction")


if __name__ == "__main__":
    unittest.main()
