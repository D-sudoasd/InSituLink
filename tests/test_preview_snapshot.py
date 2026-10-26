"""Spectra must be parsed from the exact bytes bound by the workflow."""

import bz2
import gzip
from pathlib import Path
import tempfile
import unittest

import numpy as np

from insitulink.errors import InputError
from insitulink.preview import extract_features


class SpectrumSnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.path = self.base / "spectrum.csv"
        self.frames = [dict(frame_index=0, t_mid_s=0.)]
        self.config = dict(type="spectra", axis="d_A", baseline_frames=[0],
                           files=[dict(frame_index=0, path=self.path.name)],
                           windows=[dict(id="peak", lower=1.9, upper=2.1)])

    def write_spectrum(self, center):
        x = np.linspace(1.8, 2.2, 801)
        y = 8 + 100 * np.exp(-4 * np.log(2) * ((x - center) / .018) ** 2)
        np.savetxt(self.path, np.c_[x, y], delimiter=",")

    def extract(self, **kwargs):
        return extract_features(self.config, self.frames, self.base, self.base / "unused", **kwargs)

    def test_explicit_snapshot_keeps_bound_content_after_source_changes(self):
        self.write_spectrum(2.)
        snapshot = {str(self.path.resolve()): self.path.read_bytes()}
        self.write_spectrum(2.02)
        captured = self.extract(source_bytes=snapshot)
        current = self.extract()
        self.assertAlmostEqual(captured["rows"][0]["center"], 2., delta=1e-6)
        self.assertAlmostEqual(current["rows"][0]["center"], 2.02, delta=1e-6)
        self.assertEqual(captured["metadata"]["dependencies"], [
            dict(path=str(self.path.resolve()), role="spectrum")])
        self.assertFalse((self.base / "unused").exists())
        self.assertEqual(list(self.base.iterdir()), [self.path])

    def test_snapshot_and_legacy_path_reads_have_identical_results(self):
        self.write_spectrum(2.)
        expected = self.extract()
        payload = self.path.read_bytes()
        for suffix, encode in ((".csv", lambda b: b), (".csv.gz", gzip.compress), (".csv.bz2", bz2.compress)):
            with self.subTest(suffix=suffix):
                self.path = self.base / ("spectrum" + suffix)
                self.path.write_bytes(encode(payload))
                self.config["files"][0]["path"] = self.path.name
                self.config["delimiter"] = ","
                direct = self.extract()
                self.assertEqual(direct["rows"], expected["rows"])
                self.assertEqual(self.extract(source_bytes={str(self.path.resolve()): self.path.read_bytes()}), direct)
                self.assertEqual(self.extract(source_bytes=None), direct)

    def test_incomplete_snapshot_does_not_reopen_unbound_source(self):
        self.write_spectrum(2.)
        with self.assertRaises(InputError) as raised:
            self.extract(source_bytes={})
        self.assertEqual(raised.exception.code, "preview_missing_snapshot")

    def test_peak_rows_keep_canonical_identity_without_copying_frame_provenance(self):
        self.write_spectrum(2.)
        self.frames[0].update(frame_uid="stable-frame", sequence_index=0, source_file="raw.h5",
                              dataset="/images", frame_number=1, original_id="detector-1",
                              t_start_s=0., t_end_s=1., exposure_s=1., time_origin_s="1767225600",
                              arbitrary_input_column="existing custom value", user_notes={"sample": "A"})
        expected = self.extract()["rows"]
        self.frames[0].update(source_file_declared="raw.h5", time_provenance={"source": "relative table"},
                              physical_source={"mapping": ["large frame-specific address record"]},
                              dataset_declared="images", declared_source={"source_file": "raw.h5"},
                              identity_issues=[{"code": "address_unverified"}])
        relative = self.extract()["rows"]
        self.frames[0]["source_file_declared"] = str((self.base / "raw.h5").resolve())
        self.frames[0]["time_provenance"] = {"source": "absolute table"}
        absolute = self.extract()["rows"]
        self.assertEqual(relative, expected)
        self.assertEqual(absolute, expected)
        self.assertEqual(relative[0]["frame_uid"], "stable-frame")
        self.assertEqual(relative[0]["source_file"], str((self.base / "raw.h5").resolve()))
        self.assertEqual(relative[0]["time_origin_s"], "1767225600")
        self.assertEqual(relative[0]["exposure_s"], 1.)
        self.assertEqual(relative[0]["arbitrary_input_column"], "existing custom value")
        self.assertEqual(relative[0]["user_notes"], {"sample": "A"})


if __name__ == "__main__":
    unittest.main()
