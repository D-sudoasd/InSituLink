"""Temporary-source fixtures for the generic frame and external-log readers."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from insitulink.errors import InputError
from insitulink.inputs import canonical_source_file, read_frames, read_source, referenced_paths

try:
    import h5py
except ImportError:
    h5py = None
try:
    import openpyxl
except ImportError:
    openpyxl = None


class InputsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)

    def test_csv_sequence_identity_and_numerical_features(self):
        (self.base / "frames.csv").write_text(
            "file,index,label,start,exp,eps_hkl\na.cbf,9,0009,20,100,0.002\nb.tiff,2,0002,10,200,0.003\n",
            encoding="utf-8")
        config = {"path": "frames.csv", "columns": {"source_file": "file", "frame_index": "index",
                  "id": "label", "start": "start", "exposure": "exp"}, "time": {"unit": "ms"}}
        rows, meta = read_frames(config, self.base)
        self.assertEqual([row["frame_index"] for row in rows], [9, 2])
        self.assertEqual(rows[0]["frame_number"], 10)
        self.assertEqual(rows[0]["original_id"], "0009")
        self.assertEqual(rows[0]["source_file"], str((self.base / "a.cbf").resolve()))
        self.assertAlmostEqual(rows[0]["t_start_s"], 0.02)
        self.assertAlmostEqual(rows[0]["t_mid_s"], 0.07)
        self.assertAlmostEqual(rows[0]["eps_hkl"], 0.002)
        self.assertEqual(meta["nonmonotonic_rows"], [1])
        self.assertEqual(referenced_paths(config, self.base), [self.base / "frames.csv", self.base / "a.cbf", self.base / "b.tiff"])

    def test_inline_defaults_missing_interior_and_duplicates(self):
        rows, meta = read_frames({"data": [{"t_start_s": 0, "exposure_s": 1},
                {"t_start_s": None}, {"t_start_s": 0, "exposure_s": 2}, {"t_start_s": 4}]}, self.base)
        self.assertEqual(len(rows), 4)
        self.assertEqual([row["frame_index"] for row in rows], [0, 1, 2, 3])
        self.assertEqual(meta["missing_time_indices"], [1])
        self.assertEqual(meta["duplicate_time_rows"], [2])
        self.assertIsNone(rows[1]["t_mid_s"])
        self.assertIsNone(rows[0]["source_file"])

    def test_source_requires_explicit_file_local_index_including_mapping(self):
        for index in (None, "", " "):
            config = {"data": [{"file": "one.tiff", "index": index, "t_start_s": 0}],
                      "columns": {"source_file": "file", "frame_index": "index"}}
            with self.subTest(index=index), self.assertRaises(InputError) as caught:
                read_frames(config, self.base)
            self.assertEqual(caught.exception.code, "missing_frame_index")
            self.assertEqual(caught.exception.details["row"], 0)
            self.assertEqual(caught.exception.details["frame_index_column"], "index")
        rows, _ = read_frames({"data": [{"source_file": f"{name}.tiff", "frame_index": 0,
                                        "t_start_s": i} for i, name in enumerate(("one", "two"))]}, self.base)
        self.assertEqual([r["frame_index"] for r in rows], [0, 0])

    def test_canonical_source_file_preserves_reader_missing_and_path_semantics(self):
        for value in (None, "", " "):
            self.assertIsNone(canonical_source_file(value, self.base))
        expected = str((self.base / "raw.tiff").resolve())
        self.assertEqual(canonical_source_file("nested/../raw.tiff", self.base), expected)
        self.assertEqual(canonical_source_file(expected, self.base), expected)

    def test_csv_headers_are_validated_before_reading_values(self):
        for header, duplicate, empty in (("time,time,T", ["time"], []),
                                         ("time,T,T", ["T"], []),
                                         ("frame_id,frame_id,t_start_s", ["frame_id"], []),
                                         ("time,,T", [], [1]), ("time, ,T", [], [1])):
            with self.subTest(header=header):
                (self.base / "bad.csv").write_text(header + "\n0,1,2\n", encoding="utf-8")
                with self.assertRaises(InputError) as caught:
                    read_frames({"path": "bad.csv"}, self.base)
                self.assertEqual(caught.exception.code, "invalid_table_header")
                self.assertEqual(caught.exception.details["duplicate_columns"], duplicate)
                self.assertEqual(caught.exception.details["empty_column_positions"], empty)
        (self.base / "distinct.csv").write_text("time, time\n0,1\n", encoding="utf-8")
        rows, _ = read_frames({"path": "distinct.csv", "columns": {"start": "time"}}, self.base)
        self.assertEqual(rows[0][" time"], 1)

    def test_missing_time_requires_confirmed_period(self):
        for extra in ({}, {"constant_period": {"exposure_s": 1, "interval_s": 2}},
                      {"constant_period": {"confirmed": False, "exposure_s": 1, "interval_s": 2}}):
            with self.subTest(extra=extra), self.assertRaises(InputError):
                read_frames({"data": [{"frame_id": "a"}], **extra}, self.base)

    def test_confirmed_period_uses_physical_index(self):
        rows, meta = read_frames({"data": [{"frame_index": 4, "frame_id": "f4"}, {"frame_index": 7}],
                 "constant_period": {"confirmed": True, "exposure_s": 1, "interval_s": 2, "start_s": 10}}, self.base)
        self.assertEqual([row["t_start_s"] for row in rows], [18, 24])
        self.assertEqual(rows[0]["t_end_s"], 19)
        self.assertTrue(meta["reconstructed"])
        self.assertEqual(rows[0]["clock_source"], "confirmed_constant_period")
        self.assertTrue(rows[0]["raw_time_missing"])

    def test_reconstruction_respects_epoch_origin(self):
        rows, _ = read_frames({"data": [{"t_start_s": 0}, {"t_start_s": None}],
            "time": {"start": "2026-01-01T00:00:00Z"},
            "constant_period": {"confirmed": True, "exposure_s": 1, "interval_s": 2}}, self.base)
        self.assertEqual(rows[1]["t_start_s"] - rows[0]["t_start_s"], 2)

    def test_invalid_period_or_frame_index(self):
        for period in ({"exposure_s": 2, "interval_s": 1}, {"exposure_s": -1, "interval_s": 1},
                       {"exposure_s": 0, "interval_s": 0}):
            with self.subTest(period=period), self.assertRaises(InputError):
                read_frames({"data": [{}], "constant_period": {"confirmed": True, **period}}, self.base)
        for index in (-1, 1.2, "bad"):
            with self.subTest(index=index), self.assertRaises(InputError):
                read_frames({"data": [{"frame_index": index, "t_start_s": 0}]}, self.base)

    def test_iso_timezone_and_origin(self):
        expected = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
        rows, meta = read_frames({"data": [{"t_start_s": "2026-01-01T08:00:00+08:00", "exposure_s": 1}]}, self.base)
        self.assertEqual(rows[0]["t_start_s"], expected)
        self.assertEqual(meta["time_basis"], "utc_epoch")
        rows, _ = read_frames({"data": [{"t_start_s": "2026-01-01T08:00:00"}], "time": {"timezone": "+08:00"}}, self.base)
        self.assertEqual(rows[0]["t_start_s"], expected)
        rows, _ = read_frames({"data": [{"t_start_s": 2}], "time": {"unit": "min", "scaling_factor": 0.5,
                                "start": "2026-01-01T00:00:00Z"}}, self.base)
        self.assertEqual(rows[0]["t_start_s"], expected + 60)

    def test_malformed_naive_iso_and_invalid_units(self):
        for value in ("2026-01-01T08:00:00", "not-a-time", "2026-99-99"):
            with self.subTest(value=value), self.assertRaises(InputError):
                read_frames({"data": [{"t_start_s": value}]}, self.base)
        with self.assertRaises(InputError):
            read_frames({"data": [{"t_start_s": 0}], "time": {"unit": "fortnight"}}, self.base)

    def test_json_and_original_input_preserved(self):
        data = [{"t_start_s": 0, "frame_id": "raw", "temperature": 300}]
        (self.base / "frames.json").write_text(json.dumps(data), encoding="utf-8")
        rows, _ = read_frames({"path": "frames.json"}, self.base)
        self.assertEqual(rows[0]["temperature"], 300)
        self.assertEqual(rows[0]["original_id"], "raw")
        self.assertEqual(json.loads((self.base / "frames.json").read_text()), data)

    @unittest.skipIf(openpyxl is None, "openpyxl unavailable")
    def test_xlsx_sheet_and_columns(self):
        book = openpyxl.Workbook()
        book.active.title = "frames"
        book.active.append(["when", "exposure_s", "frame_id"])
        book.active.append(["2026-01-01T00:00:00Z", 0.2, "x"])
        book.save(self.base / "frames.xlsx")
        book.close()
        rows, _ = read_frames({"path": "frames.xlsx", "sheet": "frames", "columns": {"start": "when"}}, self.base)
        self.assertEqual(rows[0]["original_id"], "x")
        self.assertAlmostEqual(rows[0]["exposure_s"], .2)

    def source(self, **updates):
        return {"id": "thermal", "data": [{"time": 0, "T": 20}, {"time": 1, "T": 21}],
                "channels": [{"name": "temperature", "column": "T", "unit": "degC"}], **updates}

    def test_generic_external_channels_and_missing_values(self):
        config = self.source(data=[{"time": 0, "T": 20, "phase": 0}, {"time": 1000, "T": "", "phase": 1},
                                   {"time": 2000, "T": float("inf"), "phase": 1}],
                    time={"unit": "ms"}, clock={"method": "manual", "offset_s": 2})
        config["channels"].append({"name": "phase", "unit": "1", "kind": "step", "required": False,
                                  "phase_tolerance": .1, "max_gap_s": 3})
        result = read_source(config, self.base)
        np.testing.assert_array_equal(result["t_s"], [0, 1, 2])
        self.assertTrue(np.isnan(result["channels"]["temperature"]["values"][1]))
        self.assertTrue(np.isinf(result["channels"]["temperature"]["values"][2]))
        self.assertEqual(result["channels"]["phase"]["kind"], "step")
        self.assertFalse(result["channels"]["phase"]["required"])
        self.assertEqual(result["clock"]["offset_s"], 2)

    def test_external_invalid_times(self):
        for times in ([0, 0], [1, 0], [0, None], [0, float("inf")], ["bad", 1]):
            with self.subTest(times=times), self.assertRaises(InputError):
                read_source(self.source(data=[{"time": value, "T": 1} for value in times]), self.base)

    def test_external_channel_validation(self):
        for channels in ([], [{"name": "T"}], [{"name": "T", "unit": "K", "kind": "other"}],
                         [{"name": "T", "unit": "K"}] * 2,
                         [{"name": "T", "unit": "K", "max_gap_s": -1}],
                         [{"name": "missing", "unit": "N"}],
                         [{"name": "T", "unit": "K", "required": "yes"}]):
            with self.subTest(channels=channels), self.assertRaises(InputError):
                read_source(self.source(channels=channels), self.base)
        with self.assertRaises(InputError):
            read_source(self.source(data=[{"time": 0, "T": "not numeric"}]), self.base)

    def test_external_csv_and_json(self):
        (self.base / "log.csv").write_text("time,T\n0,20\n1,21\n", encoding="utf-8")
        config = self.source()
        config.pop("data")
        config["path"] = "log.csv"
        np.testing.assert_array_equal(read_source(config, self.base)["channels"]["temperature"]["values"], [20, 21])
        (self.base / "log.json").write_text('[{"time":0,"T":20}]', encoding="utf-8")
        config["path"] = "log.json"
        self.assertEqual(len(read_source(config, self.base)["t_s"]), 1)

    @unittest.skipIf(openpyxl is None, "openpyxl unavailable")
    def test_external_xlsx(self):
        book = openpyxl.Workbook()
        book.active.append(["time", "T"])
        book.active.append([0, 300])
        book.active.append([1, 301])
        book.save(self.base / "log.xlsx")
        book.close()
        config = self.source()
        config.pop("data")
        result = read_source(dict(config, path="log.xlsx"), self.base)
        np.testing.assert_array_equal(result["channels"]["temperature"]["values"], [300, 301])

    def test_missing_and_malformed_table(self):
        for config in ({}, {"path": "missing.csv"}, {"data": []}, {"data": [1]}, {"data": {"a": 1}}):
            with self.subTest(config=config), self.assertRaises(InputError):
                read_frames(config, self.base)
        (self.base / "bad.json").write_text("{bad", encoding="utf-8")
        with self.assertRaises(InputError):
            read_frames({"path": "bad.json"}, self.base)

    def hdf5(self, starts, exposure=0.5, *, data_count=5, attrs=None):
        path = self.base / "master.h5"
        with h5py.File(path, "w") as file:
            file.create_dataset("/entry/data/data", shape=(data_count, 2, 2), dtype="u1", chunks=(1, 2, 2))
            if starts is not None:
                stamp = file.create_dataset("/entry/instrument/detector/start_time", data=starts)
                for key, value in (attrs or {}).items():
                    stamp.attrs[key] = value
            if exposure is not None:
                file.create_dataset("/entry/instrument/detector/count_time", data=exposure)
        return {"type": "hdf5", "path": "master.h5"}

    @unittest.skipIf(h5py is None, "h5py unavailable")
    def test_hdf5_huge_virtual_length_padding_and_no_pixel_reads(self):
        config = self.hdf5([0, np.nan, 2, np.nan, np.nan], data_count=100_000_000)
        original = h5py.Dataset.__getitem__
        def checked(dataset, key):
            if dataset.name == "/entry/data/data":
                raise AssertionError("Detector pixels must never be read")
            return original(dataset, key)
        with patch.object(h5py.Dataset, "__getitem__", checked):
            rows, meta = read_frames(config, self.base)
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["frame_index"] for row in rows], [0, 1, 2])
        self.assertEqual(meta["missing_time_indices"], [1])
        self.assertEqual(meta["dropped_padded_indices"], [3, 4])
        self.assertEqual(meta["data_frame_count"], 100_000_000)
        self.assertIn("data_dataset", meta["count_mismatches"])
        self.assertTrue(meta["frame_count_unresolved"])
        self.assertEqual(meta["observed_timestamp_frame_count"], 3)
        self.assertEqual(rows[2]["dataset"], "/entry/data/data")
        self.assertEqual(rows[2]["source_file"], str((self.base / "master.h5").resolve()))

    @unittest.skipIf(h5py is None, "h5py unavailable")
    def test_hdf5_nexus_time_attributes_and_exposure_vector(self):
        config = self.hdf5([0, 1000, 2000], [100, 200], attrs={"units": "ms", "scaling_factor": .5,
                            "start": "2026-01-01T00:00:00Z"})
        with h5py.File(self.base / "master.h5", "r+") as file:
            file["/entry/instrument/detector/count_time"].attrs["units"] = "ms"
        rows, meta = read_frames(config, self.base)
        self.assertEqual(rows[1]["t_start_s"] - rows[0]["t_start_s"], .5)
        self.assertAlmostEqual(rows[1]["exposure_s"], .2)
        self.assertIsNone(rows[2]["exposure_s"])
        self.assertIn("exposure_dataset", meta["count_mismatches"])

    @unittest.skipIf(h5py is None, "h5py unavailable")
    def test_hdf5_units_since_and_explicit_override(self):
        config = self.hdf5([0, 1], attrs={"units": "seconds since 2026-01-01T00:00:00Z"})
        rows, _ = read_frames(config, self.base)
        self.assertEqual(rows[1]["t_start_s"] - rows[0]["t_start_s"], 1)
        config["time"] = {"unit": "min"}
        rows, _ = read_frames(config, self.base)
        self.assertEqual(rows[1]["t_start_s"] - rows[0]["t_start_s"], 60)

    @unittest.skipIf(h5py is None, "h5py unavailable")
    def test_hdf5_scalar_acquisition_start_requires_confirmation(self):
        config = self.hdf5(np.bytes_("2026-01-01T00:00:00Z"), data_count=3)
        with self.assertRaises(InputError):
            read_frames(config, self.base)
        config["constant_period"] = {"confirmed": True, "exposure_s": .5, "interval_s": 2}
        rows, meta = read_frames(config, self.base)
        self.assertEqual([row["t_start_s"] - rows[0]["t_start_s"] for row in rows], [0, 2, 4])
        self.assertEqual(meta["reconstructed_rows"], [1, 2])
        self.assertEqual(meta["timestamp_count"], 1)

    @unittest.skipIf(h5py is None, "h5py unavailable")
    def test_hdf5_configurable_paths_and_numeric_string_times(self):
        config = self.hdf5([np.bytes_("0"), np.bytes_("1")], [0.1, 0.2])
        with h5py.File(self.base / "master.h5", "r+") as file:
            file.move("/entry/instrument/detector/start_time", "/time")
            file.move("/entry/instrument/detector/count_time", "/duration")
            file.move("/entry/data/data", "/images")
        config.update(start_dataset="/time", exposure_dataset="/duration", data_dataset="/images")
        rows, _ = read_frames(config, self.base)
        self.assertEqual(rows[1]["t_start_s"], 1)
        self.assertEqual(rows[1]["dataset"], "/images")
        self.assertEqual(rows[1]["exposure_s"], .2)

    @unittest.skipIf(h5py is None, "h5py unavailable")
    def test_hdf5_reconstruction_and_bounded_count(self):
        config = self.hdf5(None, data_count=3)
        with self.assertRaises(InputError):
            read_frames(config, self.base)
        config["constant_period"] = {"confirmed": True, "exposure_s": .5, "interval_s": 1}
        rows, meta = read_frames(config, self.base)
        self.assertEqual([row["t_start_s"] for row in rows], [0, 1, 2])
        self.assertTrue(meta["reconstructed"])
        config["frame_count"] = 100_000_000
        with self.assertRaises(InputError):
            read_frames(config, self.base)

    @unittest.skipIf(h5py is None, "h5py unavailable")
    def test_hdf5_rejects_timestamps_beyond_physical_stack(self):
        config = self.hdf5([0, 1, 2], data_count=1)
        with self.assertRaises(InputError) as caught:
            read_frames(config, self.base)
        details = caught.exception.as_dict()
        self.assertEqual(details["code"], "frame_count_conflict")
        self.assertEqual(details["conflict_source"], "timestamps")
        self.assertEqual(details["data_shape"], [1, 2, 2])
        self.assertEqual(details["data_frame_count"], 1)
        self.assertEqual(details["invalid_timestamp_indices"], [1, 2])

    @unittest.skipIf(h5py is None, "h5py unavailable")
    def test_hdf5_2d_image_is_exactly_one_frame(self):
        config = self.hdf5([0], data_count=1)
        with h5py.File(self.base / "master.h5", "r+") as file:
            del file["/entry/data/data"]
            file.create_dataset("/entry/data/data", shape=(10, 20), dtype="u1")
        rows, meta = read_frames(config, self.base)
        self.assertEqual(len(rows), 1)
        self.assertEqual(meta["data_frame_count"], 1)
        with h5py.File(self.base / "master.h5", "r+") as file:
            del file["/entry/instrument/detector/start_time"]
            file.create_dataset("/entry/instrument/detector/start_time", data=[0, 1])
        with self.assertRaises(InputError) as caught:
            read_frames(config, self.base)
        self.assertEqual(caught.exception.details["data_shape"], [10, 20])
        self.assertEqual(caught.exception.details["invalid_timestamp_indices"], [1])

    @unittest.skipIf(h5py is None, "h5py unavailable")
    def test_hdf5_reconstruction_cannot_manufacture_physical_frames(self):
        for starts in (None, 0.0, [0.0]):
            with self.subTest(starts=starts):
                config = self.hdf5(starts, data_count=1)
                config.update(frame_count=3, constant_period={"confirmed": True, "exposure_s": 1, "interval_s": 1})
                with self.assertRaises(InputError) as caught:
                    read_frames(config, self.base)
                self.assertEqual(caught.exception.code, "frame_count_conflict")
                self.assertEqual(caught.exception.details["conflict_source"], "frame_count")
                self.assertEqual(caught.exception.details["frame_count"], 3)

    @unittest.skipIf(h5py is None, "h5py unavailable")
    def test_hdf5_timestamp_padding_can_extend_beyond_physical_shape(self):
        config = self.hdf5([0, np.nan, np.nan], data_count=1)
        rows, meta = read_frames(config, self.base)
        self.assertEqual(len(rows), 1)
        self.assertEqual(meta["dropped_padded_indices"], [1, 2])
        self.assertFalse(meta["frame_count_unresolved"])

    @unittest.skipIf(h5py is None, "h5py unavailable")
    def test_hdf5_explicit_count_preserves_tail_and_reconstruction(self):
        for starts in ([0, np.nan, np.nan], [0], 0.0):
            for reconstruct in (False, True):
                with self.subTest(starts=starts, reconstruct=reconstruct):
                    config = self.hdf5(starts, data_count=3)
                    config["frame_count"] = 3
                    if reconstruct:
                        config["constant_period"] = {"confirmed": True, "interval_s": 1, "exposure_s": .5}
                    rows, meta = read_frames(config, self.base)
                    self.assertEqual([r["frame_index"] for r in rows], [0, 1, 2])
                    self.assertEqual([r["t_start_s"] for r in rows], [0, 1, 2] if reconstruct else [0, None, None])
                    self.assertEqual(meta["reconstructed_frame_indices"], [1, 2] if reconstruct else [])
                    self.assertEqual(meta["frame_count_source"], "frame_count")
                    self.assertFalse(meta["frame_count_unresolved"])

    @unittest.skipIf(h5py is None, "h5py unavailable")
    def test_hdf5_explicit_count_bounded_before_padding(self):
        config = self.hdf5([0], data_count=100_000_000)
        config["frame_count"] = 100_000_000
        with self.assertRaisesRegex(InputError, "bounded"):
            read_frames(config, self.base)

    @unittest.skipIf(h5py is None, "h5py unavailable")
    def test_hdf5_exposure_unit_override_and_origin(self):
        config = self.hdf5([0], exposure=.5, data_count=1)
        with h5py.File(self.base / "master.h5", "r+") as file:
            file["/entry/instrument/detector/count_time"].attrs["units"] = "ms"
        rows, meta = read_frames(config, self.base)
        self.assertEqual(rows[0]["exposure_s"], .0005)
        self.assertEqual(meta["exposure_unit_source"], "exposure_dataset.units")
        config["exposure_unit"] = "s"
        rows, meta = read_frames(config, self.base)
        self.assertEqual(rows[0]["exposure_s"], .5)
        self.assertEqual(meta["exposure_unit"], "s")
        self.assertEqual(meta["exposure_unit_source"], "config.exposure_unit")


if __name__ == "__main__":
    unittest.main()
