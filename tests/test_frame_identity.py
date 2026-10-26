"""Physical frame identity checks use temporary files and never read pixels."""
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from insitulink.identity import identify_frames
from insitulink.storage import digest


class IdentityFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def context(self, **extra):
        return {"experiment_id": "experiment-1", "acquisition_id": "scan-1", "detector_id": "detector-1",
                "base_dir": str(self.root), "request_id": str(self.root / "request.json"), **extra}


class RecordIdentityTests(IdentityFixture):
    def test_metadata_original_ids_survive_reordering_and_time_changes(self):
        rows = [{"frame_index": i, "original_id": label, "t_start_s": i} for i, label in enumerate(("A", "B"))]
        summary = identify_frames(rows, context=self.context())
        first = {row["original_id"]: row["frame_uid"] for row in rows}
        rows.reverse()
        for i, row in enumerate(rows):
            row.update(frame_index=i, t_start_s=i + 20)
        identify_frames(rows, context=self.context())
        self.assertEqual(first, {row["original_id"]: row["frame_uid"] for row in rows})
        self.assertEqual(summary["n_metadata_only"], 2)
        self.assertFalse(summary["physical_identity_complete"])
        self.assertEqual([row["sequence_index"] for row in rows], [0, 1])
        self.assertEqual(rows[0]["identity_scope"], "declared_acquisition_record")

    def test_metadata_without_namespace_states_request_scope(self):
        rows = [{"frame_index": 0, "original_id": "A"}]
        identify_frames(rows, context={"request_id": "one.json"})
        first = rows[0]["frame_uid"]
        self.assertEqual(rows[0]["identity_scope"], "request_record")
        identify_frames(rows, context={"request_id": "two.json"})
        self.assertNotEqual(first, rows[0]["frame_uid"])

    def test_basic_records_and_image_references_do_not_import_h5py(self):
        raw = self.root / "image.cbf"
        raw.write_bytes(b"Header-only fixture is intentionally not decoded")
        rows = [{"frame_index": 0}, {"source_file": str(raw), "frame_index": 0}]
        with patch.dict("sys.modules", {"h5py": None}):
            summary = identify_frames(rows)
        self.assertEqual(summary["n_metadata_only"], 1)
        self.assertEqual(summary["n_unverified"], 1)
        self.assertFalse(summary["issues"])
        self.assertFalse(summary["content_verified"])
        self.assertEqual(rows[1]["identity_status"], "unverified")

    def test_missing_source_and_missing_hdf_dataset_are_invalid(self):
        raw = self.root / "raw.h5"
        raw.write_bytes(b"No HDF open is necessary to reject a missing dataset declaration")
        rows = [{"source_file": str(self.root / "missing.cbf"), "frame_index": 0},
                {"source_file": str(raw), "frame_index": 0}]
        summary = identify_frames(rows)
        self.assertEqual(summary["n_invalid"], 2)
        self.assertEqual({issue["code"] for issue in summary["issues"]}, {"raw_source_missing", "missing_data_dataset"})

    def test_exact_legacy_uid_is_retained(self):
        row = {"source_file": None, "dataset": None, "frame_index": 12, "original_id": "frame-12"}
        expected = digest({key: row[key] for key in ("source_file", "dataset", "frame_index")})[:24]
        identify_frames([row], context=self.context())
        self.assertEqual(row["legacy_frame_uid"], expected)

    def test_non_hdf_asset_uid_survives_relocation(self):
        first, second = self.root / "one.cbf", self.root / "elsewhere.cbf"
        first.write_bytes(b"fixture")
        shutil.copyfile(first, second)
        rows = [{"source_file": str(first), "frame_index": 0}]
        identify_frames(rows, context=self.context(assets={str(first): "asset-1"}))
        uid = rows[0]["frame_uid"]
        rows[0]["source_file"] = str(second)
        identify_frames(rows, context=self.context(assets={str(second): "asset-1"}))
        self.assertEqual(rows[0]["frame_uid"], uid)
        self.assertEqual(rows[0]["identity_scope"], "declared_asset")
        self.assertEqual(rows[0]["identity_status"], "unverified")

    def test_same_basename_different_directories_is_not_duplicate(self):
        rows = []
        for directory in ("one", "two"):
            folder = self.root / directory
            folder.mkdir()
            raw = folder / "same.cbf"
            raw.write_bytes(b"fixture")
            rows.append({"source_file": str(raw), "frame_index": 0})
        summary = identify_frames(rows)
        self.assertFalse(summary["issues"])
        self.assertNotEqual(rows[0]["frame_uid"], rows[1]["frame_uid"])


@unittest.skipUnless(importlib.util.find_spec("h5py"), "optional h5py unavailable")
class HdfFrameIdentityTests(IdentityFixture):
    def setUp(self):
        super().setUp()
        import h5py
        self.h5py = h5py

    def stack(self, name="raw.h5", shape=(3, 2, 2)):
        path = self.root / name
        with self.h5py.File(path, "w") as handle:
            handle.create_dataset("images", shape=shape, dtype="u2")
        return path

    def row(self, path, index=0, dataset="/images"):
        return {"source_file": str(path), "dataset": dataset, "frame_index": index}

    def check_without_pixels(self, rows, **kwargs):
        with patch.object(self.h5py.Dataset, "__getitem__", side_effect=AssertionError("Pixel access forbidden")):
            return identify_frames(rows, **kwargs)

    def virtual(self, source, count, mappings):
        path = self.root / "virtual.h5"
        layout = self.h5py.VirtualLayout(shape=(count, 2, 2), dtype="u2")
        with self.h5py.File(source, "r") as handle:
            shape = handle["images"].shape
        physical = self.h5py.VirtualSource(str(source), "/images", shape=shape)
        for destination, origin in mappings:
            layout[destination, :, :] = physical[origin, :, :]
        with self.h5py.File(path, "w", libver="latest") as handle:
            handle.create_virtual_dataset("images", layout, fillvalue=0)
        return path

    def test_regular_stack_and_2d_indices(self):
        path = self.stack()
        rows = [self.row(path, i) for i in range(3)]
        summary = self.check_without_pixels(rows)
        self.assertTrue(summary["physical_identity_complete"], summary)
        self.assertEqual(len({row["frame_uid"] for row in rows}), 3)
        image = self.stack("image.h5", (2, 2))
        rows = [self.row(image, 0), self.row(image, 1), self.row(path, 3)]
        summary = self.check_without_pixels(rows)
        self.assertEqual([row["identity_status"] for row in rows], ["verified", "invalid", "invalid"])
        self.assertEqual({issue["code"] for issue in summary["issues"]}, {"frame_index_out_of_range"})

    def test_aliases_and_external_link_duplicate_all_conflicting_rows(self):
        path = self.stack()
        with self.h5py.File(path, "r+") as handle:
            handle["alias"] = handle["images"]
            handle["soft"] = self.h5py.SoftLink("/images")
        master = self.root / "master.h5"
        with self.h5py.File(master, "w") as handle:
            handle["images"] = self.h5py.ExternalLink(path.name, "/images")
        rows = [self.row(path, dataset=name) for name in ("images", "/images", "/alias", "/soft")]
        rows.append(self.row(master))
        summary = self.check_without_pixels(rows)
        self.assertEqual(summary["n_invalid"], 5)
        self.assertEqual(len({row["frame_uid"] for row in rows}), 1)
        self.assertEqual(len({row["legacy_frame_uid"] for row in rows}), 5)
        self.assertEqual(summary["issues"][0]["conflicting_rows"], list(range(5)))
        self.assertEqual(rows[-1]["physical_source"]["source_file"], str(path))
        self.assertEqual(set(rows[-1]["physical_source"]["storage_paths"]), {str(path), str(master)})
        self.assertEqual(rows[-1]["declared_source"]["source_file"], str(master))

    def test_vds_hole_and_duplicate_are_invalid(self):
        source = self.stack(shape=(1, 2, 2))
        path = self.virtual(source, 3, [(slice(0, 1), slice(0, 1)), (slice(2, 3), slice(0, 1))])
        rows = [self.row(path, i) for i in range(3)]
        summary = self.check_without_pixels(rows)
        self.assertEqual(summary["n_invalid"], 3, summary)
        self.assertEqual(rows[0]["physical_source"]["frame_index"], 0)
        self.assertEqual(rows[2]["physical_source"]["frame_index"], 0)
        self.assertIsNone(rows[1]["physical_source"])
        self.assertEqual(rows[1]["identity_issues"][0]["code"], "vds_unmapped_frame")
        self.assertEqual(rows[0]["identity_issues"][0]["conflicting_rows"], [0, 2])

    def test_vds_concatenation_and_mapping_provenance(self):
        source = self.stack()
        path = self.virtual(source, 2, [(slice(0, 2), slice(1, 3))])
        rows = [self.row(path, i) for i in range(2)]
        summary = self.check_without_pixels(rows)
        self.assertTrue(summary["physical_identity_complete"], summary)
        self.assertEqual([row["physical_source"]["frame_index"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["physical_source"]["mapping"][0]["virtual_selection"]["start"], [0, 0, 0])
        from insitulink.hdf5_sources import discover_dependencies
        with patch.object(self.h5py.Dataset, "__getitem__", side_effect=AssertionError("Pixel access forbidden")):
            dependencies = discover_dependencies(rows)
        self.assertTrue(dependencies["complete"])
        self.assertIn("source_selection", dependencies["datasets"][0]["virtual_sources"][0])

    def test_vds_tiles_are_explicitly_unsupported(self):
        source = self.stack(shape=(1, 2, 2))
        path = self.root / "tiles.h5"
        layout = self.h5py.VirtualLayout(shape=(1, 2, 2), dtype="u2")
        physical = self.h5py.VirtualSource(str(source), "/images", shape=(1, 2, 2))
        layout[:, :, 0:1] = physical[:, :, 0:1]
        with self.h5py.File(path, "w", libver="latest") as handle:
            handle.create_virtual_dataset("images", layout)
        rows = [self.row(path)]
        self.check_without_pixels(rows)
        self.assertEqual(rows[0]["identity_status"], "invalid")
        self.assertEqual(rows[0]["identity_issues"][0]["code"], "unsupported_vds_mapping")
        self.assertIn("unsupported", rows[0]["identity_issues"][0]["message"])

    def test_nested_vds_and_two_dimensional_source(self):
        image = self.stack("image.h5", (2, 2))
        inner = self.root / "inner.h5"
        layout = self.h5py.VirtualLayout(shape=(1, 2, 2), dtype="u2")
        layout[0, :, :] = self.h5py.VirtualSource(str(image), "/images", shape=(2, 2))
        with self.h5py.File(inner, "w", libver="latest") as handle:
            handle.create_virtual_dataset("images", layout)
        outer = self.virtual(inner, 1, [(slice(0, 1), slice(0, 1))])
        rows = [self.row(outer)]
        summary = self.check_without_pixels(rows)
        self.assertTrue(summary["physical_identity_complete"], summary)
        self.assertEqual(rows[0]["physical_source"]["source_file"], str(image))
        self.assertEqual(rows[0]["physical_source"]["frame_index"], 0)
        self.assertEqual(len(rows[0]["physical_source"]["mapping"]), 2)

    def test_missing_vds_source_and_dataset_are_invalid(self):
        source = self.stack()
        path = self.virtual(source, 1, [(slice(0, 1), slice(0, 1))])
        source.unlink()
        rows = [self.row(path), self.row(path, dataset="/absent")]
        summary = self.check_without_pixels(rows)
        self.assertEqual(summary["n_invalid"], 2)
        self.assertEqual(rows[0]["identity_issues"][0]["code"], "hdf5_missing_storage")

    def test_declared_hdf_asset_identity_survives_copy_and_reordering(self):
        source = self.stack()
        rows = [self.row(source, i) for i in range(3)]
        self.check_without_pixels(rows, context=self.context(assets={source.name: "asset-1"}))
        previous = {row["frame_index"]: row["frame_uid"] for row in rows}
        relocated = self.root / "relocated.h5"
        shutil.copyfile(source, relocated)
        rows.reverse()
        for row in rows:
            row.update(source_file=str(relocated), t_start_s=99)
        self.check_without_pixels(rows, context=self.context(assets={relocated.name: "asset-1"}))
        self.assertEqual(previous, {row["frame_index"]: row["frame_uid"] for row in rows})
        self.assertTrue(all(row["identity_scope"] == "declared_asset" for row in rows))
        self.assertTrue(all(row["identity_basis"] == "hdf5_object_address" for row in rows))

    def test_vds_multiple_search_candidates_remain_unverified(self):
        source = self.stack("source.h5")
        alternate = self.root / "prefix"
        alternate.mkdir()
        shutil.copyfile(source, alternate / source.name)
        path = self.root / "virtual.h5"
        layout = self.h5py.VirtualLayout(shape=(3, 2, 2), dtype="u2")
        layout[:] = self.h5py.VirtualSource(source.name, "/images", shape=(3, 2, 2))
        with self.h5py.File(path, "w", libver="latest") as handle:
            handle.create_virtual_dataset("images", layout)
        rows = [self.row(path)]
        with patch.dict(os.environ, {"HDF5_VDS_PREFIX": str(alternate)}):
            summary = self.check_without_pixels(rows)
        self.assertFalse(summary["issues"], summary)
        self.assertEqual(rows[0]["identity_status"], "unverified")
        self.assertEqual(len(rows[0]["physical_source"]["candidates"]), 2)
        self.assertFalse(summary["physical_identity_complete"])


if __name__ == "__main__":
    unittest.main()
