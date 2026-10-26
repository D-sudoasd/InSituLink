"""Discover selected detector storage from HDF5 metadata, without reading pixels.

Only the requested object paths and their storage mappings are traversed.  A
partial graph is explicitly incomplete; HDF5 fill values are not evidence that
a missing virtual source is available.
"""
from __future__ import annotations

from contextlib import ExitStack
import os
from pathlib import Path
import posixpath

from .inputs import canonical_source_file


def _text(value):
    return os.fsdecode(value)


def _canonical(value):
    return canonical_source_file(value, Path.cwd())


def _selection_description(space, h5py):
    """Serialize a selection without expanding a potentially enormous block list."""
    kind = space.get_select_type()
    result = {"type": int(kind)}
    if kind == h5py.h5s.SEL_HYPERSLABS and space.is_regular_hyperslab():
        start, stride, count, block = space.get_regular_hyperslab()
        result.update(start=list(start), stride=list(stride), count=list(count), block=list(block))
    return result


class _UnsupportedFrameMapping(ValueError):
    pass


def _rectangle(space, shape, h5py):
    """Return one contiguous selection; arbitrary image transforms are unsupported."""
    if space.get_select_type() == h5py.h5s.SEL_ALL:
        return (0,) * len(shape), tuple(shape)
    if space.get_select_type() != h5py.h5s.SEL_HYPERSLABS or not space.is_regular_hyperslab():
        raise _UnsupportedFrameMapping("Only contiguous, whole-frame VDS selections are supported")
    start, stride, count, block = space.get_regular_hyperslab()
    if len(start) != len(shape) or any(c > 1 and s != b for s, c, b in zip(stride, count, block)):
        raise _UnsupportedFrameMapping("Only contiguous, whole-frame VDS selections are supported")
    size = tuple(c * b for c, b in zip(count, block))
    if any(n <= 0 or a + n > total for a, n, total in zip(start, size, shape)):
        raise _UnsupportedFrameMapping("VDS selection exceeds the available dataset extent")
    return tuple(start), size


class _Discovery:
    def __init__(self, h5py, stack, paths, issues):
        self.h5py, self.stack = h5py, stack
        self.paths, self.issues = paths, issues
        self.files, self.datasets, self.visited, self.active = {}, [], set(), set()
        self.frame_objects = {}
        self.object_traces = {}
        self.virtual_frame_mappings = {}

    def issue(self, code, message, **details):
        self.issues.append({"code": code, "message": message, **details})

    def open(self, path):
        path = _canonical(path)
        self.paths.add(path)
        key = os.path.normcase(path)
        if key not in self.files:
            self.files[key] = self.stack.enter_context(self.h5py.File(path, "r"))
        return self.files[key]

    def locate(self, filename, owner, kind, prefix=""):
        """Bind plausible storage files without guessing the VDS search winner.

        HDF5 builds can differ in whether environment prefixes take effect for
        VDS and external raw storage. Bind every existing candidate in those
        cases; an external link is separately checked against the opened object.
        """
        filename = _text(filename)
        if not filename or "%" in filename or "${" in filename:
            self.issue("hdf5_dynamic_source", "Cannot enumerate dynamic HDF5 storage names", filename=filename, owner=owner, kind=kind)
            return []
        name = Path(filename)
        if kind == "external_raw":
            # H5Pset_efile_prefix defaults to cwd, not the HDF5 file directory.
            candidates = []
            for candidate_prefix in (os.environ.get("HDF5_EXTFILE_PREFIX", ""), prefix, ""):
                candidate_prefix = candidate_prefix.replace("${ORIGIN}", str(Path(owner).parent))
                if "${" in candidate_prefix:
                    self.issue("hdf5_unresolved_prefix", "Cannot resolve external storage prefix", prefix=candidate_prefix, owner=owner)
                    continue
                candidates.append(name if name.is_absolute() else Path(candidate_prefix) / name)
        else:
            candidates = [name] if name.is_absolute() else []
            if name.is_absolute() and not name.is_file():
                name = Path(name.name)
            if not name.is_absolute():
                env = os.environ.get("HDF5_EXT_PREFIX" if kind == "external_link" else "HDF5_VDS_PREFIX", "")
                prefixes = env.split(os.pathsep) if kind == "external_link" else [env]
                for candidate_prefix in [*prefixes, prefix]:
                    if not candidate_prefix:
                        continue
                    candidate_prefix = candidate_prefix.replace("${ORIGIN}", str(Path(owner).parent))
                    if "${" in candidate_prefix:
                        self.issue("hdf5_unresolved_prefix", "Cannot resolve HDF5 search prefix", prefix=candidate_prefix, owner=owner)
                        continue
                    candidates.append(Path(candidate_prefix) / name)
                candidates.extend((Path(owner).parent / name, name))
        found = []
        for candidate in candidates:
            if candidate.is_file():
                resolved = _canonical(candidate)
                self.paths.add(resolved)
                if resolved not in found:
                    found.append(resolved)
                if kind == "external_link":
                    return found
        if found:
            return sorted(found)
        # Keep the expected path even when HDF5 would silently supply fill data.
        missing = _canonical(candidates[0])
        self.paths.add(missing)
        self.issue("hdf5_missing_storage", "HDF5 storage file is unavailable", path=missing, owner=owner, kind=kind)
        return []

    def resolve(self, handle, name, active=(), trace=None):
        name = posixpath.normpath("/" + name.lstrip("/"))
        owner = _canonical(handle.filename)
        if trace is not None:
            trace.append({"source_file": owner, "dataset": name})
        state = (os.path.normcase(owner), name)
        if state in active or len(active) >= 128:
            self.issue("hdf5_link_cycle", "HDF5 link traversal is cyclic or exceeds the supported depth", path=owner, dataset=name)
            return None
        active = (*active, state)
        current = handle["/"]
        parts = [part for part in name.split("/") if part and part != "."]
        for i, part in enumerate(parts):
            link = current.get(part, getlink=True)
            remaining = parts[i + 1:]
            if isinstance(link, self.h5py.SoftLink):
                target = link.path if link.path.startswith("/") else posixpath.join(current.name, link.path)
                return self.resolve(handle, posixpath.join(target, *remaining), active, trace)
            if isinstance(link, self.h5py.ExternalLink):
                target_file = self.locate(link.filename, owner, "external_link")
                if not target_file:
                    return None
                return self.resolve(self.open(target_file[0]), posixpath.join(link.path, *remaining), active, trace)
            if not isinstance(link, self.h5py.HardLink):
                self.issue("hdf5_unresolved_link", "Selected HDF5 object path is unavailable or unsupported", path=owner, dataset=name)
                return None
            current = current[part]
            if remaining and not isinstance(current, self.h5py.Group):
                self.issue("hdf5_invalid_path", "An intermediate HDF5 object is not a group", path=owner, dataset=name)
                return None
        return current

    def visit(self, handle, name):
        if len(self.active) >= 128:
            self.issue("hdf5_storage_depth", "HDF5 storage mapping exceeds the supported depth", dataset=name)
            return
        obj = self.resolve(handle, name)
        if obj is None:
            return
        if not isinstance(obj, self.h5py.Dataset):
            self.issue("hdf5_not_dataset", "Selected HDF5 object is not a dataset", path=_canonical(handle.filename), dataset=name)
            return
        actual = _canonical(obj.file.filename)
        self.paths.add(actual)
        # Confirm traversal against HDF5 itself (including its filesystem search).
        confirmed = handle[name]
        if os.path.normcase(_canonical(confirmed.file.filename)) != os.path.normcase(actual) or confirmed.id != obj.id:
            self.paths.add(_canonical(confirmed.file.filename))
            self.issue("hdf5_resolution_mismatch", "HDF5 resolved a different storage object", path=_canonical(handle.filename), dataset=name)
            return
        key = (os.path.normcase(actual), int(self.h5py.h5o.get_info(obj.id).addr))
        if key in self.active:
            self.issue("hdf5_storage_cycle", "Virtual HDF5 storage mappings contain a cycle", path=actual, dataset=obj.name)
            return
        if key in self.visited:
            return
        self.active.add(key)
        try:
            record = {"path": actual, "dataset": obj.name, "shape": list(obj.shape), "dtype": str(obj.dtype),
                      "search_context": {"cwd": str(Path.cwd()),
                                         **{key: os.environ.get(key) for key in ("HDF5_EXT_PREFIX", "HDF5_VDS_PREFIX", "HDF5_EXTFILE_PREFIX")}}}
            self.datasets.append(record)
            access = obj.id.get_access_plist()
            external = obj.external
            if external:
                prefix = _text(access.get_efile_prefix())
                record["search_context"]["external_access_prefix"] = prefix
                record["external"] = [{"filename": _text(n), "offset": int(o), "size": int(s)} for n, o, s in external]
                for entry in record["external"]:
                    entry["candidate_paths"] = self.locate(entry["filename"], actual, "external_raw", prefix)
            if obj.is_virtual:
                prefix = _text(access.get_virtual_prefix())
                record["search_context"]["virtual_access_prefix"] = prefix
                mappings = obj.virtual_sources()
                record["virtual_sources"] = [{"filename": _text(m.file_name), "dataset": _text(m.dset_name),
                                              "virtual_selection": _selection_description(m.vspace, self.h5py),
                                              "source_selection": _selection_description(m.src_space, self.h5py)} for m in mappings]
                for mapping in record["virtual_sources"]:
                    filename, dataset = mapping["filename"], mapping["dataset"]
                    if not dataset or "%" in dataset:
                        self.issue("hdf5_dynamic_source", "Cannot enumerate dynamic virtual dataset names", path=actual, dataset=dataset)
                        continue
                    candidates = [actual] if filename == "." else self.locate(filename, actual, "virtual", prefix)
                    mapping["candidate_paths"] = candidates
                    for source in candidates:
                        try:
                            self.visit(self.open(source), dataset)
                        except Exception as exc:
                            self.issue("hdf5_dependency_unavailable", f"Cannot inspect virtual source candidate: {exc}", path=source, dataset=dataset)
            self.visited.add(key)
        finally:
            self.active.remove(key)

    def frame_object(self, handle, name):
        """Resolve one object through HDF5 and retain metadata for repeated frames."""
        key = (os.path.normcase(_canonical(handle.filename)), name)
        if key not in self.frame_objects:
            trace = []
            obj = self.resolve(handle, name, trace=trace)
            if not isinstance(obj, self.h5py.Dataset):
                raise ValueError("The selected HDF5 object is not an available dataset")
            confirmed = handle[name]
            if confirmed.id != obj.id or os.path.normcase(_canonical(confirmed.file.filename)) != os.path.normcase(_canonical(obj.file.filename)):
                raise ValueError("HDF5 resolved a different physical storage object")
            self.frame_objects[key] = obj
            self.object_traces[key] = trace
        return self.frame_objects[key]

    def frame_source(self, handle, name, index, active=()):
        """Map a logical image to complete physical images using metadata only."""
        obj = self.frame_object(handle, name)
        owner = _canonical(obj.file.filename)
        address = int(self.h5py.h5o.get_info(obj.id).addr)
        key = (os.path.normcase(owner), address, index)
        if key in active or len(active) >= 128:
            raise ValueError("Cyclic or excessively deep HDF5 frame mapping")
        if obj.ndim not in (2, 3) or any(n <= 0 for n in obj.shape):
            raise ValueError("Detector data must contain nonempty 2D images or a 3D frame stack")
        count = 1 if obj.ndim == 2 else obj.shape[0]
        if index < 0 or index >= count:
            raise IndexError("frame_index exceeds the physical dataset extent")
        trace = self.object_traces[(os.path.normcase(_canonical(handle.filename)), name)]
        result = {"source_file": owner, "dataset": obj.name, "frame_index": index,
                  "object_address": address, "shape": list(obj.shape), "resolution_chain": trace,
                  "storage_paths": sorted({owner, *(step["source_file"] for step in trace)}), "mapping": []}
        if not obj.is_virtual:
            for filename, _, _ in obj.external or []:
                candidates = self.locate(filename, owner, "external_raw", _text(obj.id.get_access_plist().get_efile_prefix()))
                if not candidates:
                    raise FileNotFoundError("External HDF5 raw storage is unavailable")
                result["storage_paths"].extend(candidates)
                if len(candidates) > 1:
                    result["storage_resolution"] = "multiple_candidates"
            result["storage_paths"] = sorted(set(result["storage_paths"]))
            return result

        mapping_key = (os.path.normcase(owner), address)
        if mapping_key not in self.virtual_frame_mappings:
            self.virtual_frame_mappings[mapping_key] = [
                (mapping, *_rectangle(mapping.vspace, obj.shape, self.h5py)) for mapping in obj.virtual_sources()]
        matches = []
        for mapping, begin, extent in self.virtual_frame_mappings[mapping_key]:
            if obj.ndim == 3 and not begin[0] <= index < begin[0] + extent[0]:
                continue
            if begin[-2:] != (0, 0) or extent[-2:] != obj.shape[-2:]:
                raise _UnsupportedFrameMapping("VDS tiles, cropped images and partial-frame mappings are unsupported")
            matches.append((mapping, begin, extent))
        if not matches:
            raise LookupError("The VDS frame has no physical source mapping; fill values are not acquired frames")
        if len(matches) != 1:
            raise _UnsupportedFrameMapping("The VDS frame has overlapping source mappings")
        mapping, begin, extent = matches[0]
        filename, dataset = _text(mapping.file_name), _text(mapping.dset_name)
        if not dataset or "%" in dataset:
            raise _UnsupportedFrameMapping("Dynamic VDS source dataset names are unsupported")
        candidates = [owner] if filename == "." else self.locate(filename, owner, "virtual", _text(obj.id.get_access_plist().get_virtual_prefix()))
        if not candidates:
            raise FileNotFoundError("The VDS physical source file is unavailable")
        physical = []
        for path in candidates:
            source_handle = self.open(path)
            source = self.frame_object(source_handle, dataset)
            if source.ndim not in (2, 3):
                raise _UnsupportedFrameMapping("VDS source is not a 2D image or 3D frame stack")
            source_begin, source_extent = _rectangle(mapping.src_space, source.shape, self.h5py)
            if (source_begin[-2:] != (0, 0) or source_extent[-2:] != source.shape[-2:]
                    or source.shape[-2:] != obj.shape[-2:]
                    or (extent[0] if obj.ndim == 3 else 1) != (source_extent[0] if source.ndim == 3 else 1)):
                raise _UnsupportedFrameMapping("VDS mapping must preserve complete images and one-to-one frame order")
            source_index = (source_begin[0] if source.ndim == 3 else 0) + (index - begin[0] if obj.ndim == 3 else 0)
            found = self.frame_source(source_handle, dataset, source_index, (*active, key))
            step = {"source_file": owner, "dataset": obj.name, "frame_index": index,
                    "resolution_chain": trace,
                    "virtual_selection": _selection_description(mapping.vspace, self.h5py),
                    "source_selection": _selection_description(mapping.src_space, self.h5py)}
            found["mapping"] = [step, *found["mapping"]]
            found["storage_paths"] = sorted(set([*result["storage_paths"], *found["storage_paths"]]))
            physical.append(found)
        if len(physical) == 1:
            return physical[0]
        # All possible resolutions remain visible; metadata cannot establish the winner.
        result.update(candidates=physical, storage_resolution="multiple_candidates",
                      storage_paths=sorted(set(path for item in physical for path in item["storage_paths"])))
        return result


def resolve_frame_sources(frames: list[dict]) -> dict:
    """Verify HDF5 frame addresses without pixels; preserve every declared row.

    This is separate from dependency enumeration: a complete dependency graph
    does not prove that a VDS has a one-to-one physical mapping for each frame.
    """
    try:
        import h5py
    except ImportError:
        return {"frames": [{"physical_source": None, "issues": [{"code": "hdf5_missing_dependency",
                 "message": "Reading HDF5 frame identity requires h5py", "blocking": True}]} for _ in frames], "paths": []}
    paths, issues, resolved = set(), [], []
    with ExitStack() as stack:
        discovery = _Discovery(h5py, stack, paths, issues)
        for frame in frames:
            try:
                source = discovery.frame_source(discovery.open(frame["source_file"]), frame["dataset"], frame["frame_index"])
                warnings = []
                if source.get("storage_resolution") == "multiple_candidates" or any(item.get("storage_resolution") == "multiple_candidates" for item in source.get("candidates", [])):
                    warnings.append({"code": "hdf5_storage_candidates", "message": "Multiple physical storage candidates remain; no search winner was inferred", "blocking": False})
                resolved.append({"physical_source": source, "issues": warnings})
            except Exception as exc:
                code = ("unsupported_vds_mapping" if isinstance(exc, _UnsupportedFrameMapping) else
                        "frame_index_out_of_range" if isinstance(exc, IndexError) else
                        "hdf5_missing_storage" if isinstance(exc, FileNotFoundError) else
                        "vds_unmapped_frame" if isinstance(exc, LookupError) else "hdf5_frame_unverified")
                resolved.append({"physical_source": None, "issues": [{"code": code, "message": str(exc), "blocking": True}]})
    return {"frames": resolved, "paths": sorted(paths)}


def discover_dependencies(frames: list[dict]) -> dict:
    """Return canonical source paths, completeness and selected storage metadata.

    Non-HDF frames only contribute their source filenames. HDF5 is imported
    lazily, and neither dataset indexing nor image readers are used here.
    """
    paths, issues, selections = set(), [], set()
    for frame in frames:
        source = _canonical(frame.get("source_file"))
        if source is not None:
            paths.add(source)
            if frame.get("dataset"):
                selections.add((source, str(frame["dataset"])))
    if not selections:
        return {"paths": sorted(paths), "complete": True, "issues": [], "datasets": []}
    try:
        import h5py
    except ImportError as exc:
        return {"paths": sorted(paths), "complete": False,
                "issues": [{"code": "hdf5_missing_dependency", "message": str(exc)}], "datasets": []}
    with ExitStack() as stack:
        discovery = _Discovery(h5py, stack, paths, issues)
        for source, dataset in sorted(selections):
            try:
                discovery.visit(discovery.open(source), dataset)
            except Exception as exc:
                # Metadata plugins, unreadable files and failed enumeration must
                # invalidate reuse rather than quietly become an empty graph.
                discovery.issue("hdf5_dependency_unavailable", f"Cannot inspect HDF5 storage: {exc}", path=source, dataset=dataset)
        return {"paths": sorted(paths), "complete": not issues, "issues": issues, "datasets": discovery.datasets}
