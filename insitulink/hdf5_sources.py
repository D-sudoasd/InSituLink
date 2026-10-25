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


class _Discovery:
    def __init__(self, h5py, stack, paths, issues):
        self.h5py, self.stack = h5py, stack
        self.paths, self.issues = paths, issues
        self.files, self.datasets, self.visited, self.active = {}, [], set(), set()

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

    def resolve(self, handle, name, active=()):
        name = posixpath.normpath("/" + name.lstrip("/"))
        owner = _canonical(handle.filename)
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
                return self.resolve(handle, posixpath.join(target, *remaining), active)
            if isinstance(link, self.h5py.ExternalLink):
                target_file = self.locate(link.filename, owner, "external_link")
                if not target_file:
                    return None
                return self.resolve(self.open(target_file[0]), posixpath.join(link.path, *remaining), active)
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
                record["virtual_sources"] = [{"filename": _text(m.file_name), "dataset": _text(m.dset_name)} for m in mappings]
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
