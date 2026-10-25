"""Atomic local artifacts and content-bound stage receipts."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from .errors import InputError


def json_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def canonical(value):
    return json.dumps(json_value(value), sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".insitulink-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_text(path, json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class FileDigests:
    """Hash each unchanged file once per invocation, checking for concurrent writes."""

    def __init__(self):
        self.values = {}

    def file(self, path):
        path = Path(path).resolve()
        try:
            before = path.stat()
            signature = (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino)
            key = (str(path), signature)
            if key not in self.values:
                checksum = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        checksum.update(block)
                after = path.stat()
                if signature != (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
                    raise InputError(f"Input changed while being read: {path}", "input_changed")
                self.values[key] = checksum.hexdigest()
            return {"path": str(path), "sha256": self.values[key]}
        except OSError as exc:
            raise InputError(f"Cannot read input {path}: {exc}", "unreadable_input") from exc


class Stages:
    def __init__(self, output, state, files):
        self.root = Path(output) / ".insitulink" / "cache"
        self.output = Path(output)
        self.state = state
        self.files = files

    def run(self, name, binding, function, modules):
        code = [self.files.file(Path(__file__).parent / module) for module in dict.fromkeys(modules + ["storage.py", "errors.py", "__init__.py"])]
        packages = {"numpy"}
        if "inputs.py" in modules:
            packages.update(("openpyxl", "h5py"))
        if "preview.py" in modules:
            packages.update(("scipy", "pyFAI", "fabio", "h5py"))
        versions = {}
        for package in sorted(packages):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = None
        fingerprint = digest({"binding": binding, "code": code, "dependencies": versions})
        receipt = self.root / f"{name}.json"
        try:
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            if saved["fingerprint"] == fingerprint and saved["result_sha256"] == digest(saved["result"]):
                self.state["stages"][name] = {"status": "reused", "fingerprint": fingerprint}
                self._log(name, "reused")
                return saved["result"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
        started = time.perf_counter()
        self.state["current_stage"] = name
        self.state["stages"][name] = {"status": "running", "fingerprint": fingerprint}
        write_json(self.output / "status.json", self.state)
        self._log(name, "started")
        try:
            result = json_value(function())
        except Exception:
            self.state["stages"][name] = {"status": "incomplete", "fingerprint": fingerprint}
            self._log(name, "incomplete")
            raise
        write_json(receipt, {"fingerprint": fingerprint, "result_sha256": digest(result), "result": result})
        self.state["stages"][name] = {"status": "computed", "fingerprint": fingerprint, "elapsed_s": time.perf_counter() - started}
        self._log(name, "computed")
        return result

    def _log(self, name, status):
        with (self.output / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{name}: {status}\n")


def _alive(pid):
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return kernel.GetLastError() == 5
        try:
            code = ctypes.c_ulong()
            return not kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextmanager
def output_lock(output):
    path = Path(output) / ".insitulink.lock"
    for _ in range(2):
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            break
        except FileExistsError:
            try:
                owner = json.loads(path.read_text(encoding="utf-8"))
                if _alive(int(owner["pid"])):
                    raise InputError("Another process owns this result directory", "output_locked", pid=owner["pid"])
                path.unlink()
            except (ValueError, KeyError, OSError) as exc:
                raise InputError("Cannot verify the existing output lock", "output_locked") from exc
    else:
        raise InputError("Result directory lock changed during acquisition", "output_locked")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid()}, stream)
        yield
    finally:
        path.unlink(missing_ok=True)
