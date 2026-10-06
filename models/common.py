"""Shared by the scripts of models/: pinned downloads, SHA-256, and safetensors files written the
same, byte for byte, on every run.

The safetensors library writes a file's metadata in a random order (a Rust HashMap: version
0.8.0 gave three SHA-256s for three saves of the same tensors and metadata). seedvr2x pins each
file's SHA-256 and a script run again must give the same bytes, so the files are written here, in
the library's layout, then read back with the library to check them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

CHUNK = 16 << 20
HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"
CACHE = Path.home() / ".cache" / "seedvr2x-models"

# Bytes per element of the safetensors dtypes the scripts write.
ITEMSIZE = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def sha256_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def check_file(path: Path, sha256: str, size: int | None = None) -> None:
    """Refuses path unless it has this size and SHA-256."""
    if size is not None and path.stat().st_size != size:
        raise SystemExit(f"{path}: {path.stat().st_size} bytes, {size} expected")
    got = sha256_file(path)
    if got != sha256:
        raise SystemExit(f"{path}: SHA-256 {got}, {sha256} expected")


def fetch(url: str, dest: Path, sha256: str, size: int | None = None) -> Path:
    """dest, downloaded from url unless it is there already. A partial download (dest.part) is
    resumed. The file is refused unless it has this size and SHA-256."""
    if dest.exists():
        check_file(dest, sha256, size)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, 11):
        have = part.stat().st_size if part.exists() else 0
        if size is not None and have >= size:
            break
        headers = {"User-Agent": "seedvr2x-models"}
        if have:
            headers["Range"] = f"bytes={have}-"
        log(f"download {url} -> {part}" + (f" (resuming at {have} bytes)" if have else ""))
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=60
            ) as r:
                if have and r.status != 206:
                    have = 0  # the server sent the whole file
                t0, done, shown = time.time(), have, have
                with open(part, "ab" if have else "wb") as f:
                    while b := r.read(CHUNK):
                        f.write(b)
                        done += len(b)
                        if done - shown >= 4 << 30:
                            rate = (done - have) / (time.time() - t0) / 2**20
                            log(f"  {done / 2**30:.1f} GiB, {rate:.0f} MiB/s")
                            shown = done
            break
        except (OSError, urllib.error.URLError) as e:
            log(f"download interrupted ({e}), attempt {attempt} of 10")
            time.sleep(10 * attempt)
    else:
        raise SystemExit(f"{url}: download failed")
    check_file(part, sha256, size)
    os.replace(part, dest)
    return dest


def find_or_fetch(
    name: str, url: str, sha256: str, size: int, dirs: list[Path], cache: Path
) -> Path:
    """The file called name in the first of dirs that has one, read where it is, never moved;
    else downloaded into cache. Refused unless it has this size and SHA-256."""
    for d in dirs:
        p = d / name
        if p.exists():
            log(f"{name}: checking {p}")
            check_file(p, sha256, size)
            return p
    return fetch(url, cache / name, sha256, size)


class Entry(NamedTuple):
    """A tensor to write: its name, safetensors dtype and shape, and a function giving its bytes
    (C order, little-endian), called once, when the tensor is written."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data: Callable[[], memoryview]


def write_safetensors(path: Path, entries: list[Entry], metadata: dict[str, str]) -> None:
    """Writes a safetensors file in the library's layout: the tensors sorted by decreasing element
    size, then by name, packed from offset 0; a compact JSON header, the metadata first with its
    keys sorted, padded with spaces to a multiple of 8 bytes. Written to path.part, then renamed."""
    names = [e.name for e in entries]
    if len(set(names)) != len(names) or "__metadata__" in names:
        raise ValueError(f"{path}: duplicate or reserved tensor names")
    order = sorted(entries, key=lambda e: (-ITEMSIZE[e.dtype], e.name))
    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = dict(sorted(metadata.items()))
    end = 0
    for e in order:
        n = ITEMSIZE[e.dtype] * math.prod(e.shape)
        header[e.name] = {"dtype": e.dtype, "shape": list(e.shape), "data_offsets": [end, end + n]}
        end += n
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    raw += b" " * (-len(raw) % 8)
    part = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(part, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for e in order:
            b = e.data().cast("B")
            a, z = header[e.name]["data_offsets"]  # type: ignore[index]
            if b.nbytes != z - a:
                raise ValueError(f"{path}: {e.name} has {b.nbytes} bytes, {z - a} expected")
            f.write(b)
    os.replace(part, path)


def read_header(path: Path) -> tuple[dict, int]:
    """A safetensors file's header (tensors in file order, and __metadata__ if any), and the
    offset its data starts at."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n


def check_output(path: Path, pinned: str | None) -> str:
    """path's SHA-256, refused if it differs from the pinned one: a run with the pinned inputs and
    versions gives the same bytes."""
    got = sha256_file(path)
    if pinned is None:
        log(f"{path.name}: SHA-256 {got} (not pinned yet)")
    elif got != pinned:
        raise SystemExit(f"{path}: SHA-256 {got}, the pinned output is {pinned}")
    else:
        log(f"{path.name}: SHA-256 {got}, the pinned output")
    return got
