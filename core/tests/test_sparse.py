"""`library_fixtures.sparse`: the synthetic model files marked sparse on NTFS before they grow.

Linux holds the calls with kernel32 and the platform faked; Windows itself holds the allocation."""

import ctypes
import sys
import types
from pathlib import Path
from typing import Any

import library_fixtures
import pytest
from library_fixtures import FSCTL_SET_SPARSE, grow, sparse

GIB = 2**30


def test_on_other_platforms_nothing_is_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        library_fixtures, "_kernel32", lambda: pytest.fail("kernel32 is a Windows library")
    )
    with sparse((tmp_path / "x").open("wb")) as file:
        file.truncate(10)
    assert (tmp_path / "x").stat().st_size == 10


def test_windows_is_asked_for_a_sparse_file_once_before_the_file_grows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[str, Any]] = []
    path = tmp_path / "x.safetensors"

    class Control:
        argtypes: Any = None
        restype: Any = None

        def __call__(self, handle: int, code: int, *rest: Any) -> int:
            events.append(("control", (handle, code, rest, path.stat().st_size)))
            return 1

    class Kernel:
        DeviceIoControl = Control()

    fake = types.ModuleType("msvcrt")
    fake.get_osfhandle = lambda descriptor: events.append(("handle", descriptor)) or 4242  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(library_fixtures, "_kernel32", lambda: Kernel())
    monkeypatch.setattr(sys, "platform", "win32")
    with sparse(path.open("wb")) as file:
        events.append(("descriptor", file.fileno()))
        file.write(b"header")
        grow(file, GIB)
    kinds = [kind for kind, _ in events]
    assert kinds.count("control") == 1 and kinds.index("control") < len(kinds) - 1
    control = next(value for kind, value in events if kind == "control")
    assert control[0] == 4242 and control[1] == FSCTL_SET_SPARSE == 0x000900C4
    assert control[3] == 0  # the file was still empty
    assert path.stat().st_size == GIB


class Recording:
    """A file that tells the calls grow makes."""

    def __init__(self, file: Any) -> None:
        self.file, self.calls = file, []

    def __getattr__(self, name: str) -> Any:
        def call(*args: Any) -> Any:
            self.calls.append((name, args))
            return getattr(self.file, name)(*args)

        return call


def test_windows_grows_by_a_seek_and_a_byte_never_a_truncate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    with (tmp_path / "x").open("wb") as raw:
        raw.write(b"head")
        recording = Recording(raw)
        grow(recording, 1000)  # type: ignore[arg-type]
        assert [name for name, _ in recording.calls] == ["tell", "seek", "write", "seek"]
        assert recording.calls[1][1] == (999,) and recording.calls[2][1] == (b"\0",)
        assert raw.tell() == 4
    data = (tmp_path / "x").read_bytes()
    assert data == b"head" + bytes(996)


def test_elsewhere_grow_is_truncate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    with (tmp_path / "x").open("wb") as raw:
        raw.write(b"head")
        recording = Recording(raw)
        grow(recording, 1000)  # type: ignore[arg-type]
        assert recording.calls == [("truncate", (1000,))]
    assert (tmp_path / "x").read_bytes() == b"head" + bytes(996)


def test_windows_refusing_the_control_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Refusing:
        argtypes: Any = None
        restype: Any = None

        def __call__(self, *rest: Any) -> int:
            return 0

    class Kernel:
        DeviceIoControl = Refusing()

    fake = types.ModuleType("msvcrt")
    fake.get_osfhandle = lambda descriptor: 1  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(library_fixtures, "_kernel32", lambda: Kernel())
    monkeypatch.setattr(sys, "platform", "win32")
    with (tmp_path / "x").open("wb") as file, pytest.raises(OSError, match="FSCTL_SET_SPARSE"):
        sparse(file)


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS's sparse files are Windows'")
def test_on_windows_a_gib_file_takes_under_a_mib_on_disk(tmp_path: Path) -> None:
    assert sys.platform == "win32"
    path = tmp_path / "big.safetensors"
    with sparse(path.open("wb")) as file:
        file.write(b"head")
        grow(file, GIB)
    high = ctypes.c_ulong(0)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCompressedFileSizeW.restype = ctypes.c_ulong
    kernel.GetCompressedFileSizeW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong)]
    low = kernel.GetCompressedFileSizeW(str(path), ctypes.byref(high))
    assert (high.value << 32) + low < 2**20
    assert path.stat().st_size == GIB
    with path.open("rb") as read:
        assert read.read(4) == b"head"
        read.seek(GIB // 2)
        assert read.read(4096) == bytes(4096)
