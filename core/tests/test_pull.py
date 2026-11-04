"""seedvr2x's own model files, pulled from its Hugging Face repo at a pinned revision, each checked
by its pinned size and SHA-256 (seedvr2x/pull.py; DESIGN.md, Weights).

On the CPU, huggingface_hub's two calls replaced (Hub), and the pins by two small files' (pins):
the pinned check, the size before the hash; each file hashed once a run, the manifest taking that
hash; the library never called with --model-dir, and called with the repo, the pinned revision,
the file's name and no token without it; a pinned name in the other role refused before any
lookup; the same bytes cached at another revision taken from the cache, offline too; each of the
library's failures refused, saying what to do, and its log in seedvr2x's, once, without the Hub's
advice to set a token; the free space checked before each download against the downloads left, a
partial file left there said; SIGTERM during a download unwinding it; the log naming the cache; a
cache behind a link; a header refusal of one of seedvr2x's own files saying how to fetch it again,
in its own role only; a file changed between its hash and its load refused, before and after its
load; all of it before torch. Hugging Face's telemetry off unless the user set it, before the
library is imported. The shot detector's file, TransNetV2's weights, resolved on its own (the CLI
doesn't ask for it yet): with the others, in the same fetch, its space counted; from --model-dir
by its name; checked by its pin, and its refusals saying how to fetch it again.

Then the revision itself, skipped offline: SHA256SUMS, a few kilobytes, downloaded at the pinned
revision through seedvr2x's own fetch, into a temporary HF_HOME behind a link, and checked by its
own pin; every pinned file's SHA-256 is its line there, and every pinned size the Hub's. With
SEEDVR2X_TESTS_PULL=1, the four v1 files themselves, into Hugging Face's cache: about 33 GB the
first time.

Every test that could reach the Hub replaces the library's two calls, or runs in a process of its
own whose HF_HOME is the test's, HF_HUB_OFFLINE=1 in its environment but for the revision's own
test: none downloads a model file. hf_xet is imported in such a process only, since its import
writes a log file under HF_HOME, and one of them holds seedvr2x to never importing it.

The tests that run the command line over the pull are test_pull_cli.py's (apps/seedvr2x/tests/); the
helpers both use (Hub, pins, ...) are this file's, which that one imports."""

import hashlib
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from huggingface_hub import constants, file_download
from huggingface_hub.errors import (
    FileMetadataError,
    LocalEntryNotFoundError,
)
from test_weights import SHOTS, as_dtype, write

from seedvr2x import files, model, pull, weights
from seedvr2x.pull import Pin
from seedvr2x.weights import ModelError

REVISION = "c14a2bc4aab04cf38ad9b0d324014c4213f07048"
SHORT = f"hekmon/seedvr2x at {REVISION[:8]}"
ELSEWHERE = "or download the files elsewhere and give --model-dir with their directory"
UNAVAILABLE = (
    "the repository or the file is unavailable: download the files elsewhere and give --model-dir"
    " with their directory"
)
NO_TOKEN = (
    "hekmon/seedvr2x is public: seedvr2x downloads from it with no token, never sending one stored"
    " for huggingface_hub"
)

# Two small files in place of seedvr2x's own (pins), as the DiT and the VAE: the stand-in's model
# check passes them (stand_in_model).
DIT, VAE = "small_dit.safetensors", "small_vae.safetensors"
CONTENT = {DIT: b"the DiT's bytes " * 40, VAE: b"the VAE's bytes " * 25}


class Hub:
    """huggingface_hub's two calls seedvr2x makes, in place of the library's: the cache's lookup
    finds a file in the pinned revision's snapshot, laid out as the library lays it out; the
    download records its arguments, then writes the file there from served, under the cache's
    directory resolved as the library resolves it (file_download.py:999), tells downloaded, or
    raises error."""

    def __init__(self, cache: Path) -> None:
        self.cache = cache
        self.served: dict[str, bytes] = dict(CONTENT)
        self.error: BaseException | None = None
        self.calls: list[dict[str, Any]] = []
        self.lookups: list[str] = []
        self.downloaded: Callable[[str], None] | None = None

    def snapshot(self, name: str, cache: Path | None = None) -> Path:
        base = self.cache if cache is None else cache
        return base / "models--hekmon--seedvr2x" / "snapshots" / REVISION / name

    def try_to_load_from_cache(
        self,
        repo_id: str,
        filename: str,
        cache_dir: Any = None,
        revision: Any = None,
        repo_type: Any = None,
    ) -> str | None:
        assert (repo_id, Path(cache_dir), revision) == ("hekmon/seedvr2x", self.cache, REVISION)
        self.lookups.append(filename)
        path = self.snapshot(filename)
        return str(path) if path.is_file() else None

    def hf_hub_download(self, repo_id: str, filename: str, **options: Any) -> str:
        self.calls.append({"repo_id": repo_id, "filename": filename, **options})
        if self.error is not None:
            raise self.error
        path = self.snapshot(filename, Path(options["cache_dir"]).resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.served[filename])
        if self.downloaded is not None:
            self.downloaded(filename)
        return str(path)


def faked(monkeypatch: pytest.MonkeyPatch, cache: Path) -> Hub:
    """The library's two calls replaced by a Hub of cache, Hugging Face's cache there, online."""
    fake = Hub(cache)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(fake.cache))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(file_download, "try_to_load_from_cache", fake.try_to_load_from_cache)
    monkeypatch.setattr(file_download, "hf_hub_download", fake.hf_hub_download)
    return fake


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Hub:
    return faked(monkeypatch, tmp_path / "hf" / "hub")


@pytest.fixture
def pins(monkeypatch: pytest.MonkeyPatch) -> dict[str, Pin]:
    """seedvr2x's pins, the small files' in their place, in their roles."""
    small = {
        name: Pin(len(data), hashlib.sha256(data).hexdigest(), "dit" if name == DIT else "vae")
        for name, data in CONTENT.items()
    }
    monkeypatch.setattr(pull, "PINNED", small)
    return small


@pytest.fixture
def hashed(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """The files read to be hashed (files.sha256), in order."""
    read: list[Path] = []
    real = files.sha256

    def sha256(path: Path) -> str:
        read.append(path)
        return real(path)

    monkeypatch.setattr(files, "sha256", sha256)
    return read


def model_dir(tmp_path: Path) -> Path:
    """The small files, as they should be, in a directory of their own."""
    directory = tmp_path / "models"
    directory.mkdir(exist_ok=True)
    for name, data in CONTENT.items():
        (directory / name).write_bytes(data)
    return directory


def url(name: str) -> str:
    return f"https://huggingface.co/hekmon/seedvr2x/resolve/{REVISION}/{name}"


def from_the_cache(name: str, path: Path, cache: Path) -> str:
    """How a refusal of a file of the cache, path, says to fetch it again."""
    command = (
        f"hf download hekmon/seedvr2x {name} --revision {REVISION} --cache-dir {cache}"
        " --force-download"
    )
    return (
        f"remove {path.resolve()}, which the next run downloads again, or download it again with"
        f" huggingface_hub's command:\n{command}"
    )


INTO_MODEL_DIR = "or leave out --model-dir, which takes it from Hugging Face's cache"


def test_pins_recorded() -> None:
    # The model conversation's record of the upload (2026-10-08), the int8 files' as
    # models/seedvr2_int8.py pins their SHA-256 (OUTPUTS); test_pins_are_the_revisions holds them
    # to the revision itself. Each in its role.
    assert (pull.REPO, pull.REVISION) == ("hekmon/seedvr2x", REVISION)
    assert {name: (pin.size, pin.sha256[:8], pin.role) for name, pin in pull.PINNED.items()} == {
        "seedvr2x_ema_7b_fp16.safetensors": (16_479_335_080, "071cab5e", "dit"),
        "seedvr2x_ema_7b_sharp_fp16.safetensors": (16_479_335_088, "5eb47fde", "dit"),
        "seedvr2x_ema_7b_int8_convrot.safetensors": (8_333_648_992, "7eb2c784", "dit"),
        "seedvr2x_ema_7b_sharp_int8_convrot.safetensors": (8_333_649_000, "d83aeaa4", "dit"),
        "seedvr2x_ema_vae_fp16.safetensors": (501_325_454, "b9c6ebf0", "vae"),
        "transnetv2.safetensors": (30_482_632, "bb8c8388", "detector"),
    }
    assert pull.DETECTOR == "transnetv2.safetensors"
    assert (pull.SUMS, pull.SUMS_PIN.size, pull.SUMS_PIN.sha256[:8], pull.SUMS_PIN.role) == (
        "SHA256SUMS",
        2335,
        "57cf5a7b",
        None,
    )


def test_size_refused_before_the_hash(
    tmp_path: Path, pins: dict[str, Pin], hashed: list[Path]
) -> None:
    # Named like one of seedvr2x's own, another size: refused by a stat, unread, both sizes said,
    # and how to fetch it again into --model-dir; the file left as it is.
    path = tmp_path / DIT
    path.write_bytes(CONTENT[DIT] + b"!")
    size = pins[DIT].size
    with pytest.raises(ModelError) as error:
        pull.check_pin(DIT, path, pins[DIT], pull.Hashes(), None)
    assert str(error.value) == (
        f"{path}: {size + 1:,} bytes, where hekmon/seedvr2x's {DIT} at {REVISION[:8]} has"
        f" {size:,}: not the file seedvr2x pins, left as it is; fetch it again from {url(DIT)},"
        f" {INTO_MODEL_DIR}"
    )
    assert hashed == []
    assert path.read_bytes() == CONTENT[DIT] + b"!"


def test_hash_refused(tmp_path: Path, pins: dict[str, Pin], hashed: list[Path]) -> None:
    # The pinned size, other bytes: refused once hashed, both hashes said, and how to fetch it
    # again into Hugging Face's cache: the file the cache's link leads to, removed, or
    # huggingface_hub's own command; the file left as it is.
    cache = tmp_path / "hf" / "hub"
    blob = cache / "models--hekmon--seedvr2x" / "blobs" / "0123abcd"
    blob.parent.mkdir(parents=True)
    other = bytes(len(CONTENT[DIT]))
    blob.write_bytes(other)
    link = cache / "models--hekmon--seedvr2x" / "snapshots" / REVISION / DIT
    link.parent.mkdir(parents=True)
    link.symlink_to(Path("..", "..", "blobs", blob.name))
    with pytest.raises(ModelError) as error:
        pull.check_pin(DIT, link, pins[DIT], pull.Hashes(), cache)
    assert str(error.value) == (
        f"{link}: SHA-256 {hashlib.sha256(other).hexdigest()}, where hekmon/seedvr2x's {DIT} at"
        f" {REVISION[:8]} has {pins[DIT].sha256}: not the file seedvr2x pins, left as it is;"
        f" {from_the_cache(DIT, blob, cache)}"
    )
    assert hashed == [link]
    assert blob.read_bytes() == other and link.resolve() == blob.resolve()


def test_pinned_hashed_once(
    tmp_path: Path, pins: dict[str, Pin], hashed: list[Path], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    path = model_dir(tmp_path) / VAE
    hashes = pull.Hashes()
    pull.check_pin(VAE, path, pins[VAE], hashes, None)
    assert hashes.sha256(path) == pins[VAE].sha256
    assert hashed == [path]
    assert f"{VAE}: SHA-256 {pins[VAE].sha256}, in" in caplog.text
    said = f"{VAE}: its size and SHA-256 as seedvr2x pins them, {SHORT}"
    assert said in caplog.messages


def test_a_file_vouched_for_by_size_and_time_is_not_read(
    tmp_path: Path, pins: dict[str, Pin], hashed: list[Path]
) -> None:
    # DESIGN.md, Pause and resume (decided on 2026-10-29): a finished job takes the SHA-256 its
    # record holds for a file whose size and modification time (nanoseconds) are the record's,
    # and reads nothing; the time it gives back is the record's. Either differing, the file is
    # not vouched for and is hashed when asked, as before.
    path = model_dir(tmp_path) / VAE
    status = path.stat()
    digest = pins[VAE].sha256
    for size, time_ns in ((status.st_size + 1, status.st_mtime_ns), (status.st_size, 1)):
        refused = pull.Hashes()
        assert not refused.vouch(path, size, time_ns, "0" * 64)
        assert refused.sha256(path) == digest and hashed == [path]
        hashed.clear()
    vouched = pull.Hashes()
    assert vouched.vouch(path, status.st_size, status.st_mtime_ns, digest)
    pull.check_pin(VAE, path, pins[VAE], vouched, None)
    assert vouched.sha256(path) == digest and hashed == []
    assert vouched.mtime_ns(path) == status.st_mtime_ns
    # Held as hashed: a file changed before its load is refused (Hashes.unchanged).
    vouched.unchanged(path)
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns + 10**9))
    with pytest.raises(pull.ModelError, match="changed since its SHA-256 was read"):
        vouched.unchanged(path)
    # A file that is gone is not vouched for.
    assert not pull.Hashes().vouch(tmp_path / "gone", 1, 1, digest)


def test_a_loaded_file_is_hashed_by_its_load(
    tmp_path: Path, pins: dict[str, Pin], hashed: list[Path], caplog: pytest.LogCaptureFixture
) -> None:
    # DESIGN.md, Weights (decided on 2026-11-03): a file the run loads is read once, by its load,
    # which hashes it as it reads it. Its pin's check takes its size, by a stat, and leaves its
    # SHA-256 to the load: nothing read here; the SHA-256 taken for the record is the pin's until
    # the load proves it (Hashes.proved), said then; another SHA-256 read by the load is refused
    # at its end in the words of the pin's refusal, into Hugging Face's cache and into
    # --model-dir alike, the file left as it is.
    caplog.set_level(logging.INFO)
    cache = tmp_path / "hf" / "hub"
    blob = cache / "models--hekmon--seedvr2x" / "blobs" / "0123abcd"
    blob.parent.mkdir(parents=True)
    other = bytes(len(CONTENT[DIT]))
    blob.write_bytes(other)
    link = cache / "models--hekmon--seedvr2x" / "snapshots" / REVISION / DIT
    link.parent.mkdir(parents=True)
    link.symlink_to(Path("..", "..", "blobs", blob.name))
    directory = model_dir(tmp_path)
    for path, where, refusal in (
        (link, cache, from_the_cache(DIT, blob, cache)),
        (directory / DIT, None, f"fetch it again from {url(DIT)}, {INTO_MODEL_DIR}"),
    ):
        hashes = pull.Hashes()
        caplog.clear()
        pull.check_pin(DIT, path, pins[DIT], hashes, where, load=True)
        assert hashed == [] and hashes.known(path)
        assert caplog.messages == [
            f"{DIT}: its size as seedvr2x pins it, {SHORT}; its SHA-256 checked as it loads"
        ]
        assert hashes.sha256(path) == pins[DIT].sha256 and hashed == []
        assert hashes.mtime_ns(path) == path.stat().st_mtime_ns
        found = hashlib.sha256(path.read_bytes()).hexdigest()
        if found == pins[DIT].sha256:
            caplog.clear()
            hashes.proved(path, found, path.stat())
            assert caplog.messages == [
                f"{DIT}: its size and SHA-256 as seedvr2x pins them, {SHORT}"
            ]
            hashes.proved(path, found, path.stat())  # said once
            assert len(caplog.messages) == 1
            continue
        with pytest.raises(ModelError) as error:
            hashes.proved(path, found, path.stat())
        assert str(error.value) == (
            f"{path}: SHA-256 {found}, where hekmon/seedvr2x's {DIT} at {REVISION[:8]} has"
            f" {pins[DIT].sha256}: not the file seedvr2x pins, left as it is; {refusal}"
        )
        assert blob.read_bytes() == other
    # The size still refused first, by a stat, before any load.
    (directory / DIT).write_bytes(CONTENT[DIT] + b"!")
    with pytest.raises(ModelError, match=r"bytes, where hekmon/seedvr2x's"):
        pull.check_pin(DIT, directory / DIT, pins[DIT], pull.Hashes(), None, load=True)
    assert hashed == []


def test_a_sha256_known_before_the_load_is_checked_at_once_and_proved_by_it(
    tmp_path: Path, pins: dict[str, Pin], hashed: list[Path], caplog: pytest.LogCaptureFixture
) -> None:
    # A SHA-256 the run has before the load, from a record's size and time or a whole read at the
    # start, is held to the pin at once (check_pin, load or not), as before; the load then
    # proves it, and refuses a file whose bytes changed since, in the words of the way it came: a
    # record's for a file seedvr2x doesn't pin (with a refusal of the caller's, or the library's
    # own), the pin's for one it pins, which say how to fetch it again (DESIGN.md, Pause and
    # resume, decided on 2026-11-03), a whole read's.
    caplog.set_level(logging.INFO)
    path = model_dir(tmp_path) / VAE
    status = path.stat()
    vouched = pull.Hashes()
    assert vouched.vouch(path, status.st_size, status.st_mtime_ns, "0" * 64)
    with pytest.raises(ModelError, match=f"SHA-256 {'0' * 64}, where hekmon"):
        pull.check_pin(VAE, path, pins[VAE], vouched, None, load=True)
    assert hashed == []
    digest = pins[VAE].sha256
    for words in (None, lambda found: f"the caller's words, {found}"):
        recorded = pull.Hashes()
        assert recorded.vouch(path, status.st_size, status.st_mtime_ns, digest, words)
        with pytest.raises(ModelError) as error:
            recorded.proved(path, "1" * 64, status)
        assert str(error.value) == (
            f"{path}: SHA-256 {'1' * 64} as it loaded, where the record has {digest} for it,"
            " with the size and modification time it has: changed since, its size and time kept"
            if words is None
            else f"the caller's words, {'1' * 64}"
        )
        recorded.proved(path, digest, status)
    pinned = pull.Hashes()
    assert pinned.vouch(path, status.st_size, status.st_mtime_ns, digest, lambda found: "record")
    pull.check_pin(VAE, path, pins[VAE], pinned, None, load=True)
    assert hashed == []
    with pytest.raises(ModelError) as error:
        pinned.proved(path, "1" * 64, status)
    assert str(error.value) == (
        f"{path}: SHA-256 {'1' * 64}, where hekmon/seedvr2x's {VAE} at {REVISION[:8]} has"
        f" {digest}: not the file seedvr2x pins, left as it is; fetch it again from {url(VAE)},"
        f" {INTO_MODEL_DIR}"
    )
    # With `whole`, a SHA-256 nothing vouches for is read at once, its size passed first: a job
    # recorded whose record no longer vouches for the file.
    (tmp_path / "long").write_bytes(CONTENT[VAE] + b"!")
    with pytest.raises(ModelError, match="bytes, where hekmon"):
        pull.check_pin(
            VAE, tmp_path / "long", pins[VAE], pull.Hashes(), None, load=True, whole=True
        )
    assert hashed == []
    whole = pull.Hashes()
    pull.check_pin(VAE, path, pins[VAE], whole, None, load=True, whole=True)
    assert hashed == [path] and whole.proven(path)
    hashed.clear()
    read = pull.Hashes()
    pull.check_pin(VAE, path, pins[VAE], read, None)
    assert hashed == [path]
    with pytest.raises(ModelError) as error:
        read.proved(path, "1" * 64, status)
    assert str(error.value) == (
        f"{path}: changed since its SHA-256 was read, before its load ended: run again, which"
        " checks it again"
    )
    caplog.clear()
    read.proved(path, digest, status)  # proved already by the whole read: nothing said
    assert caplog.messages == []


def test_a_file_unknown_before_its_load_takes_the_load_s_sha256(
    tmp_path: Path, hashed: list[Path], caplog: pytest.LogCaptureFixture
) -> None:
    # A file seedvr2x doesn't pin, on a run that needs its SHA-256 for a record made after the
    # load (a picture's, the image server's): the load's, said once, with the file as the load
    # opened it; nothing read besides.
    caplog.set_level(logging.INFO)
    path = tmp_path / "numz.safetensors"
    path.write_bytes(b"numz's bytes")
    hashes = pull.Hashes()
    assert not hashes.known(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes.proved(path, digest, path.stat())
    assert hashes.sha256(path) == digest and hashed == []
    assert hashes.mtime_ns(path) == path.stat().st_mtime_ns
    assert caplog.messages == [f"numz.safetensors: SHA-256 {digest}, read as it loaded"]
    hashes.unchanged(path)
    os.utime(path, ns=(0, path.stat().st_mtime_ns + 1))
    with pytest.raises(ModelError, match="changed since its SHA-256 was read"):
        hashes.unchanged(path)


def test_a_long_hash_is_named_before_it_starts(
    tmp_path: Path,
    pins: dict[str, Pin],
    hashed: list[Path],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # DESIGN.md, Architecture, What it prints (decided on 2026-10-28, built on 2026-10-29): the
    # silences of a start are named. A hash that takes about HASH_SAID_FROM seconds or more,
    # at pull.HASH_SECONDS_PER_BYTE (the DiT's 84 s for its 16.5 GB, 2026-10-29), is announced by
    # a line saying what is read and for about how long, before the read; a short one is not.
    caplog.set_level(logging.INFO)
    directory = model_dir(tmp_path)
    hashes = pull.Hashes()
    hashes.sha256(directory / VAE)
    assert not [m for m in caplog.messages if "to check its SHA-256" in m]
    # The rate itself: 84 s for the real DiT's size, at the real pin.
    assert pull.HASH_SECONDS_PER_BYTE * 16_479_335_080 == pytest.approx(84.0)
    size = (directory / DIT).stat().st_size
    monkeypatch.setattr(pull, "HASH_SECONDS_PER_BYTE", 84.0 / size)
    caplog.clear()
    assert hashes.sha256(directory / DIT) == pins[DIT].sha256
    said = [m for m in caplog.messages if "to check its SHA-256" in m]
    assert said == [
        f"{DIT}: reading its {size / 1024:.0f} KiB to check its SHA-256, about 1 min 24 s"
    ]
    # Before the line that gives the hash and the time it took.
    assert caplog.messages.index(said[0]) < next(
        i for i, m in enumerate(caplog.messages) if m.startswith(f"{DIT}: SHA-256 ")
    )
    # Read once: the second call says nothing.
    caplog.clear()
    hashes.sha256(directory / DIT)
    assert not caplog.messages


# What --dit-model takes without --model-dir, as a refusal lists them: seedvr2x's own DiTs.
DITS = (
    "seedvr2x_ema_7b_fp16.safetensors, seedvr2x_ema_7b_sharp_fp16.safetensors,"
    " seedvr2x_ema_7b_int8_convrot.safetensors or seedvr2x_ema_7b_sharp_int8_convrot.safetensors"
)


def test_pinned_name_in_the_other_role_refused(hub: Hub) -> None:
    # One of seedvr2x's own files given in the other role: refused by its name, before any lookup
    # or download (16.5 GB for the 7B as the VAE, which the header check would then refuse),
    # saying which option takes it and what this one takes.
    with pytest.raises(ModelError) as error:
        pull.resolve(None, "seedvr2x_ema_vae_fp16.safetensors", "seedvr2x_ema_7b_fp16.safetensors")
    assert str(error.value).splitlines() == [
        "--dit-model seedvr2x_ema_vae_fp16.safetensors: seedvr2x's VAE, which --vae-model takes"
        f" (--dit-model: {DITS})",
        "--vae-model seedvr2x_ema_7b_fp16.safetensors: one of seedvr2x's DiTs, which --dit-model"
        " takes (--vae-model: seedvr2x_ema_vae_fp16.safetensors)",
    ]
    assert hub.calls == hub.lookups == []


# A small file in place of TransNetV2's weights, under the shot detector's name (detector_pin).
DETECTOR_CONTENT = b"TransNetV2's bytes " * 30


@pytest.fixture
def detector_pin(pins: dict[str, Pin], hub: Hub) -> Pin:
    """seedvr2x's pins, the small files' in their place (pins), the shot detector's with them, in
    its role, the Hub serving it."""
    pin = Pin(len(DETECTOR_CONTENT), hashlib.sha256(DETECTOR_CONTENT).hexdigest(), "detector")
    pins[pull.DETECTOR] = pin  # pull.PINNED itself, until the test ends (pins)
    hub.served[pull.DETECTOR] = DETECTOR_CONTENT
    return pin


def test_detector_pulled_with_the_others(
    hub: Hub, detector_pin: Pin, caplog: pytest.LogCaptureFixture
) -> None:
    # With the shot detector, its file comes with the DiT's and the VAE's, last, in the same
    # fetch: asked of the library at the pinned revision with no token, then found in the cache;
    # checked by its pin, how to fetch it again said for its role. Without, never looked up.
    caplog.set_level(logging.INFO)
    found = pull.resolve(None, DIT, VAE, detector=True)
    assert [call["filename"] for call in hub.calls] == [DIT, VAE, pull.DETECTOR]
    assert hub.calls[-1] == {
        "repo_id": "hekmon/seedvr2x",
        "filename": pull.DETECTOR,
        "revision": REVISION,
        "cache_dir": hub.cache,
        "token": False,
    }
    snapshot = hub.snapshot(pull.DETECTOR)
    assert found.detector == pull.ModelFile(pull.DETECTOR, snapshot)
    assert found.roles() == [("dit", found.dit), ("vae", found.vae), ("detector", found.detector)]
    size = f"{detector_pin.size:,} bytes"
    assert f"{pull.DETECTOR}: not in the cache, downloading {size} from {SHORT}" in caplog.messages
    pull.check_pinned(found)
    said = f"{pull.DETECTOR}: its size and SHA-256 as seedvr2x pins them, {SHORT}"
    assert said in caplog.messages
    assert pull.advice(found)["detector"] == from_the_cache(pull.DETECTOR, snapshot, hub.cache)
    hub.calls.clear()
    assert pull.resolve(None, DIT, VAE, detector=True).detector == found.detector
    assert hub.calls == []
    hub.lookups.clear()
    without = pull.resolve(None, DIT, VAE)
    assert without.detector is None and hub.lookups == [DIT, VAE]
    assert without.roles() == [("dit", without.dit), ("vae", without.vae)]
    assert "detector" not in pull.advice(without)


def test_detector_counted_in_the_space(
    monkeypatch: pytest.MonkeyPatch, hub: Hub, pins: dict[str, Pin], detector_pin: Pin
) -> None:
    # The free space checked before the first download counts the shot detector's file with the
    # others': a byte short of the three, refused before any; enough, the three downloaded.
    free = [pins[DIT].size + pins[VAE].size + detector_pin.size - 1]
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=free[0]))
    with pytest.raises(ModelError) as error:
        pull.resolve(None, DIT, VAE, detector=True)
    assert str(error.value) == (
        f"Hugging Face's cache, {hub.cache}: {free[0]:,} bytes free on its disk, where downloading"
        f" {DIT}, {VAE} and {pull.DETECTOR} needs {free[0] + 1:,} bytes: free some space there,"
        f" set HF_HOME (or HF_HUB_CACHE) to a directory on a disk with room, {ELSEWHERE}"
    )
    assert hub.calls == []
    free[0] += 1
    pull.resolve(None, DIT, VAE, detector=True)
    assert [call["filename"] for call in hub.calls] == [DIT, VAE, pull.DETECTOR]


def test_detector_from_model_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detector_pin: Pin, hashed: list[Path]
) -> None:
    # With --model-dir, the shot detector's file is read there by its name, the library never
    # called: refused by its pin, saying how to fetch it again into --model-dir, and left as it
    # is; as pinned, accepted, hashed once with the others.
    def called(*args: object, **kwargs: object) -> None:
        pytest.fail("huggingface_hub called")

    monkeypatch.setattr(file_download, "try_to_load_from_cache", called)
    monkeypatch.setattr(file_download, "hf_hub_download", called)
    directory = model_dir(tmp_path)
    path = directory / pull.DETECTOR
    path.write_bytes(DETECTOR_CONTENT + b"!")
    found = pull.resolve(directory, DIT, VAE, detector=True)
    assert found.detector == pull.ModelFile(pull.DETECTOR, path) and found.cache is None
    with pytest.raises(ModelError) as error:
        pull.check_pinned(found)
    assert str(error.value) == (
        f"{path}: {detector_pin.size + 1:,} bytes, where hekmon/seedvr2x's {pull.DETECTOR} at"
        f" {REVISION[:8]} has {detector_pin.size:,}: not the file seedvr2x pins, left as it is;"
        f" fetch it again from {url(pull.DETECTOR)}, {INTO_MODEL_DIR}"
    )
    assert path.read_bytes() == DETECTOR_CONTENT + b"!"
    path.write_bytes(DETECTOR_CONTENT)
    hashed.clear()
    found = pull.resolve(directory, DIT, VAE, detector=True)
    pull.check_pinned(found)
    assert hashed == [directory / DIT, directory / VAE, path]
    assert found.hashes.sha256(path) == detector_pin.sha256


@pytest.mark.parametrize("where", ["cache", "model-dir"])
def test_detector_header_refusal_says_how_to_fetch_again(
    tmp_path: Path, hub: Hub, detector_pin: Pin, where: str
) -> None:
    # The header check of the shot detector's file, one of seedvr2x's own in its own role: its
    # refusal says how to fetch it again, as the pin's does, from the cache or into --model-dir.
    directory = model_dir(tmp_path) if where == "model-dir" else None
    if directory is not None:
        (directory / pull.DETECTOR).write_bytes(DETECTOR_CONTENT)
    found = pull.resolve(directory, DIT, VAE, detector=True)
    assert found.detector is not None
    with pytest.raises(ModelError) as error:
        weights.check_models(
            found.dit.path, found.vae.path, pull.advice(found), found.detector.path
        )
    path = found.detector.path
    again = (
        from_the_cache(pull.DETECTOR, path, hub.cache)
        if directory is None
        else f"fetch it again from {url(pull.DETECTOR)}, {INTO_MODEL_DIR}"
    )
    lines = str(error.value).splitlines()
    # The cache's command is on a line of its own (DESIGN.md, What it prints, 2026-10-29): the
    # refusal takes two lines there, the second the command.
    assert len(lines) == (6 if directory is None else 3)
    assert "\n".join(lines[-1 - (directory is None) :]) == (
        f"{path}: neither safetensors, GGUF nor a PyTorch checkpoint: {SHOTS}; {again}"
    )


def test_detector_missing_from_model_dir_said_for_what_it_is(
    tmp_path: Path, detector_pin: Pin
) -> None:
    # --model-dir without the shot detector's file, which no option names: the refusal says what
    # the file is, that a cut list runs without it, and how to fetch it, never "again" for a
    # file that was never there.
    directory = model_dir(tmp_path)
    found = pull.resolve(directory, DIT, VAE, detector=True)
    assert found.detector is not None
    with pytest.raises(ModelError) as error:
        weights.check_models(
            found.dit.path, found.vae.path, pull.advice(found), found.detector.path
        )
    assert str(error.value).splitlines()[-1] == (
        f"{directory / pull.DETECTOR}: no such model file: the shot detector's weights,"
        " TransNetV2's, which a run that detects its shots reads in --model-dir (--cuts, a cut"
        f" list, runs without them); fetch it from {url(pull.DETECTOR)}, {INTO_MODEL_DIR}"
    )
    # Without the advice (a name of the user's own): what it is, all the same.
    with pytest.raises(ModelError) as error:
        weights.check_models(found.dit.path, found.vae.path, None, found.detector.path)
    assert str(error.value).splitlines()[-1].endswith("(--cuts, a cut list, runs without them)")


def test_detector_named_in_another_role_refused(hub: Hub) -> None:
    # TransNetV2's file given as the DiT or the VAE: refused by its name, before any lookup or
    # download, saying what takes it.
    with pytest.raises(ModelError) as error:
        pull.resolve(None, pull.DETECTOR, pull.DETECTOR)
    assert str(error.value).splitlines() == [
        "--dit-model transnetv2.safetensors: TransNetV2's weights, which seedvr2x's shot detection"
        f" takes (--dit-model: {DITS})",
        "--vae-model transnetv2.safetensors: TransNetV2's weights, which seedvr2x's shot detection"
        " takes (--vae-model: seedvr2x_ema_vae_fp16.safetensors)",
    ]
    assert hub.calls == hub.lookups == []


def unreachable() -> LocalEntryNotFoundError:
    """The library's error when the Hub can't be reached, the network's own as its cause."""
    error = LocalEntryNotFoundError("An error happened while trying to locate the file on the Hub")
    error.__cause__ = httpx.ConnectError("[Errno -3] Temporary failure in name resolution")
    return error


def no_commit(endpoint: str) -> str:
    """The library's words when the file's HEAD request at endpoint is answered without the
    commit's header (file_download.py:1789-1795), as a server on 127.0.0.1 answering 200 got them
    on 2026-10-09."""
    return (
        f"Response from {endpoint}/hekmon/seedvr2x/resolve/{REVISION}/{DIT} is missing the"
        " 'X-Repo-Commit' header, so it does not seem to be served by a Hugging Face Hub endpoint."
        " If HF_ENDPOINT is set, check that it points to a Hub-compatible endpoint. Otherwise,"
        " check your firewall and proxy settings and make sure your SSL certificates are updated."
    )


def not_the_hub(endpoint: str = "https://huggingface.co") -> LocalEntryNotFoundError:
    """The library's error when the file's HEAD request is answered without the Hub's headers, as
    by an HF_ENDPOINT that isn't a Hub, a proxy or a captive portal: its FileMetadataError as its
    cause (file_download.py:1954-1958)."""
    cause = FileMetadataError(no_commit(endpoint))
    error = LocalEntryNotFoundError(
        f"{cause} We also cannot find the requested files in the local cache."
    )
    error.__cause__ = cause
    return error


@pytest.mark.usefixtures("pins")
def test_transformers_offline_named(monkeypatch: pytest.MonkeyPatch, hub: Hub) -> None:
    # TRANSFORMERS_OFFLINE sets huggingface_hub offline when HF_HUB_OFFLINE is unset
    # (constants.py:194): the refusal names the variable set.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    with pytest.raises(ModelError) as error:
        pull.resolve(None, DIT, VAE)
    assert str(error.value) == (
        f"{DIT} and {VAE}: not in Hugging Face's cache, {hub.cache}, and TRANSFORMERS_OFFLINE,"
        " which huggingface_hub takes for HF_HUB_OFFLINE, keeps the network out: unset it to"
        f" download 1,040 bytes, set HF_HOME (or HF_HUB_CACHE) to a cache holding them,"
        f" {ELSEWHERE}"
    )
    assert hub.calls == []


@pytest.mark.usefixtures("pins")
@pytest.mark.parametrize("kind", ["unreachable", "not-the-hub"])
def test_refusal_names_the_endpoint(monkeypatch: pytest.MonkeyPatch, hub: Hub, kind: str) -> None:
    # HF_ENDPOINT elsewhere: the refusal names the endpoint the library asked, which couldn't be
    # reached or answered without the Hub's headers.
    mirror = "https://hub.mirror.example"
    monkeypatch.setattr(constants, "ENDPOINT", mirror)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=10**12))
    hub.error = unreachable() if kind == "unreachable" else not_the_hub(mirror)
    with pytest.raises(ModelError) as error:
        pull.resolve(None, DIT, VAE)
    if kind == "unreachable":
        said = (
            f"{DIT}: not in the cache, and {mirror} couldn't be reached to download it ([Errno -3]"
            f" Temporary failure in name resolution): run again once it can be, {ELSEWHERE}"
        )
    else:
        said = (
            f"{DIT}: not in the cache, and {mirror} answered the request to download it without"
            f" the Hub's headers ({no_commit(mirror)}): check HF_ENDPOINT, if set, and any proxy on"
            f" the way to the Hub, {ELSEWHERE}"
        )
    assert str(error.value) == said


# hf_xet's two errors for a file its storage doesn't have or won't give, raised by the library's
# download, in a process of its own whose HF_HOME is the test's: importing hf_xet writes a log
# under it. The library's two calls are replaced, its cache argv[1]; the error, argv[2].
XET_REFUSED = """
import shutil
import sys
from types import SimpleNamespace

import hf_xet
from huggingface_hub import constants, file_download

from seedvr2x import pull
from seedvr2x.weights import ModelError

errors = {
    "not-found": hf_xet.XetObjectNotFoundError("Not found: the file's object"),
    "authentication": hf_xet.XetAuthenticationError("Authentication error: the token refused"),
}


def download(repo_id, filename, **options):
    raise errors[sys.argv[2]]


constants.HF_HUB_CACHE = sys.argv[1]
constants.HF_HUB_OFFLINE = False  # the library faked: nothing reaches the network
file_download.try_to_load_from_cache = lambda *args, **kwargs: None
file_download.hf_hub_download = download
shutil.disk_usage = lambda path: SimpleNamespace(free=10**12)
pull.PINNED = {
    "small_dit.safetensors": pull.Pin(640, "0" * 64, "dit"),
    "small_vae.safetensors": pull.Pin(400, "1" * 64, "vae"),
}
try:
    pull.resolve(None, "small_dit.safetensors", "small_vae.safetensors")
except ModelError as error:
    print(error)
"""


@pytest.mark.parametrize(
    ("kind", "said"),
    [
        ("not-found", "Not found: the file's object"),
        ("authentication", "Authentication error: the token refused"),
    ],
)
def test_xet_refusals(tmp_path: Path, kind: str, said: str) -> None:
    # Neither changes with a rerun: the repository or the file unavailable, --model-dir offered.
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_XET_CACHE")
    }
    environment |= {"HF_HOME": str(tmp_path / "home"), "HF_HUB_OFFLINE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", XET_REFUSED, str(tmp_path / "hf" / "hub"), kind],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    refused = f"small_dit.safetensors: the Hub's storage refused it ({said}): {UNAVAILABLE}"
    assert result.stdout == f"{refused}\n"


# An OSError of the library's download, the cache unwritable, in a process of its own whose HF_HOME
# is the test's: its refusal looks hf_xet's errors up first (pull._xet_refused), among the modules
# imported. The library's two calls are replaced, its cache argv[1].
XET_UNIMPORTED = """
import errno
import shutil
import sys
from types import SimpleNamespace

from huggingface_hub import constants, file_download

from seedvr2x import pull
from seedvr2x.weights import ModelError


def download(repo_id, filename, **options):
    raise PermissionError(errno.EACCES, "Permission denied", sys.argv[1])


constants.HF_HUB_CACHE = sys.argv[1]
constants.HF_HUB_OFFLINE = False  # the library faked: nothing reaches the network
file_download.try_to_load_from_cache = lambda *args, **kwargs: None
file_download.hf_hub_download = download
shutil.disk_usage = lambda path: SimpleNamespace(free=10**12)
pull.PINNED = {
    "small_dit.safetensors": pull.Pin(640, "0" * 64, "dit"),
    "small_vae.safetensors": pull.Pin(400, "1" * 64, "vae"),
}
assert "hf_xet" not in sys.modules, "imported already"
try:
    pull.resolve(None, "small_dit.safetensors", "small_vae.safetensors")
except ModelError as error:
    print(error)
print("hf_xet imported" if "hf_xet" in sys.modules else "hf_xet not imported")
"""


def test_hf_xet_never_imported(tmp_path: Path) -> None:
    # The refusal of an OSError, through the lookup of hf_xet's errors: hf_xet left unimported,
    # whose import writes a log under HF_HOME (its xet/logs/), as seedvr2x promises.
    hidden = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_XET_CACHE")
    environment = {name: value for name, value in os.environ.items() if name not in hidden}
    home = tmp_path / "home"
    environment |= {"HF_HOME": str(home), "HF_HUB_OFFLINE": "1"}
    cache = tmp_path / "hf" / "hub"
    result = subprocess.run(
        [sys.executable, "-c", XET_UNIMPORTED, str(cache)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    refused = (
        f"small_dit.safetensors: Hugging Face's cache, {cache}, can't be written (Permission"
        " denied): set HF_HOME (or HF_HUB_CACHE) to a directory you can write, on a disk with"
        f" 1,040 bytes free, {ELSEWHERE}"
    )
    assert result.stdout == f"{refused}\nhf_xet not imported\n"
    assert not (home / "xet").exists()


# seedvr2x's handling of the library's log, in a fresh interpreter whose HF_HOME is the test's,
# before seedvr2x has imported anything of the library, as a run of the command line starts: the
# library's handler, which the import of its logging module adds, gone all the same.
LIBRARY_HANDLER = """
import logging
import sys

from seedvr2x import pull

assert "huggingface_hub.utils.logging" not in sys.modules, "imported already"
pull._library_logging()
from huggingface_hub import file_download  # what the cache's lookup imports next

print(len(logging.getLogger("huggingface_hub").handlers))
"""


def test_library_handler_removed_before_its_import(tmp_path: Path) -> None:
    environment = {**os.environ, "HF_HOME": str(tmp_path / "home"), "HF_HUB_OFFLINE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", LIBRARY_HANDLER],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "0\n"


def test_changed_after_its_hash_refused(tmp_path: Path) -> None:
    # A file whose stat changed between its hash and its load: another size, modification time
    # or inode, its bytes the same or not.
    path = tmp_path / "w.safetensors"
    path.write_bytes(b"hashed")
    hashes = pull.Hashes()
    hashes.sha256(path)
    hashes.unchanged(path)
    hashes.unchanged(tmp_path / "never hashed")
    state = path.stat()

    def touched() -> None:
        os.utime(path, ns=(state.st_atime_ns, state.st_mtime_ns + 1))

    def replaced() -> None:
        copy = tmp_path / "copy"
        copy.write_bytes(b"hashed")
        os.utime(copy, ns=(state.st_atime_ns, state.st_mtime_ns))
        copy.replace(path)

    def grown() -> None:
        with path.open("ab") as file:
            file.write(b"!")
        os.utime(path, ns=(state.st_atime_ns, state.st_mtime_ns))

    for change in (touched, replaced, grown):
        change()
        with pytest.raises(ModelError) as error:
            hashes.unchanged(path)
        assert str(error.value) == (
            f"{path}: changed since its SHA-256 was read, before its load ended: run again, which"
            " checks it again"
        )
        path.write_bytes(b"hashed")
        os.utime(path, ns=(state.st_atime_ns, state.st_mtime_ns))
        hashes = pull.Hashes()
        hashes.sha256(path)


@pytest.mark.parametrize(
    ("touched", "when", "loaded"),
    [
        ("dit", "before", []),
        ("vae", "before", []),
        ("vae", "during", ["vae"]),
        ("dit", "during", ["vae", "dit"]),
    ],
)
def test_load_refuses_a_file_changed_since_its_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, touched: str, when: str, loaded: list[str]
) -> None:
    # load_models takes the paths found (pull.resolve) and checks each against its hash before
    # building anything, and again right after its own load (the VAE's first): a file touched
    # before is refused with nothing loaded, one touched during its load once loaded, before the
    # next.
    import torch

    dit = write(tmp_path / "seedvr2x_ema_7b_fp16.safetensors", as_dtype("7b", "F16"))
    vae = write(tmp_path / "seedvr2x_ema_vae_fp16.safetensors", as_dtype("vae", "F16"))
    paths = {"dit": dit, "vae": vae}
    monkeypatch.setattr(files, "sha256", lambda path: "0" * 64)  # the 0.5 GB of holes unread
    hashes = pull.Hashes()
    for path in paths.values():
        hashes.sha256(path)

    def touch(path: Path) -> None:
        os.utime(path, ns=(0, path.stat().st_mtime_ns + 1))

    done: list[str] = []

    def load_weights(
        module: object, path: Path, *args: object, **more: object
    ) -> dict[str, object]:
        role = "dit" if path == dit else "vae"
        done.append(role)
        if when == "during" and role == touched:
            touch(path)
        return {}  # no part away from the device (model._load_weights)

    monkeypatch.setattr(model, "_load_weights", load_weights)
    if when == "before":
        touch(paths[touched])
    with pytest.raises(ModelError) as error:
        model.load_models(dit, vae, torch.device("cpu"), hashes)
    assert str(error.value) == (
        f"{paths[touched]}: changed since its SHA-256 was read, before its load ended: run again,"
        " which checks it again"
    )
    assert done == loaded


@pytest.mark.parametrize("wrong", ["vae", "dit"])
def test_load_refuses_a_file_whose_bytes_are_not_those_expected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wrong: str
) -> None:
    # load_models hands each file's load the Hashes' proof (model._load_weights's hashed): a file
    # whose bytes, read by its load, aren't the SHA-256 the run took (here a pin's) is refused at
    # the end of its own load, the VAE's first, before the DiT is read or anything is made.
    import torch

    dit = write(tmp_path / "seedvr2x_ema_7b_fp16.safetensors", as_dtype("7b", "F16"))
    vae = write(tmp_path / "seedvr2x_ema_vae_fp16.safetensors", as_dtype("vae", "F16"))
    paths = {"dit": dit, "vae": vae}
    hashes = pull.Hashes()
    for role, path in paths.items():
        hashes.expect(path, f"{role} pinned", lambda found, role=role: f"{role}: not {found}", "ok")
    done: list[str] = []

    def load_weights(module: object, path: Path, *args: object, **more: Any) -> dict[str, object]:
        role = "dit" if path == dit else "vae"
        done.append(role)
        more["hashed"]("another" if role == wrong else f"{role} pinned", path.stat())
        return {}  # no part away from the device (model._load_weights)

    monkeypatch.setattr(model, "_load_weights", load_weights)
    with pytest.raises(ModelError) as error:
        model.load_models(dit, vae, torch.device("cpu"), hashes)
    assert str(error.value) == f"{wrong}: not another"
    assert done == (["vae"] if wrong == "vae" else ["vae", "dit"])


def _offline() -> str | None:
    """Why the network can't be had for the tests reading the revision, if it can't."""
    if constants.HF_HUB_OFFLINE:
        return "HF_HUB_OFFLINE is set"
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
    except OSError as error:
        return f"no network to huggingface.co: {error}"
    return None


# SHA256SUMS fetched at the pinned revision by seedvr2x's own fetch and checked by its own pin, in
# a process of its own, whose HF_HOME is the test's; then the pinned files' sizes, from the Hub's
# API, with no token.
REVISION_READ = """
import json
import logging
import sys

from seedvr2x import pull

logging.basicConfig(level=logging.INFO)
found = pull.fetch({pull.SUMS: pull.SUMS_PIN})[pull.SUMS]
pull.check_pin(pull.SUMS, found, pull.SUMS_PIN, pull.Hashes(), pull.cache())
torch = "torch" in sys.modules
from huggingface_hub import HfApi

infos = HfApi(token=False).get_paths_info(pull.REPO, list(pull.PINNED), revision=pull.REVISION)
sizes = {info.path: info.size for info in infos}
print(json.dumps({"path": str(found), "cache": str(pull.cache()), "torch": torch, "sizes": sizes}))
"""


def test_pins_are_the_revisions(tmp_path: Path) -> None:
    # Into an HF_HOME behind a link, which the library resolves (file_download.py:999): paths of
    # the library and of seedvr2x compared resolved. The Hub's advice to set a token, which it
    # sends answering a request without one, never in the log, nor httpx's requests.
    reason = _offline()
    if reason is not None:
        pytest.skip(reason)
    (tmp_path / "hf-target").mkdir()
    home = tmp_path / "hf"
    home.symlink_to(tmp_path / "hf-target")
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HUB_OFFLINE")
    }
    result = subprocess.run(
        [sys.executable, "-c", REVISION_READ],
        capture_output=True,
        text=True,
        check=False,
        env={**environment, "HF_HOME": str(home)},
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    found = json.loads(result.stdout.splitlines()[-1])
    path = Path(found["path"])
    assert found["cache"] == str(home / "hub")
    assert path.resolve().is_relative_to((home / "hub").resolve())
    assert f"{SHORT}, in Hugging Face's cache, {home / 'hub'}" in result.stderr
    assert "SHA256SUMS: not in the cache, downloading 2,335 bytes from" in result.stderr
    assert NO_TOKEN in result.stderr
    assert pull.TOKEN_ADVICE not in result.stderr and "HTTP Request" not in result.stderr
    assert not found["torch"]
    lines = (line.split("  ", 1) for line in path.read_text().splitlines())
    sums = {name: digest for digest, name in lines}
    for name, pin in pull.PINNED.items():
        assert sums[name] == pin.sha256, name
        assert found["sizes"][name] == pin.size, name


# The four v1 files the opt-in pull fetches: the int8 files, 16.7 GB more, not among them.
V1_FILES = (
    "seedvr2x_ema_7b_fp16.safetensors",
    "seedvr2x_ema_7b_sharp_fp16.safetensors",
    "seedvr2x_ema_vae_fp16.safetensors",
    pull.DETECTOR,
)


@pytest.mark.skipif(
    os.environ.get("SEEDVR2X_TESTS_PULL") != "1",
    reason="SEEDVR2X_TESTS_PULL=1 pulls seedvr2x's v1 files into Hugging Face's cache, about 33 GB"
    " the first time",
)
def test_pull_v1_files(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    # The four v1 files through seedvr2x's own fetch into the cache HF_HOME names, each checked
    # by its pin; the cache read only, once they are there. The library's paths are resolved
    # (file_download.py:999), the cache's as seedvr2x names it maybe not: compared resolved.
    caplog.set_level(logging.INFO)
    pins = {name: pin for name, pin in pull.PINNED.items() if name in V1_FILES}
    found = pull.fetch(pins)
    hashes = pull.Hashes()
    for name, pin in pins.items():
        assert found[name].resolve().is_relative_to(pull.cache().resolve())
        pull.check_pin(name, found[name], pin, hashes, pull.cache())
    with capsys.disabled():
        print("\n" + "\n".join(caplog.messages))
