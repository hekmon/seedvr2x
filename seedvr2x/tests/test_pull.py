"""seedvr2x's own model files, pulled from its Hugging Face repo at a pinned revision, each checked
by its pinned size and SHA-256 (runtime/pull.py; DESIGN.md, Weights).

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
writes a log file under HF_HOME, and one of them holds seedvr2x to never importing it."""

import errno
import hashlib
import json
import logging
import os
import re
import shutil
import signal
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
    DisabledRepoError,
    FileMetadataError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from test_cli_run import STAND_IN_MODELS, source, stand_in_model
from test_weights import SHOTS, V1, as_dtype, write

from seedvr2x import cli
from seedvr2x.media import ffmpeg, files
from seedvr2x.runtime import model, pull, weights
from seedvr2x.runtime.pull import Pin
from seedvr2x.runtime.weights import ModelError

TESTS = Path(__file__).resolve().parent
REVISION = "c14a2bc4aab04cf38ad9b0d324014c4213f07048"
# Another revision's commit, as a download of the repo's head (hf download without --revision)
# leaves the same files under it.
OTHER = "0123456789abcdef0123456789abcdef01234567"
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
# The real check and lookup, which tests replace (stand_in_model, Hub).
CHECK_MODELS = weights.check_models
LOOKUP = file_download.try_to_load_from_cache

# Two small files in place of seedvr2x's own (pins), as the DiT and the VAE: the stand-in's model
# check passes them (stand_in_model).
DIT, VAE = "small_dit.safetensors", "small_vae.safetensors"
CONTENT = {DIT: b"the DiT's bytes " * 40, VAE: b"the VAE's bytes " * 25}


def _usable() -> bool:
    try:
        ffmpeg.check()
    except ffmpeg.MediaError:
        return False
    return True


needs_ffmpeg = pytest.mark.skipif(
    not _usable(), reason="needs ffmpeg 7.1 or later with zscale, idet and ffv1"
)


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


@pytest.fixture
def stand_in(monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in_model(monkeypatch.setattr)


def upscale(tmp_path: Path, output: str, *options: str) -> int:
    """seedvr2x run in this process on test_cli_run.py's source, the model's stand-in's frames
    written as stored."""
    input_path = tmp_path / "in.mkv"
    if not input_path.exists():
        source(input_path)
    return cli.main(
        [
            *(str(input_path), "-o", str(tmp_path / output), "--resolution", "96"),
            *("--seed", "42", "--color-correction", "none", "--format", "gbrp16le", *options),
        ]
    )


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
        f" huggingface_hub's command: {command}"
    )


INTO_MODEL_DIR = "or leave out --model-dir, which takes it from Hugging Face's cache"


def test_pins_recorded() -> None:
    # The model conversation's record of the upload (2026-10-08); test_pins_are_the_revisions
    # holds them to the revision itself. Each in its role.
    assert (pull.REPO, pull.REVISION) == ("hekmon/seedvr2x", REVISION)
    assert {name: (pin.size, pin.sha256[:8], pin.role) for name, pin in pull.PINNED.items()} == {
        "seedvr2x_ema_7b_fp16.safetensors": (16_479_335_080, "071cab5e", "dit"),
        "seedvr2x_ema_7b_sharp_fp16.safetensors": (16_479_335_088, "5eb47fde", "dit"),
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


def test_help(capsys: pytest.CaptureFixture[str]) -> None:
    # --model-dir optional, and where the files come from without it.
    with pytest.raises(SystemExit) as exited:
        cli.main(["--help"])
    assert exited.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "[--model-dir MODEL_DIR]" in text
    assert (
        "a directory holding the model files, each read there by its name; nothing is downloaded."
        in text
    )
    assert (
        "taken from Hugging Face's cache (HF_HOME moves it), downloaded into it from"
        " hekmon/seedvr2x at a pinned revision when missing (HF_HUB_OFFLINE keeps the network"
        " out). Either way, seedvr2x's own files are checked by their pinned size and SHA-256"
        in text
    )
    assert "without --model-dir, seedvr2x_ema_7b_fp16.safetensors or the default" in text


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


@needs_ffmpeg
@pytest.mark.usefixtures("stand_in")
@pytest.mark.parametrize("where", ["model-dir", "cache"])
def test_hashed_once_a_run(
    tmp_path: Path, hub: Hub, pins: dict[str, Pin], hashed: list[Path], where: str
) -> None:
    # The pinned check reads each file of seedvr2x's own, and the manifest's record takes its hash:
    # one read per file a run, whether --model-dir holds it or the cache.
    if where == "model-dir":
        options = ["--model-dir", str(model_dir(tmp_path))]
    else:
        options = []
    assert upscale(tmp_path, "out", *options, "--dit-model", DIT, "--vae-model", VAE) == 0
    paths = {
        name: (tmp_path / "models" / name if where == "model-dir" else hub.snapshot(name))
        for name in CONTENT
    }
    assert [path for path in hashed if path.name in CONTENT] == [paths[DIT], paths[VAE]]
    settings = json.loads((tmp_path / "out" / "manifest.json").read_text())["settings"]
    for key, name in (("dit_model", DIT), ("vae_model", VAE)):
        assert settings[key] == {"name": name, "size": pins[name].size, "sha256": pins[name].sha256}
    assert (hub.calls == []) == (where == "model-dir")


@needs_ffmpeg
@pytest.mark.usefixtures("stand_in")
def test_not_pinned_recorded(
    tmp_path: Path, hub: Hub, hashed: list[Path], caplog: pytest.LogCaptureFixture
) -> None:
    # Not one of seedvr2x's own: run as the model check accepts it, said unpinned, hashed once,
    # for the manifest alone; with no manifest, never.
    caplog.set_level(logging.INFO)
    path = tmp_path / "w.safetensors"
    path.write_bytes(b"numz's, say")
    options = ["--model-dir", str(tmp_path), "--dit-model", path.name, "--vae-model", path.name]
    assert upscale(tmp_path, "out", *options) == 0
    said = "w.safetensors: not one of seedvr2x's pinned files, so not checked against a pin"
    assert caplog.messages.count(said) == 1
    assert [each for each in hashed if each.name == path.name] == [path]
    settings = json.loads((tmp_path / "out" / "manifest.json").read_text())["settings"]
    digest = hashlib.sha256(b"numz's, say").hexdigest()
    assert settings["dit_model"] == settings["vae_model"]
    assert settings["dit_model"] == {"name": path.name, "size": 11, "sha256": digest}
    hashed.clear()
    assert upscale(tmp_path, "one.mkv", *options) == 0
    assert [each for each in hashed if each.name == path.name] == []
    assert hub.calls == hub.lookups == []


@needs_ffmpeg
@pytest.mark.usefixtures("stand_in", "pins")
def test_model_dir_never_calls_the_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def called(*args: object, **kwargs: object) -> None:
        pytest.fail("huggingface_hub called")

    monkeypatch.setattr(file_download, "try_to_load_from_cache", called)
    monkeypatch.setattr(file_download, "hf_hub_download", called)
    options = ["--model-dir", str(model_dir(tmp_path)), "--dit-model", DIT, "--vae-model", VAE]
    assert upscale(tmp_path, "out", *options) == 0


@needs_ffmpeg
@pytest.mark.usefixtures("stand_in")
def test_downloaded_with_the_pin(
    tmp_path: Path, hub: Hub, pins: dict[str, Pin], caplog: pytest.LogCaptureFixture
) -> None:
    # Without --model-dir: each file asked of the library by itself, at the pinned revision, into
    # the cache, with no token, said once; the log names the cache, then each file, downloaded the
    # first time, from the cache the next.
    caplog.set_level(logging.INFO)
    options = ["--dit-model", DIT, "--vae-model", VAE]
    assert upscale(tmp_path, "one.mkv", *options) == 0
    assert hub.calls == [
        {
            "repo_id": "hekmon/seedvr2x",
            "filename": name,
            "revision": REVISION,
            "cache_dir": hub.cache,
            "token": False,
        }
        for name in (DIT, VAE)
    ]
    assert f"{SHORT}, in Hugging Face's cache, {hub.cache}" in caplog.messages
    assert caplog.messages.count(NO_TOKEN) == 1
    for name in (DIT, VAE):
        size = f"{pins[name].size:,} bytes"
        assert f"{name}: not in the cache, downloading {size} from {SHORT}" in caplog.messages
        assert f"{name}: downloaded in 0 s, {hub.snapshot(name)}" in caplog.messages
    caplog.clear()
    hub.calls.clear()
    assert upscale(tmp_path, "two.mkv", *options) == 0
    assert hub.calls == []
    assert NO_TOKEN not in caplog.messages
    for name in (DIT, VAE):
        assert f"{name}: in the cache, {hub.snapshot(name)}" in caplog.messages


@needs_ffmpeg
@pytest.mark.usefixtures("stand_in")
@pytest.mark.parametrize(
    ("option", "name", "listing"),
    [
        (
            "--dit-model",
            "seedvr2_ema_7b_fp16.safetensors",
            "seedvr2x_ema_7b_fp16.safetensors or seedvr2x_ema_7b_sharp_fp16.safetensors",
        ),
        ("--vae-model", "ema_vae_fp16.safetensors", "seedvr2x_ema_vae_fp16.safetensors"),
    ],
)
def test_not_pinned_refused_without_model_dir(
    tmp_path: Path, hub: Hub, caplog: pytest.LogCaptureFixture, option: str, name: str, listing: str
) -> None:
    # A file seedvr2x doesn't pin, without --model-dir: refused, listing seedvr2x's own files of
    # its role, before the library is asked anything; nothing written.
    assert upscale(tmp_path, "out", option, name) == 1
    assert caplog.messages[-1] == (
        f"{option} {name}: not one of seedvr2x's own model files, the only ones it downloads"
        f" ({option}: {listing}); give --model-dir with the directory holding another file"
    )
    assert hub.calls == hub.lookups == []
    assert not (tmp_path / "out").exists()


def test_pinned_name_in_the_other_role_refused(hub: Hub) -> None:
    # One of seedvr2x's own files given in the other role: refused by its name, before any lookup
    # or download (16.5 GB for the 7B as the VAE, which the header check would then refuse),
    # saying which option takes it and what this one takes.
    with pytest.raises(ModelError) as error:
        pull.resolve(None, "seedvr2x_ema_vae_fp16.safetensors", "seedvr2x_ema_7b_fp16.safetensors")
    assert str(error.value).splitlines() == [
        "--dit-model seedvr2x_ema_vae_fp16.safetensors: seedvr2x's VAE, which --vae-model takes"
        " (--dit-model: seedvr2x_ema_7b_fp16.safetensors or"
        " seedvr2x_ema_7b_sharp_fp16.safetensors)",
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
    assert len(lines) == 3
    assert lines[-1] == (
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
        " takes (--dit-model: seedvr2x_ema_7b_fp16.safetensors or"
        " seedvr2x_ema_7b_sharp_fp16.safetensors)",
        "--vae-model transnetv2.safetensors: TransNetV2's weights, which seedvr2x's shot detection"
        " takes (--vae-model: seedvr2x_ema_vae_fp16.safetensors)",
    ]
    assert hub.calls == hub.lookups == []


def other_revision(cache: Path, layout: str) -> dict[str, Path]:
    """The small files as a download at another revision (OTHER) leaves them in cache: each blob
    in the repository's blobs/, named by its SHA-256, linked from that revision's snapshot; with
    layout "shared", a link itself, into the shared store of Xet files, cache's blobs/, whose
    names are Xet hashes, not SHA-256s (utils/_shared_blobs.py:16-19). Each file's blob, by
    name."""
    repository = cache / "models--hekmon--seedvr2x"
    blobs: dict[str, Path] = {}
    for name, data in CONTENT.items():
        digest = hashlib.sha256(data).hexdigest()
        blob = repository / "blobs" / digest
        blob.parent.mkdir(parents=True, exist_ok=True)
        if layout == "shared":
            xet = hashlib.sha256(b"a Xet hash of " + data).hexdigest()
            stored = cache / "blobs" / xet[:2] / xet
            stored.parent.mkdir(parents=True, exist_ok=True)
            stored.write_bytes(data)
            blob.symlink_to(os.path.relpath(stored, blob.parent))
        else:
            blob.write_bytes(data)
        link = repository / "snapshots" / OTHER / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(Path("..", "..", "blobs", digest))
        blobs[name] = blob
    return blobs


@needs_ffmpeg
@pytest.mark.usefixtures("stand_in", "pins")
@pytest.mark.parametrize("layout", ["own", "shared"])
@pytest.mark.parametrize("offline", [False, True], ids=["online", "offline"])
def test_another_revisions_download_taken(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hub: Hub,
    caplog: pytest.LogCaptureFixture,
    layout: str,
    offline: bool,
) -> None:
    # The files downloaded at another revision, the same bytes: found by their SHA-256, in the
    # cache, so neither downloaded nor any space asked, offline too; then checked by their pins.
    caplog.set_level(logging.INFO)
    blobs = other_revision(hub.cache, layout)
    asked: list[Path] = []
    monkeypatch.setattr(shutil, "disk_usage", lambda path: asked.append(Path(path)))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", offline)
    assert upscale(tmp_path, "one.mkv", "--dit-model", DIT, "--vae-model", VAE) == 0
    assert hub.calls == [] and asked == []
    assert hub.lookups == [DIT, VAE]
    for name in (DIT, VAE):
        said = f"{name}: in the cache, from another revision's download, {blobs[name]}"
        assert said in caplog.messages
        assert f"{name}: its size and SHA-256 as seedvr2x pins them, {SHORT}" in caplog.messages
    assert NO_TOKEN not in caplog.messages


@needs_ffmpeg
@pytest.mark.usefixtures("stand_in")
def test_another_revisions_blob_of_another_size_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hub: Hub,
    pins: dict[str, Pin],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A blob of another size where the pinned file's would be: taken, as the library would link
    # it rather than download (file_download.py:1256-1261), and refused by its pin's size, saying
    # to remove it; no space asked, nothing downloaded, nothing written, the blob left as it is.
    blobs = other_revision(hub.cache, "own")
    blobs[VAE].write_bytes(CONTENT[VAE] + b"!")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: pytest.fail("space asked"))
    assert upscale(tmp_path, "out", "--dit-model", DIT, "--vae-model", VAE) == 1
    size = pins[VAE].size
    assert caplog.messages[-1] == (
        f"{blobs[VAE]}: {size + 1:,} bytes, where hekmon/seedvr2x's {VAE} at {REVISION[:8]} has"
        f" {size:,}: not the file seedvr2x pins, left as it is;"
        f" {from_the_cache(VAE, blobs[VAE], hub.cache)}"
    )
    assert hub.calls == [] and not (tmp_path / "out").exists()
    assert blobs[VAE].read_bytes() == CONTENT[VAE] + b"!"


def http_error(kind: type[HfHubHTTPError], status: int, message: str) -> HfHubHTTPError:
    """An HTTP error of the library's, as hf_raise_for_status raises it."""
    return kind(message, response=httpx.Response(status, request=httpx.Request("HEAD", url(DIT))))


def unreachable() -> LocalEntryNotFoundError:
    """The library's error when the Hub can't be reached, the network's own as its cause."""
    error = LocalEntryNotFoundError("An error happened while trying to locate the file on the Hub")
    error.__cause__ = httpx.ConnectError("[Errno -3] Temporary failure in name resolution")
    return error


def forbidden_head() -> LocalEntryNotFoundError:
    """The library's error when the Hub answers the file's HEAD with a 403, as its cause
    (file_download.py:1959-1965)."""
    error = LocalEntryNotFoundError("An error happened while trying to locate the file on the Hub")
    error.__cause__ = http_error(HfHubHTTPError, 403, "403 Forbidden: None.")
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


CACHE = "{cache}"  # stands for the fake cache's directory in the messages below
REFUSALS: list[tuple[str, BaseException | None, str]] = [
    (
        "offline",
        None,
        f"{DIT} and {VAE}: not in Hugging Face's cache, {CACHE}, and HF_HUB_OFFLINE keeps the"
        f" network out: unset it to download 1,040 bytes, set HF_HOME (or HF_HUB_CACHE) to a"
        f" cache holding them, {ELSEWHERE}",
    ),
    (
        "unreachable",
        unreachable(),
        f"{DIT}: not in the cache, and https://huggingface.co couldn't be reached to download it"
        f" ([Errno -3] Temporary failure in name resolution): run again once it can be,"
        f" {ELSEWHERE}",
    ),
    (
        # Reached, but answering without the Hub's headers: the endpoint or a proxy to check,
        # where a rerun alone changes nothing.
        "not-the-hub",
        not_the_hub(),
        f"{DIT}: not in the cache, and https://huggingface.co answered the request to download it"
        f" without the Hub's headers ({no_commit('https://huggingface.co')}): check HF_ENDPOINT,"
        f" if set, and any proxy on the way to the Hub, {ELSEWHERE}",
    ),
    (
        "entry",
        http_error(RemoteEntryNotFoundError, 404, "404 Client Error. Entry Not Found"),
        f"{DIT}: not in hekmon/seedvr2x at {REVISION}, the revision this seedvr2x pins (the Hub:"
        " entry not found): download it elsewhere and give --model-dir with its directory",
    ),
    (
        "revision",
        http_error(RevisionNotFoundError, 404, "404 Client Error. Revision Not Found"),
        f"{DIT}: hekmon/seedvr2x has no revision {REVISION}, the one this seedvr2x pins (the Hub:"
        " revision not found): download the files elsewhere and give --model-dir with their"
        " directory",
    ),
    (
        "repository",
        http_error(RepositoryNotFoundError, 401, "401 Client Error. Repository Not Found"),
        f"{DIT}: the Hub has no repository hekmon/seedvr2x, or not a public one (the Hub:"
        " repository not found): download the files elsewhere and give --model-dir with their"
        " directory",
    ),
    (
        "refused",
        http_error(HfHubHTTPError, 429, "429 Client Error: Too Many Requests"),
        f"{DIT}: the Hub refused its download (429 Client Error: Too Many Requests): run again,"
        f" which downloads it again, {ELSEWHERE}",
    ),
    (
        "unauthorized",
        http_error(HfHubHTTPError, 401, "401 Client Error: Unauthorized"),
        f"{DIT}: the Hub refused it (401 Client Error: Unauthorized): {UNAVAILABLE}",
    ),
    (
        "forbidden",
        http_error(HfHubHTTPError, 403, "403 Forbidden: None."),
        f"{DIT}: the Hub refused it (403 Forbidden: None.): {UNAVAILABLE}",
    ),
    (
        "forbidden-head",
        forbidden_head(),
        f"{DIT}: the Hub refused it (403 Forbidden: None.): {UNAVAILABLE}",
    ),
    (
        "disabled",
        http_error(DisabledRepoError, 403, "Access to this resource is disabled."),
        f"{DIT}: the Hub refused it (Access to this resource is disabled.): {UNAVAILABLE}",
    ),
    (
        "permissions",
        PermissionError(errno.EACCES, "Permission denied", "{cache}/models--hekmon--seedvr2x"),
        f"{DIT}: Hugging Face's cache, {CACHE}, can't be written (Permission denied): set HF_HOME"
        " (or HF_HUB_CACHE) to a directory you can write, on a disk with 1,040 bytes free,"
        f" {ELSEWHERE}",
    ),
    (
        # hf_xet's I/O error, its OS error's number in its message alone.
        "full",
        OSError("IO Error: No space left on device (os error 28)"),
        f"{DIT}: Hugging Face's cache, {CACHE}, can't be written (No space left on device): set"
        " HF_HOME (or HF_HUB_CACHE) to a directory you can write, on a disk with 1,040 bytes"
        f" free, {ELSEWHERE}",
    ),
    (
        "network-lost",
        ConnectionError("Network error: connection reset"),
        f"{DIT}: its download failed (Network error: connection reset): run again, which"
        f" downloads it again, {ELSEWHERE}",
    ),
    (
        "timed-out",
        httpx.ReadTimeout("The read operation timed out"),
        f"{DIT}: its download failed (The read operation timed out): run again, which downloads"
        f" it again, {ELSEWHERE}",
    ),
    (
        # httpx's errors that are RequestErrors but not TransportErrors.
        "decoding",
        httpx.DecodingError("Error -3 while decompressing data: invalid stored block lengths"),
        f"{DIT}: its download failed (Error -3 while decompressing data: invalid stored block"
        f" lengths): run again, which downloads it again, {ELSEWHERE}",
    ),
    (
        "redirects",
        httpx.TooManyRedirects("Exceeded maximum allowed redirects."),
        f"{DIT}: its download failed (Exceeded maximum allowed redirects.): run again, which"
        f" downloads it again, {ELSEWHERE}",
    ),
    (
        "integrity",
        RuntimeError("Data integrity error: hash mismatch"),
        f"{DIT}: its download failed (Data integrity error: hash mismatch): run again, which"
        f" downloads it again, {ELSEWHERE}",
    ),
    (
        # hf_xet's Configuration error (xet_pkg/src/error.rs:308), and the library's own when the
        # Hub's answer lacks the Xet headers (utils/_xet.py:179-181).
        "xet-configuration",
        ValueError("Configuration error: invalid range"),
        f"{DIT}: its download failed (Configuration error: invalid range): check HF_ENDPOINT, if"
        f" set, and any proxy on the way to the Hub, {ELSEWHERE}",
    ),
    (
        "xet-headers",
        ValueError("Xet headers have not been correctly set by the server."),
        f"{DIT}: its download failed (Xet headers have not been correctly set by the server.):"
        f" check HF_ENDPOINT, if set, and any proxy on the way to the Hub, {ELSEWHERE}",
    ),
]


@needs_ffmpeg
@pytest.mark.usefixtures("stand_in", "pins")
@pytest.mark.parametrize(("kind", "error", "said"), REFUSALS, ids=[kind for kind, _, _ in REFUSALS])
def test_failures_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hub: Hub,
    caplog: pytest.LogCaptureFixture,
    kind: str,
    error: BaseException | None,
    said: str,
) -> None:
    # Each of the library's failures, its errors caught by name: exit status 1, saying what to
    # do, nothing written. HF_HUB_OFFLINE refuses before the library is asked to download.
    if error is None:
        monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    hub.error = error
    assert upscale(tmp_path, "out", "--dit-model", DIT, "--vae-model", VAE) == 1
    assert caplog.messages[-1] == said.replace(CACHE, str(hub.cache))
    assert [call["filename"] for call in hub.calls] == ([] if error is None else [DIT])
    assert not (tmp_path / "out").exists()
    assert not hub.cache.exists()


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


@needs_ffmpeg
@pytest.mark.usefixtures("stand_in", "pins")
@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a directory whatever its permissions")
@pytest.mark.parametrize("unreadable", ["snapshots", "blobs"])
def test_cache_unreadable_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hub: Hub,
    caplog: pytest.LogCaptureFixture,
    unreadable: str,
) -> None:
    # The cache's own lookup, the library's, listing the snapshots' directory
    # (file_download.py:1606), and seedvr2x's of a blob, which stats into the repository's
    # blobs/: a directory that can't be read refused, naming the cache and what to do, exit 1,
    # nothing downloaded, nothing written.
    monkeypatch.setattr(file_download, "try_to_load_from_cache", LOOKUP)
    repository = hub.cache / "models--hekmon--seedvr2x"
    for name in ("snapshots", "blobs"):
        (repository / name).mkdir(parents=True)
    locked = repository / unreadable
    locked.chmod(0)
    try:
        assert upscale(tmp_path, "out", "--dit-model", DIT, "--vae-model", VAE) == 1
    finally:
        locked.chmod(0o755)
    named = (
        locked if unreadable == "snapshots" else locked / hashlib.sha256(CONTENT[DIT]).hexdigest()
    )
    assert caplog.messages[-1] == (
        f"Hugging Face's cache, {hub.cache}, can't be read (Permission denied: {named}): make it"
        f" readable, set HF_HOME (or HF_HUB_CACHE) to another cache, {ELSEWHERE}"
    )
    assert hub.calls == [] and not (tmp_path / "out").exists()


# hf_xet's two errors for a file its storage doesn't have or won't give, raised by the library's
# download, in a process of its own whose HF_HOME is the test's: importing hf_xet writes a log
# under it. The library's two calls are replaced, its cache argv[1]; the error, argv[2].
XET_REFUSED = """
import shutil
import sys
from types import SimpleNamespace

import hf_xet
from huggingface_hub import constants, file_download

from seedvr2x.runtime import pull
from seedvr2x.runtime.weights import ModelError

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

from seedvr2x.runtime import pull
from seedvr2x.runtime.weights import ModelError


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


@needs_ffmpeg
@pytest.mark.usefixtures("stand_in")
def test_space_checked_before_a_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hub: Hub,
    pins: dict[str, Pin],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The cache's disk, its free space mocked: too little for the files to download, refused
    # before the first, naming the cache, HF_HOME and --model-dir; enough, downloaded; once in
    # the cache, no space needed, nor checked.
    asked: list[Path] = []
    free = [pins[DIT].size + pins[VAE].size - 1]

    def disk_usage(path: Path) -> SimpleNamespace:
        asked.append(Path(path))
        return SimpleNamespace(free=free[0])

    monkeypatch.setattr(shutil, "disk_usage", disk_usage)
    options = ["--dit-model", DIT, "--vae-model", VAE]
    assert upscale(tmp_path, "out", *options) == 1
    assert caplog.messages[-1] == (
        f"Hugging Face's cache, {hub.cache}: 1,039 bytes free on its disk, where downloading"
        f" {DIT} and {VAE} needs 1,040 bytes: free some space there, set HF_HOME (or"
        f" HF_HUB_CACHE) to a directory on a disk with room, {ELSEWHERE}"
    )
    assert hub.calls == [] and not (tmp_path / "out").exists()
    # The nearest directory there is of the one the file goes in.
    assert asked == [tmp_path]
    free[0] += 1
    asked.clear()
    assert upscale(tmp_path, "out", *options) == 0
    assert [call["filename"] for call in hub.calls] == [DIT, VAE]
    assert len(asked) == 2
    asked.clear()
    free[0] = 0
    assert upscale(tmp_path, "again", *options) == 0
    assert asked == []


@needs_ffmpeg
@pytest.mark.usefixtures("stand_in")
def test_space_left_after_each_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: Hub, pins: dict[str, Pin]
) -> None:
    # The disk's free space lowered by each file downloaded, from exactly what both need: once
    # the DiT is there, the VAE's check asks for the downloads left, the VAE alone.
    free = [pins[DIT].size + pins[VAE].size]
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=free[0]))

    def landed(name: str) -> None:
        free[0] -= pins[name].size

    hub.downloaded = landed
    assert upscale(tmp_path, "out", "--dit-model", DIT, "--vae-model", VAE) == 0
    assert [call["filename"] for call in hub.calls] == [DIT, VAE]
    assert free == [0]


@needs_ffmpeg
@pytest.mark.usefixtures("stand_in")
@pytest.mark.parametrize("count", [1, 2])
def test_space_refusal_names_partial_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hub: Hub,
    pins: dict[str, Pin],
    caplog: pytest.LogCaptureFixture,
    count: int,
) -> None:
    # What interrupted downloads left beside the blobs, <blob>.<8 hex digits>.incomplete as
    # huggingface_hub 1.33.0 names it (file_download.py:2003), <blob>.incomplete as older
    # releases did: said in the refusal of too little space, each one's path and size, and left
    # as they are.
    blobs = hub.cache / "models--hekmon--seedvr2x" / "blobs"
    blobs.mkdir(parents=True)
    names = [f"{pins[DIT].sha256}.0123abcd.incomplete", f"{pins[VAE].sha256}.incomplete"]
    sizes = {blobs / name: size for name, size in zip(names[:count], (300, 400), strict=False)}
    for path, size in sizes.items():
        path.write_bytes(bytes(size))
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=100))
    assert upscale(tmp_path, "out", "--dit-model", DIT, "--vae-model", VAE) == 1
    said = (
        f"Hugging Face's cache, {hub.cache}: 100 bytes free on its disk, where downloading {DIT}"
        f" and {VAE} needs 1,040 bytes: free some space there, set HF_HOME (or HF_HUB_CACHE) to a"
        f" directory on a disk with room, {ELSEWHERE}."
    )
    listed = [f"{path} ({size:,} bytes)" for path, size in sorted(sizes.items())]
    if count == 1:
        said += (
            f" {listed[0]} was left there by a download stopped before its end, or is one under"
            " way: no later download takes it up, and removing it frees its space"
        )
    else:
        said += (
            f" {listed[0]} and {listed[1]} were left there by downloads stopped before their end,"
            " or are under way: no later download takes them up, and removing them frees their"
            " space"
        )
    assert caplog.messages[-1] == said
    assert {path: path.stat().st_size for path in sizes} == sizes
    assert hub.calls == []


# The command line in a process of its own, whose HF_HOME is the test's, the library's two calls
# replaced: the cache empty, its directory argv[1], and a download that says it started, then
# waits, its finally writing argv[2] as the library's own removes its partial file
# (file_download.py:2042-2045).
TERMINATED = """
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from huggingface_hub import constants, file_download

from seedvr2x import cli

cleaned = Path(sys.argv[2])


def download(repo_id, filename, **options):
    try:
        print("DOWNLOADING", file=sys.stderr, flush=True)
        time.sleep(60)
    finally:
        cleaned.write_text(filename)


constants.HF_HUB_CACHE = sys.argv[1]
constants.HF_HUB_OFFLINE = False  # the library faked: nothing reaches the network
file_download.try_to_load_from_cache = lambda *args, **kwargs: None
file_download.hf_hub_download = download
shutil.disk_usage = lambda path: SimpleNamespace(free=10**12)
sys.exit(cli.main(sys.argv[3:]))
"""


@needs_ffmpeg
def test_sigterm_during_a_download(tmp_path: Path) -> None:
    # SIGTERM while a file downloads, before the run's own handlers: raised as Terminated, so that
    # the download unwinds, its finally run, and the run stops at once, exit 143, nothing written.
    source(tmp_path / "in.mkv")
    cleaned = tmp_path / "cleaned"
    hidden = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_XET_CACHE")
    environment = {name: value for name, value in os.environ.items() if name not in hidden}
    environment |= {"HF_HOME": str(tmp_path / "home"), "HF_HUB_OFFLINE": "1"}
    process = subprocess.Popen(
        [
            *(sys.executable, "-c", TERMINATED, str(tmp_path / "hf" / "hub"), str(cleaned)),
            *(str(tmp_path / "in.mkv"), "-o", str(tmp_path / "out")),
        ],
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    assert process.stderr is not None
    lines: list[str] = []
    for line in process.stderr:
        lines.append(line)
        if line.startswith("DOWNLOADING"):
            process.send_signal(signal.SIGTERM)
    status = process.wait(timeout=60)
    text = "".join(lines)
    assert status == 143, text
    assert cleaned.read_text() == "seedvr2x_ema_7b_sharp_fp16.safetensors"
    assert "stopped at once (SIGTERM)" in text
    assert not (tmp_path / "out").exists()


# The command line's logging in a process of its own, whose HF_HOME is the test's, the library's
# two calls replaced: the cache empty, its directory argv[1], and a download that logs as the
# library logs, the Hub's advice to set a token (utils/_http.py:972-993) and another warning,
# httpx a request (its _client.py:1025), then fails.
LOGGED = """
import logging
import shutil
import sys
from types import SimpleNamespace

from huggingface_hub import constants, file_download

from seedvr2x import cli
from seedvr2x.runtime import pull


def download(repo_id, filename, **options):
    http = logging.getLogger("huggingface_hub.utils._http")
    http.warning(
        "Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to"
        " enable higher rate limits and faster downloads."
    )
    http.warning("another warning of the library's")
    request = 'HTTP Request: HEAD %s "HTTP/1.1 302 Found"'
    logging.getLogger("httpx").info(request, pull.url(filename))
    raise RuntimeError("the download stops here")


constants.HF_HUB_CACHE = sys.argv[1]
constants.HF_HUB_OFFLINE = False  # the library faked: nothing reaches the network
file_download.try_to_load_from_cache = lambda *args, **kwargs: None
file_download.hf_hub_download = download
shutil.disk_usage = lambda path: SimpleNamespace(free=10**12)
sys.exit(cli.main(sys.argv[2:]))
"""


@needs_ffmpeg
def test_library_log_once(tmp_path: Path) -> None:
    # huggingface_hub's warnings once, in seedvr2x's log, as its lines are, but the Hub's advice
    # to set a token, which seedvr2x never sends, saying so; httpx's requests not logged.
    source(tmp_path / "in.mkv")
    hidden = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_XET_CACHE")
    environment = {name: value for name, value in os.environ.items() if name not in hidden}
    environment |= {"HF_HOME": str(tmp_path / "home"), "HF_HUB_OFFLINE": "1"}
    result = subprocess.run(
        [
            *(sys.executable, "-c", LOGGED, str(tmp_path / "hf" / "hub")),
            *(str(tmp_path / "in.mkv"), "-o", str(tmp_path / "out.mkv")),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 1, result.stderr
    assert pull.TOKEN_ADVICE not in result.stderr
    assert result.stderr.count("another warning of the library's") == 1
    line = r"\d{4}-\d\d-\d\d [\d:,]+ WARNING huggingface_hub\.utils\._http: another warning"
    assert re.search(line, result.stderr), result.stderr
    assert "HTTP Request" not in result.stderr
    assert result.stderr.count(NO_TOKEN) == 1
    failed = "its download failed (the download stops here)"
    assert f"seedvr2x_ema_7b_sharp_fp16.safetensors: {failed}" in result.stderr


# seedvr2x's handling of the library's log, in a fresh interpreter whose HF_HOME is the test's,
# before seedvr2x has imported anything of the library, as a run of the command line starts: the
# library's handler, which the import of its logging module adds, gone all the same.
LIBRARY_HANDLER = """
import logging
import sys

from seedvr2x.runtime import pull

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


# python -m seedvr2x in a process of its own, its arguments argv[1:]: the telemetry variable as
# huggingface_hub reads it, when first imported (constants.py:244-248 in 1.33.0), and the
# library's constant after the run, which imports it.
TELEMETRY = """
import os
import runpy
import sys

seen = []


class Watch:
    def find_spec(self, name, path=None, target=None):
        if name == "huggingface_hub" and not seen:
            seen.append(os.environ.get("HF_HUB_DISABLE_TELEMETRY"))


sys.meta_path.insert(0, Watch())
sys.argv = ["seedvr2x", *sys.argv[1:]]
try:
    runpy.run_module("seedvr2x", run_name="__main__", alter_sys=True)
except SystemExit as exit:
    status = exit.code
imported = "huggingface_hub" in sys.modules
from huggingface_hub import constants

print(status, imported, seen, constants.HF_HUB_DISABLE_TELEMETRY)
"""


@needs_ffmpeg
@pytest.mark.parametrize("given", [None, "0", "1"], ids=["unset", "0", "1"])
def test_telemetry_off_unless_set(tmp_path: Path, given: str | None) -> None:
    # Hugging Face's telemetry off unless the user set it (DESIGN.md, Weights): seedvr2x's entry
    # sets HF_HUB_DISABLE_TELEMETRY before anything imports huggingface_hub, the user's own value
    # kept. A run without --model-dir, in a process of its own whose HF_HOME is the test's,
    # offline: it imports the library to look in the cache, empty, then refuses the files
    # missing, exit 1. The library's other two switches unset (constants.py:246-247).
    source(tmp_path / "in.mkv")
    hidden = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_XET_CACHE")
    switches = ("HF_HUB_DISABLE_TELEMETRY", "DISABLE_TELEMETRY", "DO_NOT_TRACK")
    environment = {
        name: value for name, value in os.environ.items() if name not in (*hidden, *switches)
    }
    environment |= {"HF_HOME": str(tmp_path / "home"), "HF_HUB_OFFLINE": "1"}
    if given is not None:
        environment["HF_HUB_DISABLE_TELEMETRY"] = given
    result = subprocess.run(
        [
            *(sys.executable, "-c", TELEMETRY),
            *(str(tmp_path / "in.mkv"), "-o", str(tmp_path / "out.mkv")),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    read = "1" if given is None else given
    assert result.stdout == f"1 True ['{read}'] {read == '1'}\n", result.stderr
    assert "not in Hugging Face's cache" in result.stderr


@needs_ffmpeg
@pytest.mark.usefixtures("stand_in")
def test_cache_behind_a_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pins: dict[str, Pin],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # HF_HOME a link: huggingface_hub's constants give the cache through it, as seedvr2x names
    # it, and the library resolves it to download (file_download.py:999). Downloaded, taken from
    # the cache the next run, refused saying the file to remove where it is, and the command
    # with the cache as given.
    caplog.set_level(logging.INFO)
    (tmp_path / "hf-target").mkdir()
    (tmp_path / "hf").symlink_to(tmp_path / "hf-target")
    hub = faked(monkeypatch, tmp_path / "hf" / "hub")
    options = ["--dit-model", DIT, "--vae-model", VAE]
    assert upscale(tmp_path, "one.mkv", *options) == 0
    assert [call["cache_dir"] for call in hub.calls] == [hub.cache, hub.cache]
    target = tmp_path / "hf-target" / "hub"
    for name in (DIT, VAE):
        assert f"{name}: downloaded in 0 s, {hub.snapshot(name, target)}" in caplog.messages
    caplog.clear()
    assert upscale(tmp_path, "two.mkv", *options) == 0
    for name in (DIT, VAE):
        assert f"{name}: in the cache, {hub.snapshot(name)}" in caplog.messages
    other = bytes(len(CONTENT[VAE]))
    hub.snapshot(VAE).write_bytes(other)
    target_vae = hub.snapshot(VAE, target)
    assert upscale(tmp_path, "three.mkv", *options) == 1
    assert caplog.messages[-1] == (
        f"{hub.snapshot(VAE)}: SHA-256 {hashlib.sha256(other).hexdigest()}, where"
        f" hekmon/seedvr2x's {VAE} at {REVISION[:8]} has {pins[VAE].sha256}: not the file"
        f" seedvr2x pins, left as it is; {from_the_cache(VAE, target_vae, hub.cache)}"
    )


@needs_ffmpeg
@pytest.mark.usefixtures("stand_in")
def test_cache_refusal_says_the_cache(
    tmp_path: Path, hub: Hub, pins: dict[str, Pin], caplog: pytest.LogCaptureFixture
) -> None:
    # A file of the cache of its pinned size, other bytes: refused saying how to fetch it again
    # into the cache, the file to remove or huggingface_hub's command, and never to leave out the
    # --model-dir no one gave.
    other = bytes(len(CONTENT[DIT]))
    hub.served[DIT] = other
    assert upscale(tmp_path, "out", "--dit-model", DIT, "--vae-model", VAE) == 1
    link = hub.snapshot(DIT)
    assert caplog.messages[-1] == (
        f"{link}: SHA-256 {hashlib.sha256(other).hexdigest()}, where hekmon/seedvr2x's {DIT} at"
        f" {REVISION[:8]} has {pins[DIT].sha256}: not the file seedvr2x pins, left as it is;"
        f" {from_the_cache(DIT, link, hub.cache)}"
    )
    assert not (tmp_path / "out").exists()


@needs_ffmpeg
@pytest.mark.usefixtures("stand_in")
@pytest.mark.parametrize("change", ["size", "bytes"])
def test_pinned_name_in_a_subdirectory(
    tmp_path: Path, pins: dict[str, Pin], caplog: pytest.LogCaptureFixture, change: str
) -> None:
    # --dit-model sub/<pinned name> in --model-dir names the pinned file all the same, by the
    # name's last part: checked by its pin, and refused, by its size or its SHA-256.
    directory = model_dir(tmp_path)
    (directory / "sub").mkdir()
    path = directory / "sub" / DIT
    data = CONTENT[DIT] + b"!" if change == "size" else bytes(len(CONTENT[DIT]))
    path.write_bytes(data)
    options = ["--model-dir", str(directory), "--dit-model", f"sub/{DIT}", "--vae-model", VAE]
    assert upscale(tmp_path, "out", *options) == 1
    if change == "size":
        found, pinned = f"{len(data):,} bytes", f"{pins[DIT].size:,}"
    else:
        found, pinned = f"SHA-256 {hashlib.sha256(data).hexdigest()}", pins[DIT].sha256
    assert caplog.messages[-1] == (
        f"{path}: {found}, where hekmon/seedvr2x's {DIT} at {REVISION[:8]} has {pinned}: not the"
        f" file seedvr2x pins, left as it is; fetch it again from {url(DIT)}, {INTO_MODEL_DIR}"
    )
    assert not (tmp_path / "out").exists()


@needs_ffmpeg
@pytest.mark.usefixtures("stand_in")
def test_header_refusal_says_how_to_fetch_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hub: Hub,
    pins: dict[str, Pin],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The header check, which comes before the pins, refusing seedvr2x's own files: each refusal
    # says how to fetch the file again, as the pin's would. In --model-dir, a file missing and one
    # that isn't a model file: its URL. In the cache, one that isn't a model file and one cut
    # short: the file to remove or huggingface_hub's command, in place of "fetch it again".
    monkeypatch.setattr(weights, "check_models", CHECK_MODELS)
    directory = tmp_path / "models"
    directory.mkdir()
    (directory / VAE).write_bytes(CONTENT[VAE])
    options = ["--model-dir", str(directory), "--dit-model", DIT, "--vae-model", VAE]
    assert upscale(tmp_path, "out", *options) == 1
    other = "neither safetensors, GGUF nor a PyTorch checkpoint"
    assert caplog.messages[-1].splitlines() == [
        f"{directory / DIT}: no such model file; fetch it from {url(DIT)}, {INTO_MODEL_DIR}",
        f"{directory / VAE}: {other}: {V1}; fetch it again from {url(VAE)}, {INTO_MODEL_DIR}",
    ]
    hub.served[VAE] = (100).to_bytes(8, "little") + b"{"
    assert upscale(tmp_path, "out", "--dit-model", DIT, "--vae-model", VAE) == 1
    dit, vae = hub.snapshot(DIT), hub.snapshot(VAE)
    assert caplog.messages[-1].splitlines() == [
        f"{dit}: {other}: {V1}; {from_the_cache(DIT, dit, hub.cache)}",
        f"{vae}: cut short inside its header, 9 bytes of at least 108;"
        f" {from_the_cache(VAE, vae, hub.cache)}",
    ]
    assert not (tmp_path / "out").exists()


@needs_ffmpeg
@pytest.mark.usefixtures("stand_in")
@pytest.mark.parametrize("held", ["own", "other", "missing"])
def test_fetch_advice_in_its_own_role_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, held: str
) -> None:
    # seedvr2x's 7B and VAE in --model-dir, synthetic, each file holding its own header, the
    # other's, or missing. Given in each other's roles, which only --model-dir lets through, no
    # refusal says how to fetch them again: neither the URL, the pinned file fetched again being
    # refused in that role all the same, nor leaving out --model-dir, resolve refusing the swap.
    # The header check says the file in the wrong role, or missing; the pins, once the headers
    # pass, the sizes. Given in their own roles, the advice as before.
    monkeypatch.setattr(weights, "check_models", CHECK_MODELS)
    dit, vae = "seedvr2x_ema_7b_fp16.safetensors", "seedvr2x_ema_vae_fp16.safetensors"
    directory = tmp_path / "models"
    directory.mkdir()
    models = {"own": {dit: "7b", vae: "vae"}, "other": {dit: "vae", vae: "7b"}, "missing": {}}
    sizes = {
        name: write(directory / name, as_dtype(model, "F16")).stat().st_size
        for name, model in models[held].items()
    }
    assert all(size != pull.PINNED[name].size for name, size in sizes.items())

    def unpinned(name: str) -> str:
        return (
            f"{directory / name}: {sizes[name]:,} bytes, where hekmon/seedvr2x's {name} at"
            f" {REVISION[:8]} has {pull.PINNED[name].size:,}: not the file seedvr2x pins, left as"
            " it is"
        )

    def again(name: str) -> str:
        return f"; fetch it again from {url(name)}, {INTO_MODEL_DIR}"

    vae_as_dit = f"SeedVR2's VAE in fp16, given as the DiT (--dit-model): {V1}"
    dit_as_vae = f"SeedVR2's 7B DiT in fp16, given as the VAE (--vae-model): {V1}"
    if held == "own":
        swapped = [f"{directory / vae}: {vae_as_dit}", f"{directory / dit}: {dit_as_vae}"]
        own = [unpinned(dit) + again(dit), unpinned(vae) + again(vae)]
    elif held == "other":
        swapped = [unpinned(vae), unpinned(dit)]
        own = [
            f"{directory / dit}: {vae_as_dit}{again(dit)}",
            f"{directory / vae}: {dit_as_vae}{again(vae)}",
        ]
    else:
        # Never fetched there: "fetch it", no "again".
        swapped = [f"{directory / name}: no such model file" for name in (vae, dit)]
        own = [
            f"{directory / name}: no such model file; fetch it from {url(name)}, {INTO_MODEL_DIR}"
            for name in (dit, vae)
        ]
    options = ["--model-dir", str(directory)]
    assert upscale(tmp_path, "out", *options, "--dit-model", vae, "--vae-model", dit) == 1
    assert caplog.messages[-1].splitlines() == swapped
    assert upscale(tmp_path, "out", *options, "--dit-model", dit, "--vae-model", vae) == 1
    assert caplog.messages[-1].splitlines() == own
    assert not (tmp_path / "out").exists()


@needs_ffmpeg
@pytest.mark.usefixtures("stand_in")
def test_changed_during_the_first_pass_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pins: dict[str, Pin],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A file of seedvr2x's own touched after its check, during the first pass: the load, given the
    # run's hashes, finds it changed and refuses it, nothing recorded. The stand-in's load does
    # what load_models does with them (runtime/model.py), the model built none.
    from seedvr2x.media import source as media

    directory = model_dir(tmp_path)
    first_pass = media.first_pass

    def touched(*args: Any, **kwargs: Any) -> Any:
        state = (directory / DIT).stat()
        os.utime(directory / DIT, ns=(state.st_atime_ns, state.st_mtime_ns + 1))
        return first_pass(*args, **kwargs)

    def load_models(dit: Path, vae: Path, device: Any, hashes: pull.Hashes | None = None) -> Any:
        for path in (dit, vae):
            if hashes is not None:
                hashes.unchanged(path)
        return STAND_IN_MODELS

    monkeypatch.setattr(media, "first_pass", touched)
    monkeypatch.setattr(model, "load_models", load_models)
    options = ["--model-dir", str(directory), "--dit-model", DIT, "--vae-model", VAE]
    assert upscale(tmp_path, "out", *options) == 1
    assert caplog.messages[-1] == (
        f"{directory / DIT}: changed since its SHA-256 was read, before its load ended: run"
        " again, which checks it again"
    )
    assert not (tmp_path / "out" / "manifest.json").exists()


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

    def load_weights(module: object, path: Path, *args: object) -> None:
        role = "dit" if path == dit else "vae"
        done.append(role)
        if when == "during" and role == touched:
            touch(path)

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


# The command line, exiting 4 should the first pass run, 3 once torch is imported and 5 once
# huggingface_hub is, a failed assert exiting 1, as a refusal does. huggingface_hub's two calls are
# replaced as Hub does, its cache argv[1]; a download links the file from argv[2], where the test
# made it.
PULLED = """
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from seedvr2x import cli
from seedvr2x.media import source

cache, made, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]


def ran(*args, **kwargs):
    sys.exit(4)


def snapshot(name):
    return cache / "models--hekmon--seedvr2x" / "snapshots" / "{revision}" / name


def cached(repo_id, filename, cache_dir=None, revision=None, repo_type=None):
    return str(snapshot(filename)) if snapshot(filename).is_file() else None


def download(repo_id, filename, **options):
    snapshot(filename).parent.mkdir(parents=True, exist_ok=True)
    os.link(made / filename, snapshot(filename))
    return str(snapshot(filename))


source.first_pass = ran
if mode != "model-dir":
    from huggingface_hub import constants, file_download

    constants.HF_HUB_CACHE = str(cache)
    constants.HF_HUB_OFFLINE = False  # the library faked: nothing reaches the network
    file_download.try_to_load_from_cache = cached
    file_download.hf_hub_download = download
    shutil.disk_usage = lambda path: SimpleNamespace(free=10**12)
status = cli.main(sys.argv[4:])
if "torch" in sys.modules:
    sys.exit(3)
sys.exit(5 if mode == "model-dir" and "huggingface_hub" in sys.modules else status)
""".replace("{revision}", REVISION)


@needs_ffmpeg
@pytest.mark.parametrize("mode", ["cached", "downloaded", "model-dir"])
def test_pulled_before_torch(tmp_path: Path, mode: str) -> None:
    # The default files, from the cache, downloaded into it, or in --model-dir: their headers
    # accepted, their sizes not the pinned ones (synthetic files, their data's holes), refused
    # before the first pass, unhashed, without torch; with --model-dir, without huggingface_hub.
    made = tmp_path / "made"
    made.mkdir()
    dit, vae = "seedvr2x_ema_7b_sharp_fp16.safetensors", "seedvr2x_ema_vae_fp16.safetensors"
    sizes = {
        dit: write(made / dit, as_dtype("7b", "F16")).stat().st_size,
        vae: write(made / vae, as_dtype("vae", "F16")).stat().st_size,
    }
    assert sizes[dit] != pull.PINNED[dit].size and sizes[vae] != pull.PINNED[vae].size
    cache = tmp_path / "hf" / "hub"
    where = cache / "models--hekmon--seedvr2x" / "snapshots" / REVISION
    if mode == "cached":
        where.mkdir(parents=True)
        for name in (dit, vae):
            os.link(made / name, where / name)
    source(tmp_path / "in.mkv")
    args = [str(tmp_path / "in.mkv"), "-o", str(tmp_path / "out")]
    if mode == "model-dir":
        args += ["--model-dir", str(made)]
        where = made
    # The user's cache out of reach, and the network: HF_HUB_OFFLINE=1, which the script lifts in
    # the library only where it replaces its two calls.
    hidden = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_XET_CACHE")
    environment = {name: value for name, value in os.environ.items() if name not in hidden}
    environment |= {"HF_HOME": str(tmp_path / "home"), "HF_HUB_OFFLINE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", PULLED, str(cache), str(made), mode, *args],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 1, result.stderr
    for name in (dit, vae):
        pin = pull.PINNED[name]
        said = (
            f"{where / name}: {sizes[name]:,} bytes, where hekmon/seedvr2x's {name} at"
            f" {REVISION[:8]} has {pin.size:,}: not the file seedvr2x pins, left as it is;"
        )
        assert said in result.stderr
    assert "SHA-256" not in result.stderr
    assert not (tmp_path / "out").exists()


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

from seedvr2x.runtime import pull

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
    found = pull.fetch(pull.PINNED)
    hashes = pull.Hashes()
    for name, pin in pull.PINNED.items():
        assert found[name].resolve().is_relative_to(pull.cache().resolve())
        pull.check_pin(name, found[name], pin, hashes, pull.cache())
    with capsys.disabled():
        print("\n" + "\n".join(caplog.messages))
