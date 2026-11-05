"""seedvr2x's own model files: fetched from its Hugging Face repo at a revision pinned here, each
checked by a size and a SHA-256 pinned here too, so that a version always runs the same bytes
(DESIGN.md, Weights, One Hugging Face repo).

Without --model-dir, the files come from Hugging Face's cache, file by file, only those the run
uses, downloaded into it when missing by huggingface_hub, which seedvr2x imports to look in the
cache first. With --model-dir, they are read from that directory and nothing is downloaded:
seedvr2x doesn't import huggingface_hub then, though diffusers, which the vendored model imports,
brings it into every run all the same. Either way, a file named like one of seedvr2x's own is
checked against its pin before it loads (check_pinned), its SHA-256 read once per run (Hashes),
and the manifest's record takes that hash."""

import errno
import importlib
import logging
import os
import re
import shlex
import shutil
import sys
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePath

from seedvr2x import files, weights
from seedvr2x.clock import spent, taking, wall
from seedvr2x.errors import (
    Message,
    ModelChanged,
    ModelError,
    ModelNoRoom,
    ModelNotPinned,
    ModelsRefused,
    ModelUnknown,
    ModelUnreachable,
    ModelUnreadable,
    Say,
    Setting,
)
from seedvr2x.weights import Architecture, Role

logger = logging.getLogger(__name__)

REPO = "hekmon/seedvr2x"
# The upload of 2026-10-08, v1's files and phase 2's, each read back equal to SHA256SUMS. The
# repo's head moved to 502436ca the same day with a new card alone, no model byte changed: the pin
# stays (DESIGN.md, Weights).
REVISION = "c14a2bc4aab04cf38ad9b0d324014c4213f07048"


@dataclass(frozen=True)
class Pin:
    """A file of REPO at REVISION: its size, in bytes, its SHA-256, in hex, and the role it takes
    as a model file (the DiT's, the VAE's or the shot detector's), None for another file."""

    size: int
    sha256: str
    role: Role | None = None


# The shot detector's file, TransNetV2's weights (DESIGN.md, Input, Shot detection), named by
# itself, with no option of its own (design, 2026-10-10): with --model-dir, read from that
# directory by this name (DESIGN.md, Weights: "the files are read from it by name").
DETECTOR = "transnetv2.safetensors"

# v1's files (DESIGN.md, Weights), read from REVISION on 2026-10-09 without downloading them: each
# SHA-256 is the file's line of SHA256SUMS there, which the Hub's API gives as its LFS SHA-256 too,
# and each size the API's (HfApi.get_paths_info); both as the model conversation recorded them at
# the upload. tests/test_pull.py holds them to the revision. TransNetV2's weights, the shot
# detector's, the same way, its size and SHA-256 also those of the file downloaded at REVISION
# that day.
PINNED: dict[str, Pin] = {
    "seedvr2x_ema_7b_fp16.safetensors": Pin(
        16_479_335_080, "071cab5e5ef7a4471e1df0023c26cc16deeb14e58f8ad5c9196d2a08f96da5f2", "dit"
    ),
    "seedvr2x_ema_7b_sharp_fp16.safetensors": Pin(
        16_479_335_088, "5eb47fdee4b620765573a697b6f82442234e7dc7aa0beeb3ba72fc63b917a817", "dit"
    ),
    # The two 7Bs' int8 files (DESIGN.md, Weights, "int8 before v0.1.0 is tagged"), pinned the
    # same way on 2026-11-02: each SHA-256 its line of SHA256SUMS at REVISION, which
    # models/seedvr2_int8.py's OUTPUTS gives too, each size the API's. Downloaded only when
    # --dit-model names one, as the regular 7B's file is.
    "seedvr2x_ema_7b_int8_convrot.safetensors": Pin(
        8_333_648_992, "7eb2c7841b1c480d3302d83c8583d345cb9bb80dc169ec83e9a72f3585ab33da", "dit"
    ),
    "seedvr2x_ema_7b_sharp_int8_convrot.safetensors": Pin(
        8_333_649_000, "d83aeaa4687b43755658d85eadcf2e28bb10b70c0196798b303171cf9a16de43", "dit"
    ),
    # The two 7Bs' dynamic GGUF files (DESIGN.md, Weights, "The dynamic GGUF before v0.1.0 too"),
    # pinned the same way on 2026-11-04, each SHA-256 its line of SHA256SUMS at REVISION and each
    # size the file's: downloaded only when --dit-model names one.
    "seedvr2x_ema_7b_dyn.gguf": Pin(
        4_757_055_264, "f8c0c50d233dad8a26fb82d5eda4072c2ad935100e842b3fee953994849a487d", "dit"
    ),
    "seedvr2x_ema_7b_sharp_dyn.gguf": Pin(
        4_757_129_024, "90ba80c85e25ee08162882be58c0e05f1b4fc19c8dd9837b6f87c0efc4fad779", "dit"
    ),
    "seedvr2x_ema_vae_fp16.safetensors": Pin(
        501_325_454, "b9c6ebf0b14107be595825f476b9f89029a067d5b13e9db39c1351a608265468", "vae"
    ),
    DETECTOR: Pin(
        30_482_632, "bb8c838811a5e52e23be70e2794646a758d2bf4dcbd110dec8211ae7b8cbefdf", "detector"
    ),
}
# Every file's SHA-256 at REVISION, a few kilobytes, which tests/test_pull.py fetches to hold the
# pins to the revision: its size and SHA-256 read from the revision as the files' are (a git blob,
# not LFS), its SHA-256 the one the model conversation recorded.
SUMS = "SHA256SUMS"
SUMS_PIN = Pin(2_335, "57cf5a7b00bec79fac929f6776d1819ef80f28fc957a8b131cf2b6887cd09b53")

# The setting naming each role's file, None for the shot detector's, which no option names
# (DETECTOR: "seedvr2x's shot detection" takes it); and what one of seedvr2x's own files in that
# role is.
ROLE_SETTINGS: dict[Role, str | None] = {"dit": "dit_model", "vae": "vae_model", "detector": None}
KINDS: dict[Role, str] = {
    "dit": "one of seedvr2x's DiTs",
    "vae": "seedvr2x's VAE",
    "detector": "TransNetV2's weights",
}

# The errors of a cache that can't be written: permissions, a read-only or full disk, a quota.
UNWRITABLE = frozenset({errno.EACCES, errno.EPERM, errno.EROFS, errno.ENOSPC, errno.EDQUOT})
# hf_xet raises an I/O error as an OSError of its message alone, which ends as Rust writes an OS
# error, "(os error 28)" (hf-xet 1.6.0: xet_pkg/src/error.rs:214-218 and 307): its number is read
# back from it.
OS_ERROR = re.compile(r"\(os error (\d+)\)")
# The setting for a directory of model files, which every refusal of a file seedvr2x can't have
# offers besides (elsewhere): the apps word it as their option.
MODEL_DIR = Setting("model_dir")


def elsewhere(say: Say) -> str:
    """What every refusal of a file seedvr2x can't have offers besides."""
    return f"or download the files elsewhere and give {say(MODEL_DIR)} with their directory"


def unavailable(say: Say) -> str:
    """What a refusal says of a file the Hub refuses to anyone, which no rerun changes."""
    return (
        "the repository or the file is unavailable: download the files elsewhere and give"
        f" {say(MODEL_DIR)} with their directory"
    )


def check_endpoint(say: Say) -> str:
    """What a refusal says of answers that aren't the Hub's as the library expects them, as from
    an HF_ENDPOINT that isn't a Hub or a proxy on the way: a rerun alone changes nothing."""
    return f"check HF_ENDPOINT, if set, and any proxy on the way to the Hub, {elsewhere(say)}"


# The Hub's advice to set a token, which it sends in a warning header answering a request made
# without one, not every time, and huggingface_hub logs as a warning (utils/_http.py:972-993);
# logged on 2026-10-09 answering SHA256SUMS' download: "Warning: You are sending unauthenticated
# requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster
# downloads."
TOKEN_ADVICE = "You are sending unauthenticated requests to the HF Hub"


# The field of a model's record (main._model) holding its file's modification time in nanoseconds,
# beside its size and SHA-256: a finished job compares the file by size and time with it and reads
# nothing where both match (Hashes.vouch; DESIGN.md, Pause and resume, decided on 2026-10-29).
# Information a resume doesn't compare as a setting (resume.unstamped), a touched file of
# the same content being the same model.
TIMES = "mtime_ns"

# The time to hash a model file a run says before it starts (Hashes.sha256), per byte: the DiT's
# 16,479,335,080 bytes took 84 s on the RTX 5090 machine's disk on 2026-10-29 (impl-v010's list
# from its real jobs). A figure of one disk, the line saying "about": a faster disk finishes
# sooner, a slower one later, and the hash's own "in N s" line follows with the measure.
# PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md asks "a line says what is read, and for about how
# long" and leaves the rate open).
HASH_SECONDS_PER_BYTE = 84 / PINNED["seedvr2x_ema_7b_fp16.safetensors"].size
# The estimate under which no line is said: a hash that short is no silence worth a line.
# PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md leaves the bound open; 5 s keeps the VAE's
# 0.5 GB and the shot detector's 30 MB, about 3 s and 0.2 s at that rate, quiet).
HASH_SAID_FROM = 5.0


@dataclass
class _Known:
    """A model file's SHA-256 as the run knows it (Hashes), with the file as it was then (its
    device and inode, size and modification time: _state); `refusal`, the refusal of a load
    that reads another SHA-256, given the one it read; `proved`, whether a read of the whole file
    gave it (a hash at the start, or a load); `said`, the line said once a load proves it."""

    sha256: str
    state: tuple[int, int, int, int]
    refusal: Callable[[str], ModelError]
    proved: bool
    said: str | None = None


class Hashes:
    """The SHA-256 of the run's model files, each file read once a run where its load vouches for
    it (DESIGN.md, Weights, decided on 2026-11-03): its load hashes every byte it reads
    (blockswap.read, the shot detector's load) and proves what the run took for its SHA-256
    before the load (proved), or gives it; the pin's check, a job's record or a whole read at the
    start (sha256) give it before. The pinned check and the manifest's or the record's SHA-256
    take the same, with each file as it was then, its device and inode, size and modification
    time, which its load finds again (unchanged)."""

    def __init__(self) -> None:
        self._read: dict[Path, _Known] = {}

    def sha256(self, path: Path) -> str:
        """The SHA-256 of the file at path, in hex: the one known already (vouched for, expected,
        or proved by its load), or read whole now, the first time only, its time logged."""
        if path not in self._read:
            state = _state(path)
            # The silence named before it starts (DESIGN.md, Architecture, What it prints,
            # decided on 2026-10-28; built on 2026-10-29): what is read, and for about how long.
            expected = wall(state[2] * HASH_SECONDS_PER_BYTE)
            if expected >= HASH_SAID_FROM:
                logger.info(
                    "%s: reading its %s to check its SHA-256, %s",
                    path.name,
                    files.size_said(state[2]),
                    taking(expected),
                )
            started = time.monotonic()
            digest = files.sha256(path)
            logger.info(
                "%s: SHA-256 %s, in %.1f s", path.name, digest, spent(time.monotonic() - started)
            )
            self._read[path] = _Known(digest, state, _changed(path), proved=True)
        return self._read[path].sha256

    def known(self, path: Path) -> bool:
        """Whether the run knows the SHA-256 of the file at path, so that sha256 reads nothing."""
        return path in self._read

    def proven(self, path: Path) -> bool:
        """Whether a read of the whole file at path gave its SHA-256 this run: a hash at the start
        (sha256) or its load (proved), not a record's or a pin's word for it."""
        known = self._read.get(path)
        return known is not None and known.proved

    def vouch(
        self,
        path: Path,
        size: int,
        mtime_ns: int,
        sha256: str,
        refusal: Callable[[str], ModelError] | None = None,
    ) -> bool:
        """Take the SHA-256 a job's record holds for the file at path, and read nothing, when the
        file's size and modification time (nanoseconds) are those the record gives beside it:
        True. Else False, the file hashed when asked (sha256), as before. DESIGN.md, Pause and
        resume, decided on 2026-10-29: a finished job compares its model files by size and time
        with its record, where the DiT's SHA-256 alone took 84 s on the RTX 5090. The file is then
        held as hashed (unchanged), and its load, if it runs, proves the record's SHA-256
        (proved), refused otherwise with `refusal`'s, given the SHA-256 read (the library's own
        without it)."""
        if path in self._read:
            return True
        try:
            state = _state(path)
        except OSError:
            return False
        if state[2] != size or state[3] != mtime_ns:
            return False
        self._read[path] = _Known(sha256, state, refusal or _recorded(path, sha256), proved=False)
        return True

    def expect(
        self, path: Path, sha256: str, refusal: Callable[[str], ModelError], said: str
    ) -> None:
        """Take sha256 for the SHA-256 of the file at path, its pin's, its size checked against the
        pin already (check_pin), and read nothing: its load proves it (proved), refused otherwise
        with `refusal`'s, given the SHA-256 read, and says `said` once it has."""
        self._read[path] = _Known(sha256, _state(path), refusal, proved=False, said=said)

    def proved(self, path: Path, sha256: str, opened: os.stat_result) -> None:
        """The SHA-256 of the file at path as its load read it, every byte, and its status as the
        load opened it (opened): the SHA-256 known before (vouched for, expected or read at the
        start) proved, else a ModelError in the words of the way it came (a pin's, a record's, a
        whole read's), before anything is made of the model (DESIGN.md, Weights, decided on
        2026-11-03); a file unknown before taken with this SHA-256."""
        known = self._read.get(path)
        if known is None:
            state = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            self._read[path] = _Known(sha256, state, _changed(path), proved=True)
            logger.info("%s: SHA-256 %s, read as it loaded", path.name, sha256)
            return
        if sha256 != known.sha256:
            raise known.refusal(sha256)
        if not known.proved:
            logger.info("%s", known.said or f"{path.name}: SHA-256 as expected, read as it loaded")
        known.proved = True

    def mtime_ns(self, path: Path) -> int:
        """The modification time, in nanoseconds, of the file at path as it was when its SHA-256
        was read (sha256), vouched for (vouch) or expected (expect): what the manifest records
        beside the hash."""
        return self._read[path].state[3]

    def unchanged(self, path: Path) -> None:
        """Refuse (ModelError) the file at path if its SHA-256 is known this run and it isn't that
        file any more: another size, modification time or inode. A file runs the bytes its hash
        checked: its load checks it before and after (seedvr2x/model.py), and hashes it as it
        reads it (proved)."""
        if path not in self._read:
            return
        # Provisional (implementation, 2026-10-09; DESIGN.md doesn't say how): a stat, the load's
        # hash proving the bytes (proved). The inode is compared too, correctness first, though
        # a FUSE filesystem mounted without libfuse's use_ino option numbers its inodes itself and
        # may give a file that didn't change another one: such a file is refused, and runs again.
        try:
            now = _state(path)
        except OSError as error:
            raise ModelChanged(
                f"{path}: {error.strerror}, since its SHA-256 was read",
                file=path.name,
                figure="sha256",
            ) from None
        if now != self._read[path].state:
            raise _changed(path)("")


def _changed(path: Path) -> Callable[[str], ModelError]:
    """The refusal of a file found changed since its SHA-256 was read whole or vouched for, by its
    stat or by its load's hash: run again, which reads it again."""

    def refusal(found: str) -> ModelError:
        return ModelChanged(
            f"{path}: changed since its SHA-256 was read, before its load ended: run again, which"
            " checks it again",
            file=path.name,
            figure="sha256",
            found=found or None,
        )

    return refusal


def _recorded(path: Path, sha256: str) -> Callable[[str], ModelError]:
    """The refusal of a file vouched for by a record's size and modification time whose load reads
    another SHA-256 than the record's: changed with its size and time kept."""

    def refusal(found: str) -> ModelError:
        return ModelChanged(
            f"{path}: SHA-256 {found} as it loaded, where the record has {sha256} for it, with the"
            " size and modification time it has: changed since, its size and time kept",
            file=path.name,
            figure="sha256",
            expected=sha256,
            found=found,
        )

    return refusal


@dataclass(frozen=True)
class ModelFile:
    """A model file of the run: the name it was given (--dit-model, --vae-model; DETECTOR, the
    shot detector's) and its path, in --model-dir or in Hugging Face's cache. `absent`: one of
    seedvr2x's own, not in that cache and not downloaded, for a job that loads no model
    (resolve's `download`), held to its pin (PINNED) in its place: its path is where the cache
    would hold it, and nothing reads it."""

    name: str
    path: Path
    absent: bool = False


@dataclass(frozen=True)
class ModelFiles:
    """The run's model files, the DiT's and the VAE's; the directory of Hugging Face's cache they
    come from, None for --model-dir's; their SHA-256, each read once (hashes); and the shot
    detector's, DETECTOR, when the run detects shots, else None."""

    dit: ModelFile
    vae: ModelFile
    cache: Path | None
    hashes: Hashes
    detector: ModelFile | None = None

    def roles(self) -> list[tuple[Role, ModelFile]]:
        """The run's model files, each with the role it was given, the shot detector's last."""
        given: list[tuple[Role, ModelFile]] = [("dit", self.dit), ("vae", self.vae)]
        if self.detector is not None:
            given.append(("detector", self.detector))
        return given


def resolve(
    model_dir: Path | None, dit: str, vae: str, detector: bool = False, download: bool = True
) -> ModelFiles:
    """The run's model files, the DiT's named dit and the VAE's named vae, and with detector the
    shot detector's, DETECTOR (DESIGN.md, Weights: TransNetV2's weights with the shot detector).
    With model_dir, each is read there by its name, and nothing is downloaded: huggingface_hub
    isn't imported here. Without, each must be one of seedvr2x's own (PINNED) in its role, taken
    from Hugging Face's cache, and downloaded into it when missing, in one fetch, whose check of
    the free space counts every file it downloads. ModelError says what to do for each one
    refused, before anything is fetched.

    `download` False: a job that loads no model, one with every segment made (main._run). Its
    DiT's or its VAE's file of seedvr2x's own missing from the cache is then held to its pin,
    said in the log, and neither downloaded nor read (ModelFile.absent): the job is compared with
    its record by the pin's size and SHA-256 (main._model), as its record has them where the file
    passed its pin check. DESIGN.md, Pause and resume (decided on 2026-10-13): "a finished job
    whose own model file isn't there is held to the file's pin and fetches nothing, where today
    it downloads it before saying finished; a file the user names must be there": one in
    --model-dir must, and is checked as any job's. Until then such a job downloaded 16.5 GB to
    say it was finished. The shot detector's, with detector, is fetched as any run's: such a
    job's first pass runs again with the detector where its frame index is gone, and that pass
    loads it (DESIGN.md, Pause and resume, decided on 2026-10-16: "a job that loads no model
    fetches none even where its first pass must run again", where it downloaded the DiT's 16.5
    GB for an assembly alone left). A file in the cache is checked as any job's, by its header
    and its pin."""
    if model_dir is not None:
        logger.info("model files from %s (--model-dir), nothing downloaded", model_dir)
        return ModelFiles(
            ModelFile(dit, model_dir / dit),
            ModelFile(vae, model_dir / vae),
            None,
            Hashes(),
            ModelFile(DETECTOR, model_dir / DETECTOR) if detector else None,
        )
    refusals: list[ModelError] = []
    given: tuple[tuple[Role, str], ...] = (("dit", dit), ("vae", vae))
    for role, name in given:
        pin = PINNED.get(name)
        if pin is None:
            refusals.append(_not_own(role, name))
        elif pin.role is not None and pin.role != role:
            refusals.append(_other_role(role, name, pin.role))
    if refusals:
        raise ModelsRefused(refusals)
    names = (dit, vae, DETECTOR) if detector else (dit, vae)
    # The files the run loads, which it fetches: every one, or where it loads no model, the
    # shot detector's alone, where its first pass runs.
    loads = names if download else names[2:]
    found = fetch({name: PINNED[name] for name in names}, loads)
    where = cache()

    def file(name: str) -> ModelFile:
        from huggingface_hub import file_download

        if name in found:
            return ModelFile(name, found[name])
        logger.info(
            "%s: not in Hugging Face's cache, and not downloaded: the job has every segment made"
            " and loads no model; held to its pin, %s's at %s, SHA-256 %s",
            *(name, REPO, REVISION[:8], PINNED[name].sha256),
        )
        # Where the library would put it (_cached), for the messages: nothing reads it.
        folder = file_download.repo_folder_name(repo_id=REPO, repo_type="model")
        return ModelFile(name, where / folder / "snapshots" / REVISION / name, absent=True)

    return ModelFiles(file(dit), file(vae), where, Hashes(), file(DETECTOR) if detector else None)


def _named(role: Role, say: Say, value: str = "") -> str:
    """How a refusal names the setting that gives the file of this role, "--dit-model NAME" for
    an application whose option it is; the shot detector's has none."""
    setting = ROLE_SETTINGS[role]
    return "seedvr2x's shot detection" if setting is None else say(Setting(setting, value))


def _takes(role: Role, say: Say) -> str:
    """What the setting of this role takes: the own files in that role, "--dit-model: A or B"."""
    own = [each for each, other in PINNED.items() if other.role == role]
    return f"{_named(role, say)}: {_listing(own, 'or')}"


def _not_own(role: Role, name: str) -> ModelUnknown:
    """The refusal of a file named for a role that isn't one of seedvr2x's own, the only ones it
    downloads."""

    def message(say: Say) -> str:
        return (
            f"{_named(role, say, name)}: not one of seedvr2x's own model files, the only ones it"
            f" downloads ({_takes(role, say)}); give {say(MODEL_DIR)} with the directory holding"
            " another file"
        )

    settings = (Setting(ROLE_SETTINGS[role] or "dit_model", name), MODEL_DIR)
    return ModelUnknown(message, role=role, file=name, found="", settings=settings)


def _other_role(role: Role, name: str, owner: Role) -> ModelUnknown:
    """The refusal of a pinned file given as another role's, by its name, before a download (16.5
    GB for a DiT given as the VAE) that the header check would refuse once done."""

    def message(say: Say) -> str:
        return (
            f"{_named(role, say, name)}: {KINDS[owner]}, which {_named(owner, say)} takes"
            f" ({_takes(role, say)})"
        )

    return ModelUnknown(
        message,
        role=role,
        file=name,
        found=owner,
        settings=(Setting(ROLE_SETTINGS[role] or "dit_model", name),),
    )


def cache() -> Path:
    """The directory of Hugging Face's cache, as huggingface_hub reads it when imported:
    HF_HUB_CACHE, else HF_HOME's hub/, else ~/.cache/huggingface/hub (its constants.py), as given,
    links unresolved, where the library resolves it (file_download.py:999)."""
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def cached(name: str) -> Path | None:
    """Where Hugging Face's cache holds REPO's file name at REVISION, one of seedvr2x's own
    (PINNED), found without a request (_cached): what fetch takes; None when it doesn't hold it.
    ModelError when the cache can't be read."""
    found = _cached(cache(), name, PINNED[name])
    return None if found is None else found[0]


def fetch(pins: Mapping[str, Pin], download: Collection[str] | None = None) -> dict[str, Path]:
    """The files of REPO at REVISION that pins name, by name, each from Hugging Face's cache
    (_cached), or downloaded into it, one at a time, never the repository whole (DESIGN.md,
    Weights). Before each download, the free space of the cache's disk is checked against what
    the downloads need from there on (_check_space); HF_HUB_OFFLINE refuses any. The library's log
    goes through seedvr2x's (_library_logging). ModelError says what to do when a file can't be
    had. With `download`, the names of those that may be downloaded (None: every one): any other
    the cache doesn't hold is left out, neither downloaded nor refused (resolve)."""
    from huggingface_hub import constants

    _library_logging()
    where = cache()
    logger.info("%s at %s, in Hugging Face's cache, %s", REPO, REVISION[:8], where)
    found: dict[str, Path] = {}
    for name, pin in pins.items():
        hit = _cached(where, name, pin)
        if hit is None:
            continue
        found[name], linked = hit
        how = "" if linked else "from another revision's download, "
        logger.info("%s: in the cache, %s%s", name, how, found[name])
    missing = [
        name for name in pins if name not in found and (download is None or name in download)
    ]
    if missing and constants.HF_HUB_OFFLINE:
        needed = sum(pins[name].size for name in missing)
        offline = _offline()
        names = _listing(missing)
        holding = "it" if len(missing) == 1 else "them"

        def offline_message(say: Say) -> str:
            return (
                f"{names}: not in Hugging Face's cache, {where}, and {offline} keeps"
                f" the network out: unset it to download {gigabytes(needed)}, set HF_HOME (or"
                f" HF_HUB_CACHE) to a cache holding {holding}, {elsewhere(say)}"
            )

        raise ModelUnreachable(
            offline_message, why="offline", file=names, settings=(MODEL_DIR,), expected=needed
        )
    if missing:
        logger.info(
            "%s is public: seedvr2x downloads from it with no token, never sending one stored for"
            " huggingface_hub",
            REPO,
        )
    for index, name in enumerate(missing):
        # Provisional (implementation, 2026-10-09; DESIGN.md checks each file's size): against
        # the downloads left, this one's and the next ones', so that a default run short of room
        # is refused before its first 16.5 GB, not after.
        left = {each: pins[each] for each in missing[index:]}
        _check_space(where, left)
        found[name] = _download(name, pins[name], where, sum(pin.size for pin in left.values()))
    return {name: found[name] for name in pins if name in found}


def check_pinned(model_files: ModelFiles, *, load: bool = False, whole: bool = False) -> None:
    """Check each of seedvr2x's own files among the run's against its pin (check_pin), by the name
    it was given, wherever it is read from: a file of --model-dir named like one of seedvr2x's own
    is checked as one, its refusal saying how to fetch it again only when given in the pinned
    file's own role (advice). Every file refused is said at once (ModelError, a line each, and
    another for the command a cache's refusal advises). A file seedvr2x doesn't pin is said so in
    the log, and runs as the header check accepted it. With `load`, the run loads the files, and
    a SHA-256 not known yet is left to the load, but with `whole`, where it is read at once, once
    the size has passed (check_pin)."""
    refusals: list[ModelError] = []
    # Each file once, in the first role it was given.
    first: dict[ModelFile, Role] = {}
    for role, file in model_files.roles():
        first.setdefault(file, role)
    for file, role in first.items():
        if file.absent:
            continue  # held to its pin, as resolve said, nothing to read
        name = _pinned_name(file)
        if name is None:
            logger.info(
                "%s: not one of seedvr2x's pinned files, so not checked against a pin", file.name
            )
            continue
        try:
            check_pin(
                *(name, file.path, PINNED[name], model_files.hashes, model_files.cache, role),
                load=load,
                whole=whole,
            )
        except ModelError as error:
            refusals.append(error)
    if refusals:
        raise ModelsRefused(refusals)


def advice(model_files: ModelFiles) -> dict[Role, Message]:
    """How to fetch each of the run's files named like one of seedvr2x's own again, by its role,
    when given in the pinned file's own role: what a refusal of its header check adds
    (weights.check_models), as a refusal of its pin says it (check_pin), from Hugging Face's cache
    or into --model-dir. A pinned name given in another role, which only --model-dir lets
    through (resolve refuses it), gets none: the pinned file fetched again would be refused in
    that role all the same, and leaving out --model-dir would be refused by resolve."""
    found: dict[Role, Message] = {}
    for role, file in model_files.roles():
        name = _pinned_name(file)
        if name is not None and PINNED[name].role == role:
            found[role] = _again(name, file.path, model_files.cache)
    return found


@dataclass(frozen=True)
class Smaller:
    """A smaller file of a 7B than the one the job runs (smaller_files): what it is called,
    `label`; the pinned file, `name`, and its size in bytes; the format the planner counts it by,
    `architecture` (peaks.DITS); and `model`, "the sharp 7B" or "the 7B" where the job's file is
    not a pinned one and so does not say which 7B it holds, else None."""

    label: str
    name: str
    size: int
    architecture: Architecture
    model: str | None = None


@dataclass(frozen=True)
class Swaps:
    """What a plan makes of a job in a smaller file's format (smaller_planning's result for the
    apps that name the files better than the plan's own): the DiT's blocks it swaps, and the
    longest window it runs in latents, None for a picture (one latent, no window). An app words
    the files, the options that choose them and these figures itself."""

    swapped: int
    window: int | None = None


SHARP, REGULAR = "the sharp 7B", "the 7B"
# Which 7B each of seedvr2x's pinned DiT files holds (DESIGN.md, Weights): the tensors of the fp16
# and int8 files tell the two apart in no way, so it is stated here, and tests/test_pull.py holds
# the table to PINNED's DiT files.
MODEL_OF = {
    "seedvr2x_ema_7b_fp16.safetensors": REGULAR,
    "seedvr2x_ema_7b_sharp_fp16.safetensors": SHARP,
    "seedvr2x_ema_7b_int8_convrot.safetensors": REGULAR,
    "seedvr2x_ema_7b_sharp_int8_convrot.safetensors": SHARP,
    "seedvr2x_ema_7b_dyn.gguf": REGULAR,
    "seedvr2x_ema_7b_sharp_dyn.gguf": SHARP,
}
# The smaller files of each 7B, in the order of their size (DESIGN.md, Weights, "The dynamic GGUF
# before v0.1.0 too": "where the card or the host's memory is short for float16, the plan names
# the smaller files"): the int8 file and the dynamic GGUF.
_SMALLER = {
    model: tuple(
        Smaller(label, name, PINNED[name].size, architecture) for label, name, architecture in named
    )
    for model, named in {
        REGULAR: (
            ("int8", "seedvr2x_ema_7b_int8_convrot.safetensors", weights.DIT_7B_INT8),
            ("the dynamic GGUF", "seedvr2x_ema_7b_dyn.gguf", weights.DIT_7B_DYN),
        ),
        SHARP: (
            ("int8", "seedvr2x_ema_7b_sharp_int8_convrot.safetensors", weights.DIT_7B_INT8),
            ("the dynamic GGUF", "seedvr2x_ema_7b_sharp_dyn.gguf", weights.DIT_7B_SHARP_DYN),
        ),
    }.items()
}


def smaller_files(name: str, dit_model: Architecture) -> list[Smaller]:
    """The pinned files smaller than the DiT file `name`, which is in the format dit_model: for
    fp16 its int8 file and its dynamic GGUF, for int8 the GGUF, for a GGUF none. A pinned file
    (MODEL_OF) gets those of its own 7B. Any other file, numz's or a renamed one, does not say
    which 7B it holds, and is never guessed from its name: the sharp 7B's files and the 7B's
    both, each marked with its model (Smaller.model)."""
    held = MODEL_OF.get(name)
    models = [(held, None)] if held is not None else [(SHARP, SHARP), (REGULAR, REGULAR)]
    if dit_model == weights.DIT_7B:
        keep = 0
    elif dit_model == weights.DIT_7B_INT8:
        keep = 1
    else:
        return []
    return [replace(one, model=said) for model, said in models for one in _SMALLER[model][keep:]]


def smaller_planning[T](
    name: str, dit_model: Architecture, plans: Callable[[Architecture], T | None]
) -> list[tuple[Smaller, T]]:
    """The smaller files (smaller_files) the job plans with, each with what `plans` makes of its
    format, in the order of their size; `plans` returns None for a format the job doesn't plan
    with (the planner's refusal, caught there). Planning imports no torch and is cheap."""
    found: list[tuple[Smaller, T]] = []
    for each in smaller_files(name, dit_model):
        made = plans(each.architecture)
        if made is not None:
            found.append((each, made))
    return found


def check_pin(
    name: str,
    path: Path,
    pin: Pin,
    hashes: Hashes,
    cache: Path | None,
    role: Role | None = None,
    *,
    load: bool = False,
    whole: bool = False,
) -> None:
    """Refuse (ModelError) the file at path unless it is REPO's file name at REVISION as pin pins
    it: its size first, by a stat, then its SHA-256, read once per run (hashes). With `load`, a
    file the run loads, whose SHA-256 no whole read gave yet (Hashes.proven), is read by its load
    alone: its size checked here, a SHA-256 a record vouches for held to the pin, the pin's
    proved at the load's end, refused there in the same words (Hashes.expect, Hashes.proved;
    DESIGN.md, Weights, decided on 2026-11-03); with `whole` too, a SHA-256 nothing vouches for
    is read now, once its size has passed, as a job's record that no longer vouches for its file
    asks (DESIGN.md, Pause and resume, decided on 2026-11-03). The refusal says
    both sizes or both hashes, and how to fetch the file again, into Hugging Face's cache (its
    directory, cache) or into --model-dir (cache None), unless it was given in a role (role) other
    than the pinned file's, as advice has it. Nothing is deleted, where numz's downloader deletes
    a file named like its own whose SHA-256 differs (bug 22, item 4)."""
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ModelUnreadable(f"{path}: {error.strerror}", file=path.name) from None

    def refusal(found: str, pinned: str, sha256: bool = False) -> ModelNotPinned:
        again = role is None or role == pin.role

        def message(say: Say) -> str:
            said = (
                f"{path}: {found}, where {REPO}'s {name} at {REVISION[:8]} has {pinned}: not the"
                " file seedvr2x pins, left as it is"
            )
            if again:
                said += f"; {_again(name, path, cache)(say)}"
            return said

        return ModelNotPinned(
            message,
            role=role or "",
            file=name,
            figure="sha256" if sha256 else "size",
            expected=pin.sha256 if sha256 else pin.size,
            found=found.removeprefix("SHA-256 ") if sha256 else size,
            settings=() if cache is not None or not again else (MODEL_DIR,),
            command=_command(name, cache) if again and cache is not None else "",
        )

    if size != pin.size:
        raise refusal(f"{size:,} bytes", f"{pin.size:,}")
    said = f"{name}: its size and SHA-256 as seedvr2x pins them, {REPO} at {REVISION[:8]}"
    if load and not hashes.proven(path) and (hashes.known(path) or not whole):
        # Its load hashes it as it reads it, and refuses it at its end in the same words
        # (DESIGN.md, Weights, decided on 2026-11-03): one read, not two. A SHA-256 a job's
        # record vouches for (Hashes.vouch) is held to the pin at once, as before; the load then
        # proves the pin's, and a file changed since is refused in the pin's words, which say
        # how to fetch it again.
        if hashes.known(path) and (vouched := hashes.sha256(path)) != pin.sha256:
            raise refusal(f"SHA-256 {vouched}", pin.sha256, True)
        hashes.expect(
            path, pin.sha256, lambda found: refusal(f"SHA-256 {found}", pin.sha256, True), said
        )
        logger.info(
            "%s: its size as seedvr2x pins it, %s at %s; its SHA-256 checked as it loads",
            *(name, REPO, REVISION[:8]),
        )
        return
    digest = hashes.sha256(path)
    if digest != pin.sha256:
        raise refusal(f"SHA-256 {digest}", pin.sha256, True)
    logger.info("%s", said)


def url(name: str) -> str:
    """The URL of REPO's file name at REVISION, as the Hub serves it."""
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"


def _pinned_name(file: ModelFile) -> str | None:
    """The name of the pinned file the run's file is named like, if any: the last part of the
    name it was given. Provisional (implementation, 2026-10-09): DESIGN.md says by its name, and
    --dit-model sub/<name> in --model-dir names <name> all the same."""
    name = PurePath(file.name).name
    return name if name in PINNED else None


def _command(name: str, cache: Path) -> str:
    """The command of huggingface_hub that downloads REPO's file name again into the cache at
    cache, whose copy is refused."""
    command = shlex.join(
        ["hf", "download", REPO, name, "--revision", REVISION, "--cache-dir", str(cache)]
    )
    return f"{command} --force-download"


def _again(name: str, path: Path, cache: Path | None) -> Callable[[Say], str]:
    """How to fetch REPO's file name again, its copy at path refused: from Hugging Face's cache
    (cache, its directory), remove the file its link leads to, which the next run downloads
    again, or have huggingface_hub's command download it again; into a directory of model files
    (cache None), its URL."""
    if cache is None:
        return lambda say: (
            f"fetch it again from {url(name)}, or leave out {say(MODEL_DIR)}, which takes it from"
            " Hugging Face's cache"
        )
    command = _command(name, cache)
    # The command on a line of its own, at the message's end, so that it can be copied whole
    # (DESIGN.md, What it prints: a refusal's advised command "on a line of its own", decided on
    # 2026-10-29 for this one).
    return lambda _say: (
        f"remove {path.resolve()}, which the next run downloads again, or download it again with"
        f" huggingface_hub's command:\n{command}"
    )


def _cached(where: Path, name: str, pin: Pin) -> tuple[Path, bool] | None:
    """Where Hugging Face's cache, where, holds REPO's file name at REVISION, pinned by pin, found
    without a request: its link in the revision's snapshot (True), else its blob, by its SHA-256
    (False); None when neither is there. ModelError when the cache can't be read."""
    from huggingface_hub import file_download

    try:
        # The library's own lookup, the cache alone, with no request: a file found there needs no
        # download, so no space and no check, as hf_hub_download itself returns it for a commit
        # hash (huggingface_hub 1.33.0, file_download.py:1099-1113).
        linked = file_download.try_to_load_from_cache(
            REPO, name, cache_dir=where, revision=REVISION
        )
        if isinstance(linked, str):
            return Path(linked), True
        # The same bytes downloaded at another revision: the repo's head, 502436ca, holds v1's
        # files unchanged, and hf download takes the head without --revision. The library keeps
        # an LFS file once, in the repository's blobs/, named by its ETag, the Hub's SHA-256 of
        # it (file_download.py:1200; a revision's cached listing gives the same, 1923-1926), as a
        # link into the shared store of Xet files when it is there (utils/_shared_blobs.py:
        # 16-19), whose own names are Xet hashes, which seedvr2x doesn't pin. Asked to download
        # it, the library would link that blob into the revision's snapshot with no download
        # (file_download.py:1256-1261), but only after a request, and offline not at all: it
        # links one without the network only from a listing of the revision cached by
        # snapshot_download (1736-1747), never with local_files_only (1724-1734). So the blob is
        # taken where it is, whatever its size: the pin's check refuses any but the pinned one,
        # saying to remove it. Provisional (implementation, 2026-10-09; DESIGN.md doesn't say).
        folder = file_download.repo_folder_name(repo_id=REPO, repo_type="model")
        blob = where / folder / "blobs" / pin.sha256
        if blob.is_file():
            return blob, False
    except OSError as error:
        # The library's lookup lists the snapshots' directory (file_download.py:1606), and Python
        # 3.13's is_file raises on a directory it can't search (EACCES).
        raise _unreadable(where, error) from None
    return None


def _check_space(where: Path, downloads: Mapping[str, Pin]) -> None:
    """Refuse (ModelError) when the disk of Hugging Face's cache, where, has less free space than
    downloads need: each file's size, once. The refusal names the partial files this repository's
    downloads left there, and deletes none."""
    # What a download writes on the cache's disk, huggingface_hub 1.33.0 with hf-xet 1.6.0: the
    # file, once. It is written beside its blob as <blob>.<8 hex digits>.incomplete, then renamed
    # into place (file_download.py:1968-2045, the rename at 2094), and a Xet download renamed again
    # into the shared store, <cache>/blobs/, its blob a link to it (utils/_shared_blobs.py:
    # 432-480); the links, the lock and the store's list of references take a few blocks. hf_xet
    # keeps no chunk cache for a download: its download group gives it none
    # (xet_pkg/src/xet_session/file_download_group.rs:153), and its default build sets its size to
    # 0 (hf_xet/Cargo.toml:44). Its shard cache serves uploads only
    # (xet_data/src/processing/file_upload_session.rs:95), and its logs go to HF_XET_CACHE/logs,
    # pruned to 250 MB (xet_runtime/src/config/groups/log.rs:51). The library's own check of the
    # space only warns (file_download.py:758-781).
    from huggingface_hub import file_download

    blobs = where / file_download.repo_folder_name(repo_id=REPO, repo_type="model") / "blobs"
    needed = sum(pin.size for pin in downloads.values())
    try:
        existing = next(path for path in (blobs, *blobs.parents) if path.exists())
        free = shutil.disk_usage(existing).free
        if free >= needed:
            return
        # A download killed before its end (SIGKILL, a power cut) leaves its partial file: each
        # download writes one of a name of its own, never taken up again (file_download.py:
        # 1995-2004), and older huggingface_hub releases named it <blob>.incomplete.
        partial: list[tuple[Path, int]] = []
        for path in sorted(blobs.glob("*.incomplete")):
            try:
                partial.append((path, path.stat().st_size))
            except FileNotFoundError:
                continue  # renamed into place meanwhile, by a download under way
    except OSError as error:
        raise _unreadable(where, error) from None
    listed = _listing(list(downloads))
    left = ""
    if partial:
        listing = _listing([f"{path} ({gigabytes(size)})" for path, size in partial])
        if len(partial) == 1:
            said = (
                "was left there by a download stopped before its end, or is one under way: no"
                " later download takes it up, and removing it frees its space"
            )
        else:
            said = (
                "were left there by downloads stopped before their end, or are under way: no"
                " later download takes them up, and removing them frees their space"
            )
        left = f". {listing} {said}"

    def refusal(say: Say) -> str:
        return (
            f"Hugging Face's cache, {where}: {gigabytes(free)} free on its disk, where downloading"
            f" {listed} needs {gigabytes(needed)}: free some space there, set"
            f" HF_HOME (or HF_HUB_CACHE) to a directory on a disk with room, {elsewhere(say)}"
            f"{left}"
        )

    raise ModelNoRoom(refusal, need=needed, free=free, settings=(MODEL_DIR,))


def _download(name: str, pin: Pin, where: Path, needed: int) -> Path:
    """REPO's file name at REVISION, pinned by pin, downloaded into Hugging Face's cache, where,
    with no token: the repo is public, and token=False keeps a token the user stored from being
    sent (utils/_headers.py:125-133). Each of the library's failures is refused (ModelError),
    saying what to do; needed is what the downloads left need on the cache's disk, in bytes, this
    one's and the next ones', as _check_space checked it."""
    from huggingface_hub import constants, file_download
    from huggingface_hub.errors import (
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RemoteEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    logger.info(
        "%s: not in the cache, downloading %s from %s at %s",
        *(name, gigabytes(pin.size), REPO, REVISION[:8]),
    )
    started = time.monotonic()

    def again(say: Say) -> str:
        return f"run again, which downloads it again, {elsewhere(say)}"

    # The errors are huggingface_hub 1.33.0's (errors.py, file_download.py, utils/_xet.py) and
    # those of hf-xet 1.6.0, which downloads Xet files (xet_pkg/src/error.rs:293-318), every one
    # caught by name: no bare Exception. Each message is built from locals bound in its clause
    # (Python unbinds the clause's name at its end, and a message is worded later, by an app).
    try:
        # Strict typing finds its signature partly unknown, for user_agent's bare dict and
        # tqdm_class's bare tqdm, neither of which is passed.
        path = file_download.hf_hub_download(  # pyright: ignore[reportUnknownMemberType]
            REPO, name, revision=REVISION, cache_dir=where, token=False
        )
    except LocalEntryNotFoundError as error:
        # The Hub not reached, or not answering as the Hub, the file not in the cache
        # (file_download.py:1931-1966): its cause the network's error or the Hub's refusal of the
        # file (1959-1965), but a 401's, raised as itself (1948-1953), or a FileMetadataError, an
        # answer lacking the Hub's headers (1954-1958).
        cause, endpoint = _cause(error), constants.ENDPOINT
        if _refused(error):
            raise _unreachable(
                name,
                "refused",
                lambda say: f"{name}: the Hub refused it ({cause}): {unavailable(say)}",
            ) from None
        if _not_the_hub(error):
            raise _unreachable(
                name,
                "not-the-hub",
                lambda say: (
                    f"{name}: not in the cache, and {endpoint} answered the request to"
                    f" download it without the Hub's headers ({cause}): {check_endpoint(say)}"
                ),
            ) from None
        raise _unreachable(
            name,
            "unreachable",
            lambda say: (
                f"{name}: not in the cache, and {endpoint} couldn't be reached to download it"
                f" ({cause}): run again once it can be, {elsewhere(say)}"
            ),
        ) from None
    except RemoteEntryNotFoundError:
        raise _unreachable(
            name,
            "not-found",
            lambda say: (
                f"{name}: not in {REPO} at {REVISION}, the revision this seedvr2x pins (the Hub:"
                f" entry not found): download it elsewhere and give {say(MODEL_DIR)} with its"
                " directory"
            ),
        ) from None
    except RevisionNotFoundError:
        # Unreachable at a commit hash as the Hub answered on 2026-10-09 (the review of this
        # step): a commit it doesn't have gets "Entry Not Found", refused above. Kept, for a Hub
        # that would answer RevisionNotFound, which the library raises on that error code
        # (utils/_http.py:850-852).
        raise _unreachable(
            name,
            "not-found",
            lambda say: (
                f"{name}: {REPO} has no revision {REVISION}, the one this seedvr2x pins (the Hub:"
                f" revision not found): download the files elsewhere and give {say(MODEL_DIR)}"
                " with their directory"
            ),
        ) from None
    except RepositoryNotFoundError:
        raise _unreachable(
            name,
            "not-found",
            lambda say: (
                f"{name}: the Hub has no repository {REPO}, or not a public one (the Hub:"
                " repository not found): download the files elsewhere and give"
                f" {say(MODEL_DIR)} with their directory"
            ),
        ) from None
    except HfHubHTTPError as error:
        said = _words(error)
        if _refused(error):
            raise _unreachable(
                name,
                "refused",
                lambda say: f"{name}: the Hub refused it ({said}): {unavailable(say)}",
            ) from None
        raise _unreachable(
            name,
            "failed",
            lambda say: f"{name}: the Hub refused its download ({said}): {again(say)}",
        ) from None
    except OSError as error:
        said = _words(error)
        if _xet_refused(error):
            raise _unreachable(
                name,
                "refused",
                lambda say: f"{name}: the Hub's storage refused it ({said}): {unavailable(say)}",
            ) from None
        number = error.errno if error.errno is not None else _os_error(error)
        if number is not None and number in UNWRITABLE:
            reason = os.strerror(number)

            def cannot_write(say: Say) -> str:
                return (
                    f"{name}: Hugging Face's cache, {where}, can't be written ({reason}):"
                    f" set HF_HOME (or HF_HUB_CACHE) to a directory you can write, on a disk with"
                    f" {gigabytes(needed)} free, {elsewhere(say)}"
                )

            raise ModelUnreadable(cannot_write, file=name, settings=(MODEL_DIR,)) from None
        # The network lost during a Xet download (ConnectionError, TimeoutError), or another I/O
        # error; a plain download's sizes not matching (file_download.py:491).
        raise _unreachable(
            name,
            "failed",
            lambda say: f"{name}: its download failed ({said}): {again(say)}",
        ) from None
    except (*_request_errors(), RuntimeError) as error:
        # A plain download's network lost past its retries (file_download.py:468-472), a proxy's
        # failure (1822), a body that doesn't decode or redirects without end (httpx's
        # DecodingError and TooManyRedirects, RequestErrors but not TransportErrors); hf_xet's
        # other failures, its data's integrity among them (xet_pkg/src/error.rs:309-317).
        said = _words(error)
        raise _unreachable(
            name,
            "failed",
            lambda say: f"{name}: its download failed ({said}): {again(say)}",
        ) from None
    except ValueError as error:
        # hf_xet's Configuration and InvalidTaskID errors (xet_pkg/src/error.rs:308), and the
        # library's when the Hub's answer lacks the Xet headers it needs (utils/_xet.py:131-132,
        # 179-181), as from a proxy or an HF_ENDPOINT mirror that doesn't pass them on: no rerun
        # changes them.
        said = _words(error)
        raise _unreachable(
            name,
            "failed",
            lambda say: f"{name}: its download failed ({said}): {check_endpoint(say)}",
        ) from None
    logger.info("%s: downloaded in %.0f s, %s", name, spent(time.monotonic() - started), path)
    return Path(path)


def _unreachable(name: str, why: str, message: Message) -> ModelUnreachable:
    """The refusal of a download that failed: `why` it did (ModelUnreachable), in `message`."""
    return ModelUnreachable(message, why=why, file=name, settings=(MODEL_DIR,))


def _request_errors() -> tuple[type[Exception], ...]:
    """The base class of the request errors of the HTTP library the installed huggingface_hub
    uses: httpx's up to 1.x (1.33.0), httpx2's from 2.x (2.3.0, errors.py:6, whose requests raise
    httpx2's own classes, not httpx's). Each is taken only if it is installed, imported here and
    not at import: the Hub has imported the one it uses by now, and the other may be absent."""
    found: list[type[Exception]] = []
    for module in ("httpx", "httpx2"):
        try:
            found.append(importlib.import_module(module).RequestError)
        except ImportError:
            continue
    return tuple(found)


def _library_logging() -> None:
    """huggingface_hub's log and httpx's in seedvr2x's: each of the library's records once, through
    seedvr2x's handler, but the Hub's advice to set a token; httpx's and httpcore's warnings
    alone."""
    # httpx logs every request at INFO (httpx 0.28.1: _client.py:1025-1032, 1740-1747), the
    # library's own among them (its registry of AI agents, utils/_detect_agent.py:197), which a
    # run's log isn't for; httpcore's are debug traces.
    # huggingface_hub 2.x moved to httpx2, a fork whose loggers are named httpx2 and httpcore2
    # (httpx2 2.16.0: _client.py:110, httpcore2/_sync/http11.py:27); 1.x, the CLI's lock, keeps
    # httpx's. Both generations' names are set, whichever the Hub uses.
    for name in ("httpx", "httpcore", "httpx2", "httpcore2"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # huggingface_hub's logging module gives the library's logger a handler of its own as it is
    # imported, writing the message alone to stderr, and leaves the logger propagating to the
    # root's handler, seedvr2x's (its utils/logging.py:67-70, 179): each warning came out twice.
    # Its handler is removed, rather than its propagation stopped, so that its records come out
    # once, through seedvr2x's handler, with the time, the level and the logger's name as every
    # line of the log. The module is imported first, which adds the handler unless it is there
    # already: what seedvr2x imported of the library before doesn't matter (its constants don't
    # import it, its file_download does). The library's level stays its own, WARNING (40, 67-70).
    # It logs in a run of seedvr2x only from here on: its lookup and its download (fetch).
    importlib.import_module("huggingface_hub.utils.logging")
    library = logging.getLogger("huggingface_hub")
    for handler in list(library.handlers):
        library.removeHandler(handler)
    # The Hub's advice to set a token (TOKEN_ADVICE), alone, dropped: seedvr2x sends none by
    # design (token=False, _download), the repository being public, and says so (fetch).
    advising = logging.getLogger("huggingface_hub.utils._http")
    if _no_token_advice not in advising.filters:
        advising.addFilter(_no_token_advice)


def _no_token_advice(record: logging.LogRecord) -> bool:
    """False for the Hub's advice to set a token (TOKEN_ADVICE), which is dropped; True for any
    other record."""
    return TOKEN_ADVICE not in record.getMessage()


def _refused(error: BaseException) -> bool:
    """Whether error, or an error of its chain of causes, is the Hub refusing the file to anyone
    asking without a token, which no rerun changes: an HTTP 401 or 403, or the repository
    disabled (huggingface_hub's DisabledRepoError, errors.py:365)."""
    from huggingface_hub.errors import DisabledRepoError, HfHubHTTPError

    current: BaseException | None = error
    while current is not None:
        if isinstance(current, DisabledRepoError) or (
            isinstance(current, HfHubHTTPError) and current.response.status_code in (401, 403)
        ):
            return True
        current = current.__cause__
    return False


def _not_the_hub(error: BaseException) -> bool:
    """Whether error, or an error of its chain of causes, is the library's FileMetadataError: the
    file's HEAD request answered without a header the Hub sends, its commit, its ETag or a Xet
    file's size (file_download.py:1789-1809), as by an HF_ENDPOINT that isn't a Hub, a proxy or a
    captive portal (1840-1846), where the Hub not reached is a network's error (1825-1828)."""
    from huggingface_hub.errors import FileMetadataError

    current: BaseException | None = error
    while current is not None:
        if isinstance(current, FileMetadataError):
            return True
        current = current.__cause__
    return False


def _xet_refused(error: OSError) -> bool:
    """Whether error is hf_xet's for a file its storage doesn't have or won't give
    (XetObjectNotFoundError, XetAuthenticationError: hf-xet 1.6.0, xet_pkg/src/error.rs:270-274,
    303-304), which no rerun changes. hf_xet is looked up among the modules imported, never
    imported here: its import writes a log file under HF_HOME (xet_pkg/src/lib.rs:82-94), and an
    error of its own means it is imported already."""
    xet = sys.modules.get("hf_xet")
    return xet is not None and isinstance(
        error, (xet.XetObjectNotFoundError, xet.XetAuthenticationError)
    )


def _unreadable(where: Path, error: OSError) -> ModelUnreadable:
    """The refusal of Hugging Face's cache, where, which can't be read (error)."""
    what = error.strerror or _words(error)
    if error.filename is not None:
        what = f"{what}: {error.filename}"

    def message(say: Say) -> str:
        return (
            f"Hugging Face's cache, {where}, can't be read ({what}): make it readable, set"
            f" HF_HOME (or HF_HUB_CACHE) to another cache, {elsewhere(say)}"
        )

    return ModelUnreadable(message, settings=(MODEL_DIR,))


def _offline() -> str:
    """The variable that keeps huggingface_hub offline, in words: HF_HUB_OFFLINE, or
    TRANSFORMERS_OFFLINE, which the library reads when HF_HUB_OFFLINE is unset or empty
    (constants.py:194)."""
    if not os.environ.get("HF_HUB_OFFLINE") and os.environ.get("TRANSFORMERS_OFFLINE"):
        return "TRANSFORMERS_OFFLINE, which huggingface_hub takes for HF_HUB_OFFLINE,"
    return "HF_HUB_OFFLINE"


def _state(path: Path) -> tuple[int, int, int, int]:
    """The file at path as a stat sees it: its device and inode, its size, and its modification
    time in nanoseconds."""
    status = path.stat()
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns


def _os_error(error: OSError) -> int | None:
    """The number of the OS error an OSError of hf_xet's names in its message (OS_ERROR), if any."""
    match = OS_ERROR.search(str(error))
    return int(match[1]) if match else None


def _cause(error: BaseException) -> str:
    """The first error of error's chain of causes, in words: the network's own, under the
    library's."""
    while error.__cause__ is not None:
        error = error.__cause__
    return _words(error)


def _words(error: BaseException) -> str:
    """An error's message on one line, or its type's name when it has none."""
    return " ".join(str(error).split()) or type(error).__name__


def gigabytes(count: int) -> str:
    """A number of bytes, in GB (10^9 bytes, as DESIGN.md gives the files' sizes) from 0.1 GB; the
    apps say a file's size with it."""
    return f"{count / 1e9:.1f} GB" if count >= 100_000_000 else f"{count:,} bytes"


def _listing(names: list[str], word: str = "and") -> str:
    """Names, joined by commas, the last by word."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} {word} {names[-1]}"


# SIGTERM during a download (main.py raises it as Terminated around resolve, stop.py's
# terminable): a plain download is Python's loop over the response (file_download.py:459-467),
# between whose steps the handler runs, and the library's finally removes the partial file
# (2042-2045). A Xet download, huggingface_hub 1.33.0 with hf-xet 1.6.0, read in their sources:
# xet_get queues the file and waits for it as its download group's with block ends
# (file_download.py:589-600; hf_xet/src/py_file_download_group.rs:135-151, 163-179), the download a
# task of its own (xet_pkg/src/xet_session/file_download_group.rs:324-343). That wait is
# blocking_call_with_signal_check (hf_xet/src/utils.rs:38-78): the main thread waits 100 ms at a
# time, the GIL released, and between two waits runs Python's signal handlers (check_signals). So
# SIGTERM's Terminated, as Ctrl-C's KeyboardInterrupt, comes out of the with block within 100 ms,
# and the same finally removes the partial file. hf_xet's own SIGINT handler
# (hf_xet/src/legacy/runtime.rs:17-28) belongs to its legacy functions (legacy/functions.rs:92,
# 174, 215, 250), which xet_get doesn't call. On KeyboardInterrupt xet_get aborts its session
# (file_download.py:601-603), on Terminated it doesn't: the download's task writes on into the
# file it opened once (xet_data/src/processing/file_download_session.rs:266-296,
# file_reconstruction/file_reconstructor.rs:138), removed by then, whose blocks the system frees
# as the process exits, at once (main returns 143); the runtime's own shutdown waits 5 s at most
# (xet_runtime/src/core/runtime/native.rs:592-631).
