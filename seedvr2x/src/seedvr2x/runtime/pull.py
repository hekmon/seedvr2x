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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath

from seedvr2x.media import files
from seedvr2x.runtime.weights import ModelError, Role

logger = logging.getLogger(__name__)

REPO = "hekmon/seedvr2x"
# The upload of 2026-10-08, v1's files and phase 2's, each read back equal to SHA256SUMS. The
# repo's head moved to 502436ca the same day with a new card alone, no model byte changed: the pin
# stays (DESIGN.md, Weights).
REVISION = "c14a2bc4aab04cf38ad9b0d324014c4213f07048"


@dataclass(frozen=True)
class Pin:
    """A file of REPO at REVISION: its size, in bytes, its SHA-256, in hex, and the role it takes
    as a model file (the DiT's or the VAE's), None for another file."""

    size: int
    sha256: str
    role: Role | None = None


# v1's files (DESIGN.md, Weights), read from REVISION on 2026-10-09 without downloading them: each
# SHA-256 is the file's line of SHA256SUMS there, which the Hub's API gives as its LFS SHA-256 too,
# and each size the API's (HfApi.get_paths_info); both as the model conversation recorded them at
# the upload. tests/test_pull.py holds them to the revision. TransNetV2's weights come with the
# shot detector.
PINNED: dict[str, Pin] = {
    "seedvr2x_ema_7b_fp16.safetensors": Pin(
        16_479_335_080, "071cab5e5ef7a4471e1df0023c26cc16deeb14e58f8ad5c9196d2a08f96da5f2", "dit"
    ),
    "seedvr2x_ema_7b_sharp_fp16.safetensors": Pin(
        16_479_335_088, "5eb47fdee4b620765573a697b6f82442234e7dc7aa0beeb3ba72fc63b917a817", "dit"
    ),
    "seedvr2x_ema_vae_fp16.safetensors": Pin(
        501_325_454, "b9c6ebf0b14107be595825f476b9f89029a067d5b13e9db39c1351a608265468", "vae"
    ),
}
# Every file's SHA-256 at REVISION, a few kilobytes, which tests/test_pull.py fetches to hold the
# pins to the revision: its size and SHA-256 read from the revision as the files' are (a git blob,
# not LFS), its SHA-256 the one the model conversation recorded.
SUMS = "SHA256SUMS"
SUMS_PIN = Pin(2_335, "57cf5a7b00bec79fac929f6776d1819ef80f28fc957a8b131cf2b6887cd09b53")

# The option naming each role's file, and what one of seedvr2x's own files in that role is.
OPTIONS: dict[Role, str] = {"dit": "--dit-model", "vae": "--vae-model"}
KINDS: dict[Role, str] = {"dit": "one of seedvr2x's DiTs", "vae": "seedvr2x's VAE"}

# The errors of a cache that can't be written: permissions, a read-only or full disk, a quota.
UNWRITABLE = frozenset({errno.EACCES, errno.EPERM, errno.EROFS, errno.ENOSPC, errno.EDQUOT})
# hf_xet raises an I/O error as an OSError of its message alone, which ends as Rust writes an OS
# error, "(os error 28)" (hf-xet 1.6.0: xet_pkg/src/error.rs:214-218 and 307): its number is read
# back from it.
OS_ERROR = re.compile(r"\(os error (\d+)\)")
# What every refusal of a file seedvr2x can't have offers besides.
ELSEWHERE = "or download the files elsewhere and give --model-dir with their directory"
# What a refusal says of a file the Hub refuses to anyone, which no rerun changes.
UNAVAILABLE = (
    "the repository or the file is unavailable: download the files elsewhere and give --model-dir"
    " with their directory"
)
# What a refusal says of answers that aren't the Hub's as the library expects them, as from an
# HF_ENDPOINT that isn't a Hub or a proxy on the way: a rerun alone changes nothing.
CHECK_ENDPOINT = f"check HF_ENDPOINT, if set, and any proxy on the way to the Hub, {ELSEWHERE}"
# The Hub's advice to set a token, which it sends in a warning header answering a request made
# without one, not every time, and huggingface_hub logs as a warning (utils/_http.py:972-993);
# logged on 2026-10-09 answering SHA256SUMS' download: "Warning: You are sending unauthenticated
# requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster
# downloads."
TOKEN_ADVICE = "You are sending unauthenticated requests to the HF Hub"


class Hashes:
    """The SHA-256 of the run's model files, each read once (16.5 GB for the DiT, tens of seconds:
    the pinned check and the manifest's record take the same), with each file as it was then, its
    device and inode, size and modification time, which its load finds again (unchanged)."""

    def __init__(self) -> None:
        self._read: dict[Path, tuple[str, tuple[int, int, int, int]]] = {}

    def sha256(self, path: Path) -> str:
        """The SHA-256 of the file at path, in hex, read the first time only, its time logged."""
        if path not in self._read:
            state = _state(path)
            started = time.monotonic()
            digest = files.sha256(path)
            logger.info("%s: SHA-256 %s, in %.1f s", path.name, digest, time.monotonic() - started)
            self._read[path] = (digest, state)
        return self._read[path][0]

    def unchanged(self, path: Path) -> None:
        """Refuse (ModelError) the file at path if it was hashed this run and isn't that file any
        more: another size, modification time or inode. A file runs the bytes its hash checked:
        its load checks it before and after (runtime/model.py)."""
        if path not in self._read:
            return
        # Provisional (implementation, 2026-10-09; DESIGN.md doesn't say how): a stat, not a second
        # read of 16.5 GB, so a file rewritten in place to the same size within its filesystem's
        # timestamp granularity would pass. The inode is compared too, correctness first, though
        # a FUSE filesystem mounted without libfuse's use_ino option numbers its inodes itself and
        # may give a file that didn't change another one: such a file is refused, and runs again.
        try:
            now = _state(path)
        except OSError as error:
            raise ModelError(f"{path}: {error.strerror}, since its SHA-256 was read") from None
        if now != self._read[path][1]:
            raise ModelError(
                f"{path}: changed since its SHA-256 was read, before its load ended: run again,"
                " which checks it again"
            )


@dataclass(frozen=True)
class ModelFile:
    """A model file of the run: the name it was given (--dit-model, --vae-model) and its path, in
    --model-dir or in Hugging Face's cache."""

    name: str
    path: Path


@dataclass(frozen=True)
class ModelFiles:
    """The run's model files, the DiT's and the VAE's; the directory of Hugging Face's cache they
    come from, None for --model-dir's; and their SHA-256, each read once (hashes)."""

    dit: ModelFile
    vae: ModelFile
    cache: Path | None
    hashes: Hashes


def resolve(model_dir: Path | None, dit: str, vae: str) -> ModelFiles:
    """The run's model files, the DiT's named dit and the VAE's named vae. With model_dir, each is
    read there by its name, and nothing is downloaded: huggingface_hub isn't imported here.
    Without, each must be one of seedvr2x's own (PINNED) in its role, taken from Hugging Face's
    cache, and downloaded into it when missing (fetch). ModelError says what to do for each one
    refused, before anything is fetched."""
    if model_dir is not None:
        logger.info("model files from %s (--model-dir), nothing downloaded", model_dir)
        return ModelFiles(
            ModelFile(dit, model_dir / dit), ModelFile(vae, model_dir / vae), None, Hashes()
        )
    refusals: list[str] = []
    given: tuple[tuple[Role, str], ...] = (("dit", dit), ("vae", vae))
    for role, name in given:
        option, pin = OPTIONS[role], PINNED.get(name)
        own = [each for each, other in PINNED.items() if other.role == role]
        takes = f"{option}: {_listing(own, 'or')}"
        if pin is None:
            refusals.append(
                f"{option} {name}: not one of seedvr2x's own model files, the only ones it"
                f" downloads ({takes}); give --model-dir with the directory holding another file"
            )
        elif pin.role is not None and pin.role != role:
            # Refused by its name, before a download (16.5 GB for a DiT given as the VAE) that the
            # header check would refuse once done.
            refusals.append(
                f"{option} {name}: {KINDS[pin.role]}, which {OPTIONS[pin.role]} takes ({takes})"
            )
    if refusals:
        raise ModelError("\n".join(refusals))
    found = fetch({name: PINNED[name] for name in (dit, vae)})
    return ModelFiles(ModelFile(dit, found[dit]), ModelFile(vae, found[vae]), cache(), Hashes())


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


def fetch(pins: Mapping[str, Pin]) -> dict[str, Path]:
    """The files of REPO at REVISION that pins name, by name, each from Hugging Face's cache
    (_cached), or downloaded into it, one at a time, never the repository whole (DESIGN.md,
    Weights). Before each download, the free space of the cache's disk is checked against what
    the downloads need from there on (_check_space); HF_HUB_OFFLINE refuses any. The library's log
    goes through seedvr2x's (_library_logging). ModelError says what to do when a file can't be
    had."""
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
    missing = [name for name in pins if name not in found]
    if missing and constants.HF_HUB_OFFLINE:
        needed = sum(pins[name].size for name in missing)
        raise ModelError(
            f"{_listing(missing)}: not in Hugging Face's cache, {where}, and {_offline()} keeps"
            f" the network out: unset it to download {_size(needed)}, set HF_HOME (or"
            f" HF_HUB_CACHE) to a cache holding {'it' if len(missing) == 1 else 'them'},"
            f" {ELSEWHERE}"
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
    return {name: found[name] for name in pins}


def check_pinned(model_files: ModelFiles) -> None:
    """Check each of seedvr2x's own files among the run's against its pin (check_pin), by the name
    it was given, wherever it is read from: a file of --model-dir named like one of seedvr2x's own
    is checked as one, its refusal saying how to fetch it again only when given in the pinned
    file's own role (advice). Every file refused is said at once (ModelError, a line each). A file
    seedvr2x doesn't pin is said so in the log, and runs as the header check accepted it."""
    refusals: list[str] = []
    for file in dict.fromkeys((model_files.dit, model_files.vae)):
        name = _pinned_name(file)
        if name is None:
            logger.info(
                "%s: not one of seedvr2x's pinned files, so not checked against a pin", file.name
            )
            continue
        role: Role = "dit" if file == model_files.dit else "vae"
        try:
            check_pin(name, file.path, PINNED[name], model_files.hashes, model_files.cache, role)
        except ModelError as error:
            refusals.append(str(error))
    if refusals:
        raise ModelError("\n".join(refusals))


def advice(model_files: ModelFiles) -> dict[Role, str]:
    """How to fetch each of the run's files named like one of seedvr2x's own again, by its role,
    when given in the pinned file's own role: what a refusal of its header check adds
    (weights.check_models), as a refusal of its pin says it (check_pin), from Hugging Face's cache
    or into --model-dir. A pinned name given in the other role, which only --model-dir lets
    through (resolve refuses it), gets none: the pinned file fetched again would be refused in
    that role all the same, and leaving out --model-dir would be refused by resolve."""
    found: dict[Role, str] = {}
    given: tuple[tuple[Role, ModelFile], ...] = (("dit", model_files.dit), ("vae", model_files.vae))
    for role, file in given:
        name = _pinned_name(file)
        if name is not None and PINNED[name].role == role:
            found[role] = _again(name, file.path, model_files.cache)
    return found


def check_pin(
    name: str, path: Path, pin: Pin, hashes: Hashes, cache: Path | None, role: Role | None = None
) -> None:
    """Refuse (ModelError) the file at path unless it is REPO's file name at REVISION as pin pins
    it: its size first, by a stat, then its SHA-256, read once per run (hashes). The refusal says
    both sizes or both hashes, and how to fetch the file again, into Hugging Face's cache (its
    directory, cache) or into --model-dir (cache None), unless it was given in a role (role) other
    than the pinned file's, as advice has it. Nothing is deleted, where numz's downloader deletes
    a file named like its own whose SHA-256 differs (bug 22, item 4)."""
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ModelError(f"{path}: {error.strerror}") from None
    if size != pin.size:
        found, pinned = f"{size:,} bytes", f"{pin.size:,}"
    else:
        digest = hashes.sha256(path)
        if digest == pin.sha256:
            logger.info(
                "%s: its size and SHA-256 as seedvr2x pins them, %s at %s",
                *(name, REPO, REVISION[:8]),
            )
            return
        found, pinned = f"SHA-256 {digest}", pin.sha256
    refusal = (
        f"{path}: {found}, where {REPO}'s {name} at {REVISION[:8]} has {pinned}: not the file"
        " seedvr2x pins, left as it is"
    )
    if role is None or role == pin.role:
        refusal += f"; {_again(name, path, cache)}"
    raise ModelError(refusal)


def url(name: str) -> str:
    """The URL of REPO's file name at REVISION, as the Hub serves it."""
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"


def _pinned_name(file: ModelFile) -> str | None:
    """The name of the pinned file the run's file is named like, if any: the last part of the
    name it was given. Provisional (implementation, 2026-10-09): DESIGN.md says by its name, and
    --dit-model sub/<name> in --model-dir names <name> all the same."""
    name = PurePath(file.name).name
    return name if name in PINNED else None


def _again(name: str, path: Path, cache: Path | None) -> str:
    """How to fetch REPO's file name again, its copy at path refused: from Hugging Face's cache
    (cache, its directory), remove the file its link leads to, which the next run downloads
    again, or have huggingface_hub's command download it again; into --model-dir (cache None),
    its URL."""
    if cache is None:
        return (
            f"fetch it again from {url(name)}, or leave out --model-dir, which takes it from"
            " Hugging Face's cache"
        )
    command = shlex.join(
        ["hf", "download", REPO, name, "--revision", REVISION, "--cache-dir", str(cache)]
    )
    return (
        f"remove {path.resolve()}, which the next run downloads again, or download it again with"
        f" huggingface_hub's command: {command} --force-download"
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
        raise ModelError(_unreadable(where, error)) from None
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
        raise ModelError(_unreadable(where, error)) from None
    refusal = (
        f"Hugging Face's cache, {where}: {_size(free)} free on its disk, where downloading"
        f" {_listing(list(downloads))} needs {_size(needed)}: free some space there, set HF_HOME"
        f" (or HF_HUB_CACHE) to a directory on a disk with room, {ELSEWHERE}"
    )
    if partial:
        listing = _listing([f"{path} ({_size(size)})" for path, size in partial])
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
        refusal += f". {listing} {said}"
    raise ModelError(refusal)


def _download(name: str, pin: Pin, where: Path, needed: int) -> Path:
    """REPO's file name at REVISION, pinned by pin, downloaded into Hugging Face's cache, where,
    with no token: the repo is public, and token=False keeps a token the user stored from being
    sent (utils/_headers.py:125-133). Each of the library's failures is refused (ModelError),
    saying what to do; needed is what the downloads left need on the cache's disk, in bytes, this
    one's and the next ones', as _check_space checked it."""
    import httpx
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
        *(name, _size(pin.size), REPO, REVISION[:8]),
    )
    started = time.monotonic()
    again = f"run again, which downloads it again, {ELSEWHERE}"
    # The errors are huggingface_hub 1.33.0's (errors.py, file_download.py, utils/_xet.py) and
    # those of hf-xet 1.6.0, which downloads Xet files (xet_pkg/src/error.rs:293-318), every one
    # caught by name: no bare Exception.
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
        if _refused(error):
            raise ModelError(
                f"{name}: the Hub refused it ({_cause(error)}): {UNAVAILABLE}"
            ) from None
        if _not_the_hub(error):
            raise ModelError(
                f"{name}: not in the cache, and {constants.ENDPOINT} answered the request to"
                f" download it without the Hub's headers ({_cause(error)}): {CHECK_ENDPOINT}"
            ) from None
        raise ModelError(
            f"{name}: not in the cache, and {constants.ENDPOINT} couldn't be reached to download it"
            f" ({_cause(error)}): run again once it can be, {ELSEWHERE}"
        ) from None
    except RemoteEntryNotFoundError:
        raise ModelError(
            f"{name}: not in {REPO} at {REVISION}, the revision this seedvr2x pins (the Hub: entry"
            f" not found): download it elsewhere and give --model-dir with its directory"
        ) from None
    except RevisionNotFoundError:
        # Unreachable at a commit hash as the Hub answered on 2026-10-09 (the review of this
        # step): a commit it doesn't have gets "Entry Not Found", refused above. Kept, for a Hub
        # that would answer RevisionNotFound, which the library raises on that error code
        # (utils/_http.py:850-852).
        raise ModelError(
            f"{name}: {REPO} has no revision {REVISION}, the one this seedvr2x pins (the Hub:"
            f" revision not found): download the files elsewhere and give --model-dir with their"
            " directory"
        ) from None
    except RepositoryNotFoundError:
        raise ModelError(
            f"{name}: the Hub has no repository {REPO}, or not a public one (the Hub: repository"
            " not found): download the files elsewhere and give --model-dir with their directory"
        ) from None
    except HfHubHTTPError as error:
        if _refused(error):
            raise ModelError(
                f"{name}: the Hub refused it ({_words(error)}): {UNAVAILABLE}"
            ) from None
        raise ModelError(
            f"{name}: the Hub refused its download ({_words(error)}): {again}"
        ) from None
    except OSError as error:
        if _xet_refused(error):
            raise ModelError(
                f"{name}: the Hub's storage refused it ({_words(error)}): {UNAVAILABLE}"
            ) from None
        number = error.errno if error.errno is not None else _os_error(error)
        if number is not None and number in UNWRITABLE:
            raise ModelError(
                f"{name}: Hugging Face's cache, {where}, can't be written ({os.strerror(number)}):"
                f" set HF_HOME (or HF_HUB_CACHE) to a directory you can write, on a disk with"
                f" {_size(needed)} free, {ELSEWHERE}"
            ) from None
        # The network lost during a Xet download (ConnectionError, TimeoutError), or another I/O
        # error; a plain download's sizes not matching (file_download.py:491).
        raise ModelError(f"{name}: its download failed ({_words(error)}): {again}") from None
    except (httpx.RequestError, RuntimeError) as error:
        # A plain download's network lost past its retries (file_download.py:468-472), a proxy's
        # failure (1822), a body that doesn't decode or redirects without end (httpx's
        # DecodingError and TooManyRedirects, RequestErrors but not TransportErrors); hf_xet's
        # other failures, its data's integrity among them (xet_pkg/src/error.rs:309-317).
        raise ModelError(f"{name}: its download failed ({_words(error)}): {again}") from None
    except ValueError as error:
        # hf_xet's Configuration and InvalidTaskID errors (xet_pkg/src/error.rs:308), and the
        # library's when the Hub's answer lacks the Xet headers it needs (utils/_xet.py:131-132,
        # 179-181), as from a proxy or an HF_ENDPOINT mirror that doesn't pass them on: no rerun
        # changes them.
        raise ModelError(
            f"{name}: its download failed ({_words(error)}): {CHECK_ENDPOINT}"
        ) from None
    logger.info("%s: downloaded in %.0f s, %s", name, time.monotonic() - started, path)
    return Path(path)


def _library_logging() -> None:
    """huggingface_hub's log and httpx's in seedvr2x's: each of the library's records once, through
    seedvr2x's handler, but the Hub's advice to set a token; httpx's and httpcore's warnings
    alone."""
    # httpx logs every request at INFO (httpx 0.28.1: _client.py:1025-1032, 1740-1747), the
    # library's own among them (its registry of AI agents, utils/_detect_agent.py:197), which a
    # run's log isn't for; httpcore's are debug traces.
    for name in ("httpx", "httpcore"):
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


def _unreadable(where: Path, error: OSError) -> str:
    """The refusal of Hugging Face's cache, where, which can't be read (error)."""
    what = error.strerror or _words(error)
    if error.filename is not None:
        what = f"{what}: {error.filename}"
    return (
        f"Hugging Face's cache, {where}, can't be read ({what}): make it readable, set HF_HOME (or"
        f" HF_HUB_CACHE) to another cache, {ELSEWHERE}"
    )


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


def _size(count: int) -> str:
    """A number of bytes, in GB (10^9 bytes, as DESIGN.md gives the files' sizes) from 0.1 GB."""
    return f"{count / 1e9:.1f} GB" if count >= 100_000_000 else f"{count:,} bytes"


def _listing(names: list[str], word: str = "and") -> str:
    """Names, joined by commas, the last by word."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} {word} {names[-1]}"


# SIGTERM during a download (cli.py raises it as Terminated around resolve, runtime/stop.py's
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
