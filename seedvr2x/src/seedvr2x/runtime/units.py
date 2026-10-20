"""Where a run keeps its units (DESIGN.md, Pause and resume): each shot's latent after its encode
and each window's DiT output, until the shot is decoded, and its output segment finished; and,
for `split` (DESIGN.md, Colour correction), each shot's input copy, made during its encode.

With several runs a shot (DESIGN.md, Several runs of a shot), the units are the runs': each
run's latent and windows, in its shot's directory, named for the run (run_file); the input copy
is the shot's, made by its first run's encode, which reads the shot whole, and read by every
run's correction from that run's first frame. Indices are runs' (Manifest.runs), with one run a
shot the shots'.

A run's windows are only ever taken in order: the run's latent is kept until its windows are
all done, and they are kept until its shot's decode.

A unit is kept in a file, written from a copy in system memory, never from the GPU's own
tensor, with its values and its memory layout (shot.moved): what the GPU holds mustn't grow with
the shot (DESIGN.md, Memory planner, Per-phase peaks)."""

import shutil
from pathlib import Path

import torch
from torch import Tensor

from seedvr2x.media.files import make_directories, partial_path, replace_whole
from seedvr2x.runtime.manifest import STATE, Manifest
from seedvr2x.runtime.shot import CPU, moved

# A shot's files besides its units (DESIGN.md, Colour correction): the copy of its input frames,
# an FFV1 gbrp16le file at the input's size, which split's reference is rebuilt from, and its
# frames' checksums (media/checksums.py).
COPY = "input.mkv"
CHECKSUMS = "input.crc32"
# The frames a shot's runs but its last give, kept in a file in the shot's directory while its
# runs are decoded in turn, until the last run's decode interleaves them with its own (run.py,
# _decode_runs): derived data, never recorded, gone when the shot's decode ends, a kill's left
# for the resume to discard (DESIGN.md, Several runs of a shot).
KEPT = "runs.kept"


def run_file(offset: int, name: str) -> str:
    """The name of a unit's file of the run starting at frame `offset` of its shot: the name
    itself for the first run, so that a job of one run a shot keeps its files as before the
    option; run<offset>_<name> for a later one."""
    return name if offset == 0 else f"run{offset}_{name}"


class Units:
    """Units kept in the job's directory, the output directory or a .mkv path's work directory,
    beside the manifest, for a resume: under resume/shot_<start>/, the shot's latent.pt, then
    window_<k>.pt for each window, each written whole (files.replace_whole), then recorded in the
    manifest, which so only ever names whole files; a later run's named for it (run_file). The
    latent goes once the windows are all done, a shot's directory once its segment is finished,
    and resume/ once every segment is. With `split`, the shot's input copy (COPY) and its
    checksums (CHECKSUMS) are made whole before its first run's latent is recorded, and
    recorded with it; a tiled pass's scratch goes in the shot's directory too (run._scratch), and
    the frames its runs keep for its decode (KEPT).

    The files are CPU copies saved by torch.save, which keeps a tensor's values and memory layout:
    the noise drawn from a latent depends on its layout (sample_windows), and the decode is fed the
    windows as the DiT gave them. A shot's decode reads its windows back, so that a resumed run
    and an uninterrupted one decode the same tensors, and the memory used stays one shot's."""

    def __init__(self, directory: Path, manifest: Manifest) -> None:
        self.directory, self.manifest = directory, manifest
        self.root = directory / STATE

    def shot_directory(self, run: int) -> Path:
        """Where the files of run `run`'s shot go: its runs' units, its input copy and the copy's
        checksums, a tiled pass's scratch, and the frames its runs keep (KEPT)."""
        shot = self.manifest.shots[self.manifest.shot_of(run)]
        return self.root / f"shot_{shot.start:06d}"

    def copy_path(self, run: int) -> Path:
        """The input copy of run `run`'s shot, which its first run makes."""
        return self.shot_directory(run) / COPY

    def checksums_path(self, run: int) -> Path:
        return self.shot_directory(run) / CHECKSUMS

    def kept_path(self, run: int) -> Path:
        """The file of the frames run `run`'s shot keeps between its runs' decodes (KEPT)."""
        return self.shot_directory(run) / KEPT

    def _unit(self, run: int, name: str) -> Path:
        return self.shot_directory(run) / run_file(self.manifest.runs[run].offset, name)

    def windows_done(self, run: int) -> int:
        """The windows of run `run` kept, its first ones."""
        return self.manifest.windows_done[run]

    def latent(self, run: int) -> Tensor | None:
        """Run `run`'s latent, when kept (encode_shot), read back, on the CPU."""
        if not self.manifest.encoded[run]:
            return None
        return _load(self._unit(run, "latent.pt"))

    def save_latent(self, run: int, latent: Tensor) -> Tensor:
        """Keep run `run`'s latent, (T', h, w, 16) as its encode gives it, in system memory
        (shot.encode_shot). Returns the one written for a resume: itself, or its copy in system
        memory were it given on the GPU, values and memory layout kept (shot.moved), so that a
        run samples its windows from it (shot.sample_windows) and holds no second copy of a
        latent, nor reads back a file it wrote."""
        kept = _save(latent, self._unit(run, "latent.pt"))
        self.manifest.shot_encoded(run)
        return kept

    def save_window(self, run: int, window: int, sampled: Tensor) -> None:
        """Keep window `window` of run `run`, (t, h, w, 16) as the DiT gives it, wherever it
        is: written from its copy on the CPU, values and memory layout kept (shot.moved), which
        is given up once the file is whole."""
        _save(sampled, self._unit(run, f"window_{window:04d}.pt"))
        self.manifest.window_done(run, window)

    def drop_latent(self, run: int) -> None:
        """Give up run `run`'s latent: its windows are all done."""
        self._unit(run, "latent.pt").unlink(missing_ok=True)

    def finished(self, segment: int) -> bool:
        """Whether segment `segment` is finished already."""
        return self.manifest.finished[segment]

    def take_windows(self, run: int) -> list[Tensor]:
        """Run `run`'s windows, in order, read back, on the CPU; they stay on disk until its
        segment is finished, which a resume may decode again."""
        return [
            _load(self._unit(run, f"window_{window:04d}.pt"))
            for window in range(self.windows_done(run))
        ]

    def segment_finished(self, segment: int) -> None:
        """Record segment `segment` as finished, then remove what its shots kept."""
        self.manifest.segment_finished(segment, size(self.directory / self.manifest.files[segment]))
        for shot in self.manifest.segment_shots(segment):
            start = self.manifest.shots[shot].start
            shutil.rmtree(self.root / f"shot_{start:06d}", ignore_errors=True)
        if all(self.manifest.finished) and self.root.is_dir() and not any(self.root.iterdir()):
            self.root.rmdir()


def size(path: Path) -> int:
    """The bytes of a file, or of a directory's files (a PNG segment)."""
    if path.is_dir():
        return sum(entry.stat().st_size for entry in path.iterdir())
    return path.stat().st_size


def _save(tensor: Tensor, path: Path) -> Tensor:
    """Write tensor's copy on the CPU (shot.moved) whole at path, and give that copy."""
    copy = moved(tensor, CPU, str(path))
    make_directories(path.parent)
    partial = partial_path(path)
    torch.save(copy, partial)
    replace_whole(partial, path)
    return copy


def _load(path: Path) -> Tensor:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, Tensor):
        raise RuntimeError(f"{path}: not a tensor")
    return tensor
