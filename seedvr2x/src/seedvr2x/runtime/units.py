"""Where a run keeps its units (DESIGN.md, Pause and resume): each shot's latent after its encode
and each window's DiT output, until the shot is decoded, and its output segment finished.

A shot's windows are only ever taken in order: the shot's latent is kept until its windows are
all done, and they are kept until its decode."""

import shutil
from pathlib import Path

import torch
from torch import Tensor

from seedvr2x.media.files import make_directories, partial_path, replace_whole
from seedvr2x.runtime.manifest import STATE, Manifest


class Units:
    """Units kept in memory, by a run that keeps nothing once it stops: the one-file output (-o
    x.mkv), which can't resume until assembly makes it of segments (DESIGN.md, Output). A shot's
    latent and windows stay on the device until its decode, as upscale_shot keeps them."""

    persistent = False

    def __init__(self) -> None:
        self._latents: dict[int, Tensor] = {}
        self._windows: dict[int, list[Tensor]] = {}

    def windows_done(self, shot: int) -> int:
        """The windows of shot `shot` kept, its first ones."""
        return len(self._windows.get(shot, ()))

    def latent(self, shot: int) -> Tensor | None:
        """Shot `shot`'s latent, when kept (encode_shot)."""
        return self._latents.get(shot)

    def save_latent(self, shot: int, latent: Tensor) -> None:
        self._latents[shot] = latent

    def save_window(self, shot: int, window: int, sampled: Tensor) -> None:
        """Keep window `window` of shot `shot`, the windows before it being kept already."""
        kept = self._windows.setdefault(shot, [])
        if window != len(kept):
            raise ValueError(f"shot {shot}: window {window} after {len(kept)}")
        kept.append(sampled)

    def drop_latent(self, shot: int) -> None:
        """Give up shot `shot`'s latent: its windows are all done."""
        self._latents.pop(shot, None)

    def take_windows(self, shot: int) -> list[Tensor]:
        """Shot `shot`'s windows, in order, for its decode, given up here."""
        return self._windows.pop(shot)

    def segment_finished(self, segment: int) -> None:
        """Segment `segment` is whole: what its shots kept can go."""


class DiskUnits(Units):
    """Units kept in the output directory, beside the manifest, for a resume: under
    resume/shot_<start>/, the shot's latent.pt, then window_<k>.pt for each window, each written
    whole (files.replace_whole), then recorded in the manifest, which so only ever names whole
    files. The latent goes once the windows are all done, a shot's directory once its segment is
    finished, and resume/ once every segment is.

    The files are CPU copies saved by torch.save, which keeps a tensor's values and memory layout:
    the noise drawn from a latent depends on its layout (sample_windows), and the decode is fed the
    windows as the DiT gave them. A shot's decode reads its windows back, so that a resumed run
    and an uninterrupted one decode the same tensors, and the memory used stays one shot's."""

    persistent = True

    def __init__(self, directory: Path, manifest: Manifest) -> None:
        super().__init__()
        self.directory, self.manifest = directory, manifest
        self.root = directory / STATE

    def shot_directory(self, shot: int) -> Path:
        return self.root / f"shot_{self.manifest.shots[shot].start:06d}"

    def windows_done(self, shot: int) -> int:
        return self.manifest.windows_done[shot]

    def latent(self, shot: int) -> Tensor | None:
        """Shot `shot`'s latent, when kept, on the CPU."""
        if not self.manifest.encoded[shot]:
            return None
        return _load(self.shot_directory(shot) / "latent.pt")

    def save_latent(self, shot: int, latent: Tensor) -> None:
        _save(latent, self.shot_directory(shot) / "latent.pt")
        self.manifest.shot_encoded(shot)

    def save_window(self, shot: int, window: int, sampled: Tensor) -> None:
        _save(sampled, self.shot_directory(shot) / f"window_{window:04d}.pt")
        self.manifest.window_done(shot, window)

    def drop_latent(self, shot: int) -> None:
        (self.shot_directory(shot) / "latent.pt").unlink(missing_ok=True)

    def take_windows(self, shot: int) -> list[Tensor]:
        """Shot `shot`'s windows, in order, read back, on the CPU; they stay on disk until its
        segment is finished, which a resume may decode again."""
        directory = self.shot_directory(shot)
        return [
            _load(directory / f"window_{window:04d}.pt")
            for window in range(self.windows_done(shot))
        ]

    def segment_finished(self, segment: int) -> None:
        """Record segment `segment` as finished, then remove what its shots kept."""
        self.manifest.segment_finished(segment, size(self.directory / self.manifest.files[segment]))
        for shot in self.manifest.segment_shots(segment):
            shutil.rmtree(self.shot_directory(shot), ignore_errors=True)
        if all(self.manifest.finished) and self.root.is_dir() and not any(self.root.iterdir()):
            self.root.rmdir()


def size(path: Path) -> int:
    """The bytes of a file, or of a directory's files (a PNG segment)."""
    if path.is_dir():
        return sum(entry.stat().st_size for entry in path.iterdir())
    return path.stat().st_size


def _save(tensor: Tensor, path: Path) -> None:
    copy = tensor.to("cpu")
    if copy.stride() != tensor.stride():
        raise RuntimeError(f"{path}: a CPU copy of {tuple(tensor.stride())} strides changes them")
    make_directories(path.parent)
    partial = partial_path(path)
    torch.save(copy, partial)
    replace_whole(partial, path)


def _load(path: Path) -> Tensor:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, Tensor):
        raise RuntimeError(f"{path}: not a tensor")
    return tensor
