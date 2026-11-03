"""The DiT's sampling of latents (DESIGN.md, Pipeline step 2): the noise of a shot's latent drawn
once, on the CPU, from a seed, and one window's pass of the DiT over a window's noise and latent.
The library's brick (DESIGN.md, Repository and apps); the windows over a shot, their layout, the
units they are kept in and their merge are the command line's (shot.py, sample_windows)."""

import logging
import time

import torch
from torch import Tensor

from seedvr2x import model
from seedvr2x.model import COMPUTE_DTYPE, Models
from seedvr2x.passes import ENCODE_SEED_OFFSET as ENCODE_SEED_OFFSET

logger = logging.getLogger(__name__)

GIB = 1024**3
CPU = torch.device("cpu")


def moved(tensor: Tensor, device: torch.device, what: str) -> Tensor:
    """tensor on device with its values and its memory layout, its strides: itself when it is
    there already, else a copy. The noise drawn from a latent follows the latent's layout
    (shot.py, sample_windows), and the decode is fed the windows as the DiT gave them, so
    neither changes layout on its way to system memory or back (units.py). torch keeps
    the strides of a tensor that is dense, as an encode's latent and a window's output are, and
    lays the copy of any other out anew, dense in the same order of dimensions (a slice of the
    shot's latent along time): RuntimeError then, naming `what`.

    The tensor's device is polled first (model.synchronize), so that a Ctrl-C is answered while
    it finishes what the copy waits for."""
    model.synchronize(tensor.device)
    copy = tensor.to(device)
    if copy.stride() != tensor.stride():
        raise RuntimeError(
            f"{what}: its copy on {device} has strides {tuple(copy.stride())}, not its"
            f" {tuple(tensor.stride())}"
        )
    return copy


def draw(models: Models, latent: Tensor, seed: int, numz_draw: bool = False) -> Tensor:
    """The noise of a shot's latent as shot.py's sample_windows draws it (drawn), on
    its own: for the time estimates' calibration (calibration.py), which times it apart
    from the window."""
    return drawn(models, latent, seed, numz_draw)[0]


@torch.no_grad()
def drawn(
    models: Models, latent: Tensor, seed: int, numz_draw: bool = False
) -> tuple[Tensor, Tensor]:
    """A shot's noise, drawn from seed for its whole latent, and that latent: both on the CPU,
    (T', h, w, 16) in the latent's memory layout (moved), the noise in COMPUTE_DTYPE; the
    latent itself when it is given there.

    The draw is torch.randn_like of the latent in system memory, from a CPU generator of the
    draw's own seeded with seed: the generator fills the tensor in memory order, so that the
    values follow the latent's strides as well as its shape. Its own, so that nothing drawing
    from torch's global generator between the seeding and the draw can move the noise; seeded
    alike, it is the same stream as the global CPU generator numz seeds (model.set_seed, still
    called, as numz's draw is preceded by it), and gives the same values (tests/test_sampling.py
    holds their digest). A latent given on the device is copied to system memory first, strides
    kept (moved). Nothing of the draw is on the GPU, and its values are the same on every card:
    CUDA's generator lays its draw out by the card's count of processors, so that an RTX 5090
    and an RTX PRO 6000 drew other noise for one seed at a shot's size, as another seed would
    (DESIGN.md, Pipeline step 2, decided on 2026-10-13). Measured on 2026-10-16 (DESIGN.md, the
    same step), the noise then drawn from the global CPU generator, which gives the same values:
    the same noise on an RTX 5090 and an RTX PRO 6000, digest for digest, for every shot of a
    job; and the two machines' CPUs draw the same values (tests/test_sampling.py holds their digest,
    which both pass). This draw alone, on a desktop's 16-core CPU (torch 2.14.1, 2026-10-16, a
    process a shape, twice each): 8.8 to 9.0 s and 0.987 GiB for 1,000 latents at 1080p, padded
    to 1104x1920; 12.2 to 12.6 s and 1.415 GiB for 361 at 4K, padded to 2192x3840; the process's
    peak growing by the noise's bytes alone.

    numz_draw is for tests only (main.NUMZ_DRAW): the draw as numz makes it, torch.randn_like of
    the latent on the device, the latent sent there for it, strides kept (moved), both copies
    there given up when this returns. Milestone 1's regression holds the output to numz's with
    it, as does every GPU test held to a reference numz made.

    With -v, the draw is logged: where, for how many latents, its time and the noise's size."""
    started = time.monotonic()
    if numz_draw:
        there = moved(latent, models.device, "the shot's latent")
        model.set_seed(seed)
        drawn = torch.randn_like(there, dtype=COMPUTE_DTYPE)
        noise, latent = (
            moved(drawn, CPU, "the shot's noise"),
            moved(latent, CPU, "the shot's latent"),
        )
    else:
        latent = moved(latent, CPU, "the shot's latent")
        model.set_seed(seed)
        generator = torch.Generator(device=CPU).manual_seed(seed)
        noise = torch.randn_like(latent, dtype=COMPUTE_DTYPE, generator=generator)
    logger.debug(
        "noise drawn %s for the shot's %d latents in %.2f s: %.3f GiB",
        "on the device (numz's draw)" if numz_draw else "on the CPU",
        latent.shape[0],
        time.monotonic() - started,
        noise.numel() * noise.element_size() / GIB,
    )
    return noise, latent


@torch.no_grad()
def window(models: Models, noise: Tensor, latent: Tensor) -> Tensor:
    """The DiT's output for one window, (t, h, w, 16) on the device, from the window's slices of
    its shot's noise and latent, on the CPU: sent to the device for the pass, where nothing of
    them is left when this returns but the output."""
    noise, latent = noise.to(models.device), latent.to(models.device)
    return model.sample(models, noise, model.condition(models, noise, latent))
