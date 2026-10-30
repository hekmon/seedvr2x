"""The VAE compiled (DESIGN.md, Memory planner, "torch.compile: the VAE, opt-in, in v0.1.0; the
DiT, not", decided on 2026-10-24; research/docs/compile.md): `--compile`.

PyTorch's compiler turns a model's Python code into fused GPU kernels the first time the model
meets an input of a given shape, and compiles again for each new shape. A job's frame size being
fixed, the VAE's shapes are its passes' tiles (or the frame), at the slice lengths of its shots,
all known from the plan before a unit runs (planner.compile_shapes): this module compiles them
once, right after the plan, with the VAE alone on the card and before the DiT loads, so that no
unit of the run compiles and the workers the compiler starts are gone before the DiT's blocks
are pinned.

What is compiled is the study's `excl` variant (research/scripts/compile_vae.py: install_vae):
the encoder and the decoder each through torch.compile (inductor, mode default, fullgraph off,
dynamic shapes off, the recompile limit numz sets, 128), with every causal convolution left out of
the graph. numz means that exclusion and never gets it (research/bugs/15-compile-vae-
exclusion-dead-code.md: it sets an attribute torch never reads); without it the compiled VAE took
twice its memory. Here the convolutions' forward is wrapped by torch.compiler.disable, which
dynamo does read, and they run as plain code between the compiled pieces: each pass 0.79 to 0.95
of plain code's time, its peak 1.04 to 1.10 times (planner.COMPILE_PEAK).

The model code is not edited: the vendored VAE is wrapped from outside, as the study did."""

import importlib
import logging
import os
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from seedvr2x.clock import spent
from seedvr2x.runtime import blockswap, model
from seedvr2x.runtime.job import JobError
from seedvr2x.runtime.model import COMPUTE_DTYPE, Models
from seedvr2x.runtime.plan import Compilation, Shapes
from seedvr2x.runtime.tilegrid import SCALE
from seedvr2x.runtime.timing import spoken
from seedvr2x.vendor.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d

logger = logging.getLogger(__name__)

# numz's torch.compile settings for the VAE (src/core/model_configuration.py:1307-1315 at
# 4490bd1, inference_cli.py:1455-1466; research/scripts/compile_vae.py:138, NUMZ_SETTINGS): the
# study measured these. Its dynamo cache limit 64 and recompile limit 128 are one setting in this
# torch, so 128 holds (compile_vae.py:137).
SETTINGS: dict[str, Any] = {
    "backend": "inductor",
    "mode": "default",
    "fullgraph": False,
    "dynamic": False,
}
RECOMPILE_LIMIT = 128


class CompileError(JobError):
    """The VAE did not compile: a compiler torch needs missing, an error of inductor's or
    Triton's. The job is refused before the DiT loads, nothing run, and the message says so and
    names the way out, the same command without --compile. PROVISIONAL (impl-v010, 2026-10-29:
    DESIGN.md says nothing of a compilation that fails): refused, never run plain in its place,
    as a silent fallback would hand a user frames other than the ones asked for."""


def keep_deterministic() -> Callable[[], None]:
    """Keep inductor's deterministic mode on while compiling, and return the function that puts
    torch's own back.

    Dynamo calls torch.use_deterministic_algorithms(<the state before the trace>) after every
    trace (torch 2.14.1, torch/_dynamo/convert_frame.py:375), and that call sets inductor's
    flag as well (torch/__init__.py:1852), so that setting it once, or by TORCHINDUCTOR_
    DETERMINISTIC, is undone by the first trace. Without it Triton's runtime autotune picks a
    reduction's block size by timing, two processes sum in other orders and their frames
    differ; with it, cold processes, warm caches and a resume with a fresh cache gave the same
    frames, at no measured cost (DESIGN.md, same section; compile.md, "Bits"). Ported from
    research/scripts/compile_dit.py:627, keep_deterministic(torch, "inductor"): the flag is set
    again after each such call, the eager operations unchanged (the call's own mode is
    torch's). Idempotent: a second call, while the first holds, wraps nothing more.

    The returned function restores torch's function, for the tests; a run keeps it to the end
    of the process."""
    inductor_config: Any = importlib.import_module("torch._inductor.config")
    current: Any = torch.use_deterministic_algorithms
    if getattr(current, "keeps_inductor_deterministic", False):
        inductor_config.deterministic = True
        return lambda: None
    original: Any = current

    def keep(mode: bool, *, warn_only: bool = False) -> None:
        original(mode, warn_only=warn_only)
        inductor_config.deterministic = True

    keep.keeps_inductor_deterministic = True  # pyright: ignore[reportFunctionMemberAccess]
    torch.use_deterministic_algorithms = keep
    inductor_config.deterministic = True

    def restore() -> None:
        if torch.use_deterministic_algorithms is keep:
            torch.use_deterministic_algorithms = original

    return restore


def is_causal_convolution(module: torch.nn.Module) -> bool:
    """Whether a module is one of the VAE's causal convolutions, the ones left out of the graph:
    InflatedCausalConv3d, whose memory_limit_conv branches on a tensor's value, splits by it and
    keeps a cache between slices (vendor/models/video_vae_v3/modules/causal_inflation_lib.py),
    dynamic shapes and state that compile as breaks and recompilations."""
    return isinstance(module, InflatedCausalConv3d)


def exclude_causal_convolutions(vae: torch.nn.Module) -> int:
    """Leave every causal convolution of the VAE out of any graph compiled over it: each one's
    forward is wrapped by torch.compiler.disable, as the study's `excl` does
    (research/scripts/compile_vae.py:686, install_vae), and runs as plain code. Returns how many.
    The wrapper is an attribute of the instance, the class and the vendored code untouched."""
    count = 0
    for module in vae.modules():
        if is_causal_convolution(module):
            module.forward = torch.compiler.disable(module.forward)  # pyright: ignore
            count += 1
    return count


def use_cache(directory: Path) -> None:
    """Point inductor's cache, and Triton's under it, at `directory`, before the first compile:
    the job's own (manifest.COMPILE_CACHE), warm on a resume. torch reads the variable when it
    needs the cache (torch/_inductor/runtime/cache_dir_utils.py:14-21, which also exports it for
    Triton), so a user's own is replaced, said at debug."""
    # PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md puts the cache "in the job's work directory"
    # and says nothing of a user's own TORCHINDUCTOR_CACHE_DIR): the job's replaces it.
    asked = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if asked is not None and Path(asked) != directory:
        logger.debug("TORCHINDUCTOR_CACHE_DIR=%s replaced by the job's own, %s", asked, directory)
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(directory)


def shutdown_workers() -> None:
    """Shut torch's compile workers (torch._inductor.async_compile.shutdown_compile_workers,
    torch 2.14.1): their host memory, 4.4 GiB of PSS with 32 of them (16 GiB summed RSS, most of
    it shared) on the study's machine, is the compiler's, and goes before the DiT's blocks are
    pinned (compile.md, "The VAE's checks for v0.1.0"; DESIGN.md, same section)."""
    module: Any = importlib.import_module("torch._inductor.async_compile")
    module.shutdown_compile_workers()
    # Any compilation after this one runs in this process: torch starts its pool again on the
    # next compile (async_compile.py: get_compile_threads() > 1 asks for one), and the workers
    # it starts would be back, with their host memory, beside the DiT's pinned blocks (the
    # review of 2026-10-29, LOW 5). A compilation after the warm-up is said (report_late).
    config: Any = importlib.import_module("torch._inductor.config")
    config.compile_threads = 1


# The count of frames torch._dynamo had started to compile when the warm-up ended, None where the
# VAE isn't compiled. torch 2.14.1 counts one in counters["frames"]["total"] each time dynamo
# converts a frame, which happens on a cache miss alone (_dynamo/convert_frame.py:2331): a
# frame the warm-up compiled, found again, isn't counted.
_warmed: list[int | None] = [None]


def frames_compiled() -> int:
    """The frames torch._dynamo has started to compile in this process."""
    utils: Any = importlib.import_module("torch._dynamo.utils")
    return int(utils.counters["frames"]["total"])


def report_late(where: str) -> int:
    """Say at WARNING any compilation since the warm-up ended, or since the last such saying,
    naming the pass `where` (a unit's part: "shot 2's decode"), and return how many frames it
    was. A job whose VAE is compiled expects none: the warm-up compiled every shape the plan's
    passes meet (DESIGN.md, Memory planner, "torch.compile: the VAE, opt-in, in v0.1.0"; warm),
    and a pass that compiled one more ran its first call slowly, in this process (shutdown_workers
    left the compile threads at 1). Nothing is said, and 0 returned, without a warm-up.

    PROVISIONAL (impl-v010, 2026-10-29: the review asks that it be said; DESIGN.md doesn't say
    when): checked after each part of a unit, and after the calibration, so that the pass is
    named; a job-end check would not name it."""
    before = _warmed[0]
    if before is None:
        return 0
    now = frames_compiled()
    if now <= before:
        return 0
    _warmed[0] = now
    logger.warning(
        "%s: the compiler compiled %d more %s after the warm-up: the warm-up missed a shape this"
        " pass meets, so its first call took the compilation's time",
        where,
        now - before,
        "frame" if now - before == 1 else "frames",
    )
    return now - before


@contextmanager
def _on_the_device(models: Models) -> Generator[None]:
    """The VAE's weights on the device for the block, where the plan has them away from it in
    some phase (Models.vae, a part loaded in its pinned copy): brought as Models.enter brings
    them, and sent away after the device's work, as it sends them (DESIGN.md, BlockSwap,
    rewritten). Nothing to do where they never leave."""
    part = models.vae
    if part is None or part.device is not None or models.device.type != "cuda":
        yield
        return
    backend = blockswap.Cuda(models.device)
    part.enter(backend)
    try:
        yield
    finally:
        backend.wait(backend.compute())
        part.leave()
        backend.trim()


def compile_vae(models: Models, compilation: Compilation) -> None:
    """Compile the VAE of `models` and warm it, right after the plan and before the DiT loads:
    its cache in the job's directory, inductor's deterministic mode kept on, its causal
    convolutions out of the graph, a warm call of every shape the plan's passes meet
    (warm; planner.compile_shapes), the compile workers shut at once. compilation.seconds is the
    time that took, in the run's own clock.

    Called with the VAE alone on the card (model.load_models): the DiT's blocks aren't there
    yet, nor pinned. The workers are shut whatever happens, so that a failure leaves none."""
    started = time.monotonic()
    vae: Any = models.runner.vae
    use_cache(compilation.cache)
    keep_deterministic()
    convolutions = exclude_causal_convolutions(vae)
    # numz's recompile limit for the VAE (RECOMPILE_LIMIT), raised for the process: nothing else
    # is compiled in v0.1.0 (the DiT is not), and torch's own, 8, is under the shapes a plan
    # meets: the study's `excl` compiled 500 to 800 graphs a run, at 250 to 330 graph breaks,
    # one per piece between two convolutions and shape (research/docs/compile.md, the VAE).
    limit: Any = importlib.import_module("torch._dynamo.config")
    if limit.recompile_limit < RECOMPILE_LIMIT:
        limit.recompile_limit = RECOMPILE_LIMIT
    logger.info(
        "compiling the VAE (--compile): its encoder and decoder, %d causal convolutions left out"
        " of the graph, %s; the cache in %s",
        convolutions,
        _shapes_said(compilation.shapes),
        compilation.cache,
    )
    try:
        vae.encoder = torch.compile(vae.encoder, **SETTINGS)
        vae.decoder = torch.compile(vae.decoder, **SETTINGS)
        with _on_the_device(models):
            warm(models, compilation.shapes)
        _warmed[0] = frames_compiled()
    except torch.OutOfMemoryError:
        raise  # the command line's own refusal for the card's memory
    except (RuntimeError, OSError, ImportError) as error:
        # Inductor's and Triton's failures are RuntimeErrors of several kinds (a compiler
        # missing, a kernel that doesn't build); torch's own words first, one line of them. Never
        # a plain VAE in its place: the job's frames would not be those it asked for.
        said = next(
            (line for line in str(error).splitlines() if line.strip()), type(error).__name__
        )
        raise CompileError(
            f"--compile: the VAE did not compile ({type(error).__name__}: {said[:300]}); nothing"
            " was run, the same command without --compile runs the plain VAE"
        ) from error
    finally:
        shutdown_workers()
    compilation.seconds = time.monotonic() - started
    logger.info("the VAE compiled in %s", spoken(spent(compilation.seconds)))


def _shapes_said(shapes: Shapes) -> str:
    """The shapes warmed, in words, for the log."""

    def each(found: tuple[tuple[int, int], ...]) -> str:
        return ", ".join(f"{columns}x{rows}" for rows, columns in found)

    return (
        f"warmed at the encode's {each(shapes.encode)} and the decode's {each(shapes.decode)},"
        f" shots of {', '.join(str(each) for each in shapes.latents)} latents or more"
    )


@torch.no_grad()
def warm(models: Models, shapes: Shapes) -> None:
    """A warm call of every shape the plan's VAE passes meet, at every slice kind the job's shots
    have: the compiler is called once per shape and kind (dynamic shapes are off), so that the
    run's units find them compiled (DESIGN.md: "a warm call per shape the plan uses ... at the
    job's slice lengths").

    The passes are the run's own, through its own functions (model.encode_sliced,
    model.decode_sliced), fed zeros: the encode with a tile's crop as the tiled encode hands it
    (tiling._crops), a contiguous (3, t, rows, columns) tensor on the device in the VAE's dtype,
    t following model.encode_slices; the decode with the crop of a latent of the whole padded
    frame, as the tiled decode takes it (tiling._decoded), its memory the encode's
    (model.to_latent), so that the decoder is handed each slice in the memory it meets in the
    run. A shot of `n` latents in `shapes.latents` is one of 4 * (n - 1) + 1 frames, its slices
    those of model.encode_slices and model.latent_slices.

    Not whole shots' worth of work: a slice is 4 or 5 frames of a tile.

    PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md says "a warm call per shape", not with what):
    zeros, in the memory the run's inputs have as far as the code shows (a tile's crop copied to
    the device is contiguous; the decoder's latent is the encode's channel-major one, cropped
    from the whole padded frame's, as tiling._decoded crops it). Whether they match the run's
    own inputs for the compiler's guards is for a GPU run to show: no unit of the run may
    compile (torch._dynamo.utils.counters, after the first unit)."""
    padded_rows, padded_columns = shapes.padded
    latent_rows, latent_columns = padded_rows // SCALE, padded_columns // SCALE
    for latents in shapes.latents:
        frames = 4 * (latents - 1) + 1
        for rows, columns in shapes.encode:
            crops = (
                torch.zeros((3, size, rows, columns), dtype=COMPUTE_DTYPE, device=models.device)
                for size in model.encode_slices(models, frames)
            )
            for moments in model.encode_sliced(models, crops, frames):
                del moments
            model.synchronize(models.device)
        storage = torch.zeros((16, latents, latent_rows, latent_columns), dtype=COMPUTE_DTYPE)
        latent = storage.permute(1, 2, 3, 0)
        for rows, columns in shapes.decode:
            top, left = slice(0, rows // SCALE), slice(0, columns // SCALE)
            for decoded in model.decode_sliced(models, latent, top, left):
                del decoded
            model.synchronize(models.device)
