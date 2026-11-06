"""A GPU whose queued work neither completes nor fails, on demand, inside a seedvr2x run.

Reproduces the signature of the hang of 2026-11-06 (DESIGN.md, Allocator, "A hang"): a job's
decode waited on the GPU 34 minutes, the card 6% busy, the main thread in model.synchronize's
poll, and the process stayed 20 s in its teardown after SIGTERM. A real driver fault can't be
asked for twice, so this script queues work the GPU can never finish, at a place chosen on the
command line, then runs seedvr2x's command unchanged. What a stop, a wait that lasts and a stack
dump do about it (the library's waiting.py, the command's stop handling) is then seen on demand,
before and after a change: where `main` has no `command` (main.py of before the stop-at-once
change), the script calls `main.main()` in its place, so that one script serves both.

  stall_hook.py [--at decode|dit] [--shot N] [--block K] [--mode wait|sleep] [--check] -- \
      SEEDVR2X ARGS...

  --at decode   the first slice of the Nth decode of the run (default). The stall is queued on the
                stream right after that slice's own kernels (model.decode_sliced, which the tiled
                and the untiled decodes both call per slice), so the main thread's next
                model.synchronize waits on it. A tiled decode stalls in its first tile.
  --at dit      the DiT's forward, at block K (--block, 10 by default) of the first window of the
                Nth shot: a forward pre-hook on that block (nablocks), so that with no block
                swapped the CPU goes on launching the window's kernels behind the stalled one,
                the launch queue filling.
  --shot N      the Nth shot's (1 by default). Counted over the run's own shots: the calibration's
                trial window and decode, which run first, are not counted.
  --mode wait   cuStreamWaitValue32 on a device word nothing ever writes: the GPU idle, the
                stream blocked (default). Where the driver refuses stream memory operations (the
                call's CUresult is not 0, as under WSL2 it may be), the reason is printed and the
                mode becomes sleep.
  --mode sleep  torch.cuda._sleep(2**62): the GPU busy in a kernel that never ends.
  --check       install the hooks against the library's modules and exit 0, without CUDA and
                without running anything: whether the hook points exist in this tree.

Everything after `--` is seedvr2x's command line, as the script `seedvr2x` takes it. Nothing else
changes: the run's options, seed and outputs are the command's; the hook only queues one more
operation on the stream when it fires, and prints one line on stderr (flushed):

  STALL_HOOK fired: <where>, mode <wait|sleep>, CUresult <n>, pid <pid>, at <time>

Run with seedvr2x's environment (apps/seedvr2x/.venv/bin/python), from the repository's root,
holding the GPU's lock (LOCK, its file); PYTHONFAULTHANDLER=1 gives a stack dump on a fatal
signal. The commands (SOURCE a clip of two shots, MODEL_DIR the directory holding the DiT and
the VAE files; a short clip of 66 frames at 720p decodes in about a minute, so the stall comes
early; a cut list of one cut makes two shots):

  flock LOCK env PYTHONFAULTHANDLER=1 \
      apps/seedvr2x/.venv/bin/python research/scripts/stall_hook.py --at decode -- \
      SOURCE -o OUT --model-dir MODEL_DIR --resolution 720 --cuts CUTS -v

  flock LOCK env PYTHONFAULTHANDLER=1 \
      apps/seedvr2x/.venv/bin/python research/scripts/stall_hook.py --at dit --block 10 \
      --mode sleep -- SOURCE -o OUT --model-dir MODEL_DIR --resolution 720 --cuts CUTS -v

then, from another shell, `kill -USR1 PID` for the stacks, `kill -TERM PID` for a stop, and the
time the process takes to end (`ps`, `date`). The run stays stalled until it is stopped; the
script never ends it.

2026-11-06 (impl-stop). Standalone: it imports seedvr2x's library and the command's modules, and
puts apps/seedvr2x on the path for the latter, the CLI's modules being top-level."""

import argparse
import contextlib
import ctypes
import functools
import os
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

# CU_STREAM_WAIT_VALUE_EQ: the wait ends when the word equals the value (CUDA driver API,
# cuStreamWaitValue32). The word is never written, so it holds 0 and the value is 1.
WAIT_VALUE_EQ = 0x1


class State:
    """What the hooks share: the options, the counts, the armed hook and the word kept alive."""

    at = "decode"
    shot = 1
    block = 10
    mode = "wait"
    decodes = 0  # the run's decodes seen, the calibration's left out
    windows = 0  # the run's shots seen by the DiT, likewise
    decode_armed = False  # the Nth decode has begun and has not stalled yet
    fired = False
    word: Any = None  # the device word nothing writes: kept alive, the wait reads it


def _calibrating() -> bool:
    """Whether the call comes from the calibration (apps/seedvr2x/calibration.py), which runs a
    trial window and decode before the run's own and shares the functions hooked here."""
    frame = sys._getframe(1)
    while frame is not None:
        if os.path.basename(frame.f_code.co_filename) == "calibration.py":
            return True
        frame = frame.f_back  # pyright: ignore[reportAssignmentType]
    return False


def _report(where: str, mode: str, status: int) -> None:
    print(
        f"STALL_HOOK fired: {where}, mode {mode}, CUresult {status}, pid {os.getpid()},"
        f" at {time.strftime('%Y-%m-%d %H:%M:%S')}",
        file=sys.stderr,
        flush=True,
    )


def stall(where: str) -> None:
    """Queue work on the current CUDA stream that the GPU never finishes, once, and say so."""
    import torch

    State.fired = True
    mode = State.mode
    status = 0
    if mode == "wait":
        State.word = torch.zeros(1, dtype=torch.int32, device="cuda")
        stream = torch.cuda.current_stream().cuda_stream
        status = _wait_value(stream, State.word.data_ptr())
        if status != 0:
            print(
                f"STALL_HOOK: stream memory operations refused (CUresult {status}"
                f" {_error_name(status)}): falling back to --mode sleep",
                file=sys.stderr,
                flush=True,
            )
            mode = "sleep"
    if mode == "sleep":
        torch.cuda._sleep(2**62)
    _report(where, mode, status)


def _driver() -> ctypes.CDLL:
    return ctypes.CDLL("libcuda.so.1")


def _wait_value(stream: int, address: int) -> int:
    """cuStreamWaitValue32_v2(stream, address, 1, CU_STREAM_WAIT_VALUE_EQ, 0) through libcuda,
    or cuStreamWaitValue32 where the driver lacks the _v2 name: its CUresult."""
    library = _driver()
    function = getattr(library, "cuStreamWaitValue32_v2", None) or library.cuStreamWaitValue32
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
    return int(function(ctypes.c_void_p(stream), address, 1, WAIT_VALUE_EQ))


def _error_name(status: int) -> str:
    name = ctypes.c_char_p()
    try:
        if _driver().cuGetErrorName(status, ctypes.byref(name)) == 0 and name.value:
            return name.value.decode()
    except (OSError, AttributeError):
        pass
    return "unknown"


def install(shot_pass: Any, model: Any, run: Any | None) -> None:
    """Patch the library's module attributes for the option chosen; the library is never edited.

    decode: shot_pass._slices (a decode's start, counted: the Nth arms the hook) and
    model.decode_sliced (every slice of every decode, tiled or not, comes through it: the hook
    stalls after the first slice of the armed decode). dit: sample_windows, in the library and
    in the command's run module, which imported the name (a shot's windows, counted: the Nth
    registers the pre-hook on its block)."""
    if State.at == "decode":
        _hook_decode(shot_pass, model)
    else:
        _hook_dit(shot_pass, run)


def _hook_decode(shot_pass: Any, model: Any) -> None:
    started: Callable[..., Any] = shot_pass._slices
    sliced: Callable[..., Iterator[Any]] = model.decode_sliced

    @contextlib.contextmanager
    @functools.wraps(started)
    def slices(*args: Any, **kwargs: Any) -> Iterator[Any]:
        # Counted where the decode begins, so that the Nth decode is the Nth shot's, not the
        # Nth tile's or slice's.
        if not _calibrating() and not State.fired:
            State.decodes += 1
            State.decode_armed = State.decodes == State.shot
        with started(*args, **kwargs) as taken:
            yield taken

    @functools.wraps(sliced)
    def decode_sliced(*args: Any, **kwargs: Any) -> Iterator[Any]:
        inner = sliced(*args, **kwargs)
        try:
            for decoded in inner:
                # After the slice's own kernels are queued, before anything waits on them.
                if State.decode_armed and not State.fired:
                    State.decode_armed = False
                    stall(f"decode {State.shot}, first slice")
                yield decoded
        finally:
            inner.close()

    shot_pass._slices = slices
    model.decode_sliced = decode_sliced


def _hook_dit(shot_pass: Any, run: Any | None) -> None:
    original: Callable[..., Any] = shot_pass.sample_windows

    @functools.wraps(original)
    def sample_windows(models: Any, *args: Any, **kwargs: Any) -> Any:
        if not _calibrating() and not State.fired:
            State.windows += 1
            if State.windows == State.shot:
                _register(models)
        return original(models, *args, **kwargs)

    shot_pass.sample_windows = sample_windows
    if run is not None and hasattr(run, "sample_windows"):
        run.sample_windows = sample_windows


def _register(models: Any) -> None:
    blocks = models.runner.dit.blocks
    if not 0 <= State.block < len(blocks):
        raise SystemExit(f"stall_hook: --block {State.block}, the DiT has {len(blocks)} blocks")
    handles: list[Any] = []

    def before(module: Any, args: Any) -> None:
        if not State.fired:
            stall(f"dit, shot {State.shot}, block {State.block} of the first window")
        for handle in handles:
            handle.remove()

    handles.append(blocks[State.block].register_forward_pre_hook(before))


def _command(main: Any) -> int:
    """The command as its script runs it: command() where main has one, main() before it."""
    entry = getattr(main, "command", None) or main.main
    return int(entry())


def _arguments(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    cut = argv.index("--") if "--" in argv else len(argv)
    own, rest = argv[:cut], argv[cut + 1 :]
    parser = argparse.ArgumentParser(
        prog="stall_hook.py",
        description="Queue GPU work that never finishes inside a seedvr2x run, then run"
        " the command: stall_hook.py [options] -- SEEDVR2X ARGS...",
    )
    parser.add_argument("--at", choices=("decode", "dit"), default="decode")
    parser.add_argument("--shot", type=int, default=1, help="the Nth shot's (decode: the Nth)")
    parser.add_argument("--block", type=int, default=10, help="--at dit: the block's index")
    parser.add_argument("--mode", choices=("wait", "sleep"), default="wait")
    parser.add_argument("--check", action="store_true", help="install the hooks, exit 0")
    options = parser.parse_args(own)
    if options.shot < 1:
        parser.error("--shot counts from 1")
    if not rest and not options.check:
        parser.error("seedvr2x's arguments follow --")
    return options, rest


def main(argv: list[str]) -> int:
    options, rest = _arguments(argv)
    State.at, State.shot, State.block, State.mode = (
        options.at,
        options.shot,
        options.block,
        options.mode,
    )
    # The command's modules are top-level (main, run, ...), in apps/seedvr2x.
    sys.path.insert(0, str(ROOT / "apps" / "seedvr2x"))
    import main as command  # the seedvr2x command's own module
    import run
    from seedvr2x import model, shot_pass

    install(shot_pass, model, run)
    if options.check:
        print(f"stall_hook: hooks installed (--at {State.at})")
        return 0
    sys.argv = ["seedvr2x", *rest]
    return _command(command)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
