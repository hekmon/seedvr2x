#!/usr/bin/env python3
"""Run ComfyUI's main.py with its sampling and VAE calls timed and their memory peaks recorded.

  comfy_wrap.py COMFY_DIR [COMFYUI ARGS...]        (the log's path is the environment's
                                                    COMFY_PHASE_LOG)

ComfyUI's code is run, never copied or changed: this file puts its directory on the path, then
installs an import hook that wraps, the moment each module has been imported, the functions the
nodes call by attribute at run time (comfy.sample.sample, and encode, encode_tiled, decode and
decode_tiled of comfy.sd.VAE), then runs main.py as __main__. A hook rather than an import here,
because main.py sets the environment and the argument parsing before the first import of comfy.*
and torch.

Each call is bracketed by torch.cuda.synchronize() (so that the seconds are the work's, not the
launch's), torch.cuda.reset_peak_memory_stats() at its start, and gives one JSON line to the
file: the function, the seconds, torch's peak allocated and reserved memory in GiB during the call,
the allocated and reserved memory at its start, the card's used memory (the driver's, by
torch.cuda.mem_get_info: everything on the card, other programs' included) at its start and end,
and its highest value, sampled every 20 ms by a thread. torch's counters count what torch's
allocator hands out; memory that comfy-aimdo maps itself (the dynamic VRAM's weights) is in the
card's used figure and may be missing from torch's. A call inside another (decode falling back to
decode_tiled) is logged with its depth and resets nothing, so that the outer call's peak stays
whole.
"""

import importlib.abc
import importlib.util
import json
import os
import runpy
import sys
import threading
import time

GIB = 1024**3
LOG = os.environ.get("COMFY_PHASE_LOG")
VAE_METHODS = ("encode", "encode_tiled", "decode", "decode_tiled")
_state = threading.local()
_write = threading.Lock()


def _used() -> int:
    import torch

    free, total = torch.cuda.mem_get_info()
    return total - free


def _wrapped(function, label: str):
    def call(*args, **kwargs):
        import torch

        depth = getattr(_state, "depth", 0)
        torch.cuda.synchronize()
        if depth == 0:
            torch.cuda.reset_peak_memory_stats()
        before = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved(), _used())
        top = [before[2]]
        stop = threading.Event()

        def sample():
            while not stop.wait(0.02):
                top[0] = max(top[0], _used())

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        _state.depth = depth + 1
        start = time.monotonic()
        failed = None
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            failed = type(error).__name__
            raise
        finally:
            torch.cuda.synchronize()
            seconds = time.monotonic() - start
            _state.depth = depth
            stop.set()
            sampler.join()
            end_used = _used()
            top[0] = max(top[0], end_used)
            line = {
                "function": label,
                "depth": depth,
                "seconds": seconds,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
                "allocated_before_gib": before[0] / GIB,
                "reserved_before_gib": before[1] / GIB,
                "device_used_before_gib": before[2] / GIB,
                "device_used_end_gib": end_used / GIB,
                "device_used_max_gib": top[0] / GIB,
                "error": failed,
            }
            if LOG:
                with _write, open(LOG, "a") as handle:
                    handle.write(json.dumps(line) + "\n")

    call.__name__ = getattr(function, "__name__", label)
    call.__wrapped__ = function
    return call


def _patch(name: str, module) -> None:
    if name == "comfy.sample":
        module.sample = _wrapped(module.sample, "sample")
    elif name == "comfy.sd":
        for method in VAE_METHODS:
            setattr(module.VAE, method, _wrapped(getattr(module.VAE, method), f"VAE.{method}"))


class Hook(importlib.abc.MetaPathFinder):
    targets = ("comfy.sample", "comfy.sd")

    def __init__(self):
        self.busy: set[str] = set()

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.targets or fullname in self.busy:
            return None
        self.busy.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self.busy.discard(fullname)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        execute = loader.exec_module

        def exec_module(module):
            execute(module)
            _patch(fullname, module)

        loader.exec_module = exec_module  # on this loader instance only
        return spec


def main() -> None:
    if sys.argv[1:2] in (["-h"], ["--help"]):
        print(__doc__)
        return
    directory = os.path.abspath(sys.argv[1])
    sys.path.insert(0, directory)
    sys.meta_path.insert(0, Hook())
    script = os.path.join(directory, "main.py")
    sys.argv = [script, *sys.argv[2:]]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
