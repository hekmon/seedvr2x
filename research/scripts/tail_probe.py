"""hold_probe.py with the shipped decoder tail switched on.

  tail_probe.py [--] HOLD_PROBE ARGS...

seedvr2x/model.py's tail_frames (Decoder3D.tail_frames: the decoder's blocks after its last
temporal upsampler run a frame at a time, the causal convolutions' memory carried between frames)
is switched on for the VAE that load_models returns, then hold_probe.py runs with its arguments
unchanged, so that a run with this wrapper and one without differ by the tail alone. torch's
allocator is set first, from hold_probe.py's own --allocator, --gc, --split and --conf, as
slice_probe.py sets it (torch takes its backend once, at its import).

Example, the tail against the plain decoder at 5 latents and 768 px tiles, 1080p:

  tail_probe.py --dit DIT --vae VAE --allocator native --conf expandable_segments:True --without dit --size 1080 \\
      --steps d5/768

Written on 2026-11-06 for the decoder's tail by tile side (docs/small-cards.md)."""

import os
import runpy
import sys

here = os.path.dirname(os.path.abspath(__file__))
rest = sys.argv[1:]
if rest[:1] == ["--"]:
    rest = rest[1:]
sys.path.insert(0, here)
from slice_probe import allocator_conf  # noqa: E402

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = allocator_conf(rest)
from seedvr2x import model  # noqa: E402

original = model.load_models


def load(*args, **kwargs):
    made = original(*args, **kwargs)
    model.tail_frames(made.runner.vae, True)
    print("tail_probe: the shipped decoder tail switched on", flush=True)
    return made


model.load_models = load
sys.argv = [os.path.join(here, "hold_probe.py"), *rest]
runpy.run_path(sys.argv[0], run_name="__main__")
