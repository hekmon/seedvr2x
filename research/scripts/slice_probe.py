"""hold_probe.py with the VAE decode cut finer in time (the first slice, or the decoder's tail),
or with the DiT's lean attention.

  slice_probe.py [--first N] [--tail-frames] [--lean [--lean-chunk 4096] [--gelu-inplace]] \
      -- HOLD_PROBE ARGS...

seedvr2x decodes a shot's latents as numz's slicing_decode cuts them: the first latent with the
next one, 5 frames, then one latent, 4 frames, at a time (seedvr2x/model.py, latent_slices), and
the decoder takes a slice's frames through every block at once
(vendor/models/video_vae_v3/modules/attn_video_vae.py, Decoder3D.forward). At 512 px tiles the
first slice peaks at 6.39 GiB where a slice of one latent, 1 frame, peaks at 2.35 (place-study,
2026-11-05): the floor of a 1080p clip on a small card. This runs hold_probe.py with:

--first N    model.latent_slices replaced by N latents first, then one at a time;
--tail-frames  the decoder's blocks after its last temporal upsampler (the up blocks from
             temporal_up_num on, then the output norm, activation and convolution) run one frame
             at a time, each frame after the first with the causal convolutions' memory
             (MemoryState.ACTIVE), as the slices themselves follow one another. ComfyUI v0.44.0
             runs its SeedVR2 decoder's tail a frame at a time so (comfy/ldm/seedvr/vae.py,
             Decoder3D.tail_frames, read 2026-11-05: described, not copied).

--lean       dit_lean.py's attention (chunks of --lean-chunk tokens), and with --gelu-inplace its
             MLP's GELU in place: what the window steps (L) need lean.

so that its dF/T steps give the peak of such a decode and, against a reference made with the
original code, whether its frames keep their bits; its L steps the same of a lean window.

Each of these imports torch before hold_probe.py runs: the allocator's settings hold_probe.py
would set (its --allocator, --gc, --split and --conf) are set here first, the same string, since
torch takes its backend once.

Written on 2026-11-05 for place-study (Q3; --lean for the night's Q9)."""

import os
import runpy
import sys
from typing import Any


def main() -> int:
    args = sys.argv[1:]
    first, tail, lean, chunk, gelu = 0, False, False, 4096, False
    while args and args[0] != "--":
        if args[0] == "--first":
            first, args = int(args[1]), args[2:]
        elif args[0] == "--tail-frames":
            tail, args = True, args[1:]
        elif args[0] == "--lean":
            lean, args = True, args[1:]
        elif args[0] == "--lean-chunk":
            chunk, args = int(args[1]), args[2:]
        elif args[0] == "--gelu-inplace":
            gelu, args = True, args[1:]
        elif args[0] in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            raise SystemExit(__doc__)
    rest = args[1:]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = allocator_conf(rest)
    if lean:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import dit_lean

        dit_lean.install(lean=True, chunk=chunk)
        if gelu:
            dit_lean.gelu_inplace()
    if first:
        from seedvr2x import model

        original = model.latent_slices

        def cut(models: Any, latents: int) -> list[int]:
            slices = original(models, latents)
            if len(slices) == 1 and latents <= first:
                return slices
            return [first] + [1] * (latents - first)

        model.latent_slices = cut
    if tail:
        install_tail()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.argv = [os.path.join(here, "hold_probe.py"), *rest]
    runpy.run_path(sys.argv[0], run_name="__main__")
    return 0


def allocator_conf(rest: list[str]) -> str:
    """PYTORCH_CUDA_ALLOC_CONF as hold_probe.py builds it from its own options."""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--allocator", choices=("async", "native"), default="async")
    parser.add_argument("--gc", type=float)
    parser.add_argument("--split", type=int)
    parser.add_argument("--conf")
    known, _ = parser.parse_known_args(rest)
    conf = "backend:cudaMallocAsync" if known.allocator == "async" else "backend:native"
    if known.gc is not None:
        conf += f",garbage_collection_threshold:{known.gc}"
    if known.split is not None:
        conf += f",max_split_size_mb:{known.split}"
    if known.conf:
        conf += f",{known.conf}"
    return conf


def install_tail() -> None:
    """Decoder3D.forward (out of training, the branch seedvr2x runs) with its tail a frame at a
    time: the same modules in the same order on each frame."""
    import torch

    from seedvr2x.vendor.models.video_vae_v3.modules import attn_video_vae as vae
    from seedvr2x.vendor.models.video_vae_v3.modules.types import MemoryState

    def forward(
        self: Any,
        sample: Any,
        latent_embeds: Any = None,
        memory_state: Any = MemoryState.DISABLED,
    ) -> Any:
        sample = self.conv_in(sample, memory_state=memory_state)
        sample = self.mid_block(sample, latent_embeds, memory_state=memory_state)
        split = self.temporal_up_num
        for up_block in self.up_blocks[:split]:
            sample = up_block(sample, latent_embeds, memory_state=memory_state)
        if memory_state == MemoryState.DISABLED or sample.shape[2] == 1:
            frames = [sample]
        else:
            frames = list(sample.split(1, dim=2))
        del sample
        out = []
        for index, frame in enumerate(frames):
            frames[index] = None  # each frame's input dropped as it is taken
            state = memory_state if index == 0 else MemoryState.ACTIVE
            for up_block in self.up_blocks[split:]:
                frame = up_block(frame, latent_embeds, memory_state=state)
            frame = vae.causal_norm_wrapper(self.conv_norm_out, frame)
            frame = self.conv_act(frame)
            out.append(self.conv_out(frame, memory_state=state))
        return out[0] if len(out) == 1 else torch.cat(out, dim=2)

    vae.Decoder3D.forward = forward


if __name__ == "__main__":
    sys.exit(main())
