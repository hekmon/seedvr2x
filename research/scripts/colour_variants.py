#!/usr/bin/env python3
"""Colour-correction variants for the colour study (docs/colour.md, step 2), built on seedvr2x's
runtime/colour.py so that a winner ports as it is.

  colour_variants.py selftest
  colour_variants.py bench SPEC[,SPEC...] [FRAMES]   # GPU (under the lock): time and memory per 4K frame

Every variant takes content (T, 3, H, W), the VAE's decode in [-1, 1], unclamped, and reference
(T, 3, H, W), the input through the model's transform in [-1, 1], both of one shot, and returns
(T, 3, H, W) float32 in [0, 1]. A variant is named by a spec string:

  none                         the decode alone, brought to [0, 1]
  lab                          our lab, colour.correct: the split in RGB at 5 stages, then numz's
                               histogram step pooled over the shot
  split:SPACE:SL:SC[:HIST]     the split: lightness takes the reference's low band below SL à-trous
                               stages, colour below SC stages, the rest from the content. SPACE:
                               rgb (one scale, SL = SC: numz's wavelet at 5), ycc (BT.709 Y'CbCr on
                               the gamma-encoded RGB: linear, so ycc:S:S is rgb:S:S), lab (CIELAB,
                               colour.py's conversions), ok (OKLab). HIST: hist (numz's histogram
                               step after the split, as lab), histab (a* and b* only), histL<W> (L*
                               weight W, e.g. histL0.9), histY<W> (L* only, weight W, a* and b*
                               kept, e.g. histY0.8)
  guided:SPACE:SL:R:EPS        lightness as split:SPACE:SL; colour from a guided filter of the
                               reference-minus-content colour difference, guided by the content's
                               lightness (He, Sun and Tang), box radius R px, regularisation EPS (in
                               units of the guide squared)

The à-trous low band of S stages is colour.low_band with S stages instead of 5 (1: σ 0.71 px, 2:
1.58, 3: 3.24, 4: 6.52, 5: 13.06); S = 0 takes everything from the reference. Lightness and colour
are Y' and (Cb, Cr) in ycc, L* and (a*, b*) in lab, L and (a, b) in ok. All in float32, on the
tensors' device.

The baseline is loaded from COLOUR_BASELINE (a copy of seedvr2x/src/seedvr2x/runtime/colour.py at
the commit the study names), or from the repository next to this script.
"""
import importlib.util
import math
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.environ.get("COLOUR_BASELINE") or os.path.join(
    HERE, "..", "..", "seedvr2x", "src", "seedvr2x", "runtime", "colour.py")


def load_baseline(path=BASELINE):
    spec = importlib.util.spec_from_file_location("colour_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = load_baseline()

# BT.709 Y'CbCr on gamma-encoded RGB (the values themselves, any range: only differences go through it)
YCC = torch.tensor([[0.2126, 0.7152, 0.0722],
                    [-0.2126 / 1.8556, -0.7152 / 1.8556, 0.9278 / 1.8556],
                    [0.7874 / 1.5748, -0.7152 / 1.5748, -0.0722 / 1.5748]], dtype=torch.float64)
YCC_INV = torch.linalg.inv(YCC)

# OKLab (Ottosson, 2020): linear sRGB -> LMS -> cube root -> Lab, and back
OK_M1 = torch.tensor([[0.4122214708, 0.5363325363, 0.0514459929],
                      [0.2119034982, 0.6806995451, 0.1073969566],
                      [0.0883024619, 0.2817188376, 0.6299787005]], dtype=torch.float64)
OK_M2 = torch.tensor([[0.2104542553, 0.7936177850, -0.0040720468],
                      [1.9779984951, -2.4285922050, 0.4505937099],
                      [0.0259040371, 0.7827717662, -0.8086757660]], dtype=torch.float64)
OK_M2_INV = torch.tensor([[1.0, 0.3963377774, 0.2158037573],
                          [1.0, -0.1055613458, -0.0638541728],
                          [1.0, -0.0894841775, -1.2914855480]], dtype=torch.float64)
OK_M1_INV = torch.tensor([[4.0767416621, -3.3077115913, 0.2309699292],
                          [-1.2684380046, 2.6097574011, -0.3413193965],
                          [-0.0041960863, -0.7034186147, 1.7076147010]], dtype=torch.float64)


def mix(x, m):
    """The 3x3 matrix m applied to the channels of x (T, 3, H, W): float32."""
    return torch.einsum("ij,tjhw->tihw", m.to(device=x.device, dtype=torch.float32), x)


def low_band(frames, stages):
    """colour.low_band with `stages` à-trous stages (colour.py has 5): the same code, its stage count
    a parameter. 0 stages: the frames themselves. float32, the shape of frames."""
    *lead, height, width = frames.shape
    x = frames.reshape(-1, 1, height, width).to(torch.float32)
    cap = max(1, min(height, width) // 8)
    for stage in range(stages):
        d = min(2**stage, cap)
        x = F.pad(x, (d, d, d, d), mode="replicate")
        x = (x[..., :, : -2 * d] + x[..., :, 2 * d :] + 2 * x[..., :, d:-d]) * 0.25
        x = (x[..., : -2 * d, :] + x[..., 2 * d :, :] + 2 * x[..., d:-d, :]) * 0.25
    return x.reshape(*lead, height, width)


def srgb_to_linear(x):
    return torch.where(x > 0.04045, torch.pow((x.clamp(min=0) + 0.055) / 1.055, 2.4), x / 12.92)


def linear_to_srgb(x):
    return torch.where(x > 0.0031308, 1.055 * torch.pow(x.clamp(min=0), 1 / 2.4) - 0.055, 12.92 * x)


def rgb_to_ok(rgb01):
    lms = mix(srgb_to_linear(rgb01), OK_M1)
    return mix(torch.sign(lms) * lms.abs().pow(1 / 3), OK_M2)


def ok_to_rgb(ok):
    lms = mix(ok, OK_M2_INV) ** 3
    return linear_to_srgb(mix(lms, OK_M1_INV)).clamp(0, 1)


def to_space(space, x):
    """x (T, 3, H, W) in [-1, 1] (ycc: any range) to SPACE; lab and ok clamp to [0, 1] first."""
    if space == "ycc":
        return mix(x.to(torch.float32), YCC)
    if space == "lab":
        return C.rgb_to_lab(C.unit_range(x))
    if space == "ok":
        return rgb_to_ok(C.unit_range(x))
    raise ValueError(f"space {space!r}")


def from_space(space, y):
    """SPACE back to RGB in [0, 1], clamped."""
    if space == "ycc":
        return mix(y, YCC_INV).add(1.0).mul_(0.5).clamp_(0, 1)
    if space == "lab":
        return C.lab_to_rgb(y)
    if space == "ok":
        return ok_to_rgb(y)
    raise ValueError(f"space {space!r}")


def box(x, r):
    """Mean over a (2r+1)^2 window, edges reflected: x (N, 1, H, W)."""
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="reflect"), 2 * r + 1, stride=1)


def guided(guide, p, r, eps):
    """He, Sun and Tang's guided filter of p (N, 1, H, W) by guide (N, 1, H, W): locally
    q = a * guide + b, a and b averaged over the windows that hold each pixel."""
    mi, mp = box(guide, r), box(p, r)
    cov = box(guide * p, r) - mi * mp
    var = box(guide * guide, r) - mi * mi
    a = cov / (var + eps)
    b = mp - a * mi
    return box(a, r) * guide + box(b, r)


def split(content, reference, space, sl, sc):
    """The split in SPACE: [0, 1] RGB (T, 3, H, W)."""
    if space == "rgb":
        if sl != sc:
            raise ValueError("rgb has one scale: SL = SC")
        return C.unit_range(content.to(torch.float32) + (low_band(reference, sl) - low_band(content, sl)))
    c, r = to_space(space, content), to_space(space, reference)
    d = r - c
    out = torch.cat([c[:, :1] + low_band(d[:, :1], sl), c[:, 1:] + low_band(d[:, 1:], sc)], dim=1)
    return from_space(space, out)


def guided_split(content, reference, space, sl, r, eps):
    c, ref = to_space(space, content), to_space(space, reference)
    d = ref - c
    light = c[:, :1] + low_band(d[:, :1], sl)
    g = c[:, :1]
    colour = [c[:, k:k + 1] + guided(g, d[:, k:k + 1], r, eps) for k in (1, 2)]
    return from_space(space, torch.cat([light, *colour], dim=1))


def histogram_step(out01, reference, mode):
    """numz's matching, colour.py's histograms pooled over the shot: out01 (T, 3, H, W) in [0, 1]
    against the reference's CIELAB. mode: hist (a*, b* matched, L* weight colour.LUMINANCE_WEIGHT),
    histab (L* kept), histL<W> (L* weight W), histY<W> (L* only, weight W; a* and b* kept)."""
    lab = C.rgb_to_lab(out01)
    ref = C.rgb_to_lab(C.unit_range(reference))
    h = C.Histograms(lab.device)
    h.add(lab, ref)
    if mode == "hist":
        return C.lab_to_rgb(h.match(lab))
    if mode.startswith("histY"):  # L* only: W of its own and the rest matched; a* and b* kept
        w = float(mode[len("histY"):])
        light = C._match(lab[:, 0], h.content[0], h.reference[0])
        return C.lab_to_rgb(torch.stack([lab[:, 0] * w + light * (1 - w), lab[:, 1], lab[:, 2]], dim=1))
    w = 1.0 if mode == "histab" else float(mode[len("histL"):])
    light = C._match(lab[:, 0], h.content[0], h.reference[0])
    a = C._match(lab[:, 1], h.content[1], h.reference[1])
    b = C._match(lab[:, 2], h.content[2], h.reference[2])
    return C.lab_to_rgb(torch.stack([lab[:, 0] * w + light * (1 - w), a, b], dim=1))


def apply(spec, content, reference):
    """The variant named by spec on one shot: (T, 3, H, W) float32 in [0, 1]."""
    parts = spec.split(":")
    if parts[0] == "none":
        return C.unit_range(content)
    if parts[0] == "lab":
        return C.correct(content, reference)
    if parts[0] == "split":
        space, sl, sc = parts[1], int(parts[2]), int(parts[3])
        out = split(content, reference, space, sl, sc)
        return histogram_step(out, reference, parts[4]) if len(parts) > 4 else out
    if parts[0] == "guided":
        space, sl, r, eps = parts[1], int(parts[2]), int(parts[3]), float(parts[4])
        return guided_split(content, reference, space, sl, r, eps)
    raise ValueError(f"variant {spec!r}")


def per_frame(spec):
    """Whether the variant treats each frame alone (no histogram pooled over the shot)."""
    return not (spec == "lab" or any(p.startswith("hist") for p in spec.split(":")[4:]))


# ---------------------------------------------------------------- self-test (CPU)

def selftest():
    torch.manual_seed(0)
    t, h, w = 3, 96, 128
    # smooth-ish content and reference in [-1, 1], content slightly out of range in places
    base = torch.rand(t, 3, h // 8, w // 8) * 2 - 1
    ref = F.interpolate(base, size=(h, w), mode="bilinear", align_corners=False)
    content = (ref * 1.2 + 0.1 * torch.randn(t, 3, h, w)).clamp(-1.05, 1.05)
    # low_band with 5 stages is colour.py's, bit for bit
    assert torch.equal(low_band(content, 5), C.low_band(content)), "low_band(5) != colour.low_band"
    # rgb split at 5 stages = colour.transfer, clamped
    assert torch.equal(split(content, ref, "rgb", 5, 5), C.unit_range(C.transfer(content, ref)))
    # ycc at equal scales = rgb (linear transform), to float32 rounding
    d = (split(content, ref, "ycc", 4, 4) - split(content, ref, "rgb", 4, 4)).abs().max().item()
    assert d < 1e-5, f"ycc:4:4 vs rgb:4:4 {d}"
    # split:...:hist at 5 stages in rgb = our lab
    assert torch.allclose(apply("split:rgb:5:5:hist", content, ref), apply("lab", content, ref), atol=1e-6)
    # OKLab: white -> (1, 0, 0), black -> 0, round trip
    white = torch.ones(1, 3, 1, 1)
    ok = rgb_to_ok(white).flatten()
    assert abs(ok[0].item() - 1) < 1e-4 and ok[1:].abs().max().item() < 1e-4, ok
    x = torch.rand(2, 3, 16, 16)
    rt = (ok_to_rgb(rgb_to_ok(x)) - x).abs().max().item()
    assert rt < 1e-4, f"OKLab round trip {rt}"
    # a split with 0 stages takes everything from the reference
    for space in ("ycc", "lab", "ok"):
        z = (split(content, ref, space, 0, 0) - C.unit_range(ref)).abs().max().item()
        assert z < 2e-4, f"{space}:0:0 vs reference {z}"
    # guided filter: a constant difference passes unchanged; huge eps = box mean of p
    g = torch.rand(1, 1, 32, 32)
    p = torch.full((1, 1, 32, 32), 0.3)
    assert (guided(g, p, 4, 1e-3) - 0.3).abs().max().item() < 1e-5
    q = guided(g, torch.rand(1, 1, 32, 32), 4, 1e6)
    assert q.std().item() < 0.2
    # stage sigmas
    imp = torch.zeros(1, 1, 257, 257)
    imp[..., 128, 128] = 1
    yy = (torch.arange(257) - 128).double()
    for s in range(1, 6):
        k = low_band(imp, s)[0, 0].double()
        sig = math.sqrt(float((k.sum(1) * yy**2).sum()))
        print(f"{s} stages: sigma {sig:.3f} px")
    # histY1 keeps the split's output, to the CIELAB round trip
    d = (apply("split:ycc:5:3:histY1", content, ref) - apply("split:ycc:5:3", content, ref)).abs().max().item()
    assert d < 1e-4, f"histY1 vs the split alone {d}"
    for spec in ("none", "lab", "split:rgb:5:5", "split:ycc:5:3", "split:lab:5:3", "split:ok:5:3",
                 "split:ycc:5:3:hist", "split:ycc:5:3:histab", "split:ycc:5:3:histL0.9", "split:ycc:5:3:histY0.8",
                 "guided:ycc:5:4:0.001"):
        y = apply(spec, content, ref)
        assert y.shape == content.shape and y.dtype == torch.float32 and 0 <= y.min() and y.max() <= 1, spec
    print(f"baseline {os.path.abspath(BASELINE)}")
    print("selftest passed")


def bench(specs, frames=4, height=2160, width=3840, repeats=3):
    """Time and memory of each variant on the GPU, per 4K frame (DESIGN.md: it runs in the decode's
    second pass): `frames` frames at once, content in bfloat16 as the VAE decodes it, smooth random
    pictures (the cost doesn't depend on the content). The best of `repeats` runs after a warm-up;
    the torch peak above what was allocated before."""
    import time
    dev = torch.device("cuda")
    torch.manual_seed(0)
    base = torch.rand(frames, 3, height // 16, width // 16, device=dev) * 2 - 1
    ref = F.interpolate(base, size=(height, width), mode="bicubic", align_corners=False).clamp_(-1, 1)
    content = (ref * 1.1 + 0.05 * torch.randn_like(ref)).to(torch.bfloat16)
    print(f"{frames} frames {width}x{height}, {torch.cuda.get_device_name(dev)}, torch {torch.__version__}")
    print("| Variant | ms per frame | torch peak (GiB) | per frame (GiB) |")
    print("|---|---|---|---|")
    for spec in specs:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeats + 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = apply(spec, content, ref)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            del out
        peak = (torch.cuda.max_memory_allocated() - before) / 2**30
        print(f"| {spec} | {1000 * min(times[1:]) / frames:.1f} | {peak:.2f} | {peak / frames:.2f} |", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "selftest":
        selftest()
    elif len(sys.argv) >= 3 and sys.argv[1] == "bench":
        bench([s for s in sys.argv[2].split(",") if s],
              frames=int(sys.argv[3]) if len(sys.argv) > 3 else 4)
    else:
        sys.exit(__doc__)
