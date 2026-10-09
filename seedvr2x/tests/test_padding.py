"""The frames' padding before the model, on the CPU (DESIGN.md, Pipeline step 0): at the bottom, at
least 8 rows reflected from the picture, up to a multiple of 16, then 16 black ones; at the right,
columns alike when the width isn't a multiple of 16; all trimmed after the decode, and from lab's
reference. Held bit for bit to research/scripts/numerics_patch.py's Pad, which padded the variant
measured (NUM_PAD=reflect>=8+black+16), and the tests' numz padding to numz's DivisiblePad. The
GPU's side is test_regression.py's."""

import importlib.util
import math
import os
import sys
from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torchvision.transforms import Compose, InterpolationMode, Lambda, Normalize
from torchvision.transforms import functional as TVF

from seedvr2x.runtime import model
from seedvr2x.runtime.job import (
    JobError,
    check_target,
    output_size,
    padded_size,
    padding,
    target_size,
)
from seedvr2x.runtime.shot import (
    _frames,
    decode_shot,
    encode_shot,
    encoder_inputs,
    padded_length,
    reference_inputs,
)
from seedvr2x.vendor.data.image.transforms.divisible_crop import DivisiblePad

PROJECT = Path(__file__).resolve().parents[1]
NUMERICS_PATCH = PROJECT.parent / "research" / "scripts" / "numerics_patch.py"

# Heights and the rows reflected under them: 1080p's 8 (where numz pads 8 black rows), 720p's and
# 4K's 16 (where numz pads none), multiples of 16 and their neighbours, odd heights, SD's.
ROWS = {
    **{1080: 8, 720: 16, 2160: 16, 2048: 16, 1088: 16, 1096: 8, 1060: 12},
    **{1079: 9, 1081: 23, 1083: 21, 1087: 17, 1089: 15, 541: 19, 540: 20, 480: 16},
}
# Widths and the columns reflected right of them: none at a multiple of 16.
COLUMNS = {
    **{1920: 0, 1280: 0, 3840: 0, 4096: 0},
    **{1912: 8, 1916: 20, 1438: 18, 3832: 8, 1366: 10, 961: 15},
}


def test_rows_and_columns_per_size() -> None:
    for height, rows in ROWS.items():
        for width, columns in COLUMNS.items():
            assert padding((height, width)) == (rows, columns), (height, width)
            black = 16 if columns else 0
            assert padded_size((height, width)) == (height + rows + 16, width + columns + black)
    # 1080p: 8 + 16 rows (1,104 for numz's 1,088); 720p and 4K: 16 + 16 (752, 2,192).
    assert padded_size((1080, 1920)) == (1104, 1920)
    assert padded_size((720, 1280)) == (752, 1280)
    assert padded_size((2160, 3840)) == (2192, 3840)


def test_fewest_rows_from_8() -> None:
    # The rule itself, every side up to 4K and beyond: the fewest rows, 8 at least, that reach a
    # multiple of 16; columns alike unless the width is a multiple of 16 already.
    for side in range(1, 4400):
        fewest = next(k for k in range(8, 40) if (side + k) % 16 == 0)
        assert 8 <= fewest <= 23
        assert padding((side, side)) == (fewest, fewest if side % 16 else 0), side


@pytest.mark.parametrize(
    ("height", "width"),
    [(1080, 1920), (720, 1280), (1060, 1912), (1081, 1438), (541, 961), (24, 32), (17, 17)],
)
def test_reflected_then_black(height: int, width: int) -> None:
    frames = torch.rand(2, 3, height, width, generator=torch.Generator().manual_seed(height))
    padded = model.pad(frames)
    rows, columns = padding((height, width))
    assert padded.shape == (2, 3, *padded_size((height, width)))
    assert torch.equal(padded[..., :height, :width], frames)
    # Reflected, the edge row not repeated: row height + k is row height - 2 - k; columns alike,
    # and the corner from both.
    below = torch.arange(height - 2, height - 2 - rows, -1)
    right = torch.arange(width - 2, width - 2 - columns, -1)
    assert torch.equal(padded[..., height : height + rows, :width], frames[..., below, :])
    assert torch.equal(padded[..., :height, width : width + columns], frames[..., right])
    corner = padded[..., height : height + rows, width : width + columns]
    assert torch.equal(corner, frames[..., below, :][..., right])
    # Then black: 16 rows of zeros, and 16 columns when any are reflected.
    assert padded.shape[-2] - height - rows == 16
    assert padded.shape[-1] - width - columns == (16 if columns else 0)
    assert bool((padded[..., height + rows :, :] == 0).all())
    assert bool((padded[..., width + columns :] == 0).all())


@pytest.mark.parametrize("target", [(1060, 1912), (720, 1280), (1080, 1920)])
def test_black_once_normalised(target: tuple[int, int]) -> None:
    # Through the whole transform: the picture and its reflection normalised alike, the black rows
    # and columns -1 exactly, as numz's zeros are.
    height, width = target
    frames = torch.rand(2, 3, height, width, generator=torch.Generator().manual_seed(width))
    # At the target's size already: torchvision's resize returns the frames as they are.
    out = model.input_transform(target)(frames)
    rows, columns = padding(target)
    assert out.shape == (3, 2, *padded_size(target))
    picture = Normalize(0.5, 0.5)(frames).permute(1, 0, 2, 3)
    assert torch.equal(out[..., :height, :width], picture)
    below = torch.arange(height - 2, height - 2 - rows, -1)
    assert torch.equal(out[..., height : height + rows, :width], picture[..., below, :])
    assert bool((out[..., height + rows :, :] == -1).all())
    assert bool((out[..., width + columns :] == -1).all())


def load(path: Path, name: str) -> ModuleType:
    """The Python file at path, loaded as module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def numerics_patch() -> ModuleType:
    """research/scripts/numerics_patch.py, loaded: at import it only reads its NUM_ switches,
    unset here, its patches being installed by its main alone. Skipped where it is absent: where
    research/ isn't beside seedvr2x/ (the package copied alone)."""
    if not NUMERICS_PATCH.is_file():
        pytest.skip(f"{NUMERICS_PATCH}: absent")
    with pytest.MonkeyPatch.context() as patch:
        for name in [name for name in os.environ if name.startswith("NUM_")]:
            patch.delenv(name)
        return load(NUMERICS_PATCH, "numerics_patch_under_test")


def test_pad_is_numerics_patch(numerics_patch: ModuleType) -> None:
    # Every height with every width, in float32 and in bfloat16, the encoder's dtype.
    pad = numerics_patch.Pad(DivisiblePad((16, 16)), "reflect>=8+black+16")
    base = torch.rand(1, 1, max(ROWS), max(COLUMNS), generator=torch.Generator().manual_seed(0))
    for dtype in (torch.float32, torch.bfloat16):
        whole = base.to(dtype)
        for height in ROWS:
            for width in COLUMNS:
                frames = whole[..., :height, :width]
                ours, theirs = model.pad(frames), pad(frames)
                assert ours.dtype == theirs.dtype == dtype
                assert torch.equal(ours, theirs), (height, width, dtype)


def chain(target: tuple[int, int], pad: Any) -> Compose:
    """numz's transform chain (src/core/generation_utils.py:72-84), NaResize's resize to an
    explicit size as model.input_transform makes it, with pad as its padding."""
    size = list(target)
    return Compose(
        [
            Lambda(lambda x: TVF.resize(x, size, InterpolationMode.BICUBIC, antialias=True)),
            Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
            pad,
            Normalize(0.5, 0.5),
            Lambda(lambda x: x.permute(1, 0, 2, 3)),
        ]
    )


# test_regression.py's padding cases: milestone 1's input (960x540) at 1080 and 720, and its
# crop to 956x530 at 1060, (source height, width), target.
CASES = [((540, 960), (1080, 1920)), ((540, 960), (720, 1280)), ((530, 956), (1060, 1912))]


@pytest.mark.parametrize(("source", "target"), CASES)
def test_transform_is_numerics_patch(
    numerics_patch: ModuleType, source: tuple[int, int], target: tuple[int, int]
) -> None:
    # numz's chain as numerics_patch.py patches it (ts[2] = Pad(ts[2], S.pad)): the same resize,
    # clamp and normalisation, its padding in the same place. In bfloat16, as the encoder's.
    frames = torch.rand(5, 3, *source, generator=torch.Generator().manual_seed(1)).bfloat16()
    patched = chain(target, numerics_patch.Pad(DivisiblePad((16, 16)), "reflect>=8+black+16"))
    assert torch.equal(model.input_transform(target)(frames), patched(frames))


@pytest.mark.parametrize(
    ("source", "resolution", "target", "padded"),
    [
        ((960, 540), 1080, (1080, 1920), (1104, 1920)),
        ((960, 540), 720, (720, 1280), (752, 1280)),
        ((956, 530), 1060, (1060, 1912), (1088, 1936)),
    ],
)
def test_regression_sizes(
    source: tuple[int, int],
    resolution: int,
    target: tuple[int, int],
    padded: tuple[int, int],
) -> None:
    # The sizes numz gave test_regression.py's references with NUM_PAD=reflect>=8+black+16, as
    # its log says them ("Padded: 1920x1104px → Output: 1920x1080px", and alike 1280x752 for
    # 1280x720, 1936x1088 for 1912x1060: src/core/generation_utils.py:274-279): seedvr2x's
    # --resolution gives the same targets from the same sources (width, height), and the padding
    # the same frames.
    assert target_size(*source, Fraction(1), resolution) == target == output_size(target)
    assert padded_size(target) == padded


@pytest.fixture(scope="module")
def numz_divisible_pad() -> Any:
    """numz's own DivisiblePad, from its file at 4490bd1 in the submodule's git objects, read as
    tools/vendor.py reads it; skipped where the submodule isn't."""
    vendor = load(PROJECT / "tools" / "vendor.py", "vendor_tool_under_test")
    path = "src/data/image/transforms/divisible_crop.py"
    source = vendor.git_show(vendor.NUMZ, vendor.NUMZ_COMMIT, path)
    if source is None:
        pytest.skip("numz submodule not checked out")
    namespace: dict[str, Any] = {}
    exec(compile(source, f"numz {vendor.NUMZ_COMMIT[:7]}:{path}", "exec"), namespace)
    return namespace["DivisiblePad"]


@pytest.mark.parametrize("target", [(1080, 1920), (720, 1280), (1060, 1912), (541, 961)])
def test_numz_mode_is_divisible_pad(target: tuple[int, int]) -> None:
    # The tests' numz padding (cli.NUMZ_PADDING) is numz's chain, its DivisiblePad: zeros up to
    # multiples of 16 (none at 720p), bit for bit.
    frames = torch.rand(5, 3, 540, 960, generator=torch.Generator().manual_seed(2)).bfloat16()
    ours = model.input_transform(target, numz_padding=True)(frames)
    assert ours.shape[-2:] == tuple(-(-side // 16) * 16 for side in target)
    assert torch.equal(ours, chain(target, DivisiblePad((16, 16)))(frames))


def test_vendored_divisible_pad_is_numz(numz_divisible_pad: Any) -> None:
    # The vendored DivisiblePad, which the numz padding runs, is numz's own, bit for bit, at
    # every size of the table.
    ours, theirs = DivisiblePad((16, 16)), numz_divisible_pad((16, 16))
    base = torch.rand(1, 1, max(ROWS), max(COLUMNS), generator=torch.Generator().manual_seed(3))
    for dtype in (torch.float32, torch.bfloat16):
        whole = base.to(dtype)
        for height in ROWS:
            for width in COLUMNS:
                frames = whole[..., :height, :width]
                padded = cast(torch.Tensor, ours(frames))  # a tensor in, a tensor out
                assert torch.equal(padded, theirs(frames)), (height, width, dtype)


@pytest.mark.parametrize("target", [(24, 32), (30, 37)])
def test_encoder_slices_padded_as_asked(
    monkeypatch: pytest.MonkeyPatch, target: tuple[int, int]
) -> None:
    # Each slice the encoder takes, from encoder_inputs and through encode_shot: job.padded_size
    # by default, numz's multiples of 16 in the tests' numz padding (cli.NUMZ_PADDING), the mode
    # passed down every call to the transform; run.py's calls are test_cli_run.py's
    # (test_numz_padding_for_tests_only). (24, 32) is padded to (48, 32), or (32, 32) in numz's
    # padding; (30, 37) to (64, 64), or (32, 48).
    from test_shot import SLICING, Reads

    count = 6
    frames = np.random.default_rng(target[1]).random((count, 15, 20, 3), dtype=np.float32)
    sizes = model.encode_slices(SLICING, padded_length(count))  # 5 frames, then 4
    encoded: list[tuple[int, int]] = []

    def encode_stream(
        models: model.Models, inputs: Iterable[torch.Tensor], length: int
    ) -> torch.Tensor:
        encoded.extend((x.shape[-2], x.shape[-1]) for x in inputs)
        return torch.zeros(1)

    monkeypatch.setattr(model, "encode_stream", encode_stream)
    numz = (-(-target[0] // 16) * 16, -(-target[1] // 16) * 16)
    assert padded_size(target) != numz
    for numz_padding, size in ((False, padded_size(target)), (True, numz)):
        given = encoder_inputs(SLICING, Reads(frames), count, target, numz_padding=numz_padding)
        assert [(x.shape[-2], x.shape[-1]) for x in given] == [size] * len(sizes)
        encoded.clear()
        encode_shot(SLICING, Reads(frames), count, target, 0, numz_padding=numz_padding)
        assert encoded == [size] * len(sizes)


@pytest.mark.parametrize(
    "target", [(1080, 1920), (720, 1280), (2160, 3840), (1060, 1912), (1081, 1913), (17, 16)]
)
def test_decode_trimmed(monkeypatch: pytest.MonkeyPatch, target: tuple[int, int]) -> None:
    # The decode's frames at the padded size, NaN in the padding, which the crop must drop whole
    # (a NaN written stops the run: shot.finite) and no more: the output's size, the target's even
    # sides as before, and the picture as decoded.
    padded = padded_size(target)
    decoded = torch.full((3, 2, *padded), math.nan, dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(target[0])
    picture = torch.rand(3, 2, *target, generator=generator) * 2.4 - 1.2  # clamped beyond 1
    decoded[..., : target[0], : target[1]] = picture.bfloat16()
    height, width = output_size(target)
    kept = decoded[..., :height, :width].permute(1, 2, 3, 0)
    expected = kept.clamp(-1, 1).mul(0.5).add(0.5).float().numpy()
    # Two slices of one frame, as the VAE's slices come.
    slices = [decoded[:, :1], decoded[:, 1:]]
    monkeypatch.setattr(model, "decode_stream", lambda models, latent: iter(slices))
    written: list[npt.NDArray[np.float32]] = []
    models = cast(model.Models, SimpleNamespace(device=torch.device("cpu")))
    decode_shot(models, torch.zeros(1), 2, target, written.append)
    frames = np.concatenate(written)
    assert frames.shape == (2, height, width, 3)
    assert np.array_equal(frames, expected)


@pytest.mark.parametrize("target", [(24, 32), (17, 16), (30, 37), (541, 961)])
def test_reference_is_the_picture(target: tuple[int, int]) -> None:
    # lab's reference, cropped as the decode's frames are (_frames): the picture alone, whatever
    # padding the transform made, numz's included: the frames resized, clamped and normalised.
    from test_shot import SLICING, Reads

    count = 6
    rng = np.random.default_rng(target[1])
    frames = rng.random((count, 15, 20, 3), dtype=np.float32)
    height, width = output_size(target)
    take = _frames(reference_inputs(SLICING, Reads(frames), count, target), height, width)
    reference = take(count)
    video = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)
    size = list(target)
    resized = TVF.resize(video, size, InterpolationMode.BICUBIC, antialias=True).clamp(0.0, 1.0)
    picture = Normalize(0.5, 0.5)(resized)[..., :height, :width]
    assert reference.shape == (count, 3, height, width)
    assert torch.equal(reference, picture)
    for numz_padding in (False, True):
        out = model.input_transform(target, numz_padding=numz_padding)(video)
        assert torch.equal(out.permute(1, 0, 2, 3)[..., :height, :width], picture)


def test_too_small_to_pad_refused() -> None:
    # The target check refuses exactly what torch can't reflect: 17 rows and 16 columns at least.
    for height in range(1, 41):
        for width in range(1, 41):
            try:
                model.pad(torch.zeros(1, 1, height, width))
            except RuntimeError:
                reflected = False
            else:
                reflected = True
            assert reflected == (height >= 17 and width >= 16), (height, width)
            if reflected:
                check_target((height, width), 7)
            else:
                with pytest.raises(JobError, match="17 rows and 16 columns at least"):
                    check_target((height, width), 7)
    with pytest.raises(JobError) as refused:
        check_target((16, 21), 16)
    assert str(refused.value) == (
        "--resolution 16: the frames resized to 21x16 are too small for the padding, at least 8"
        " rows reflected from the picture under it, up to a multiple of 16, and columns alike: it"
        " takes 17 rows and 16 columns at least"
    )
