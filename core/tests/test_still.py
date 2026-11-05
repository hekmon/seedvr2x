"""One picture through the models (seedvr2x/still.py) on the CPU: the VAE of cpu_models (the real
configs, random weights) in bfloat16, as the pipeline runs it, and a stand-in DiT, so that what is
held is still.py's own steps: the phases entered in order and nothing loaded, the encode and the
noise seeded as a shot's first frame is, one latent sampled, the tiled passes' scratch in system
memory, split's reference the picture itself, a NaN in the decode refused. That the output is the
shot's pass's (seedvr2x/shot_pass.py), bit for bit, is the CLI's tests/test_still_cli.py's."""

import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch
from stand_ins import cpu_models  # noqa: F401
from torch import Tensor

from seedvr2x import colour, model, sampling, still, tiling
from seedvr2x.frames import output_size, padded_size
from seedvr2x.passes import Tiles
from seedvr2x.tilegrid import latent_size

# A picture of 24 x 40 resized to 48 x 80, padded to 80 x 80 (8 rows reflected, 16 black, and
# the columns the same way): 10 x 10 latents, which tiles of 64 px sharing 16 cut in 4.
PICTURE, TARGET = (24, 40), (48, 80)
TILES = Tiles(64, 16)
SEED = 5

Events = list[Any]


def pixels_of(seed: int = 0) -> npt.NDArray[np.float32]:
    """A picture as stored, (24, 40, 3) float32 in [0, 1]."""
    return np.random.default_rng(seed).random((*PICTURE, 3), dtype=np.float32)


def pipeline_models(vae_models: model.Models, events: Events | None = None) -> model.Models:
    """cpu_models' VAE in bfloat16, the pipeline's dtype (its encode and decode take
    model.COMPUTE_DTYPE), in models of their own, so that the module's float32 ones are left
    alone; enter records the phase entered in events. Neither it nor the DiT stand-in loads
    anything."""
    runner: Any = cast(Any, vae_models).runner
    vae = copy.deepcopy(runner.vae).to(torch.bfloat16)
    entered: Events = [] if events is None else events
    return cast(
        model.Models,
        SimpleNamespace(
            runner=SimpleNamespace(vae=vae, config=runner.config),
            device=torch.device("cpu"),
            enter=lambda phase: entered.append(("enter", phase)),
        ),
    )


def stand_in_dit(monkeypatch: pytest.MonkeyPatch, events: Events | None = None) -> list[Tensor]:
    """Put a DiT of arithmetic in the place of model.condition and model.sample, the two calls
    sampling.window makes of the runner: the condition (T', h, w, 17) holds the latent and a
    channel of the noise, the output (T', h, w, 16) mixes the latent with the noise. Returns the
    noise of each call."""
    noises: list[Tensor] = []

    def condition(models: model.Models, noise: Tensor, latent: Tensor) -> Tensor:
        return torch.cat([latent, noise[..., :1]], dim=-1)

    def sample(models: model.Models, noise: Tensor, cond: Tensor) -> Tensor:
        noises.append(noise)
        if events is not None:
            events.append("sample")
        return cond[..., :16] * 0.7 + noise * 0.3

    monkeypatch.setattr(model, "condition", condition)
    monkeypatch.setattr(model, "sample", sample)
    return noises


@pytest.fixture
def events() -> Events:
    return []


@pytest.fixture
def models(cpu_models: model.Models, events: Events) -> model.Models:  # noqa: F811
    return pipeline_models(cpu_models, events)


def upscale(
    models: model.Models,
    pixels: npt.NDArray[np.float32] | None = None,
    seed: int = SEED,
    *,
    encode_tiles: Tiles | None = None,
    decode_tiles: Tiles | None = None,
    correct: bool = False,
) -> npt.NDArray[np.float32]:
    return still.upscale(
        models,
        pixels_of() if pixels is None else pixels,
        TARGET,
        seed,
        encode_tiles=encode_tiles,
        decode_tiles=decode_tiles,
        correct=correct,
    )


def test_the_output_is_the_targets_picture(
    models: model.Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_in_dit(monkeypatch)
    out = upscale(models)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert out.shape == (*output_size(TARGET), 3) == (48, 80, 3)
    assert 0.0 <= out.min() and out.max() <= 1.0 and np.isfinite(out).all()
    # The decode is the picture's pixels' own, not a constant.
    assert out.std() > 0.01
    # An odd target is cropped to even sides, the padding dropped.
    odd = still.upscale(
        models,
        pixels_of(),
        (47, 79),
        SEED,
        encode_tiles=None,
        decode_tiles=None,
        correct=False,
    )
    assert odd.shape == (*output_size((47, 79)), 3) == (46, 78, 3)


def test_one_latent_sampled_from_the_seeds_of_a_shots_first_frame(
    models: model.Models, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    noises = stand_in_dit(monkeypatch, events)
    seeded: list[int] = []
    drawn: list[tuple[int, tuple[int, ...]]] = []
    set_seed, draw = model.set_seed, sampling.drawn

    def spy_seed(seed: int) -> None:
        seeded.append(seed)
        set_seed(seed)

    def spy_drawn(models: model.Models, latent: Tensor, seed: int) -> tuple[Tensor, Tensor]:
        drawn.append((seed, tuple(latent.shape)))
        return draw(models, latent, seed)

    monkeypatch.setattr(model, "set_seed", spy_seed)
    monkeypatch.setattr(sampling, "drawn", spy_drawn)
    first = upscale(models)
    padded = padded_size(TARGET)
    # The encode seeded with the seed plus the offset, nothing drawing; the noise drawn at the
    # seed (sampling.drawn seeds again, as numz's draw is preceded by it).
    assert seeded == [SEED + sampling.ENCODE_SEED_OFFSET, SEED]
    assert drawn == [(SEED, (1, padded[0] // 8, padded[1] // 8, 16))]
    # The DiT ran once, on a window of one latent, the noise of the draw.
    assert [tuple(noise.shape) for noise in noises] == [(1, 10, 10, 16)]
    # The CPU generator seeded at the seed, filling the latent's memory order (the encode's
    # channel-major layout), in bfloat16.
    like = torch.zeros(16, 1, 10, 10, dtype=model.COMPUTE_DTYPE).permute(1, 2, 3, 0)
    generator = torch.Generator().manual_seed(SEED)
    expected = torch.randn_like(like, generator=generator)
    assert noises[0].dtype == model.COMPUTE_DTYPE and torch.equal(noises[0], expected)
    # Deterministic, and the seed moves it.
    assert upscale(models).tobytes() == first.tobytes()
    assert upscale(models, seed=SEED + 1).tobytes() != first.tobytes()


def test_the_models_are_never_loaded_and_the_phases_are_entered_in_order(
    models: model.Models, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    stand_in_dit(monkeypatch, events)

    def loading(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("still.upscale loaded models")

    monkeypatch.setattr(model, "load_models", loading)
    upscale(models, correct=True, encode_tiles=TILES, decode_tiles=TILES)
    assert events == [("enter", "encode"), ("enter", "dit"), "sample", ("enter", "decode")]
    # In a loop, as a server calls it: the same phases each time.
    events.clear()
    upscale(models)
    assert events == [("enter", "encode"), ("enter", "dit"), "sample", ("enter", "decode")]


def test_tiled_passes_keep_their_scratch_in_memory(
    models: model.Models,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stand_in_dit(monkeypatch)
    padded = latent_size(padded_size(TARGET))
    assert len(tiling.grid(padded, TILES)) == 4
    stores: list[tiling.InMemory] = []

    class Counted(tiling.InMemory):
        def __init__(self) -> None:
            super().__init__()
            stores.append(self)

    monkeypatch.setattr(tiling, "InMemory", Counted)
    # Nothing on a disk: any file made while it runs is made under tmp_path, which stays empty.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    whole = upscale(models)
    assert stores == []
    encoded = upscale(models, encode_tiles=TILES)
    decoded = upscale(models, decode_tiles=TILES)
    both = upscale(models, encode_tiles=TILES, decode_tiles=TILES)
    assert len(stores) == 1 + 1 + 2
    for out in (encoded, decoded, both):
        assert out.shape == whole.shape and out.dtype == np.float32
        assert 0.0 <= out.min() and out.max() <= 1.0
        assert out.tobytes() != whole.tobytes()
    assert both.tobytes() != encoded.tobytes() and both.tobytes() != decoded.tobytes()
    assert upscale(models, encode_tiles=TILES, decode_tiles=TILES).tobytes() == both.tobytes()
    assert list(tmp_path.iterdir()) == []
    # A tile that covers the whole padded frame is no tiling: the untiled pass, its bits.
    assert upscale(models, encode_tiles=Tiles(1024), decode_tiles=Tiles(1024)).tobytes() == (
        whole.tobytes()
    )


def test_the_correction_is_split_against_the_picture(
    models: model.Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_in_dit(monkeypatch)
    pixels = pixels_of(3)
    seen: list[tuple[Tensor, Tensor, int]] = []
    split = colour.split

    def spy(content: Tensor, reference: Tensor, stages: int) -> Tensor:
        seen.append((content, reference, stages))
        return split(content, reference, stages)

    monkeypatch.setattr(colour, "split", spy)
    plain = upscale(models, pixels)
    assert seen == []
    corrected = upscale(models, pixels, correct=True)
    assert len(seen) == 1
    content, reference, stages = seen[0]
    height, width = output_size(TARGET)
    # The picture through the encoder's transform in float32, cropped as the decode is.
    frames = torch.as_tensor(pixels).unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32)
    expected = model.input_transform(TARGET)(frames).permute(1, 0, 2, 3)[:, :, :height, :width]
    assert reference.dtype == torch.float32 and reference.shape == (1, 3, height, width)
    assert torch.equal(reference, expected)
    assert content.shape == reference.shape
    assert stages == colour.colour_stages(PICTURE, TARGET)
    assert corrected.shape == plain.shape and corrected.dtype == np.float32
    assert 0.0 <= corrected.min() and corrected.max() <= 1.0
    assert corrected.tobytes() != plain.tobytes()


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_decode_holding_nan_or_inf_is_refused(
    models: model.Models, monkeypatch: pytest.MonkeyPatch, value: float
) -> None:
    stand_in_dit(monkeypatch)
    decode_stream = model.decode_stream

    def poisoned(models: model.Models, latent: Tensor) -> Any:
        for chunk in decode_stream(models, latent):
            chunk[0, 0, 3, 5] = value
            yield chunk

    monkeypatch.setattr(model, "decode_stream", poisoned)
    for correct in (False, True):
        with pytest.raises(still.NonFinite, match="decode holds NaN or inf") as caught:
            upscale(models, correct=correct)
        assert isinstance(caught.value, RuntimeError)


def test_pixels_that_are_not_a_float32_picture_are_refused(models: model.Models) -> None:
    bad: list[Any] = [pixels_of().astype(np.float64), pixels_of()[:, :, :1], pixels_of()[None]]
    for pixels in bad:
        with pytest.raises(ValueError, match=r"not \(H, W, 3\) float32"):
            upscale(models, pixels)
