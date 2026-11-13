"""The library's callbacks that tell a caller how far a pass has got (the ComfyUI nodes' bar, their
text and their Cancel): each tile of a tiled encode and decode (tiling.TileStep, through
shot_pass, clip.upscale's `tile` and still.encoded and decoded), the pieces of a model file read
(blockswap.read's `progress`, model.load_models'), the bytes of a download (pull.fetch's
`progress`, through the progress bar huggingface_hub is given) and the batches of the shot
detector's scan (detector.probabilities' `progress`).

What is held is that they only observe: the bits of every output are those of the same call
without them (the CLI's tests/test_clip_cli.py and test_still_cli.py hold the pass itself to the
shot's), they are told in order and complete, and what one raises stops the pass and leaves
nothing alive."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch
from huggingface_hub import constants, file_download
from stand_ins import cpu_models  # noqa: F401
from test_clip import LONG, clip_of
from test_held import strays
from test_pull import CONTENT, DIT, Hub, faked
from test_still import SEED, TARGET, TILES, pipeline_models, pixels_of, stand_in_dit

from seedvr2x import blockswap, clip, detector, model, pull, still, tiling
from seedvr2x.pull import Pin

Told = list[tuple[Any, ...]]


@pytest.fixture
def models(cpu_models: model.Models) -> model.Models:  # noqa: F811
    made = pipeline_models(cpu_models, [])
    made.runner.dit = torch.nn.ModuleList()  # pyright: ignore[reportAttributeAccessIssue]
    made.text_pos = made.text_neg = torch.zeros(2)  # pyright: ignore[reportAttributeAccessIssue]
    return made


class Stop(BaseException):
    """What a node's Cancel raises."""


# ---- The tiles ----------------------------------------------------------------------------------


def test_a_tiled_picture_tells_each_tile_and_the_bits_are_the_same(
    models: model.Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_in_dit(monkeypatch)
    pixels = pixels_of()
    told: Told = []
    encoded = still.encoded(
        models, pixels, TARGET, SEED, tiles=TILES, tile=lambda *each: told.append(("encode", *each))
    )
    sampled = still.sampled(models, encoded, SEED)
    out = still.decoded(
        models,
        sampled,
        pixels,
        TARGET,
        tiles=TILES,
        correct=True,
        tile=lambda *each: told.append(("decode", *each)),
    )
    # TILES cut the 10 x 10 latents of the stand-in picture in 4: 0 before the first, then each.
    expected = [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]
    assert told == [("encode", *each) for each in expected] + [
        ("decode", *each) for each in expected
    ]
    plain = still.upscale(
        models, pixels, TARGET, SEED, encode_tiles=TILES, decode_tiles=TILES, correct=True
    )
    assert out.tobytes() == plain.tobytes()


def test_an_untiled_pass_tells_no_tile(
    models: model.Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_in_dit(monkeypatch)
    told: Told = []
    pixels = pixels_of()
    latent = still.encoded(models, pixels, TARGET, SEED, tiles=None, tile=lambda *a: told.append(a))
    still.decoded(
        models,
        still.sampled(models, latent, SEED),
        pixels,
        TARGET,
        tiles=None,
        correct=False,
        tile=lambda *a: told.append(a),
    )
    assert told == []


def test_a_tiled_decode_tells_every_tile_before_its_first_slice(
    models: model.Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_in_dit(monkeypatch)
    latent = still.sampled(
        models, still.encoded(models, pixels_of(), TARGET, SEED, tiles=None), SEED
    )
    told: list[tuple[int, int]] = []
    slices = tiling.decode_tiled(
        models, latent, TILES, tiling.InMemory(), lambda done, total: told.append((done, total))
    )
    assert told == []
    next(slices)
    assert told == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]
    slices.close()


def test_a_clip_tells_its_tiles_apart_from_its_steps_and_the_bits_are_the_same(
    models: model.Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_in_dit(monkeypatch)
    frames = clip_of(LONG)
    told: Told = []
    steps: Told = []
    kwargs: dict[str, Any] = {
        "window": 5,
        "encode_tiles": TILES,
        "decode_tiles": TILES,
        "correct": True,
    }
    out = clip.upscale(
        models,
        frames,
        TARGET,
        SEED,
        step=lambda *each: steps.append(each),
        tile=lambda *each: told.append(each),
        **kwargs,
    )
    expected = [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]
    assert told == [("encode", *each) for each in expected] + [
        ("decode", *each) for each in expected
    ]
    # The steps are the same with the tiles told: no tile among them.
    assert {each[0] for each in steps} == {"encode", "window", "decode"}
    plain = clip.upscale(models, frames, TARGET, SEED, **kwargs)
    assert out.tobytes() == plain.tobytes()


def test_every_run_of_a_clip_tells_its_own_tiles_after_its_run_step(
    models: model.Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_in_dit(monkeypatch)
    events: Told = []
    out = clip.upscale(
        models,
        clip_of(LONG),
        TARGET,
        SEED,
        window=5,
        encode_tiles=TILES,
        decode_tiles=TILES,
        correct=False,
        quality="enhanced",
        step=lambda *each: events.append(("step", *each)),
        tile=lambda *each: events.append(("tile", *each)),
    )
    runs = [index for index, each in enumerate(events) if each[:2] == ("step", "run")]
    assert [events[index] for index in runs] == [("step", "run", 1, 2), ("step", "run", 2, 2)]
    for first, last in zip(runs, [*runs[1:], len(events)], strict=True):
        tiles = [each[1:] for each in events[first:last] if each[0] == "tile"]
        assert tiles == [("encode", n, 4) for n in range(5)] + [("decode", n, 4) for n in range(5)]
    plain = clip.upscale(
        models,
        clip_of(LONG),
        TARGET,
        SEED,
        window=5,
        encode_tiles=TILES,
        decode_tiles=TILES,
        correct=False,
        quality="enhanced",
    )
    assert out.tobytes() == plain.tobytes()


@pytest.mark.parametrize("pass_", ["encode", "decode"])
def test_a_tile_that_raises_stops_the_pass_and_leaves_nothing(
    models: model.Models, monkeypatch: pytest.MonkeyPatch, pass_: str
) -> None:
    stand_in_dit(monkeypatch)

    def tile(each: str, done: int, total: int) -> None:
        if each == pass_ and done == 2:
            raise Stop()

    with pytest.raises(Stop):
        clip.upscale(
            models,
            clip_of(LONG),
            TARGET,
            SEED,
            window=5,
            encode_tiles=TILES,
            decode_tiles=TILES,
            correct=False,
            tile=tile,
        )
    assert strays(models) == []
    clean = clip.upscale(
        models,
        clip_of(LONG),
        TARGET,
        SEED,
        window=5,
        encode_tiles=None,
        decode_tiles=None,
        correct=False,
    )
    assert clean.shape[0] == LONG


# ---- The pieces of a file read ------------------------------------------------------------------


def write_checkpoint(path: Path, count: int = 3, size: int = 40) -> None:
    """A safetensors file of `count` float32 tensors of `size` values."""
    header: dict[str, Any] = {}
    data = b""
    for index in range(count):
        values = np.full(size, index + 1, dtype=np.float32).tobytes()
        header[f"layer{index}.weight"] = {
            "dtype": "F32",
            "shape": [size],
            "data_offsets": [len(data), len(data) + len(values)],
        }
        data += values
    text = json.dumps(header).encode()
    path.write_bytes(len(text).to_bytes(8, "little") + text + data)


class Layers(torch.nn.Module):
    def __init__(self, count: int = 3, size: int = 40) -> None:
        super().__init__()
        for index in range(count):
            setattr(self, f"layer{index}", torch.nn.Linear(size, 1, bias=False))


@pytest.mark.parametrize("hashing", [True, False])
def test_a_read_tells_the_bytes_read_so_far_in_order_up_to_the_files_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hashing: bool
) -> None:
    path = tmp_path / "model.safetensors"
    write_checkpoint(path)
    # Pieces of 100 bytes, so that a tensor's 160 bytes are read in two.
    monkeypatch.setattr(blockswap, "PIECE", 100)
    told: list[tuple[int, int]] = []
    seen: list[str] = []
    state, _ = blockswap.read(
        Layers(),
        path,
        torch.device("cpu"),
        None,
        None,
        hashed=(lambda digest, stat: seen.append(digest)) if hashing else None,
        progress=lambda done, total: told.append((done, total)),
    )
    size = path.stat().st_size
    assert told and told[-1] == (size, size)
    assert [each[0] for each in told] == sorted(each[0] for each in told)
    assert {each[1] for each in told} == {size}
    if hashing:
        # Each tensor in two pieces of 100 and 60 bytes, the header and the rest in pieces too.
        assert len(told) > 3
        assert seen == [hashlib.sha256(path.read_bytes()).hexdigest()]
    # Observing only: the tensors are the file's.
    plain, _ = blockswap.read(Layers(), path, torch.device("cpu"), None, None)
    assert all(torch.equal(state[key], plain[key]) for key in plain)


def test_a_read_that_raises_in_its_progress_stops_there(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    write_checkpoint(path)
    calls = 0

    def progress(done: int, total: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise Stop()

    with pytest.raises(Stop):
        blockswap.read(Layers(), path, torch.device("cpu"), None, None, progress=progress)
    assert calls == 2


# ---- The bytes of a download --------------------------------------------------------------------


class Plain:
    """huggingface_hub 1.33.0 and 2.4.0's plain download of a file, as they drive a progress bar
    class (file_download.py:410-415, 447-490): one bar made with the keywords tqdm takes, updated
    by each chunk, taken back by the size resumed on a retry, closed by its context manager."""

    def __init__(self, hub: Hub, sizes: list[int], resumed: int = 0) -> None:
        self.hub, self.sizes, self.resumed = hub, sizes, resumed

    def __call__(self, repo_id: str, filename: str, **options: Any) -> str:
        bars = options["tqdm_class"]
        total = sum(self.sizes) + self.resumed
        with bars(
            desc=filename, total=total, initial=self.resumed, unit="B", unit_scale=True
        ) as bar:
            for size in self.sizes:
                bar.update(size)
            bar.update(-self.resumed)
            bar.update(self.resumed)
        return self.hub.hf_hub_download(repo_id, filename, **options)


class Xet:
    """The Xet download's two bars (utils/_xet_progress_reporting.py:98-130), each made with the
    keywords the reporter gives (position, bar_format, leave besides tqdm's own), the bytes
    received leading the bytes written."""

    def __init__(self, hub: Hub, total: int) -> None:
        self.hub, self.total = hub, total

    def __call__(self, repo_id: str, filename: str, **options: Any) -> str:
        bars = options["tqdm_class"]
        keywords = {"unit": "B", "unit_scale": True, "total": self.total, "leave": True}
        written = bars(desc="reconstruction", position=1, bar_format="{n}", **keywords)
        received = bars(desc="Downloading bytes", position=0, bar_format="{n}", **keywords)
        # No update_transfer on the bar: the library would send both counts to one bar.
        assert not hasattr(written, "update_transfer")
        step = self.total // 4
        for _ in range(4):
            received.update(step)
            written.set_postfix_str("fast", refresh=False)
            written.update(step - 1)
        written.update(self.total - 4 * (step - 1))
        received.update(self.total - 4 * step)
        received.refresh()
        received.close()
        written.close()
        return self.hub.hf_hub_download(repo_id, filename, **options)


@pytest.fixture
def small(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Hub, str, int]:
    hub = faked(monkeypatch, tmp_path / "hf" / "hub")
    monkeypatch.setattr(
        pull,
        "PINNED",
        {
            name: Pin(len(data), hashlib.sha256(data).hexdigest(), "dit" if name == DIT else "vae")
            for name, data in CONTENT.items()
        },
    )
    return hub, DIT, len(CONTENT[DIT])


def collected(
    monkeypatch: pytest.MonkeyPatch, download: Callable[..., str], name: str
) -> list[tuple[str, int, int]]:
    told: list[tuple[str, int, int]] = []
    monkeypatch.setattr(file_download, "hf_hub_download", download)
    pull.fetch({name: pull.PINNED[name]}, progress=lambda *each: told.append(each))
    return told


def check_told(told: list[tuple[str, int, int]], name: str, total: int) -> None:
    assert told[0] == (name, 0, total), "told 0 before the first byte"
    done = [each[1] for each in told]
    assert done == sorted(done) and len(set(done)) == len(done), "in order, none twice"
    assert told[-1] == (name, total, total) and {each[2] for each in told} == {total}


def test_a_plain_download_tells_its_bytes_in_order_to_the_files_size(
    small: tuple[Hub, str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub, name, total = small
    told = collected(
        monkeypatch, Plain(hub, [total // 3, total // 3, total - 2 * (total // 3)]), name
    )
    check_told(told, name, total)
    assert len(told) == 4


def test_a_resumed_download_never_goes_back(
    small: tuple[Hub, str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub, name, total = small
    resumed = total // 2
    told = collected(monkeypatch, Plain(hub, [total - resumed], resumed), name)
    check_told(told, name, total)
    assert told[1][1] == resumed, "a bar made with the bytes already there starts from them"


def test_a_xet_download_tells_the_most_either_bar_has_counted(
    small: tuple[Hub, str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub, name, total = small
    told = collected(monkeypatch, Xet(hub, total), name)
    check_told(told, name, total)


def test_a_download_with_no_progress_is_called_as_before(small: tuple[Hub, str, int]) -> None:
    hub, name, _ = small
    pull.fetch({name: pull.PINNED[name]})
    assert "tqdm_class" not in hub.calls[-1]


@pytest.mark.usefixtures("small")
def test_a_huggingface_hub_without_progress_bars_downloads_as_before_and_says_so(
    small: tuple[Hub, str, int],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # huggingface_hub before 1.1 (0.34 to 0.36, the line ComfyUI's transformers 4.x takes, and
    # 1.0): no tqdm_class, no **options.
    hub, name, _ = small

    def old(repo_id: str, filename: str, *, revision: str, cache_dir: Any, token: Any) -> str:
        return hub.hf_hub_download(
            repo_id, filename, revision=revision, cache_dir=cache_dir, token=token
        )

    monkeypatch.setattr(file_download, "hf_hub_download", old)
    told: list[tuple[str, int, int]] = []
    with caplog.at_level("WARNING", logger="seedvr2x.pull"):
        found = pull.fetch({name: pull.PINNED[name]}, progress=lambda *each: told.append(each))
    assert found[name].is_file() and told == [(name, 0, pull.PINNED[name].size)]
    assert any("gives no way to follow a download" in each for each in caplog.messages)


@pytest.mark.usefixtures("small")
def test_the_progress_is_told_for_every_file_to_download_before_the_first_begins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    told: list[tuple[str, int, int]] = []
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    pull.fetch(dict(pull.PINNED), progress=lambda *each: told.append(each))
    assert [each for each in told if each[1] == 0] == [
        (name, 0, pin.size) for name, pin in pull.PINNED.items()
    ]
    assert told[0][1] == told[1][1] == 0


def test_a_progress_that_raises_stops_a_plain_download(
    small: tuple[Hub, str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub, name, total = small
    monkeypatch.setattr(
        file_download, "hf_hub_download", Plain(hub, [total // 2, total - total // 2])
    )

    def progress(name: str, done: int, total: int) -> None:
        if done:
            raise Stop()

    with pytest.raises(Stop):
        pull.fetch({name: pull.PINNED[name]}, progress=progress)


# ---- The shot detector's batches ----------------------------------------------------------------


def test_the_detector_tells_each_batch_and_gives_the_same_probabilities(tmp_path: Path) -> None:
    from test_detector import untrained_weights

    untrained_weights(tmp_path)
    path = tmp_path / pull.DETECTOR
    images = torch.from_numpy(np.random.default_rng(3).random((450, 24, 40, 3), dtype=np.float32))
    told: list[tuple[int, int]] = []
    with detector.Detector(path, torch.device("cpu")) as found:
        probabilities: npt.NDArray[np.float32] = detector.probabilities(
            found, images, lambda done, total: told.append((done, total))
        )
    batch = 4 * detector.STEP
    assert told == [(min(first + batch, 450), 450) for first in range(0, 450, batch)]
    with detector.Detector(path, torch.device("cpu")) as plain:
        assert detector.probabilities(plain, images).tobytes() == probabilities.tobytes()
