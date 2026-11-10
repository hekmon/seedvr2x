"""seedvr2x Models as the user sees it (the user, 2026-11-08): the model folder an optional input
(stripped of quotes and spaces, an empty value the cache, a value that is no folder refused), the
downloads (the shot detector's file with the DiT's and the VAE's) on the node's own bar with its
lines, nothing downloaded for the DiT and the VAE from a folder and a file missing there refused at
this node, the detector's file the folder's alone with a folder (none there: nothing said, seedvr2x
Cuts refuses it) or the cache's, the files' places said in ComfyUI's console, and the upscale node
downloading again a file the cache has lost since."""

import logging
import sys
import threading
from pathlib import Path
from typing import Any

import comfy.model_management as management
import numpy as np
import pytest
import server
import torch
from comfy.utils import ProgressBar
from comfyui import console, models, nodes, plans, progress, words
from standins import PICTURE, RESOLUTION, StandIn, execute, upscale_image, where

from seedvr2x import pull
from seedvr2x.errors import ModelUnreachable
from seedvr2x.pull import ModelFile, ModelFiles

INT8 = "seedvr2x_ema_7b_sharp_int8_convrot.safetensors"
INT8_LABEL = plans.DIT_LABELS[INT8]
VAE = plans.VAE_FILE
CACHE = Path("/the/cache")
DETECTOR = pull.DETECTOR
SIZES = (pull.PINNED[INT8].size, pull.PINNED[VAE].size, pull.PINNED[DETECTOR].size)


def unprefixed(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The console's lines without the "[seedvr2x] " every one of seedvr2x's begins with
    (console.py, held by tests/test_console.py): the words are what these tests hold."""
    return [line.removeprefix(console.PREFIX) for line in caplog.messages]


def node_texts(node: str) -> list[str]:
    instance = server.PromptServer.instance
    assert instance is not None
    return [text for text, each in instance.texts if each == node]


def run_models(
    keep: str = "on", model_dir: str | None = None, unique_id: str = "4"
) -> models.Handle:
    inputs: dict[str, Any] = {"model": INT8_LABEL, "keep_in_ram_between_runs": keep}
    if model_dir is not None:
        inputs["model_dir"] = model_dir
    (found,) = execute(nodes.Models, unique_id, **inputs).args
    return found


class Cache:
    """Hugging Face's cache as the tests fake it: which of the two files it has, the downloads
    told to the node's callback (each file's size first, as pull.fetch tells them), and what
    resolve was asked."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, has: tuple[str, ...] = ()) -> None:
        self.has = set(has)
        self.asked: list[tuple[Path | None, str, str, bool]] = []
        self.steps: list[int] = []
        monkeypatch.setattr(pull, "cache", lambda: CACHE)
        monkeypatch.setattr(pull, "cached", lambda name: CACHE / name if name in self.has else None)
        monkeypatch.setattr(pull, "resolve", self.resolve)
        monkeypatch.setattr(pull, "fetch", self.fetch)
        self.fetched: list[tuple[str, ...]] = []

    def resolve(
        self, model_dir: Path | None, dit: str, vae: str, detector: bool = False, **options: Any
    ) -> ModelFiles:
        self.asked.append((model_dir, dit, vae, detector))
        tell = options["progress"]
        wanted = (dit, vae, DETECTOR) if detector else (dit, vae)
        missing = [name for name in wanted if name not in self.has]
        for name in missing:
            tell(name, 0, pull.PINNED[name].size)
        for step in self.steps:
            # The first file's bytes, then the second's.
            for name in missing:
                tell(name, min(step, pull.PINNED[name].size), pull.PINNED[name].size)
        for name in missing:
            self.has.add(name)
        return ModelFiles(
            ModelFile(dit, CACHE / dit), ModelFile(vae, CACHE / vae), CACHE, pull.Hashes()
        )

    def fetch(self, pins: Any, download: Any = None, progress: Any = None) -> dict[str, Path]:
        """pull.fetch: what the folder's Models node asks for the detector alone."""
        self.fetched.append(tuple(pins))
        for name in pins:
            if name not in self.has:
                progress(name, 0, pull.PINNED[name].size)
                progress(name, pull.PINNED[name].size, pull.PINNED[name].size)
                self.has.add(name)
        return {name: CACHE / name for name in pins}


def test_the_missing_files_are_downloaded_on_a_bar_of_the_nodes_with_their_time_under_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = Cache(monkeypatch)
    cache.steps = [3_100_000_000, 20_000_000_000]
    found = run_models()
    assert found == models.Handle(INT8, "", "on")
    # The DiT's and the VAE's in one resolve, then the shot detector's on its own (its failure
    # holds no other node up: test_a_detector_that_cant_be_had_is_said_and_the_handle_given).
    assert cache.asked == [(None, INT8, VAE, False)] and cache.fetched == [(DETECTOR,)]
    # One bar, in bytes, of the three downloads together, made at the first byte, ending full.
    (bar,) = ProgressBar.created
    total = sum(SIZES)
    assert set(bar.totals) == {total} and bar.updates[-1] == total
    assert bar.updates == sorted(bar.updates), "never back"
    texts = node_texts("4")
    # Two lines while it downloads, no name (the node's title has it): the bytes of all the
    # downloads together in the total's unit, over the time (tests/test_lines.py holds its
    # forms); one line at the end.
    assert all(text.count("\n") == 1 for text in texts[:-1]), texts
    assert all("seedvr2x" not in text for text in texts)
    # The total moved by the detector's 30.5 MB: 8.8 GB of the two files, 8.9 with it.
    assert total - sum(SIZES[:2]) == 30_482_632 and f"{total / 1e9:.1f}" == "8.9"
    assert (
        texts[0] == f"downloading 0.0/{total / 1e9:.1f} GB\n0 s elapsed \u00b7 estimating time left"
    )
    assert f"downloading 3.1/{total / 1e9:.1f} GB" in [where(text) for text in texts[:-1]]
    assert texts[-1] == "Sharp 7B int8 · kept in RAM"


def test_the_end_line_says_whether_the_model_is_kept_by_the_keeping_it_comes_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Cache(monkeypatch, has=(INT8, VAE, DETECTOR))
    run_models(keep="on", unique_id="1")
    run_models(keep="off", unique_id="2")
    assert node_texts("1") == ["Sharp 7B int8 · kept in RAM"]
    assert node_texts("2") == ["Sharp 7B int8 · not kept in RAM"]
    assert not ProgressBar.created, "nothing to download, no bar"
    # The longest the line gets, for the longest label, stays within the node's width.
    longest = max(len(f"{plans.short_label(name)} · not kept in RAM") for name in plans.DIT_LABELS)
    assert longest <= 40
    assert [plans.short_label(name) for name in plans.DIT_LABELS] == [
        "Sharp 7B float16",
        "Sharp 7B int8",
        "Sharp 7B dynamic GGUF",
        "7B float16",
        "7B int8",
        "7B dynamic GGUF",
    ]


def test_the_console_says_where_each_file_is_and_whether_it_was_downloaded(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="seedvr2x_comfyui")
    here = f"in Hugging Face's cache: {CACHE}; HF_HOME moves it"
    Cache(monkeypatch, has=(INT8,))
    run_models()
    # The DiT was there, the VAE and the detector are fetched; the sizes are the pins'.
    assert f"Sharp 7B int8, 8.3 GB, {here}" in unprefixed(caplog)
    assert f"VAE, 0.5 GB, downloaded into Hugging Face's cache: {CACHE}; HF_HOME moves it" in (
        unprefixed(caplog)
    )
    assert (
        f"shot detector, 30.5 MB, downloaded into Hugging Face's cache: {CACHE}; HF_HOME moves it"
        in unprefixed(caplog)
    )
    # Said each time the node runs, found now, in the order DiT, VAE, shot detector.
    caplog.clear()
    run_models()
    assert f"Sharp 7B int8, 8.3 GB, {here}" in unprefixed(caplog)
    assert f"VAE, 0.5 GB, {here}" in unprefixed(caplog)
    assert f"shot detector, 30.5 MB, {here}" in unprefixed(caplog)
    where = [line for line in unprefixed(caplog) if ", in Hugging Face's cache" in line]
    assert [line.split(",")[0] for line in where] == ["Sharp 7B int8", "VAE", "shot detector"]
    assert not any("downloaded" in line for line in unprefixed(caplog))
    assert all(
        line.startswith(console.PREFIX) and "seedvr2x:" not in line for line in caplog.messages
    )


def test_with_a_folder_the_dit_and_the_vae_are_not_downloaded_and_the_console_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("a download was asked of a model folder")

    monkeypatch.setattr(pull, "fetch", never)
    monkeypatch.setattr(pull, "resolve", never)
    caplog.set_level(logging.INFO, logger="seedvr2x_comfyui")
    for name in (INT8, VAE, DETECTOR):
        (tmp_path / name).write_bytes(b"x")
    found = run_models(model_dir=str(tmp_path))
    assert found == models.Handle(INT8, str(tmp_path), "on")
    # A folder's file by its own size, not its pin's (refused by the pin when it is read); the
    # detector's the folder's where the folder has it.
    one = plans.size_text(1)
    lines = unprefixed(caplog)
    assert f"Sharp 7B int8, {one}, in the model folder: {tmp_path}" in lines
    assert f"VAE, {one}, in the model folder: {tmp_path}" in lines
    assert f"shot detector, {one}, in the model folder: {tmp_path}" in lines
    assert not ProgressBar.created
    assert node_texts("4") == ["Sharp 7B int8 · kept in RAM"]


def test_a_detector_that_cant_be_had_is_said_and_the_handle_given(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Offline, no folder, the detector's file not in the cache (review M2 of impl-nodes-6): the
    # DiT and the VAE are there, and a picture needs nothing else, so the node gives its handle;
    # Cuts fetches the file when it runs, or refuses in Models' words.
    cache = Cache(monkeypatch, has=(INT8, VAE))

    def offline(pins: Any, download: Any = None, progress: Any = None) -> Any:
        cache.fetched.append(tuple(pins))
        raise ModelUnreachable(
            "transnetv2.safetensors: not in the cache, and offline", file=DETECTOR
        )

    monkeypatch.setattr(pull, "fetch", offline)
    caplog.set_level(logging.INFO, logger="seedvr2x_comfyui")
    found = run_models()
    assert found.dit == INT8 and cache.fetched == [(DETECTOR,)]
    lines = unprefixed(caplog)
    (warned,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert (
        warned.getMessage()
        .removeprefix(console.PREFIX)
        .startswith(
            "the shot detector's file, which seedvr2x Cuts reads, was not downloaded, and Cuts"
            " downloads it when it runs: transnetv2.safetensors: not in the cache, and offline"
        )
    )
    # Its place is said for the two that are there, none for it; the node's end line as ever.
    assert not any(line.startswith("shot detector, ") for line in lines)
    assert any(line.startswith("Sharp 7B int8, ") for line in lines)
    assert node_texts("4")[-1] == "Sharp 7B int8 · kept in RAM"


def test_a_folder_without_the_detector_makes_no_warning_and_never_asks_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A model folder is the user's choice of no cache and no download (the user, 2026-11-08):
    # one with the DiT and the VAE alone gives the handle, says nothing of the detector (seedvr2x
    # Cuts, which alone reads it, may not be wired: it refuses the file when it runs), downloads
    # nothing and reads nothing of Hugging Face's cache.
    for name in (INT8, VAE):
        (tmp_path / name).write_bytes(b"x")

    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("Hugging Face's cache asked with a model folder given")

    for name in ("fetch", "cached", "cache", "resolve"):
        monkeypatch.setattr(pull, name, never)
    caplog.set_level(logging.INFO, logger="seedvr2x_comfyui")
    found = run_models(model_dir=str(tmp_path))
    assert found == models.Handle(INT8, str(tmp_path), "on")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    shown = f"{plans.size_text(1)}, in the model folder: {tmp_path}"
    lines = unprefixed(caplog)
    assert f"Sharp 7B int8, {shown}" in lines and f"VAE, {shown}" in lines
    assert not any(line.startswith("shot detector, ") for line in lines)
    assert not ProgressBar.created
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted([INT8, VAE])
    assert node_texts("4") == ["Sharp 7B int8 · kept in RAM"]


def test_a_file_missing_from_the_folder_is_refused_at_this_node_in_the_nodes_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pull, "resolve", lambda *args, **kwargs: pytest.fail("downloaded"))
    monkeypatch.setattr(pull, "fetch", lambda *args, **kwargs: pytest.fail("downloaded"))
    (tmp_path / INT8).write_bytes(b"x")
    (tmp_path / DETECTOR).write_bytes(b"x")
    with pytest.raises(words.Refused) as refused:
        run_models(model_dir=str(tmp_path))
    text = str(refused.value)
    assert text.startswith(f"{VAE} is not in the folder wired to the")
    assert 'the "model folder" input of seedvr2x Models' in text and "\n" not in text
    assert "unwire the input to use Hugging Face's cache" in text and "--" not in text
    # Both missing: a line each.
    (tmp_path / INT8).unlink()
    with pytest.raises(words.Refused) as refused:
        run_models(model_dir=str(tmp_path))
    lines = str(refused.value).splitlines()
    assert [line.split(" ")[0] for line in lines] == [INT8, VAE], "the detector is no part of it"
    # Nothing was said on the node as if it had succeeded.
    assert not any("kept in RAM" in text for text in node_texts("4"))


def test_quotes_and_spaces_around_the_folder_are_taken_off(tmp_path: Path) -> None:
    folder = tmp_path / "my models"
    folder.mkdir()
    for given in (f'"{folder}"', f'  "{folder}"  ', f" {folder} ", f'" {folder} "'):
        assert models.model_folder(given) == str(folder), given
    # One pair only, and only a pair: a quote at one end is part of what was given.
    with pytest.raises(words.Refused, match="is not a folder"):
        models.model_folder(f'""{folder}""')
    with pytest.raises(words.Refused, match="is not a folder"):
        models.model_folder(f'"{folder}')


def test_a_tilde_that_begins_the_folder_is_the_home_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As a shell has it (review L7 of impl-nodes-5: a "~" was refused as no folder).
    home = tmp_path / "home"
    (home / "my models").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for given in ("~/my models", '"~/my models"', " ~/my models ", "~"):
        expected = home / "my models" if given.strip(' "') != "~" else home
        if sys.platform == "win32":  # expanduser keeps the "/" given after the home folder
            assert Path(models.model_folder(given)) == expected, given
        else:
            assert models.model_folder(given) == str(expected), given
    # Only where it begins the value: elsewhere it is part of a name, refused as no folder.
    with pytest.raises(words.Refused, match="is not a folder"):
        models.model_folder(str(tmp_path / "~"))
    # One that names no folder is refused by the folder it named.
    with pytest.raises(words.Refused) as refused:
        models.model_folder("~/nowhere")
    # The rest as given after the home folder ("/" kept on Windows too, by expanduser).
    assert str(refused.value).startswith(f'"{home}/nowhere" is not a folder')


@pytest.mark.parametrize("given", [None, "", "   ", '""', ' "" ', '" "'])
def test_nothing_given_or_an_empty_value_is_the_cache_and_an_empty_one_is_said(
    given: str | None, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="seedvr2x_comfyui")
    assert models.model_folder(given) == ""
    said = "the model folder given is empty: Hugging Face's cache is used" in unprefixed(caplog)
    assert said == (given is not None)


def test_a_value_that_is_no_folder_is_refused_in_the_nodes_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(pull, "resolve", lambda *args, **kwargs: pytest.fail("went on"))
    (tmp_path / "file.txt").write_text("x")
    for given in (str(tmp_path / "nowhere"), str(tmp_path / "file.txt")):
        with pytest.raises(words.Refused) as refused:
            run_models(model_dir=given)
        text = str(refused.value)
        assert text.startswith(f'"{given}" is not a folder: wire a folder that holds')
        assert 'the "model folder" input of seedvr2x Models' in text
        assert text.endswith("or unwire the input to use Hugging Face's cache")
        assert "--" not in text
    # The node says only that it refused; the text is in the error and once each in the console.
    assert node_texts("4") == ["refused: see the error"] * 2
    refusals = [line for line in unprefixed(caplog) if line.startswith("refused: ")]
    assert len(refusals) == 2 and all("is not a folder: wire a folder" in each for each in refusals)
    assert all(r.levelname == "WARNING" for r in caplog.records if "refused: " in r.getMessage())


def test_a_refusal_of_the_download_is_worded_for_the_models_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def offline(*args: object, **kwargs: object) -> None:
        raise ModelUnreachable("offline", file=INT8, why="offline")

    Cache(monkeypatch)
    monkeypatch.setattr(pull, "resolve", offline)
    with pytest.raises(words.Refused, match="Hugging Face's offline mode is on") as refused:
        run_models()
    assert (
        'unset HF_HUB_OFFLINE and restart ComfyUI, or wire a folder that has them to the "model'
        ' folder" input of seedvr2x Models'
    ) in str(refused.value)


def test_cancel_stops_a_download_and_one_reported_elsewhere_is_honoured_when_it_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = Cache(monkeypatch)
    tell: list[Any] = []
    real = cache.resolve

    def resolve(*arguments: Any, **options: Any) -> Any:
        tell.append(options["progress"])
        return real(*arguments, **options)

    monkeypatch.setattr(pull, "resolve", resolve)
    # On the nodes' thread: the bar's hook raises at the next report.
    cache.steps = [1_000_000]
    ProgressBar.on_update = lambda value: management.interrupt_current_processing()
    with pytest.raises(management.InterruptProcessingException):
        run_models()
    ProgressBar.on_update = None
    management.reset()
    ProgressBar.created.clear()

    # From the downloader's own thread (Xet): left pending, honoured when the download returns.
    cache.has.clear()
    errors: list[BaseException] = []

    def elsewhere(*arguments: Any, **options: Any) -> Any:
        found = real(*arguments, **options)
        management.interrupt_current_processing()

        def report() -> None:
            try:
                options["progress"](INT8, 5, SIZES[0])
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=report)
        thread.start()
        thread.join()
        return found

    monkeypatch.setattr(pull, "resolve", elsewhere)
    with pytest.raises(management.InterruptProcessingException):
        run_models()
    assert errors == [], "the downloader's thread was not interrupted"
    assert not any("kept in RAM" in text for text in node_texts("4")), "no end line after Cancel"


def test_the_downloading_line_is_in_the_totals_unit() -> None:
    assert progress.amount_short(3_100_000_000, 8_800_000_000) == "3.1/8.8 GB"
    assert progress.amount_short(12_100_000, 30_482_632) == "12.1/30.5 MB"
    shown: list[str] = []
    downloading = progress.Preparing(shown.append, plans.file_label)
    downloading.download(INT8, 0, SIZES[0])
    downloading.download(VAE, 0, SIZES[1])
    downloading.download(INT8, 4_000_000_000, SIZES[0])
    assert where(shown[-1]) == f"downloading 4.0/{sum(SIZES[:2]) / 1e9:.1f} GB"


# ---- The upscale node: a file the cache lost since the Models node ran ---------------------------


def test_a_file_gone_from_the_cache_since_is_downloaded_again_by_the_upscale_node_and_said(
    machine: StandIn,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="seedvr2x_comfyui")
    fetched: list[str] = []
    cache = Cache(monkeypatch, has=(plans.DEFAULT_DIT,))

    def resolve(model_dir: Path | None, dit: str, vae: str, **options: Any) -> ModelFiles:
        cache.asked.append((model_dir, dit, vae, False))
        fetched.extend(name for name in (dit, vae) if name not in cache.has)
        options["progress"](vae, 0, pull.PINNED[vae].size)
        assert options["detector"] is False, "the upscale nodes need no detector"
        return ModelFiles(
            ModelFile(dit, tmp_path / dit), ModelFile(vae, tmp_path / vae), CACHE, pull.Hashes()
        )

    monkeypatch.setattr(pull, "resolve", resolve)
    pictures = torch.from_numpy(np.random.default_rng(8).random((1, *PICTURE, 3), dtype=np.float32))
    upscale_image(
        models.Handle(plans.DEFAULT_DIT, "", "off"), plans.Settings(), pictures, RESOLUTION
    )
    assert fetched == [VAE]
    said = "VAE is no longer in Hugging Face's cache: downloading it again"
    assert unprefixed(caplog).count(said) == 1
    # The one still there is not said, and the line comes before the download's own.
    assert not any(plans.short_label(plans.DEFAULT_DIT) in line for line in unprefixed(caplog))
    # The bar for it is the upscale node's own, as before.
    assert ProgressBar.created, "the download is told on the upscale node's bar"


def test_a_file_still_in_the_cache_says_nothing_and_a_folder_has_no_such_line(
    machine: StandIn,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="seedvr2x_comfyui")
    cache = Cache(monkeypatch, has=(plans.DEFAULT_DIT, VAE))
    monkeypatch.setattr(
        pull,
        "resolve",
        lambda model_dir, dit, vae, **options: ModelFiles(
            ModelFile(dit, tmp_path / dit), ModelFile(vae, tmp_path / vae), CACHE, pull.Hashes()
        ),
    )
    pictures = torch.from_numpy(np.random.default_rng(8).random((1, *PICTURE, 3), dtype=np.float32))
    upscale_image(
        models.Handle(plans.DEFAULT_DIT, "", "off"), plans.Settings(), pictures, RESOLUTION
    )
    # A folder: the cache is not asked at all.
    cache.has.clear()
    monkeypatch.setattr(pull, "cached", lambda name: pytest.fail("the cache was asked"))
    upscale_image(
        models.Handle(plans.DEFAULT_DIT, str(tmp_path), "off"),
        plans.Settings(),
        pictures,
        RESOLUTION,
    )
    assert not any("no longer in Hugging Face's cache" in line for line in unprefixed(caplog))


def test_a_node_run_without_comfyuis_server_still_raises_its_own_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ComfyUI's PromptServer has no `instance` until its server is made (server.py): the node's
    # line is then not sent, and its refusal is what comes out, not an AttributeError (found by
    # the smoke test in ComfyUI v0.44.0, run without its server).
    monkeypatch.delattr(server.PromptServer, "instance")
    with pytest.raises(words.Refused, match="is not a folder"):
        run_models(model_dir="/no/such/folder")
