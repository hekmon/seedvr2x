"""--quality deghost on the command line (DESIGN.md, Each frame on its own: `--quality deghost`)
on the CPU, the model replaced by test_cli_run.py's stand-in: a job's frames those of jobs of a
single frame each, on the source ffmpeg debanded, seeded by their shot's seed (the definition the
library's tests/test_clip.py holds for clip.upscale); the units in their order, the shot's
encode, a unit a frame for the DiT, the segment's decode; the manifest's record of a run that
takes each frame alone; a stop inside a shot's frames and its resume, as the uninterrupted job.
The banded clip and its reference debanded by ffmpeg are test_deband_cli.py's."""

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from test_cli_run import (
    SEED,
    Steps,
    decoded,
    plan_lines,
    stand_in,  # noqa: F401
    steps,  # noqa: F401
    stopped,
    upscale,
)
from test_deband import needs_ffmpeg
from test_deband_cli import clips, run_ffmpeg

import run
from media.conversion import DEBAND
from seedvr2x import pull

pytestmark = needs_ffmpeg

# The stand-in's master frames (gbrp16le, 128x96), in bytes.
FRAME_BYTES = 3 * 96 * 128 * 2
# test_deband_cli.py's clip at 16 bits after ffmpeg's deband (yuv420p16le, 64x48), in bytes.
DEBANDED_BYTES = 64 * 48 * 3 // 2 * 2
# test_cli_run.py's JOB's cut list (cuts.txt: shots of 3, 1 and 21 frames) without its window:
# a frame alone is one latent in the DiT, and --window is refused at deghost.
SHOTS = ((0, 3), (3, 4), (4, 25))
DEGHOST = ("--cuts", "cuts.txt", "--min-segment", "0.12", "--quality", "deghost")


def frames_of(path: Path) -> list[str]:
    """The MD5 of each frame of a master as stored (gbrp16le)."""
    data = decoded(path)
    assert len(data) % FRAME_BYTES == 0
    return [
        hashlib.md5(data[k : k + FRAME_BYTES]).hexdigest() for k in range(0, len(data), FRAME_BYTES)
    ]


def masters(directory: Path) -> list[str]:
    return [frame for path in sorted(directory.glob("seg_*.mkv")) for frame in frames_of(path)]


@pytest.mark.parametrize("correction", ["split", "none"])
def test_each_frame_is_a_job_of_that_frame_alone_seeded_by_its_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    correction: str,
) -> None:
    # Frame f of a shot starting at s is a job of that frame alone, on the source ffmpeg debanded
    # (test_deband_cli.py's reference), trimmed to the frame, seeded 42 + s, so that its one shot
    # is seeded as the shot is: bit for bit as stored, and its dumped float32 frame. Each encode
    # reads its frame alone, the same frames the single jobs read.
    banded, prefiltered = clips(tmp_path, monkeypatch)
    reads: list[str] = []
    encode = steps.encode

    def reading(models: object, read: Any, count: int, *arguments: Any, **keywords: Any) -> object:
        frames = read(count)
        reads.append(hashlib.md5(frames.tobytes()).hexdigest())
        return encode(models, lambda n: frames[:n], count, *arguments, **keywords)

    monkeypatch.setattr(run, "encode_shot", reading)
    options = ("--color-correction", correction)
    dump = tmp_path / "dump"
    assert upscale(tmp_path, banded, "job", *DEGHOST, *options, "--dump-frames", str(dump)) == 0
    got, job_reads = masters(tmp_path / "job"), list(reads)
    assert len(got) == 25 and len(job_reads) == 25
    (tmp_path / "empty.txt").write_text("")
    expected: list[str] = []
    single_reads: list[str] = []
    for start, end in SHOTS:
        for frame in range(start, end):
            alone = tmp_path / f"frame_{frame}.mkv"
            run_ffmpeg(
                *("-i", str(prefiltered), "-vf"),
                f"trim=start_frame={frame}:end_frame={frame + 1},setpts=PTS-STARTPTS",
                *("-c:v", "ffv1", str(alone)),
            )
            whole = decoded(prefiltered)
            assert decoded(alone) == whole[frame * DEBANDED_BYTES : (frame + 1) * DEBANDED_BYTES]
            reads.clear()
            name, dumped = f"single_{frame}", tmp_path / f"dump_{frame}"
            assert (
                upscale(
                    *(tmp_path, alone, name, "--cuts", "empty.txt", *options),
                    *("--seed", str(SEED + start), "--dump-frames", str(dumped)),
                )
                == 0
            )
            expected += masters(tmp_path / name)
            single_reads += reads
            single = np.load(dumped / "frame_000000.npy")
            assert np.array_equal(np.load(dump / f"frame_{frame:06d}.npy"), single)
    assert got == expected
    assert job_reads == single_reads


@pytest.mark.usefixtures("stand_in")
def test_the_units_the_record_and_the_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    banded, _ = clips(tmp_path, monkeypatch)
    caplog.set_level("INFO")
    assert upscale(tmp_path, banded, "out", *DEGHOST) == 0
    # Each shot: its frames' encodes, one a frame, then its frames' windows, one a frame, each
    # seeded by the shot (the stand-in names a call by its seed less the job's: the shot's first
    # frame); then each segment's decode, a frame at a time.
    calls = [call for call in steps.calls]
    expected: list[str] = []
    for start, end in SHOTS:
        expected += [f"encode {start}"] * (end - start) + [f"window {start}:0"] * (end - start)
    decodes = [call for call in calls if call.startswith("decode")]
    assert [call for call in calls if not call.startswith("decode")] == expected
    # The stand-in names a decode by the latent's first value, its shot's first frame.
    assert decodes == [f"decode {start}" for start, end in SHOTS for _ in range(start, end)]
    content = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert content["settings"]["quality"] == "deghost"
    assert content["settings"]["deband"] == DEBAND
    for (start, end), shot in zip(SHOTS, content["shots"], strict=True):
        # One run a shot, its latents its frames, no window listed: one a frame; done whole.
        assert shot == {
            "start": start,
            "end": end,
            "seed": SEED + start,
            "latents": end - start,
            "encoded": True,
            "windows_done": end - start,
        }
    lines = plan_lines(caplog)
    assert "plan: windows: one a frame, of its one latent, the frame's own DiT step" in lines
    assert (
        "plan: shots: 3, the longest of 21 frames, each frame a latent of its own (--quality"
        " deghost)" in lines
    )
    assert (
        "plan: quality: deghost, each frame of a shot upscaled on its own, in a latent of its"
        " own, from the shot's noise, its source debanded: about 1.8 times standard's time, a"
        " frame's memory; for animation, where the model blends the drawings a latent packs,"
        " nothing measured on live action; --quality standard, enhanced or maximum packs a shot's"
        " frames in its latents"
    ) in lines
    assert any(line.endswith("(--quality deghost); the shot detection reads the source as it is")
               for line in lines)  # fmt: skip
    # The unit lines name each frame.
    assert "shot 3/3's frame 21/21" in caplog.text or "shot 3/3: frame 21/21" in caplog.text


def test_a_stop_inside_a_shots_frames_resumes_as_the_uninterrupted_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
) -> None:
    # Stopped in the third shot's 8th frame's window, after the frames before it are kept: the
    # resume samples on from that frame, from the latents kept, and the masters are the
    # uninterrupted job's; nothing of the run is left in the job's directory.
    banded, _ = clips(tmp_path, monkeypatch)
    options = (*DEGHOST, "--color-correction", "split")
    assert upscale(tmp_path, banded, "whole", *options) == 0
    uninterrupted = masters(tmp_path / "whole")
    windows = steps.windows
    count = 0

    def stopping(models: object, latent: torch.Tensor, *rest: Any) -> Iterator[torch.Tensor]:
        nonlocal count
        count += 1
        if count == 3 + 1 + 8:
            raise KeyboardInterrupt
        return windows(models, latent, *rest)

    monkeypatch.setattr(run, "sample_windows", stopping)
    stopped(tmp_path, banded, "out", *options)
    content = json.loads((tmp_path / "out" / "manifest.json").read_text())
    third = content["shots"][2]
    assert (third["encoded"], third["windows_done"]) == (True, 7)
    kept = sorted(path.name for path in (tmp_path / "out" / "resume" / "shot_000004").iterdir())
    assert kept == [
        *("input.crc32", "input.mkv"),
        *(f"latent_{frame:06d}.pt" for frame in range(7, 21)),
        *(f"window_{frame:04d}.pt" for frame in range(7)),
    ]
    steps.calls.clear()
    monkeypatch.setattr(run, "sample_windows", windows)
    assert upscale(tmp_path, banded, "out", *options) == 0
    # No encode again: the third shot's frames sampled on from its 8th.
    assert [call for call in steps.calls if not call.startswith("decode")] == ["window 4:0"] * 14
    assert masters(tmp_path / "out") == uninterrupted
    assert not (tmp_path / "out" / "resume").exists()


@pytest.mark.usefixtures("stand_in")
@pytest.mark.parametrize("plan", [False, True], ids=["run", "plan"])
def test_window_is_refused_at_deghost_before_anything_is_made(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    plan: bool,
) -> None:
    # A frame alone is one latent in the DiT, with no windows to cap: --window refused at the
    # start, as a picture refuses it, in the command line's words, before the first pass, the
    # card read or a model loaded: no job directory, no plan's file, nothing else beside the
    # source.
    banded, _ = clips(tmp_path, monkeypatch)
    before = sorted(path.name for path in tmp_path.iterdir())
    options = ("--plan", "plan.txt") if plan else ()
    assert upscale(tmp_path, banded, "out", *DEGHOST, "--window", "5", *options) == 1
    assert (
        "--window: at --quality deghost each frame is one latent in the DiT, with no windows to"
        " cap; run without --window" in caplog.text
    )
    assert "first pass" not in caplog.text and "plan: " not in caplog.text
    # The stand-in models' files upscale writes beside the source (test_cli_run.upscale) alone.
    made = {path.name for path in tmp_path.iterdir()} - set(before)
    assert made <= {"w.safetensors", pull.DETECTOR}
    assert not (tmp_path / "out").exists() and not (tmp_path / "plan.txt").exists()


@pytest.mark.usefixtures("stand_in")
def test_a_plan_at_deghost_says_every_levels_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # --plan says each level's time (main._say_times): the plan at deghost is a frame's, its
    # window a latent, and the levels that pack a shot's frames are counted uncapped there
    # (main._level_units), not laid out by that window, which no shot of theirs takes.
    banded, _ = clips(tmp_path, monkeypatch)
    caplog.set_level("INFO")
    assert upscale(tmp_path, banded, "out", *DEGHOST, "--plan", "plan.txt") == 0
    [line] = [message for message in caplog.messages if message.startswith("time at each")]
    for level in ("standard", "enhanced", "maximum"):
        assert f"{level} about " in line, line
    assert "deghost about " in line and "(this job's)" in line


def test_a_frames_window_files_are_named_past_9999() -> None:
    # A shot of each frame alone has a window a frame: past 9,999 frames, 5 digits, a unit file
    # still (resume.UNIT_FILE), as the latents of 6 digits and more are.
    import resume
    from units import frame_latent, window_file

    for name in (window_file(9999), window_file(12345), frame_latent(0), frame_latent(1234567)):
        assert resume.UNIT_FILE.fullmatch(name), name
    assert window_file(12345) == "window_12345.pt" and frame_latent(7) == "latent_000007.pt"


def test_the_help_says_what_deghost_does_and_for_what(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from test_cli import _option_help  # pyright: ignore[reportPrivateUsage]

    import main as cli

    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = _option_help(capsys.readouterr().out, "--quality")
    for said in (
        "deghost: each frame of a shot upscaled on its own, from the shot's noise, its source"
        " debanded",
        "about 1.8 times standard's time",
        "for animation, where the model blends the drawings a group packs",
        "not measured on live action",
    ):
        assert said in text, said
