"""Milestone 1 as a regression test: seedvr2x's output stays bit-identical to numz's, in numz's
padding; and in seedvr2x's own (DESIGN.md, Pipeline step 0), to numz's with that padding patched
in.

Needs a GPU (pytest -m gpu), the weights (SEEDVR2X_MODEL_DIR) and milestone 1's reference, in
the m1/ directory of SEEDVR2X_REFERENCE_DIR:
- input_rgb.mkv: 45 frames of 8-bit RGB FFV1 at 960x540, read alike by numz (OpenCV) and by us:
  frames 48-92 of SEGMENT, a 1920x1080 yuv420p FFV1 segment of an animated episode (limited
  range, chroma left, no colour tags, 23.976 fps, 377 frames; its first 26 black, then a fade-in
  to about frame 50), BT.709 limited range to full-range RGB, its chroma upsampled and the
  picture brought to 960x540 by zimg's spline36, with ffmpeg n9.0.2:

    VF=trim=start_frame=48:end_frame=93,setpts=PTS-STARTPTS
    VF=$VF,zscale=w=960:h=540:filter=spline36:matrixin=709:transferin=709:primariesin=709
    VF=$VF:rangein=limited:transfer=709:primaries=709:range=full:dither=none,format=gbrp
    ffmpeg -nostdin -y -i SEGMENT -map 0:v:0 -vf "$VF" -fps_mode passthrough \
      -c:v ffv1 -level 3 -g 1 -pix_fmt bgr0 -an -sn input_rgb.mkv

  Remade so on 2026-10-09: the same frames and packets, the two files differing only in the
  Matroska muxer's random UIDs.
- numz.mkv and numz_frames/: numz 4490bd1's output for input_rgb.mkv, captured by
  research/scripts/ffv1_out.py, from numz's checkout:

    FFV1_OUT_PATH=numz.mkv FFV1_OUT_KEEP=0 FFV1_OUT_DUMP=numz_frames \
    FFV1_OUT_DUMP_FRAMES=$(seq -s, 0 44) python ffv1_out.py inference_cli.py input_rgb.mkv \
      --output out --output_format png --model_dir MODELS \
      --dit_model seedvr2_ema_7b_fp16.safetensors --resolution 1080 --batch_size 45 \
      --seed 42 --attention_mode flash_attn_2 --color_correction none

- numz_sharp.mkv and numz_sharp_frames/: the same with the sharp 7B (made on 2026-10-08), the
  same command with FFV1_OUT_PATH=numz_sharp.mkv, FFV1_OUT_DUMP=numz_sharp_frames and
  --dit_model seedvr2_ema_7b_sharp_fp16.safetensors.

Each case is a run of its own, both model files named, in numz's padding (cli.NUMZ_PADDING, for
the tests only):
- numz-7b: numz's seedvr2_ema_7b_fp16.safetensors and ema_vae_fp16.safetensors, against numz.mkv;
- ours-7b: seedvr2x's seedvr2x_ema_7b_fp16.safetensors and seedvr2x_ema_vae_fp16.safetensors,
  the same values as numz's files (DESIGN.md, Weights), so against the same reference, bit for
  bit;
- ours-sharp: seedvr2x_ema_7b_sharp_fp16.safetensors and seedvr2x_ema_vae_fp16.safetensors, against
  numz_sharp.mkv, skipped when that reference is absent.
Every run detects its shots, as a user's does without --cuts (DESIGN.md, Shot detection): the
shot detector runs on the GPU in the first pass, before the models load, so that its settings,
scoped to its own forwards, and the memory it gives back are held to the bits too. At
--cut-threshold 0.5, where its cuts are none, so that the job is one shot, as numz's one batch,
which the log must say. At the default, 0.3, it isn't: input_rgb.mkv ends on two short shots,
which numz's batch ran through, TransNetV2 peaking at 0.3242 on frame 41 and 0.3880 on frame 43,
and at 0.1171 on frame 0, on the CPU and on the GPU alike (1.19e-07 apart; the same on its
956x530 crop): cut at 42 and 44, 3 shots, none of the run's 45 frames was numz's, where at 0.5,
1 shot and 2 possible cuts, all 45 are (the GPU box, 2026-10-10). A case whose model file, or the
shot detector's (transnetv2.safetensors), isn't in SEEDVR2X_MODEL_DIR is skipped, naming the file.
One more,
test_milestone1_from_the_cache, runs ours-sharp with the default models taken from Hugging Face's
cache (runtime/pull.py): no --model-dir, no --dit-model or --vae-model, and HF_HUB_OFFLINE=1 in
its environment, so that only the cache can serve them; it needs no SEEDVR2X_MODEL_DIR, and is
skipped, saying how to fill the cache (uv run hf download, the three files, 17 GB), unless the
run would find them there: at the pinned revision, or downloaded at another (runtime/pull.py).

seedvr2x's own padding, the default (at least 8 rows reflected from the picture, up to a multiple
of 16, then 16 black rows; columns alike when the width isn't a multiple of 16), is held to numz
patched by research/scripts/numerics_patch.py with NUM_PAD=reflect>=8+black+16, which pads the
same way, numz's 7B fp16 and VAE on both sides. Each reference was made on 2026-10-09, from numz's
checkout, with numerics_patch.py running ffv1_out.py running numz's CLI:
- pad1080: input_rgb.mkv at --resolution 1080, 1920x1080 padded to 1920x1104 (8 rows reflected,
  no column), against numz_pad1080.mkv and numz_pad1080_frames/:

    NUM_PAD='reflect>=8+black+16' FFV1_OUT_PATH=numz_pad1080.mkv FFV1_OUT_KEEP=0 \
    FFV1_OUT_DUMP=numz_pad1080_frames FFV1_OUT_DUMP_FRAMES=$(seq -s, 0 44) \
    python numerics_patch.py ffv1_out.py inference_cli.py input_rgb.mkv \
      --output out --output_format png --model_dir MODELS \
      --dit_model seedvr2_ema_7b_fp16.safetensors --resolution 1080 --batch_size 45 \
      --seed 42 --attention_mode flash_attn_2 --color_correction none

- pad720: input_rgb.mkv at --resolution 720, 1280x720 padded to 1280x752 (16 rows reflected),
  against numz_pad720.mkv and numz_pad720_frames/: the same command with
  FFV1_OUT_PATH=numz_pad720.mkv, FFV1_OUT_DUMP=numz_pad720_frames and --resolution 720;
- padcols: input_rgb_956x530.mkv, input_rgb.mkv's top left 956x530, cropped losslessly,

    ffmpeg -i input_rgb.mkv -vf crop=956:530:0:0 -c:v ffv1 -pix_fmt bgr0 input_rgb_956x530.mkv

  at --resolution 1060, 1912x1060 padded to 1936x1088 (12 rows and 8 columns reflected), against
  numz_padcols.mkv and numz_padcols_frames/: the same command with input_rgb_956x530.mkv,
  FFV1_OUT_PATH=numz_padcols.mkv, FFV1_OUT_DUMP=numz_padcols_frames and --resolution 1060.
numz logs each size (src/core/generation_utils.py:274-279):
"Padded: 1920x1104px → Output: 1920x1080px", "Padded: 1280x752px → Output: 1280x720px",
"Padded: 1936x1088px → Output: 1912x1060px"; seedvr2x's --resolution gives the same
(tests/test_padding.py). A case whose input or reference is absent is skipped, naming the file.

The run goes through the CLI in a process of its own, as a user's would: the allocator is set
before torch is imported, which a test process importing torch first wouldn't do.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from seedvr2x.cli import NUMZ_PADDING
from seedvr2x.runtime import pull

PROJECT = Path(__file__).resolve().parents[1]
MODELS = os.environ.get("SEEDVR2X_MODEL_DIR")
REFERENCE = os.environ.get("SEEDVR2X_REFERENCE_DIR")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not REFERENCE, reason="needs SEEDVR2X_REFERENCE_DIR"),
]

# numz's model files, which the references were made with.
NUMZ_DIT, NUMZ_VAE = "seedvr2_ema_7b_fp16.safetensors", "ema_vae_fp16.safetensors"
# The command line's defaults, the sharp 7B and the VAE (cli.py).
DEFAULT_DIT, DEFAULT_VAE = (
    "seedvr2x_ema_7b_sharp_fp16.safetensors",
    "seedvr2x_ema_vae_fp16.safetensors",
)


def models_present(*names: str) -> None:
    """Skip, naming the file, unless SEEDVR2X_MODEL_DIR holds each of names, and the shot
    detector's file, which every run here takes (the module's docstring)."""
    if MODELS is None:
        pytest.skip("needs SEEDVR2X_MODEL_DIR")
    for name in (*names, pull.DETECTOR):
        if not (Path(MODELS) / name).is_file():
            pytest.skip(f"{name}: not in SEEDVR2X_MODEL_DIR")


def bit_identical(
    tmp_path: Path,
    source: Path,
    resolution: int,
    models: tuple[str, str] | None,
    reference: Path,
    *,
    numz_padding: bool,
) -> str:
    """Run seedvr2x on source at resolution with the DiT and VAE files of models in
    SEEDVR2X_MODEL_DIR, or with None, its default models from Hugging Face's cache, offline
    (HF_HUB_OFFLINE=1), in numz's padding or in its own; compare its master and float32 frames
    with the reference's, <reference>.mkv and <reference>_frames/, bit for bit. Returns its
    log."""
    master, frames = tmp_path / "ours.mkv", tmp_path / "frames"
    environment = {name: value for name, value in os.environ.items() if name != NUMZ_PADDING}
    if numz_padding:
        environment[NUMZ_PADDING] = "1"
    if models is None:
        options = []
        environment["HF_HUB_OFFLINE"] = "1"
    else:
        assert MODELS is not None
        options = ["--model-dir", MODELS, "--dit-model", models[0], "--vae-model", models[1]]
    run = subprocess.run(
        [
            *(sys.executable, "-m", "seedvr2x", str(source), "-o", str(master), *options),
            *("--resolution", str(resolution), "--seed", "42", "--dump-frames", str(frames)),
            # The shots detected, and none found: at the default 0.3, the input is cut at 42 and
            # 44, 3 shots where numz ran one batch (the module's docstring).
            *("--cut-threshold", "0.5"),
            # numz's output milestone 1 holds to, without its lab (StableSR's, not vendored).
            *("--color-correction", "none"),
            # numz.mkv's format, ffv1_out.py's default: compare.py reads both masters as stored.
            *("--format", "gbrp16le"),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    # The shot detector ran in the first pass, on the GPU, and gave its memory back; no cut at
    # 0.5, the input one shot, as numz's one batch (the module's docstring).
    assert "the shot detector scored" in run.stderr, run.stderr[-3000:]
    assert "shot detector: VRAM peak" in run.stderr, run.stderr[-3000:]
    assert "1 shot: 0 cuts detected at 0.5" in run.stderr, run.stderr[-3000:]
    expected = (
        reference.parent / f"{reference.name}.mkv",
        reference.parent / f"{reference.name}_frames",
    )
    for kind, theirs, ours in (("masters", expected[0], master), ("frames", expected[1], frames)):
        compare = subprocess.run(
            [sys.executable, str(PROJECT / "tools" / "compare.py"), kind, str(theirs), str(ours)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert compare.returncode == 0, f"{kind}: {compare.stdout[-3000:]}"
    return run.stderr


@pytest.mark.parametrize(
    ("dit", "vae", "reference"),
    [
        pytest.param(NUMZ_DIT, NUMZ_VAE, "numz", id="numz-7b"),
        pytest.param(
            "seedvr2x_ema_7b_fp16.safetensors",
            "seedvr2x_ema_vae_fp16.safetensors",
            "numz",
            id="ours-7b",
        ),
        pytest.param(
            "seedvr2x_ema_7b_sharp_fp16.safetensors",
            "seedvr2x_ema_vae_fp16.safetensors",
            "numz_sharp",
            id="ours-sharp",
        ),
    ],
)
def test_milestone1_bit_identical(tmp_path: Path, dit: str, vae: str, reference: str) -> None:
    assert REFERENCE is not None
    models_present(dit, vae)
    m1 = Path(REFERENCE) / "m1"
    if reference == "numz_sharp":
        for path in (m1 / f"{reference}.mkv", m1 / f"{reference}_frames"):
            if not path.exists():
                pytest.skip(f"{path}: absent; this module's docstring says how to make it")
    source = m1 / "input_rgb.mkv"
    bit_identical(tmp_path, source, 1080, (dit, vae), m1 / reference, numz_padding=True)


def test_milestone1_from_the_cache(tmp_path: Path) -> None:
    # ours-sharp, its files taken from Hugging Face's cache, as a user's run takes them by
    # default, the shot detector's with them: offline, so that the cache alone can serve them,
    # bit for bit as SEEDVR2X_MODEL_DIR does. Skipped unless the run would find all three there
    # (pull.cached).
    assert REFERENCE is not None
    for name in (DEFAULT_DIT, DEFAULT_VAE, pull.DETECTOR):
        if pull.cached(name) is None:
            pytest.skip(
                f"{name}: not in Hugging Face's cache; fill it with the three default files, 17"
                f" GB: uv run hf download {pull.REPO} {DEFAULT_DIT} {DEFAULT_VAE} {pull.DETECTOR}"
                f" --revision {pull.REVISION}"
            )
    m1 = Path(REFERENCE) / "m1"
    for path in (m1 / "numz_sharp.mkv", m1 / "numz_sharp_frames"):
        if not path.exists():
            pytest.skip(f"{path}: absent; this module's docstring says how to make it")
    log = bit_identical(
        tmp_path, m1 / "input_rgb.mkv", 1080, None, m1 / "numz_sharp", numz_padding=True
    )
    for name in (DEFAULT_DIT, DEFAULT_VAE, pull.DETECTOR):
        assert f"{name}: in the cache, " in log
        pinned = f"its size and SHA-256 as seedvr2x pins them, {pull.REPO} at {pull.REVISION[:8]}"
        assert f"{name}: {pinned}" in log
    assert "downloading" not in log


@pytest.mark.parametrize(
    ("source", "resolution", "reference"),
    [
        pytest.param("input_rgb.mkv", 1080, "numz_pad1080", id="pad1080"),
        pytest.param("input_rgb.mkv", 720, "numz_pad720", id="pad720"),
        pytest.param("input_rgb_956x530.mkv", 1060, "numz_padcols", id="padcols"),
    ],
)
def test_padding_bit_identical(
    tmp_path: Path, source: str, resolution: int, reference: str
) -> None:
    assert REFERENCE is not None
    models_present(NUMZ_DIT, NUMZ_VAE)
    m1 = Path(REFERENCE) / "m1"
    for path in (m1 / source, m1 / f"{reference}.mkv", m1 / f"{reference}_frames"):
        if not path.exists():
            pytest.skip(f"{path}: absent; this module's docstring says how to make it")
    models = (NUMZ_DIT, NUMZ_VAE)
    bit_identical(tmp_path, m1 / source, resolution, models, m1 / reference, numz_padding=False)
