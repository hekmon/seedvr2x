"""The model check (seedvr2x/weights.py; DESIGN.md, Weights: recognised by content) through the
command line: a model refused before the first pass, saying what the file is, and the defaults
the help names. The check itself, on synthetic and real files, is test_weights.py's (core/tests/),
whose writers these tests use."""

import subprocess
import sys
from pathlib import Path

import pytest
from test_weights import V1, as_dtype, write

import main as cli
from media import ffmpeg


def _usable() -> bool:
    try:
        ffmpeg.check()
    except ffmpeg.MediaError:
        return False
    return True


@pytest.mark.skipif(not _usable(), reason="needs ffmpeg 7.1 or later with zscale, idet and ffv1")
def test_cli_refuses_before_the_first_pass(tmp_path: Path) -> None:
    # A 3B as the DiT stops the run in its checks, before the first pass (which logs the frames
    # it counted) and before torch is imported.
    source = tmp_path / "in.mkv"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=25"),
            *("-frames:v", "3", "-c:v", "ffv1", str(source)),
        ],
        check=True,
    )
    write(tmp_path / "seedvr2x_ema_vae_fp16.safetensors", as_dtype("vae", "F16"))
    write(tmp_path / "seedvr2_ema_3b_fp16.safetensors", as_dtype("3b", "F16"))
    args = [str(source), "-o", "out.mkv", "--model-dir", "."]
    args += ["--dit-model", "seedvr2_ema_3b_fp16.safetensors"]
    # Exit 3 if torch was imported: a failed assert would exit 1, as the refusal does.
    code = (
        "import sys; import main as cli; status = cli.main(sys.argv[1:]);"
        " sys.exit(3 if 'torch' in sys.modules else status)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode == 1, result.stderr
    assert f"seedvr2_ema_3b_fp16.safetensors: SeedVR2's 3B DiT in fp16: {V1}" in result.stderr
    assert "frames, 64x48" not in result.stderr
    assert "VAE seedvr2x_ema_vae_fp16.safetensors: SeedVR2's VAE in fp16" in result.stderr


def test_defaults_in_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        cli.main(["--help"])
    assert exited.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "(default: seedvr2x_ema_7b_sharp_fp16.safetensors, the sharp 7B)" in text
    assert "(default: seedvr2x_ema_vae_fp16.safetensors)" in text
    assert "SeedVR2's 7B DiT, regular or sharp, in fp16, recognised by its tensors" in text
