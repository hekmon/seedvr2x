"""A unit's VRAM peak read from its own phase (run.py, _phase_entered): torch's peak is reset once
the unit's phase is entered, so that a VAE unit right after a DiT phase counts none of the DiT's
weights the phase change sends away, and a window right after a VAE unit none of the VAE's. At
deghost on an 8 GB card, 1080p, the encode planned at 5.02 GiB and the decode at 4.73, the units
after a change of phase logged 6,116,776,960 bytes (5.70 GiB) for both, the DiT's resident
weights at the reset, the frames' peak less the same 637,944,832 bytes on two cards (impl-box-14
and impl-wsl2-11, 2026-11-25). On the CPU, the stand-in's card a fake whose bytes allocated are
each phase's resident weights, set as the stand-in models enter it: the DiT's over the VAE's, as
there, and under them, its blocks swapped, so that each direction of a change of phase is held."""

import logging
import re
from pathlib import Path

import pytest
import torch
from levels import levels
from test_cli_run import (
    JOB,
    STAND_IN_MODELS,
    Steps,
    job,
    stand_in,  # noqa: F401
    steps,  # noqa: F401
    upscale,
)

# Each phase's resident bytes on the fake card: the DiT's far above the VAE's, as at deghost on
# 8 GB (the DiT's 6,116,776,960 bytes at the reset, the VAE units' 2.10 to 2.38 GiB); and the
# DiT's under them, most of its blocks swapped, so that a window right after a VAE unit would
# read the VAE's bytes were the peak reset before the change.
RESIDENT = {
    "dit-over-vae": {"encode": 2_260_087_296, "dit": 6_116_776_960, "decode": 2_552_181_760},
    "vae-over-dit": {"encode": 2_260_087_296, "dit": 1_073_741_824, "decode": 2_552_181_760},
}


class FakeCard:
    """torch's peak counters over a card whose bytes allocated are the phase's resident ones:
    reset sets the peak to what is allocated then, as torch's reset_peak_memory_stats does."""

    def __init__(self, resident: dict[str, int]) -> None:
        self.resident = resident
        self.allocated = 0
        self.peak = 0

    def enter(self, phase: str) -> None:
        self.allocated = self.resident[phase]
        self.peak = max(self.peak, self.allocated)

    def reset(self, device: object) -> None:
        self.peak = self.allocated

    def max_allocated(self, device: object) -> int:
        return self.peak


@pytest.mark.parametrize("card", list(RESIDENT))
@pytest.mark.parametrize(
    "options",
    [
        (*JOB,),
        ("--cuts", "cuts.txt", "--min-segment", "0.12", "--quality", "deghost"),
        ("--cuts", "cuts.txt", "--min-segment", "0.12", "--quality", "maximum", "--window", "5"),
    ],
    ids=["standard", "deghost", "maximum"],
)
def test_each_units_peak_counts_no_byte_of_the_phase_before(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
    options: tuple[str, ...],
    card: str,
) -> None:
    # Every unit's debug line of its peak, an encode's, a window's and a segment's decode, is its
    # phase's resident bytes alone, whatever phase the unit before it left: before the fix, an
    # encode or a decode after a window read the DiT's where the DiT's are over the VAE's, a
    # window after an encode or a decode the VAE's where they are under.
    resident = RESIDENT[card]
    fake = FakeCard(resident)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", fake.reset)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", fake.max_allocated)
    monkeypatch.setattr(STAND_IN_MODELS, "enter", fake.enter)
    levels(caplog, logging.DEBUG)
    source = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source, "out", *options) == 0
    peaks = [
        (match[1], int(match[2]))
        for message in caplog.messages
        if (match := re.fullmatch(r"(.*): VRAM peak (\d+) bytes allocated, \d+ reserved", message))
    ]
    kinds = {"encode": "encode", "frame": "dit", "window": "dit", "decode": "decode"}
    seen: set[str] = set()
    for unit, peak in peaks:
        kind = next(
            phase for word, phase in kinds.items() if re.search(rf"(?:'s|:) {word}\b", unit)
        )
        seen.add(kind)
        assert peak == resident[kind], (unit, peak)
    assert seen == {"encode", "dit", "decode"}
