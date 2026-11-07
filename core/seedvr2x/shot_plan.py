"""The memory planner (DESIGN.md, Memory planner): one plan per job, made after the first pass,
whose shots are its input, and before the models load. No GPU here: what is free comes in as a
figure. Nor torch at import: the command line imports this module before its first refusals,
which don't wait for torch (tests/test_cli.py).

Its memory model, the bricks' (peaks.py; the constants named below are its): what each phase is
predicted to peak at on the GPU, in bytes. The DiT's and the VAE's passes by the fits of
research/docs/planner-limits.md and research/docs/vram.md, measured on numz's CLI with the 7B
fp16 (peaks.FP16), and the DiT of seedvr2x's int8 files and dynamic GGUF files by its own weights
and the fp16 fit's other terms, those of int8 provisional (peaks.INT8) and the GGUF's measured
(peaks.GGUF_7B, its buffers peaks.GGUF_BUFFERS on the RTX 5090, 2026-11-04; the job's dit_model); a
tile by the peak measured on seedvr2x's own tiled passes (MEASURED), the fits serving the tiles
cut off at the picture's edges at a size nobody measured. No VAE pass keeps anything of the shot
on the GPU (shot_pass.py, seedvr2x/tiling.py), and none is counted by its shot's length. Nor is
the DiT's phase, judged by its longest window alone: it keeps nothing of a shot beside the
window it runs, the shot's noise drawn in system memory (shot_pass.sample_windows; DESIGN.md,
Pipeline step 2). Where the weights are in each phase, and what that takes, by what
seedvr2x/blockswap.py does with them, measured on seedvr2x itself (milestone 3). The tiles and
their scratch by seedvr2x/tilegrid.py's, the arithmetic the tiled passes run on. Of system memory:
the pinned copies of what leaves the GPU, what the run holds there of a shot at the moment it
holds the most, the job's shot that holds the most, by the tensors' sizes (host_moments), and
what the scratch may take of what is left (Plan.scratch_allowance). Sizes are (height, width)
in pixels; `padded` is the frame the model takes, the target padded (frames.padded_size).

Its choices, in DESIGN.md's order (plan): the window, the biggest quality lever; the fewest
swapped blocks that hold it; the largest VAE tiles, none where the untiled pass fits; and where
the plan so made takes more system memory than the host has, the VAE's weights kept on the GPU
for the DiT's phase, then the DiT kept there for the VAE's phases and their tiles shrunk, then the
longest window whose plan fits both. A pin is taken exactly or refused, the refusal naming what
fits; a size that doesn't fit is refused, never degraded (PlanError).

Four corrections come from the planner's first two matrices on seedvr2x, emulated cards
(tests/test_shot_plan.py, test_first_matrix). Two are DESIGN.md's: no tile chosen above TILE_MAX
(Memory planner, What follows from them), and every VAE pass held to a room under the budget
besides its peak (VAE_ROOM), judged by the frames against a run with room to spare (Memory
planner, Budget; To measure, The VAE's room). PROVISIONAL (implementation, 2026-10-11), the two
others: ENCODE_EXTRA on every encode, until the encode is refitted on seedvr2x, and split's term
off a tiled decode's peak (decode_peak)."""

import logging
import math
import mmap
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from typing import Any, Literal

from seedvr2x import memory, peaks, tilegrid
from seedvr2x.errors import (
    CompileRefused,
    HostShort,
    MemoryShort,
    PinRefused,
    PlanRefused,
    Refusal,
    Say,
    Setting,
    Shortfall,
    plain,
)
from seedvr2x.frames import check_target, output_size, padded_length, padded_size
from seedvr2x.memory import Reading
from seedvr2x.passes import Phase, Plan, Tiles
from seedvr2x.peaks import MARGIN, VAE_ROOM
from seedvr2x.shots import SHARED, Shot, window_layout
from seedvr2x.weights import Architecture

logger = logging.getLogger("seedvr2x.shot_plan")


# A latent as the VAE makes it and the DiT takes it: its channels, the pixels of the padded
# frame along a side of one of its positions (vendor/models/video_vae_v3/
# s8_c16_t4_inflation_sd3.yaml: latent_channels 16, spatial_downsample_factor 8), and the bytes
# of one of its values, in the pipeline's bfloat16 (model.COMPUTE_DTYPE; tests/test_shot_plan.py
# holds the three to tensors of that shape). What host_moments counts a shot's latent and its
# noise by in system memory, their sizes: literals here since seedvr2x/model.py imports torch.
LATENT_CHANNELS = 16


LATENT_SIDE = 8


LATENT_VALUE = 2


# The process's resident system memory during the shots, which stays flat: 2.3-2.4 GiB over 6
# shots (DESIGN.md, Memory planner, Host RAM), short ones: what the run holds of a long shot
# there is counted apart (host_moments).
HOST_RESIDENT = round(2.4 * peaks.GIB)


# What the check of a tensor's values for NaN and inf holds in system memory over a tensor it
# reads there, an encode's latent (shot_pass.finite: torch.isfinite, then all), a latent of it at a
# time: that latent's absolute values, as large as it, and CHECK_MASKS masks of a byte a value,
# the two tests' and their product. For bfloat16, 2.5 times the latent's bytes, by torch's
# memory profiler on shots of 40 and of 200 latents (torch 2.14.1, 2026-10-12;
# tests/test_shot_pass.py measures it).
CHECK_MASKS = 3


# What a join of two windows holds in system memory while their shared latents are mixed
# (shot_pass._merge), in tensors as large as those latents: the two products and their sum, given up
# before the next join (torch's memory profiler, 2026-10-12; tests/test_shot_pass.py measures it).
MIXING = 3


# The most a scratch may take of system memory when what the host has available can't be read
# (main._host_available), in bytes. PROVISIONAL (implementation, 2026-10-11): DESIGN.md sets the
# allowance from the host's memory, and gives no figure without it: 8 GiB, which holds 675 frames
# at 1080p and 170 at 4K, what every scratch was allowed before the planner set it.
ALLOWANCE_UNKNOWN_HOST = 8 * peaks.GIB


# The refusals of a plan are errors.py's, by kind (PlanRefused and its subclasses); PlanError is
# the name the command line and the tests catch them by, and Shortfall the parts of a resume's.
PlanError = PlanRefused


@dataclass(frozen=True)
class Pins:
    """The choices the command line fixes (DESIGN.md, Memory planner, Pins), each taken exactly or
    refused, never adjusted; None leaves a choice to the planner."""

    window: int | None = None  # --window: the cap on the DiT windows, in latents
    encode_tile: int | None = None  # --encode-tile: a tile's side in output pixels; 0: untiled
    decode_tile: int | None = None  # --decode-tile, likewise
    swap_blocks: int | None = None  # --swap-blocks: the DiT's blocks swapped


# A shot's moments in system memory, in its order (host_moments).
MOMENTS = ("encode", "check", "windows", "merge", "decode")


def host_moments(
    padded: tuple[int, int], latents: int, cap: int | None, in_memory: bool = False
) -> dict[str, int]:
    """What the run holds in system memory of a shot of `latents` latents of padded frames, in
    bytes, at each of its moments (MOMENTS), its windows capped at `cap` latents (None: one
    window). The GPU keeps nothing of a shot between its units (units.py,
    shot_pass.sample_windows): it waits here, and its noise is drawn here.

    in_memory: the shot's windows are kept in system memory until its decode, none written to a
    file (clip.py, the library's entry for a clip of frames in memory): at the last window the
    latent and the noise are there beside all the windows, not the longest alone; and the latent
    is still alive through the merge (the clip's upscale holds it until its windows are merged),
    which the command line's run has given up by then. The default, False, is the command line's
    run.

    By the tensors' sizes: a latent is (H / 8, W / 8, 16) in bfloat16, half a byte a padded
    pixel, 1.01 MiB at 1080p, padded to 1104x1920, and 4.01 MiB at 4K, padded to 2192x3840. The
    moments, counted in latents, a shot of L in n windows, which share SHARED latents with each
    neighbour, L + SHARED x (n - 1) latents in all:
    - "encode", the encode's end: the latent and the buffer it is made from, as large, the
      posterior's mode or a tiled encode's sums (model.to_latent): 2 L;
    - "check", the latent checked for NaN and inf where the encode left it (run._sample,
      shot_pass.finite), a latent at a time: the latent, and of the one being read its absolute
      values and CHECK_MASKS masks of a byte a value: L + 2.5;
    - "windows", from the noise's draw to the last window: the latent and the noise, drawn
      beside it in place (2 L), and each window's copy as it leaves the GPU, gone once written
      to its file (units.Units.save_window), so the longest window's there;
    - "merge", the decode's start: the windows, read back from their files, and what merging
      them makes (shot_pass.merge_windows): nothing for one window, which is the merged latent; else
      the merged latent, allocated once (L), and at a join the two products and the sum of the
      latents mixed (MIXING x SHARED), given up before the next join; with in_memory the latent
      too, alive beside them (clip.upscale): L more;
    - "decode", through the decode: the merged latent, L.
    The latent and the noise are gone before the merge (run._sample; in_memory: the noise
    alone), and the windows after it: the moments follow one another, and a shot holds the
    largest, never their sum. One shot at a time: a segment's shots are sampled, then decoded,
    each one's tensors given up before the next one's (run.run_job).

    The largest is the windows', 3 L for a shot in one window, 1.5 bytes a padded pixel and a
    latent; the check for a shot of 1 latent; and the merge where a shot's n windows are
    shorter than 2 n + 4 latents, sharing more latents between them than one of them holds: a
    long shot, from 9 windows on under a cap of 20 latents.

    The moments on the CPU alone are measured, by torch's memory profiler (tests/test_shot_pass.py,
    2026-10-12): the check, 2.5 times one latent over the shot's; the merge, this count to the
    few bytes of the mix's weights, for shots of 13 to 100 latents in 1 to 33 windows; the
    encode's end, the latent's memory beside its buffer; a window's file is written from its
    copy and read into one tensor, nothing more allocated. The others move tensors between the
    GPU and system memory, read from the code: a copy each (sampling.moved).

    Not here: a tiled pass's scratch, which the plan allows apart (Plan.scratch_allowance). It
    is open through a tiled encode's tiles, beside the sums alone (L), and through a tiled
    decode, beside the merged latent alone (L): never at the moment a shot holds the most."""
    height, width = padded
    latent = LATENT_CHANNELS * LATENT_VALUE * (height // LATENT_SIDE) * (width // LATENT_SIDE)
    if cap is None or cap >= latents:
        windows = [latents]
    else:
        windows = [end - start for start, end in window_layout(latents, cap)]
    kept = sum(windows)
    merging = latents + MIXING * SHARED if len(windows) > 1 else 0
    shot = latent * latents
    return {
        "encode": 2 * shot,
        "check": shot + latent + CHECK_MASKS * latent // LATENT_VALUE,
        "windows": 2 * shot + latent * (kept if in_memory else max(windows)),
        "merge": latent * (kept + merging + (latents if in_memory else 0)),
        "decode": shot,
    }


# The phases, in a shot's order.
PHASES: tuple[Phase, ...] = ("encode", "dit", "decode")


@dataclass(frozen=True)
class Shortened:
    """A plan whose window the planner shortened for the host's memory (plan): the cap the card
    alone would have given, in latents, what the plan under it takes of system memory and what
    is available, in hundredths of a GiB, as its refusal would have said them (_unfit)."""

    window: int
    taken: int
    available: int


@dataclass(frozen=True)
class Shrunk:
    """A plan whose VAE tiles the planner shrank for the host's memory, the DiT kept on the GPU
    in the VAE's phases (_dit_kept): the encode's and the decode's tiles the card alone would
    have given at its window (None: untiled), what the plan with them takes of system memory and
    what is available, in hundredths of a GiB, as its refusal would have said them (_unfit)."""

    encode: Tiles | None
    decode: Tiles | None
    taken: int
    available: int


@dataclass(frozen=True)
class Peaks:
    """What a plan is predicted to take, in bytes where nothing else is said: which phase bounds
    it, and by how much, is what --plan prints and the manifest keeps for information."""

    free: int  # the memory the plan was made in: what it may count its peaks in (Reading.counted)
    budget: int  # free - MARGIN; free itself on a run held to the card (_Job.margin)
    encode: int  # each phase's predicted peak on the GPU
    dit: int  # the longest window's (dit_peak)
    decode: int
    host: int  # the system memory the plan takes, a shot's (`shot`) and the scratch's in it
    scratch: int  # the scratch of the longest shot's tiled passes; 0 untiled
    window: int  # the longest window the job really runs, in latents
    tokens: int  # the DiT's tokens per latent
    # How `free` was read: what the driver reported free to the process, what the whole card had
    # free where that was read, and whether the run is held to the card (seedvr2x/memory.py).
    reading: Reading
    attention: str  # the attention backend the DiT's peak is the fit of (DIT_ATTENTION)
    # What the run holds of a shot in system memory at the moment it holds the most, and that
    # moment, one of MOMENTS (host_moments): the job's shot that holds the most, of `holder`
    # latents, the longest but where a shorter one in fewer windows holds more (_shot_held);
    # counted in `host`.
    shot: int
    moment: str
    holder: int
    # Where the planner took a shorter window than the card alone holds, for the host's memory:
    # what the card alone would have given, and why not (plan); else None.
    shortened: Shortened | None = None
    # Where the planner shrank the VAE's tiles at that window for the host's memory, the DiT
    # kept on the GPU in their phases: the tiles the card alone would have given, and why not
    # (_dit_kept); else None.
    shrunk: Shrunk | None = None
    # The DiT's file's format the peaks were predicted for (peaks.DITS), which the plan's lines
    # say and the record keeps by its name (DIT_FORMATS), for information: the job's record
    # names the file itself (settings.dit_model), which a resume compares.
    dit_model: Architecture = peaks.FP16
    # What the planner leaves each phase beyond its predicted peak, in bytes (peaks.reserve, by
    # the step's P and the allocator the reading names): the default is the plan of a driver that
    # refuses, nothing beyond a window's peak and VAE_ROOM beyond a VAE pass's.
    # Not in the repr: a plan under torch's own allocator without the segments prints as it did.
    reserves: dict[Phase, int] = field(
        default_factory=lambda: {"encode": VAE_ROOM, "dit": 0, "decode": VAE_ROOM}, repr=False
    )

    def peak(self, phase: Phase) -> int:
        """The predicted peak of `phase`."""
        return {"encode": self.encode, "dit": self.dit, "decode": self.decode}[phase]

    def need(self, phase: Phase) -> int:
        """What the planner judges `phase` by against the budget: its predicted peak and its
        reserve (`reserves`): under torch's own allocator without the segments and on a driver
        that refuses, nothing for a window and VAE_ROOM for a VAE pass, as before the reserve was
        measured by the step."""
        return self.peak(phase) + self.reserves[phase]

    def reserve_taken(self) -> dict[str, int] | None:
        """The reserve a held run's plan leaves torch beyond each phase's peak, in bytes, as
        the manifest keeps it and `memory.hold` takes it: the step's own (`reserves`) under the
        segments and cudaMallocAsync, the flat memory.RESERVE for every phase under torch's own
        allocator without the segments (the figure counted off the plan's memory); None on a
        driver that refuses, which leaves none."""
        if not self.reading.held:
            return None
        if self.reading.allocator == memory.PLAIN:
            return dict.fromkeys(PHASES, memory.RESERVE)
        return {phase: each for phase, each in self.reserves.items()}

    @property
    def bound(self) -> Phase:
        """The phase that bounds the plan: the one with the least left under the budget, the
        VAE's phases counted with their room (need), as the planner judged them."""
        return max(PHASES, key=self.need)

    def record(self) -> dict[str, Any]:
        """What the manifest keeps of them beside where the weights are (Plan.record's memory),
        for information, never compared (resume.UNCOMPARED): bytes, the longest window in
        latents. `free` is the memory the plan was made in; beside it, what the driver reported
        free to the process (driver_free, the same figure on a driver that refuses) and what the
        whole card had free (card_free; null where it wasn't read), the attention backend the
        DiT's peak was predicted for, and the format of the DiT's file the peaks were predicted
        for (dit_format, DIT_FORMATS' name, read back by planner.recorded_dit_model). Of `host`, the
        system memory the plan takes, host_shot
        is one shot's part, the shot that holds the most (Peaks.shot). PROVISIONAL
        (implementation, 2026-10-10 and 12): the fields' names; and a window shortened or tiles
        shrunk for the host's memory (Peaks.shortened, Peaks.shrunk) aren't recorded: the plan's
        lines say them."""
        return {
            "free": self.free,
            "driver_free": self.reading.reported,
            "card_free": self.reading.card,
            # What a held run's plan leaves torch's allocator beyond each step's peak, by phase
            # (reserve_taken; until 2026-11-06 one number); null on a driver that refuses, which
            # leaves none.
            "reserve": self.reserve_taken(),
            "attention": self.attention,
            "dit_format": _format_name(self.dit_model),
            "budget": self.budget,
            "encode_peak": self.encode,
            "dit_peak": self.dit,
            "decode_peak": self.decode,
            "bound": self.bound,
            "host": self.host,
            "host_shot": self.shot,
            "scratch": self.scratch,
            "longest_window": self.window,
        }


# The DiT's formats by the name a plan's record gives them (Peaks.record's dit_format), peaks.DITS:
# fp16's, int8's and the two dynamic GGUF files', the regular 7B's and the sharp one's (DESIGN.md,
# Weights, "The dynamic GGUF before v0.1.0 too").
DIT_FORMATS: dict[str, Architecture] = {
    "fp16": peaks.FP16,
    "int8": peaks.INT8,
    "gguf_dyn": peaks.GGUF_7B,
    "gguf_sharp_dyn": peaks.GGUF_7B_SHARP,
}


def _format_name(dit_model: Architecture) -> str:
    """The name of dit_model's format in a plan's record (DIT_FORMATS)."""
    return next(name for name, known in DIT_FORMATS.items() if known == dit_model)


@dataclass(frozen=True)
class _Job:
    """What the planner counts a job by: the memory it may count on, its size, its shots'
    lengths, the attention backend its DiT runs on, and its DiT's file's format."""

    free: int  # bytes: what the plan may count its peaks in (reading.counted)
    target: tuple[int, int]  # frames.target_size
    lengths: tuple[int, ...]  # its shots' lengths in latents, each once, ascending
    frames: int  # its longest shot's frames
    split: bool
    reading: Reading  # how `free` was read, for the plan's words and its record
    attention: str  # DIT_ATTENTION's
    compile_vae: bool = False  # the VAE compiled (--compile): its passes at COMPILE_PEAK
    # The DiT's file's format, whose figures its phase and its weights' places are counted by
    # (peaks.DITS): fp16's, int8's or a dynamic GGUF file's.
    dit_model: Architecture = peaks.FP16
    # The weights' pinned copies exist already (plan's held): none is counted against the host.
    held: bool = False
    # The shot's windows and a tiled pass's scratch are kept in system memory whole, never in a
    # file (plan's in_memory).
    in_memory: bool = False
    # The plan's last rungs (plan's `tail_frames` and the room's), which only a plan refused
    # without them takes: the decoder's tail a frame at a time (peaks.decode_peak's `tail`), and
    # the floor of a VAE pass's reserve (peaks.reserve's `vae_floor`), None: the rule's own.
    tail: bool = False
    vae_floor: int | None = None

    @property
    def margin(self) -> int:
        """What the plan leaves of the memory it may count on, in bytes: MARGIN on a driver that
        refuses; nothing on a run held to the card, where the same need is inside the
        figure counted (memory.RESERVE, taken off already, under torch's own allocator without
        the segments: DESIGN.md, Allocator, "the planner's 0.6 GiB margin is counted inside the
        4.5") or inside a window's reserve, whose floor it is (peaks.DIT_RESERVE_FLOOR): the
        budget is then what the plan counts on."""
        return 0 if self.reading.held else MARGIN

    @property
    def budget(self) -> int:
        return self.free - self.margin

    @property
    def padded(self) -> tuple[int, int]:
        return padded_size(self.target)

    @property
    def output(self) -> tuple[int, int]:
        return output_size(self.target)

    @property
    def smallest(self) -> int:
        """The smallest cap the job can run: no cap goes under 2 * SHARED + 1 latents, a window
        in the middle of a shot keeping one of its own (DESIGN.md, Terms, Window); the longest
        shot's length when every shot is shorter, each then one window whatever the cap."""
        return min(2 * SHARED + 1, self.lengths[-1])


def plan(
    free: int | Reading,
    target: tuple[int, int],
    shots: Sequence[Shot],
    pins: Pins,
    host_available: int | None,
    *,
    split: bool = False,
    recorded: Plan | None = None,
    attention: str = peaks.FLASH_ATTN_2,
    compile_vae: bool = False,
    compile_fixed: bool = False,
    dit_model: Architecture = peaks.FP16,
    held: bool = False,
    in_memory: bool = False,
    tail_frames: bool | None = None,
    rungs: bool = True,
) -> tuple[Plan, Peaks]:
    """The plan of a job of `shots` resized to target (frames.target_size), in the GPU memory it may
    count on, less its margin, and what it is predicted to take; refused (PlanError) when a pin
    can't be taken exactly, when no plan fits that memory, or when the plan takes more system
    memory than host_available, in bytes (None: not checked). split: whether the decode corrects
    colours. An output directory and a .mkv
    path plan alike: both jobs keep their units in a directory (units.py).

    `free` is that memory: a number of bytes, what a driver that refuses reports free; or the
    reading the run made of it (memory.Reading), whose `counted` the plan is made in, and whose
    other figures only word the plan's first line, its refusals and its record: on a run held
    to the card they name what the driver reports, what the whole card has free where it was
    read, and what the plan counts on. The margin is MARGIN on a driver that refuses, and none
    on a run held to the card, whose `counted` is under torch's limit by what measures the same
    need already (_Job.margin): the budget is then all the plan counts on. attention: the
    backend the run's DiT takes (model.attention_backend), whose fit the DiT's peak is
    predicted by (DIT_ATTENTION).

    The DiT's phase is judged by the job's longest window alone (dit_peak; _dit_need): nothing
    of a shot is on the GPU beside the window it runs, its noise drawn in system memory
    (shot_pass.sample_windows), so that no shot's length bounds the phase. Until 2026-10-16 the
    noise was drawn on the GPU, the shot's latent and noise whole there before its first window, and
    the phase was judged by the larger of that draw and the window, which bounded a shot's
    length on every card.

    A tile pinned that covers the whole padded frame is no tiling, and is refused, naming 0, the
    untiled pass (DESIGN.md, Pipeline, per shot, step 3), whatever the memory.

    The choices, each made once for the job (DESIGN.md, Memory planner, Choices, in order):
    1. the window cap, the longest shot's length when the DiT's phase holds it, with as many
       blocks swapped as it takes and the VAE's weights away, else the largest cap whose longest
       window it holds: what it must hold is the longest window the job really runs, a shot of
       80 latents under a cap of 76 running two of 41; the plan's cap is then that longest
       window, which lays the shots out the same (_window);
    2. the fewest swapped blocks that hold that window, the VAE leaving the DiT's phase before
       one more block is swapped; but where the plan so made takes more system memory than
       host_available, more blocks swapped and the VAE staying, when that takes less of it
       (_vae_kept);
    3. the VAE's tiles, the decode's then the encode's: none where the untiled pass fits, else
       the largest that fit, TILE_MAX at most, each pass with its room (VAE_ROOM), the DiT
       leaving the phase before a tile shrinks; then as much of it left there as fits
       (_resident); but where the plan so made, the VAE's weights kept or not, takes more system
       memory than host_available, the DiT kept on the GPU in the VAE's phases, a block more at
       a time, and their tiles shrunk to keep it, the largest that fit the host too (_dit_kept;
       design's decision of 2026-10-13); the plan's lines say which tiles the card alone gives
       and why smaller ones are taken (Peaks.shrunk). Never a pinned pass's tiles, nor a
       recorded plan's, which shape the output.
    Then the system memory: what the run holds there of a shot at the moment it holds the most
    (_shot_held), counted with the pinned copies, and what a tiled shot's scratch may take of
    what host_available has left once both are counted (_allowance).

    Where the plan so made fits the card and takes more system memory than host_available, its
    tiles shrunk as far as they go, the planner takes by itself the longest window whose plan
    fits both, the other choices made again under it (_window_that_runs), and the plan's lines
    say which window the card alone would have given and why a shorter one was taken
    (Peaks.shortened; DESIGN.md, Memory planner, Choices: design's decision of 2026-10-12): a
    long window may take swapped blocks, each pinned in system memory, where a shorter one takes
    none. Never under --window, a pin being taken exactly or refused, its refusal naming that
    window; nor for a recorded plan, whose window shapes the output. A job is refused for the
    system memory only where no window's plan fits both, its tiles shrunk as far as they go.

    recorded is the plan of a job resumed that has made units: what shapes the output is taken
    from it as it is, the cap, the tiles and the compile flags, and where the weights are and
    the scratch's allowance planned again around it (DESIGN.md, Memory planner, A plan shapes
    the output a little); one with tiles of which one covers the frame is no plan's, refused
    (planner.check_recorded). Where the weights are is planned for the backend given, a resume's
    too: FlashAttention installed since, or gone, the recorded shape is held to today's fit.

    On a run held to the card each phase is judged by its predicted peak and its own reserve
    beyond it (peaks.reserve, by the step's P and the allocator the reading names; Peaks.reserves;
    DESIGN.md, Allocator, Small cards, decided on 2026-11-06): the reserve replaces the flat
    memory.RESERVE and the 2.5 GiB retry of before it, which retired under the segments. Under
    torch's own allocator without them (memory.PLAIN) the flat memory.RESERVE stays off the
    figure, and a job refused there is refused, saying where the segments' default would run it.
    A driver that refuses leaves MARGIN and VAE_ROOM, as before.

    compile_vae: --compile, the VAE compiled (DESIGN.md, Memory planner, "torch.compile: the
    VAE, opt-in, in v0.1.0"): each VAE pass is held to COMPILE_PEAK times its tile's plain peak
    (encode_peak), so that the tiles are chosen smaller where it doesn't fit, and only passes
    within the shapes measured compiled are made or taken (in_compile_range): the encode's tiles
    up to COMPILE_ENCODE_TILE_MAX px or the 1080p frame whole, the decode's up to
    COMPILE_DECODE_TILE_MAX px. Where a pin gives a pass beyond them, or no compiled plan fits
    where a plain one would, the job is refused, never run plain in its place (DESIGN.md, same
    section, decided on 2026-10-29): the refusal names the pass and the shapes measured and
    advises, each on a line of its own, the same command without --compile and, for a pin, the
    pin within them (_compile_refusal). A job no plain plan fits is refused as it was, whatever
    --compile asked. The plan's compile_vae is the setting's, always. A recorded plan keeps its
    own compile_vae, whatever is asked (the units made are what they are), and is refused as
    any recorded shape; compile_fixed is the same for a plan planned again around a recorded
    shape (planner.plan_again), and for the compiled plan itself, whose refusal is
    _compile_refusal's.

    held: the weights' pinned copies exist already, the models kept in system memory between
    runs (model.Models.place, passes.HELD; DESIGN.md, Repository and apps, the nodes): the plan
    counts none of them against host_available, the pinned part of the host count (_pinned) is
    0, and the choices made only to pin less of it are never taken: neither the VAE kept on the
    GPU for the DiT's phase (_vae_kept) nor the DiT kept there in the VAE's phases with tiles
    shrunk (_dit_kept), whatever the host leaves. A host short of the process's own memory, a
    shot's or a scratch's still refuses the plan or shortens its window. Not a word of the
    plan's record: the weights' place is no part of what a plan shapes. The default, False, plans
    as before.

    in_memory: the run keeps a shot's windows and a tiled pass's scratch in system memory, whole,
    where the command line's go to files (clip.py): the windows counted all together at the
    DiT's phase's end (host_moments), and the scratch's allowance the whole scratch
    (_allowance), so that a host that can't hold it refuses the plan, or shortens its window,
    as any host short of memory does. Not a word of the plan's record. The default, False, plans
    as before.

    The floor, last rungs (DESIGN.md, Allocator, Small cards), for a clip that fits no other way:
    a plan refused for the memory of a VAE pass is made again, by the pass refused (_ladder): the
    decode, (1) with the decoder's tail a frame at a time (Plan.tail_frames; peaks.decode_peak's
    `tail`), then (2) with that and the VAE passes' reserve floor down to peaks.VAE_ROOM_LEAST
    (Plan.vae_floor), under expandable segments alone; the encode, the room alone (never the tail,
    which lowers the decode alone). The first plan that fits is the plan, and a job that fits
    without them never takes one. Where no
    rung fits it the refusal is the last rung's, whose figures are the least the plan takes, said
    with the rungs it counted (_rungs_counted). `tail_frames`: None, the ladder above; True, the
    tail taken (a job resumed whose recorded plan has it, which is what shapes its output:
    `recorded` says it too); False, never the tail, the room alone as a rung (a resumed job
    whose plan has not the tail). No rung under --compile: a compiled decode with the tail is
    not measured. `rungs`: False, no rung at all (the plain plan a compiled job's refusal is
    judged by first, _compile_refusal: a job no plain plan fits, rungs or not, is refused as it
    was before the rungs; one that the plain plan fits only with a rung is refused saying that it
    runs without --compile, with that rung, _rung_refusal)."""
    if recorded is not None:
        compile_vae, compile_fixed = recorded.compile_vae, True
        tail_frames = recorded.tail_frames
    if compile_vae and not compile_fixed:
        try:
            return plan(
                free,
                target,
                shots,
                pins,
                host_available,
                split=split,
                recorded=None,
                attention=attention,
                compile_vae=True,
                compile_fixed=True,
                dit_model=dit_model,
                held=held,
                in_memory=in_memory,
                tail_frames=tail_frames,
            )
        except PlanError as refused:
            raise _compile_refusal(
                refused,
                (free, target, shots, pins, host_available),
                dict(
                    split=split,
                    attention=attention,
                    dit_model=dit_model,
                    held=held,
                    in_memory=in_memory,
                ),
            ) from None
    job = _job(free, target, shots, split, attention, compile_vae, dit_model, held, in_memory)
    if pins.swap_blocks is not None and pins.swap_blocks > peaks.BLOCKS:
        swap = Setting("swap_blocks", str(pins.swap_blocks))
        raise PinRefused(
            lambda say: (
                f"{say(swap)}: the DiT has {peaks.BLOCKS} blocks, so from 0 to {peaks.BLOCKS}"
            ),
            pin=swap,
            why="range",
        )
    covering: tuple[tuple[Phase, int | None], ...] = (
        ("encode", pins.encode_tile),
        ("decode", pins.decode_tile),
    )
    for phase, pin in covering:
        if pin and peaks.covers(job.padded, Tiles(pin)):
            raise _covering(phase, pin, job.padded)
    return _climbed(job, pins, recorded, host_available, tail_frames, rungs)


# A rung of the floor (plan): the decoder's tail a frame at a time, the VAE passes' reserve floor
# (None: the rule's own) and the phases whose refusal it is tried for.
_Rung = tuple[bool, int | None, frozenset[str]]
_ENCODE, _DECODE, _BOTH = (
    frozenset({"encode"}),
    frozenset({"decode"}),
    frozenset({"encode", "decode"}),
)


def _ladder(first: str, tail_fixed: bool | None, segments: bool) -> tuple[_Rung, ...]:
    """The rungs of the floor for a plan first refused in `first` ("encode" or "decode"), in order,
    each with the ones before it. The tail lowers the decode alone (small-cards.md, "The VAE decode
    at small tiles"), so a plan refused for the decode climbs the tail, then the tail with the room,
    and one refused for the encode takes the room alone and never the tail (the tail, which changes
    the decoder's frames at some tile sides, is taken for the decode alone); if that fits the
    encode and the decode then refuses, the tail follows, with the room. The room rung (the VAE
    passes' floor down to peaks.VAE_ROOM_LEAST) only under expandable segments (`segments`), where
    it was measured: a pass packed closer than about 1 GiB can give other frames and say nothing
    under the plain allocator, the pool and a driver that refuses (peaks.VAE_ROOM). `tail_fixed`:
    the plan resumed has the tail (True) or has not (False): the tail is not a rung then."""
    if tail_fixed is not None:
        return ((tail_fixed, peaks.VAE_ROOM_LEAST, _BOTH),) if segments else ()
    if first == "decode":
        ladder: tuple[_Rung, ...] = ((True, None, _DECODE),)
        if segments:
            ladder += ((True, peaks.VAE_ROOM_LEAST, _BOTH),)
        return ladder
    if not segments:
        return ()
    return (
        (False, peaks.VAE_ROOM_LEAST, _BOTH),
        (True, peaks.VAE_ROOM_LEAST, _DECODE),
    )


def _climbed(
    job: _Job,
    pins: Pins,
    recorded: Plan | None,
    host_available: int | None,
    tail_frames: bool | None,
    climb: bool = True,
) -> tuple[Plan, Peaks]:
    """plan's last rungs: `job` planned as it is (_attempt) and, refused for the memory of a VAE
    pass, again with each rung of the floor in turn (_ladder, by the phase refused); the last
    rung's refusal if none fits (the least the plan takes, said with the rungs it counted), or the
    refusal of a rung that the card holds and another limit refuses (the system memory). A job
    whose tail is fixed (`tail_frames` True: the plan resumed has it) starts with it and has the
    room alone left; one whose tail is fixed off (False) has the room alone. Only a pass is helped
    by them: a refusal for the system memory, a window or a pin's range is final, and so is any
    under --compile (plan)."""
    if tail_frames is True:
        job = replace(job, tail=True)
    segments = job.reading.held and job.reading.allocator == memory.SEGMENTS
    try:
        return _attempt(job, pins, recorded, host_available)
    except PlanRefused as first:
        if not climb or job.compile_vae or not _helped(first):
            raise
        last: tuple[PlanRefused, bool, int | None] | None = None
        refused = first
        for tail, floor, phases in _ladder(first.phase, tail_frames, segments):
            if refused.phase not in phases:
                break
            again = replace(job, tail=tail, vae_floor=floor)
            try:
                return _attempt(again, pins, recorded, host_available)
            except PlanRefused as refused_again:
                refused = refused_again
                if not _helped(refused):
                    # The card holds the pass with this rung: what refuses it now is another
                    # limit (the system memory, say), which is the refusal that matters.
                    raise
                last = (refused, tail, floor)
        if last is None:
            raise first from None
        # Every rung tried: the last attempt's refusal, whose figures are the least the plan
        # takes, said with the rungs it counted (a recorded plan's too: the same rule).
        refused, tail, floor = last
        counted = _rungs_counted(tail, floor)
        words = refused.words
        raise refused.reworded(lambda say: f"{words(say)}, {counted}") from None


def _rungs_counted(tail: bool, floor: int | None) -> str:
    """What the refusal of a plan with the last rungs says it counted: the decoder's tail a frame
    at a time (peaks.decode_peak's `tail`) and the VAE passes' room down to `floor` bytes, the
    least the plan takes."""
    if floor is None:
        return "with the decoder's tail a frame at a time, the least the plan takes"
    room = f"the VAE passes left {floor / peaks.GIB:.2f} GiB"
    if tail:
        return f"with the decoder's tail a frame at a time and {room}, the least the plan takes"
    return f"with {room}, the least the plan takes"


def _rungs_named(tail: bool, floor: int | None) -> str:
    """The last rungs a plan takes, named as the command line's plan lines name them
    (planner._rungs_said): the decoder's tail, its last blocks a frame at a time (`tail`), and
    the VAE's least room, the VAE passes' reserve floor lowered to `floor` bytes (None: the
    rule's own)."""
    named = ["the decoder's tail (its last blocks a frame at a time)"] if tail else []
    if floor is not None:
        named.append(
            "the VAE's least room (the VAE passes' reserve floor lowered to"
            f" {floor / peaks.GIB:.2f} GiB)"
        )
    return " and ".join(named)


def _helped(refused: PlanRefused) -> bool:
    """Whether a rung of the floor can help the refusal: a VAE pass short of memory, a plan's
    own (MemoryShort) or a pin's (PinRefused, why "memory")."""
    if refused.phase not in ("encode", "decode"):
        return False
    return isinstance(refused, MemoryShort) or (
        isinstance(refused, PinRefused) and refused.why == "memory"
    )


def _attempt(
    job: _Job, pins: Pins, recorded: Plan | None, host_available: int | None
) -> tuple[Plan, Peaks]:
    """plan's work on `job` as it is, torch's own allocator without the segments told where the
    segments would run it."""
    try:
        return _planned(job, pins, recorded, host_available)
    except PlanRefused as refused:
        if not job.reading.held or job.reading.allocator != memory.PLAIN:
            raise
        # A user's allocator without the segments refuses a step short of room: said where torch's
        # default, with them, would run the job, whose reserve is the step's (peaks.reserve). The
        # reading is turned into that allocator's with what PLAIN took off the figure given back.
        if job.reading.counted <= 0:
            raise
        words = refused.words
        segments = replace(
            job.reading,
            allocator=memory.SEGMENTS,
            counted=job.reading.counted + memory.RESERVE,
        )
        try:
            _planned(
                replace(job, free=segments.counted, reading=segments),
                pins,
                recorded,
                host_available,
            )
        except PlanRefused:
            raise refused from None
        raise refused.reworded(
            lambda say: (
                f"{words(say)}; torch's own allocator with its expandable segments, a held run's"
                " default, which the allocator's setting in the environment replaces here, would"
                " run it"
            )
        ) from None


def _covering(phase: Phase, pin: int, padded: tuple[int, int]) -> PinRefused:
    """The refusal of a tile pinned that covers the whole padded frame, which is no tiling. The
    way out is always the untiled pass, 0: the pin's `why` says it."""
    height, width = padded
    asked, untiled = Setting(f"{phase}_tile", str(pin)), Setting(f"{phase}_tile", "0")
    return PinRefused(
        lambda say: (
            f"{say(asked)}: a tile of {pin} px covers the whole padded frame,"
            f" {width}x{height}: {say(untiled)} pins an untiled {phase}"
        ),
        pin=asked,
        why="covers",
    )


def _compile_refusal(
    refused: PlanRefused,
    job: tuple[int | Reading, tuple[int, int], Sequence[Shot], Pins, int | None],
    options: dict[str, Any],
) -> PlanRefused:
    """The refusal of a job --compile can't run: the compiled plan's `refused`, where a plain
    plan of the same job fits, with the shapes measured and the advice, each on a line of its own
    (DESIGN.md, Memory planner, "torch.compile: the VAE, opt-in", decided on 2026-10-29): the same
    command without --compile and, for a tile pinned, the pin within the shapes that the planner
    takes (the untiled pass if it is within them, else the largest tile that fits). `job` is plan's
    free, target, shots, pins and host_available, `options` its split, attention, dit_model,
    held and in_memory.
    A job no plain plan fits is refused as it was, whatever --compile asked: the plain plan's
    refusal. A job the plain plan fits only with a last rung of the floor (the decoder's tail, the
    VAE's least room; no rung is taken compiled, unmeasured there), its compiled plan refused for
    a VAE pass's memory, is refused too, saying so: the compiled plan's sentence, a line saying
    that --compile takes no rung, then that the same command without --compile runs, with the
    rungs that plan takes (_rung_refusal; DESIGN.md, Allocator, Small cards, decided on
    2026-11-06).

    PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md says the refusal "names the pass and the shapes
    measured" and doesn't word the lines): the compiled plan's own sentence, which names the
    pass and its tiles, a line for each other pass pinned beyond the shapes (the plan stops at
    the first it meets), then the shapes measured where none says them, then the two pieces of
    advice."""
    free, target, shots, pins, host_available = job
    try:
        plan(*job[:5], recorded=None, compile_vae=False, compile_fixed=True, rungs=False, **options)
    except PlanRefused as plain_refusal:
        # A rung helps a VAE pass short of memory (_helped): a compiled refusal for anything else,
        # a pin beyond the shapes measured among them, keeps the plain plan's refusal, as before.
        if not _helped(refused):
            return plain_refusal
        try:
            climbed, _ = plan(
                *job[:5], recorded=None, compile_vae=False, compile_fixed=True, **options
            )
        except (PlanRefused, ValueError) as climbing:
            # The plain plan with its rungs refused too, or a pin it rejects once memory is no
            # longer what refuses it (a window too short to share): the refusal before the rungs.
            logger.debug("the plain plan with the floor's rungs refused too: %s", climbing)
            return plain_refusal
        return _rung_refusal(refused, climbed)
    padded = _job(free, target, shots, **options, compile_vae=True).padded
    lines: list[Callable[[Say], str]] = [refused.words]
    # The plan refuses at the first pass it meets: each other pass pinned beyond the shapes is
    # named too, in the same words, so that the advice below is no surprise for the second.
    others: tuple[tuple[Phase, int | None], ...] = (
        ("encode", pins.encode_tile),
        ("decode", pins.decode_tile),
    )
    for phase, pin in others:
        if pin is not None and phase != refused.phase:
            tiles = _pin_tiles(pin)
            if not peaks.in_compile_range(padded, phase, tiles):
                lines.append(
                    lambda say, phase=phase, pin=pin, tiles=tiles: (
                        f"{say(Setting(f'{phase}_tile', str(pin)))}: the VAE compiled has the"
                        f" {phase} {_tiles_said(tiles)} beyond the shapes measured"
                        f" ({peaks.compile_range_said(phase)})"
                    )
                )
    named = ("encode", "decode") if refused.phase not in ("encode", "decode") else (refused.phase,)
    said = "\n".join(line(plain) for line in lines)
    if not all(peaks.compile_range_said(phase) in said for phase in named):
        lines.append(
            lambda say: (
                f"{say(Setting('compile'))} compiles only the shapes measured: "
                + "; ".join(peaks.compile_range_said(phase) for phase in ("encode", "decode"))
            )
        )
    lines.append(lambda say: f"The same command without {say(Setting('compile'))} fits")
    pinned: dict[Phase, int | None] = {"encode": pins.encode_tile, "decode": pins.decode_tile}
    # The pins refused: a tile beyond the shapes, or the pass the refusal names.
    bad: list[Phase] = [
        phase
        for phase, pin in pinned.items()
        if pin is not None
        and (phase == refused.phase or not peaks.in_compile_range(padded, phase, _pin_tiles(pin)))
    ]
    current: Pins = pins
    within: list[Setting] = []
    for index, phase in enumerate(bad):
        later = bad[index + 1 :]
        for candidate in _within(padded, phase):
            substituted = _with_tile(current, phase, candidate)
            try:
                plan(
                    free,
                    target,
                    shots,
                    substituted,
                    host_available,
                    recorded=None,
                    compile_vae=True,
                    compile_fixed=True,
                    **options,
                )
            except PlanRefused as again:
                # Another pin still beyond the shapes, to be replaced in its turn, refuses it.
                if again.phase not in later:
                    continue
            lines.append(
                lambda say, phase=phase, candidate=candidate: (
                    f"The same command with {say(Setting(f'{phase}_tile', str(candidate)))} fits,"
                    " within the shapes measured"
                )
            )
            within.append(Setting(f"{phase}_tile", str(candidate)))
            current = substituted
            break
    return CompileRefused(
        lambda say: "\n".join(line(say) for line in lines),
        phase=refused.phase,
        instead=refused.instead,
        within=tuple(within),
    )


def _rung_refusal(refused: PlanRefused, climbed: Plan) -> CompileRefused:
    """The refusal of a job --compile can't run where the plain plan fits only with a last rung of
    the floor (`climbed`: that plan, its tail_frames or vae_floor set): the compiled plan's own
    sentence (`refused`), a line saying that --compile takes none of the rungs (a compiled decode
    with the tail, or a VAE pass compiled in the lowered room, is not measured), then that the
    same command without --compile runs, with the rungs the plain plan takes (_rungs_named), each
    on a line of its own, as _compile_refusal words its advice. Only the rungs that plan takes
    are named: the VAE's least room is never taken under torch's own allocator without the
    segments, nor under cudaMallocAsync."""
    rungs = _rungs_named(climbed.tail_frames, climbed.vae_floor)

    def words(say: Say) -> str:
        compile_ = say(Setting("compile"))
        return "\n".join(
            (
                refused.words(say),
                f"{compile_} takes none of the floor's last rungs, not measured under {compile_}",
                f"The same command without {compile_} runs, with {rungs}",
            )
        )

    return CompileRefused(
        words,
        phase=refused.phase,
        instead=refused.instead,
        tail_frames=climbed.tail_frames,
        vae_floor=climbed.vae_floor,
    )


def _with_tile(pins: Pins, phase: Phase, pin: int) -> Pins:
    """`pins` with the tile pin of a VAE phase set."""
    return replace(pins, encode_tile=pin) if phase == "encode" else replace(pins, decode_tile=pin)


def _pin_tiles(pin: int) -> Tiles | None:
    """The tiles a tile pin gives: none for 0, the untiled pass."""
    return Tiles(pin) if pin else None


def _within(padded: tuple[int, int], phase: Phase) -> list[int]:
    """The tile pins within the shapes measured compiled (in_compile_range), in the planner's
    own order: 0, the untiled pass, if it is within them, then the tiles it makes (_tile_sizes),
    the largest first."""
    pins = [0, *_tile_sizes(padded)]
    return [pin for pin in pins if peaks.in_compile_range(padded, phase, _pin_tiles(pin))]


def _planned(
    job: _Job, pins: Pins, recorded: Plan | None, host_available: int | None
) -> tuple[Plan, Peaks]:
    """plan's work, once the pins are checked: the plan of `job`, in the memory its reading
    counts on, and its peaks; refused (PlanRefused) as plan says."""
    made, refused, shrunk = _made(job, pins, recorded, host_available)
    shortened = None
    window = None
    if refused is not None and shrunk is not None:
        # Its tiles shrunk as far as they go, which the figures said before are of.
        refused += _shrunk_said(job, made)
    if refused is not None and recorded is None:
        shorter = _window_that_runs(job, pins, made, host_available)
        if shorter is not None and pins.window is None:
            # The planner's own choice: the longest window whose plan fits the host too.
            shortened = _short_of_host(job, made, host_available)
            (_, made, shrunk), refused = shorter, None
        elif shorter is not None:
            window = Setting("window", str(shorter[0]))
    shortfall = None
    if refused is not None and recorded is not None:
        shortfall = _shortfall(job, made, host_available)
    if refused is not None:
        phase = _refusing(job, made, host_available)
        text = refused

        def message(say: Say) -> str:
            if window is not None:
                return f"{text}: the longest window whose plan fits it is {say(window)}"
            if recorded is None:
                return text
            return (
                f"the recorded plan no longer fits the memory free now: {text}"
                f"{_pin_s_part(job, made, pins, say)}"
            )

        taken = _peaks(job, made)
        if phase == "host":
            assert host_available is not None
            raise HostShort(
                message,
                need=taken.host,
                available=host_available,
                short=taken.host - host_available,
                instead=window,
                shortfall=shortfall,
            )
        assert phase
        need = int(taken.need(phase) + job.margin)
        raise MemoryShort(
            message,
            need=need,
            there=int(job.free),
            short=need - int(job.free),
            freeable=_can_free(job, need),
            reserve=_reserve_fact(job, taken.reserves[phase]),
            phase=phase,
            instead=window,
            shortfall=shortfall,
        )
    return made, replace(_peaks(job, made), shortened=shortened, shrunk=shrunk)


def _can_free(job: _Job, need: float) -> bool:
    """Whether memory freed on the card gives a plan the `need` bytes it is refused for, as
    Shortfall.freeable says it: a driver that refuses reports what other programs leave, closing
    them gives it back; an early reading, held or not, is judged by what they hold by NVML
    (_freeable)."""
    return (not job.reading.held and job.reading.nvml is None) or _freeable(job, need)


def _made(
    job: _Job, pins: Pins, recorded: Plan | None, host_available: int | None
) -> tuple[Plan, str | None, Shrunk | None]:
    """The job's plan around its pins, or around a recorded shape (plan), with its scratch's
    allowance; why it doesn't fit the memory, None when it does (_unfit); and where its tiles
    were shrunk for the host's memory, the DiT kept on the GPU in the VAE's phases, the tiles
    the card alone gives (_dit_kept), else None. PlanError for a pin that can't be taken, or a
    size no plan fits (_chosen, _around)."""
    if recorded is None:
        made = _chosen(job, pins)
    else:
        _check_tiles(job.padded, recorded)
        made = _around(job, recorded, pins)
    made = replace(made, scratch_allowance=_allowance(job, made, host_available))
    refused = _unfit(job, made, host_available)
    ordered = made
    if refused is not None and pins.swap_blocks is None:
        # Where the system memory is what the plan is short of: more blocks swapped, the VAE's
        # weights staying on the GPU, when that takes less of it (_vae_kept).
        spared = _vae_kept(job, made, host_available)
        if spared is not None:
            again = _unfit(job, spared, host_available)
            if again is None or _peaks(job, spared).host < _peaks(job, made).host:
                made, refused = spared, again
    shrunk = None
    if refused is not None and recorded is None:
        # Then the DiT kept on the GPU in the VAE's phases, their tiles shrunk for it, when that
        # takes less of it (_dit_kept); a recorded plan's tiles shape the output. From the plan
        # with the VAE's weights kept first; and where that doesn't fit, from the plan as the
        # choices made it too: the VAE kept swaps one or two blocks more, each pinned whatever
        # the VAE's phases keep, so that the tiles shrunk under it may reach no plan the host
        # holds where they reach one with the VAE away, and the window would shorten for it, or
        # the job be refused: at 4K, shots of 23, 24, 150 and 600 frames with 63.0 GiB free and
        # 11.22 GiB of system memory ran windows of 12 latents where 13 plan, and a 5K shot of
        # 149 frames was refused with 43.0 GiB free where windows of 5 plan
        # (tests/test_shot_plan.py, test_the_vae_kept_first_costs_no_window). PROVISIONAL
        # (implementation, 2026-10-16): DESIGN.md orders the VAE kept before the tiles, the
        # window last; its order is kept where it fits, the plan with the VAE away taken where it
        # alone does, or where it takes less.
        least = None
        for start in dict.fromkeys((made, ordered)):
            kept = _dit_kept(job, start, host_available, pins)
            if kept is None:
                continue
            if least is None or _peaks(job, kept).host < _peaks(job, least[1]).host:
                least = start, kept
            if _unfit(job, kept, host_available) is None:
                least = start, kept
                break
        if least is not None and _peaks(job, least[1]).host < _peaks(job, made).host:
            start, kept = least
            shrunk = Shrunk(
                start.encode_tiles, start.decode_tiles, *_host_short(job, start, host_available)
            )
            made, refused = kept, _unfit(job, kept, host_available)
    return made, refused, shrunk


def _window_that_runs(
    job: _Job, pins: Pins, made: Plan, host_available: int | None
) -> tuple[int, Plan, Shrunk | None] | None:
    """For a new job whose plan fits the card and takes more system memory than the host has,
    its tiles shrunk as far as they go: the longest window under the plan's own with which the
    job is planned on this host, that plan, and the tiles the card alone gives there where they
    were shrunk, the other pins as given, by the planner's own making of each (_made); None where
    no shorter window is, or for another refusal.

    The choices are made for the GPU's memory, the longest window first, and the system memory
    is checked after them: a long window may take swapped blocks, each pinned in system memory,
    where a shorter one takes none, so that a job would be refused on a host that runs it under
    a shorter window, and refused with MORE free VRAM where it was planned with less. 1080p, a
    shot of 301 frames, 16 GiB available and 80 GiB of free VRAM: one window of 76 latents with
    35 blocks pinned takes 17.40 GiB; two windows of 39 run, nothing pinned. The planner takes
    that window by itself (plan); under --window, the refusal names it.

    The window given is the one the job really runs under the cap found (_own_cap), as the
    planner's own cap is: under --window it is told as the pin that plans the same."""
    peaks = _peaks(job, made)
    over = any(peaks.need(phase) + job.margin > job.free for phase in PHASES)
    if host_available is None or over or peaks.host <= host_available or made.window is None:
        return None
    ran = _longest(job.lengths, made.window)
    for cap in range(made.window - 1, job.smallest - 1, -1):
        # A cap under which the job runs the windows it ran under the last one tried plans the
        # same.
        if _longest(job.lengths, cap) == ran:
            continue
        ran = _longest(job.lengths, cap)
        own = _own_cap(job, cap)
        try:
            shorter, short, shrunk = _made(job, replace(pins, window=own), None, host_available)
        except PlanError:
            continue
        if short is None:
            return own, shorter, shrunk
    return None


def _short_of_host(job: _Job, made: Plan, host_available: int | None) -> Shortened:
    """What a plan the host can't hold says of itself where the planner then takes a shorter
    window (plan): its window, the system memory it takes and what is available (_host_short)."""
    assert made.window is not None
    return Shortened(made.window, *_host_short(job, made, host_available))


def _host_short(job: _Job, made: Plan, host_available: int | None) -> tuple[int, int]:
    """The system memory a plan the host can't hold takes and what is available, in hundredths of
    a GiB, as its refusal would have said them (_unfit: each part to the nearest hundredth of a
    GiB, their sum, and what is there rounded down, a hundredth under the total at least)."""
    assert host_available is not None
    taken = sum(_host_hundredths(made, _peaks(job, made).shot, job.dit_model, job.held))
    return taken, min(_hundredths(host_available, up=False), taken - 1)


def _check_tiles(padded: tuple[int, int], recorded: Plan) -> None:
    """Refuse (PlanError) a recorded plan with tiles of which one covers the whole padded frame
    (planner.check_recorded), or that the tiled passes lay no grid of (tilegrid.latent_tiles: a side
    or an overlap that isn't a multiple of 8 px), which no plan of the planner's has: that record
    was edited, and its pass has no peak (_fit). PROVISIONAL (implementation, 2026-10-12):
    DESIGN.md doesn't say what such a record gets; refused here, in tilegrid.grid's own words,
    which the tiled passes would raise for it."""
    height, width = padded
    passes: tuple[tuple[Phase, Tiles | None], ...] = (
        ("encode", recorded.encode_tiles),
        ("decode", recorded.decode_tiles),
    )
    for phase, tiles in passes:
        if tiles is None:
            continue
        try:
            tilegrid.latent_tiles(tiles)
        except ValueError as error:
            raise PinRefused(
                f"not a plan as seedvr2x records one: its {phase}_tiles: {error}",
                why="recorded",
            ) from None
        if peaks.covers(padded, tiles):
            raise PinRefused(
                f"not a plan as seedvr2x records one: its {phase}_tiles, of {tiles.size} px, with"
                f" frames padded to {width}x{height}: a tile that covers the whole frame is no"
                f" tiling, and an untiled {phase} records none",
                why="recorded",
            )
    if recorded.compile_vae:
        # A plan the planner makes with the VAE compiled has no pass beyond the shapes measured
        # (in_compile_range): a record that has one was edited, and its peaks are no one's.
        # PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md doesn't say what such a record gets):
        # refused as the covering tile is, in the same words.
        for phase, tiles in passes:
            if not peaks.in_compile_range(padded, phase, tiles):
                raise PinRefused(
                    f"not a plan as seedvr2x records one: its compile_vae is true and its"
                    f" {phase} {_tiles_said(tiles)}, beyond the shapes measured compiled"
                    f" ({peaks.compile_range_said(phase)})",
                    why="recorded",
                )


def _job(
    free: int | Reading,
    target: tuple[int, int],
    shots: Sequence[Shot],
    split: bool,
    attention: str = peaks.FLASH_ATTN_2,
    compile_vae: bool = False,
    dit_model: Architecture = peaks.FP16,
    held: bool = False,
    in_memory: bool = False,
) -> _Job:
    if not shots:
        raise ValueError("a job without a shot")
    if attention not in peaks.DIT_ATTENTION:
        raise ValueError(
            f"attention {attention!r}: the DiT's fit is {peaks.FLASH_ATTN_2}'s or {peaks.SDPA}'s"
        )
    if dit_model not in peaks.DITS:
        raise ValueError(
            f"{dit_model.name}: the DiT's figures are for its fp16, int8 and dynamic GGUF files"
        )
    # A number is a driver that refuses: what it reports free is the card's, and all the plan's.
    reading = free if isinstance(free, Reading) else Reading(free, None, free, False)
    lengths = sorted({_latents(shot.frames) for shot in shots})
    frames = max(shot.frames for shot in shots)
    return _Job(
        *(reading.counted, target, tuple(lengths), frames, split, reading),
        attention,
        compile_vae,
        dit_model,
        held,
        in_memory,
    )


def _latents(frames: int) -> int:
    """The latents of a shot of `frames` frames, padded to 4n + 1 (frames.padded_length)."""
    return (padded_length(frames) - 1) // 4 + 1


@cache
def _longest(lengths: tuple[int, ...], cap: int | None) -> int:
    """The longest window, in latents, that shots of `lengths` latents really run under a cap
    (None: each in one window): the longest of their balanced layouts' (shots.window_layout),
    which is the cap only where a shot's length makes it so. It never falls as the cap rises:
    fewer windows over a shot are longer ones (tests/test_shot_plan.py holds it)."""
    if cap is None or cap >= lengths[-1]:
        return lengths[-1]
    return max(_split(length, cap) for length in lengths)


@cache
def _split(length: int, cap: int) -> int:
    """The longest window, in latents, of a shot of `length` latents under a cap, by its balanced
    layout (shots.window_layout); its length when the cap holds it whole."""
    if cap >= length:
        return length
    return max(end - start for start, end in window_layout(length, cap))


def _dit_peak(job: _Job, cap: int | None, swapped: int, vae_in_dit: bool) -> int:
    """The predicted peak of the job's DiT phase under a cap, in bytes, `swapped` blocks swapped
    and the VAE's weights there or not: its longest window's (dit_peak), on the attention
    backend the job runs; nothing of a shot is on the GPU beside the window, its noise drawn in
    system memory (shot_pass.sample_windows). What the plan records (_peaks): real memory,
    unlike the reserve beyond it."""
    latents = _longest(job.lengths, cap)
    return peaks.dit_peak(job.padded, latents, swapped, vae_in_dit, job.attention, job.dit_model)


def _dit_reserve(job: _Job, cap: int | None) -> int:
    """What the plan leaves the job's DiT phase under a cap beyond its peak, in bytes: the
    reserve of its longest window (peaks.reserve), by the window's P (peaks.dit_working) and the
    allocator the reading names. Whatever the blocks swapped and the VAE's place: P is what the
    window takes over the weights."""
    working = peaks.dit_working(
        job.padded, _longest(job.lengths, cap), job.attention, job.dit_model
    )
    return peaks.reserve(peaks.DIT_PHASE, working, job.reading.allocator, job.reading.margin)


def _dit_need(job: _Job, cap: int | None, swapped: int, vae_in_dit: bool) -> int:
    """What the planner judges the job's DiT phase by against the budget, in bytes: its peak
    (_dit_peak) and its reserve (_dit_reserve), wherever it decides whether a window fits, as
    _unfit judges the plan made."""
    return _dit_peak(job, cap, swapped, vae_in_dit) + _dit_reserve(job, cap)


def _holds(job: _Job, cap: int | None, swapped: int) -> bool:
    """Whether the DiT's phase holds the job under a cap, `swapped` blocks swapped and the VAE's
    weights away: its longest window (_dit_need)."""
    return _dit_need(job, cap, swapped, vae_in_dit=False) <= job.budget


def _longest_cap(job: _Job, swapped: int) -> int | None:
    """The largest cap, from the longest shot's length down to the smallest the job can run, that
    the DiT's phase holds with `swapped` blocks swapped (_holds); None when it holds none.

    A film has thousands of shots and windows of hundreds of latents: the caps whose longest
    window is over the budget by its tokens alone are bisected away first (_longest never falls
    as the cap rises, nor the tokens with it), the window functions then asked from there down."""

    def possible(cap: int) -> bool:
        latents = _longest(job.lengths, cap)
        least = peaks.dit_least(job.padded, latents, swapped, False, job.attention, job.dit_model)
        return least <= job.budget

    low, high = job.smallest, job.lengths[-1]
    if not possible(low):
        return None
    while low < high:
        middle = (low + high + 1) // 2
        if possible(middle):
            low = middle
        else:
            high = middle - 1
    return next(
        (cap for cap in range(low, job.smallest - 1, -1) if _holds(job, cap, swapped)), None
    )


def _fewest(job: _Job, cap: int | None) -> tuple[int, bool] | None:
    """The fewest swapped blocks with which the DiT's phase holds the job under a cap, its
    longest window (_dit_need), and whether the VAE's weights stay on the GPU meanwhile: they
    leave before one more block is swapped. One and two swapped blocks save nothing (dit_peak),
    so are never chosen. None when every block swapped and the VAE away don't hold it."""
    for swapped in (0, *range(peaks.SLOTS + 1, peaks.BLOCKS + 1)):
        for vae_in_dit in (True, False):
            if _dit_need(job, cap, swapped, vae_in_dit) <= job.budget:
                return swapped, vae_in_dit
    return None


def _chosen(job: _Job, pins: Pins) -> Plan:
    """The plan made whole, around the pins (plan)."""
    floor = _floor(job)
    if floor is not None:
        raise _too_large(job, *floor)
    cap, swapped, vae_in_dit = _window(job, pins)
    decode = _pass(job, "decode", pins.decode_tile, swapped)
    encode = _pass(job, "encode", pins.encode_tile, swapped)
    return Plan(
        cap,
        encode,
        decode,
        swapped,
        _resident(job, "encode", encode, swapped),
        _resident(job, "decode", decode, swapped),
        vae_in_dit,
        compile_vae=job.compile_vae,
        tail_frames=job.tail,
        vae_floor=job.vae_floor,
    )


def _around(job: _Job, recorded: Plan, pins: Pins) -> Plan:
    """A recorded plan's shape as it is, where the weights are planned again around it (plan): the
    fewest swapped blocks that hold its windows, or --swap-blocks' when given, and as much of the
    DiT in the VAE's phases as fits; every block swapped and none left there when it no longer
    fits, which _unfit then says, and under --swap-blocks the pin's blocks, whose part of the
    shortfall the refusal names (_pin_s_part)."""
    cap = recorded.window
    if pins.swap_blocks is None:
        swapped, vae_in_dit = _fewest(job, cap) or (peaks.BLOCKS, False)
    else:
        swapped = pins.swap_blocks
        vae_in_dit = _dit_need(job, cap, swapped, vae_in_dit=True) <= job.budget
        fewest = _fewest(job, cap)
        if not _holds(job, cap, swapped) and fewest is not None:
            # The pin's doing, not the memory's: the recorded shape still fits with more blocks.
            # PROVISIONAL (implementation, 2026-10-10): refused as the pin is on a new job.
            raise _swap_refused(job, swapped, cap, "the recorded plan's", fewest[0])
    return replace(
        recorded,
        swapped=swapped,
        resident_encode=_resident(job, "encode", recorded.encode_tiles, swapped),
        resident_decode=_resident(job, "decode", recorded.decode_tiles, swapped),
        vae_in_dit=vae_in_dit,
        tail_frames=job.tail,
        vae_floor=job.vae_floor,
    )


def _window(job: _Job, pins: Pins) -> tuple[int, int, bool]:
    """The window cap, the swapped blocks and whether the VAE's weights stay in the DiT's phase
    (plan's choices 1 and 2), --window and --swap-blocks taken exactly or refused; the job's
    floor holds already (_floor).

    The cap is always a number, never None: the longest shot's length when it fits, else the
    longest window the job really runs under the largest cap that fits (_own_cap), not that cap,
    which a card may hold no window of: a shot of 38 latents on a 24 GB card at 1080p runs two
    windows of 20 under every cap from 20 to 37, and its plan says 20 (DESIGN.md, Memory
    planner, Choices, 1); a refused --window is told the same. A --window pinned stays the cap
    as given (DESIGN.md, Memory planner, Pins). With --swap-blocks, the cap is the largest that
    fits with those blocks swapped and the VAE's weights away, as it is without the pin: the
    VAE leaves the DiT's phase before a window shortens or a block is swapped (Choices, 1), a
    choice the planner makes around the pin as around any other (Pins), so that --swap-blocks 0
    runs 6 latents at 1080p on a 24 GB card (Per-phase peaks: every block on the GPU, the VAE
    away).

    PROVISIONAL (implementation, 2026-10-10), a choice DESIGN.md doesn't make: of two pins that
    don't fit together, --window is the one refused while some cap fits the blocks pinned,
    --swap-blocks when none does."""
    most = peaks.BLOCKS if pins.swap_blocks is None else pins.swap_blocks
    longest = _longest_cap(job, most)
    if longest is None:
        # The floor holds every block swapped: a --swap-blocks that holds no window at all.
        cap = job.smallest
        whose: str | Setting = "the shortest this job can run"
        if pins.window is not None and _holds(job, pins.window, peaks.BLOCKS):
            cap, whose = pins.window, Setting("window", str(pins.window))
        fewest = _fewest(job, cap)
        assert fewest is not None
        raise _swap_refused(job, most, cap, whose, fewest[0], phase="dit")
    if pins.window is None:
        cap = _own_cap(job, longest)
    elif _holds(job, pins.window, most):
        cap = pins.window
    else:
        latents = _longest(job.lengths, pins.window)
        need = _dit_need(job, pins.window, most, vae_in_dit=False) + job.margin
        room = _reserve_part(job, "dit", _dit_reserve(job, pins.window))
        asked = Setting("window", str(pins.window))
        fits = Setting("window", str(_own_cap(job, longest)))

        def refused(say: Say) -> str:
            blocks = (
                "every block of the DiT swapped"
                if pins.swap_blocks is None
                else say(Setting("swap_blocks", str(pins.swap_blocks)))
            )
            return (
                f"{say(asked)}: its longest window, {_count(latents, 'latent')}, takes"
                f" {_over(need)} GiB of free VRAM{room} with {blocks}, and {_there(job)}:"
                f" {_to_free(job, need)}the longest window that fits is {say(fits)}"
            )

        raise PinRefused(
            refused,
            pin=asked,
            why="memory",
            need=int(need),
            there=int(job.free),
            phase="dit",
            instead=fits,
        )
    if pins.swap_blocks is not None:
        return cap, most, _dit_need(job, cap, most, vae_in_dit=True) <= job.budget
    fewest = _fewest(job, cap)
    assert fewest is not None
    return cap, *fewest


def _own_cap(job: _Job, cap: int) -> int:
    """The cap the planner takes for a job that fits under `cap`: the longest window the job
    really runs under it (_longest), the longest shot's length when every shot is one window. It
    lays every shot out as `cap` does: a shot's balanced windows under a cap are its windows
    under any smaller cap that is still at least the longest of them, as many of that length
    still covering the shot (tests/test_shot_plan.py holds it on shots.shot_layout). 2 * SHARED + 1
    at least when a shot is split: a shot of 6 latents under a cap of 5 runs two windows of 4,
    and no layout exists for a cap that short with a longer shot (planner.check_recorded)."""
    latents = _longest(job.lengths, cap)
    return latents if latents == job.lengths[-1] else max(latents, 2 * SHARED + 1)


def _swap_refused(
    job: _Job,
    swapped: int,
    cap: int | None,
    whose: str | Setting,
    fewest: int,
    phase: str = "",
) -> PinRefused:
    """The refusal of swap_blocks: the fewest blocks that hold the job's DiT phase under a cap,
    the free VRAM said that of the phase with the blocks pinned (_dit_need), and what takes it
    named: the cap's longest window, `whose` in words, or the window pinned (a Setting) whose
    longest window it is."""
    latents = _longest(job.lengths, cap)
    need = _dit_need(job, cap, swapped, vae_in_dit=False) + job.margin
    room = _reserve_part(job, "dit", _dit_reserve(job, cap))
    asked, fits = Setting("swap_blocks", str(swapped)), Setting("swap_blocks", str(fewest))

    def message(say: Say) -> str:
        owner = whose if isinstance(whose, str) else f"{say(whose)}'s longest"
        what = f"a window of {_count(latents, 'latent')}, {owner},"
        return (
            f"{say(asked)}: {what} takes {_over(need)} GiB of free VRAM{room} with"
            f" {_count(swapped, 'block')} swapped, and {_there(job)}: {_to_free(job, need)}the"
            f" fewest blocks that hold it are {say(fits)}"
        )

    return PinRefused(
        message,
        pin=asked,
        why="memory",
        need=int(need),
        there=int(job.free),
        phase=phase,
        instead=fits,
    )


def _vae_kept(job: _Job, plan: Plan, host_available: int | None) -> Plan | None:
    """The plan's shape with the VAE's weights left on the GPU in the DiT's phase and as many
    more blocks swapped as that takes, the fewest, where the plan as made fits the GPU's memory
    and not the system's; None where it doesn't apply: the system memory unchecked or enough,
    a phase over the budget, the VAE's weights there already, or no count of swapped blocks
    holding the window with them; nor for weights held (plan's held), whose pinned copies this
    would spare nothing of. Not chosen under --swap-blocks, taken exactly (plan).

    DESIGN.md has the VAE leave the DiT's phase before a block is swapped (Memory planner,
    Choices, 1: "the VAE off the GPU for the DiT's phase first, then as many blocks swapped as
    it takes"): a swapped block costs time at every window, the VAE's leaving a copy per phase.
    But the VAE's weights away are 0.47 GiB more of pinned system memory, and a swapped block
    costs none where the VAE's phases send that block away anyway, as most plans of a small
    card do, or 0.42. So DESIGN.md has this one too, provisional there (Choices, the paragraph
    after 3: "Where the plan so made takes more system memory than the host has, the VAE's
    weights stay on the GPU for the DiT's phase and the fewest more blocks are swapped in their
    place, when that fits the host or takes less of it"): taken only where the host's memory
    refuses the plan as the choices order it, so that no plan moves on a host that holds it; it
    shapes no bit. 1080p on a 24 GB card, a shot of 301 frames, 18.3 GiB available: 29 blocks
    swapped and the VAE staying take 17.92 GiB, where 28 and the VAE away take 18.38 and were
    refused (tests/test_shot_plan.py)."""
    predicted = _peaks(job, plan)
    over = any(predicted.need(phase) + job.margin > job.free for phase in PHASES)
    if (
        job.held
        or host_available is None
        or over
        or predicted.host <= host_available
        or plan.vae_in_dit
    ):
        return None
    counts = range(max(plan.swapped, peaks.SLOTS) + 1, peaks.BLOCKS + 1)
    fits = (count for count in counts if _dit_need(job, plan.window, count, True) <= job.budget)
    swapped = next(fits, None)
    if swapped is None:
        return None
    kept = replace(
        plan,
        swapped=swapped,
        resident_encode=_resident(job, "encode", plan.encode_tiles, swapped),
        resident_decode=_resident(job, "decode", plan.decode_tiles, swapped),
        vae_in_dit=True,
    )
    return replace(kept, scratch_allowance=_allowance(job, kept, host_available))


def _dit_kept(job: _Job, plan: Plan, host_available: int | None, pins: Pins) -> Plan | None:
    """The plan's shape with more of the DiT kept on the GPU in the VAE's phases, their tiles
    shrunk to keep it, where the plan as made, the VAE's weights kept for the DiT's phase where
    that takes less (_vae_kept), fits the GPU's memory and not the system's: every block a VAE
    phase sends away waits in a pinned copy (_pinned), and the largest tiles, chosen first,
    send the most away (DESIGN.md, Memory planner, Choices, decided on 2026-10-13: "before the
    window shortens, the DiT stays on the GPU for the VAE's phases and the tiles shrink, the
    window being the larger lever"). One block more kept in both phases at a time, one block
    fewer pinned, each pass in the largest of its tiles, from its own down, that keeps as many
    (_keeping), up to every block the DiT's phase keeps: the first plan that fits the host is
    taken, with the largest tiles that do; where none does, the last one made, which takes the
    least, for the shorter windows' search and the refusal's figures (plan). None where it
    doesn't apply: the system memory unchecked or enough, a phase over the budget, or no tile
    under a pass's own that keeps one block more; nor for weights held (plan's held), the
    pinned copies this would spare being counted none.

    A pinned pass keeps its tiles, as on any host ("never under a tile pin"), the other one
    still shrinking. The swapped blocks stay as they are, under --swap-blocks or not: each is
    pinned whatever the VAE's phases keep. Jobs refused for the system memory with more free
    VRAM than one that planned, short shots whose untiled VAE pass fits only with the DiT sent
    away, 26 to 36 blocks pinned (DESIGN.md, Memory planner, Choices: 828 of a grid of 144,000
    plans); and windows that fell as the free VRAM grew (DESIGN.md, To measure, The planner on a
    short host): at 1440p a decode's tiles growing from 1600 to 1728 px kept one block less on
    the GPU, which pinned 0.42 GiB more, the window then shortened for 0.01 GiB of system memory
    (the CLI's tests/test_planner.py, test_on_a_short_host_the_tiles_shrink_before_the_window).

    PROVISIONAL (implementation, 2026-10-16): DESIGN.md orders the steps, not how far the tiles
    shrink. They shrink as little as the host needs, the same count of blocks kept in both
    phases, the largest tiles that keep it, where the smallest tiles would pin the least: tiles
    drift in colour more as they get smaller (DESIGN.md, Memory planner, A plan shapes the
    output a little)."""
    predicted = _peaks(job, plan)
    over = any(predicted.need(phase) + job.margin > job.free for phase in PHASES)
    if job.held or host_available is None or over or predicted.host <= host_available:
        return None
    passes: tuple[tuple[Phase, Tiles | None, bool], ...] = (
        ("encode", plan.encode_tiles, pins.encode_tile is not None),
        ("decode", plan.decode_tiles, pins.decode_tile is not None),
    )
    least = None
    keep = peaks.BLOCKS - _leaving(plan) + 1
    while keep <= peaks.BLOCKS - plan.swapped:
        chosen = [_keeping(job, *each, plan.swapped, keep) for each in passes]
        if not all(found for found, _ in chosen):
            break
        (_, encode), (_, decode) = chosen
        kept = replace(
            plan,
            encode_tiles=encode,
            decode_tiles=decode,
            resident_encode=_resident(job, "encode", encode, plan.swapped),
            resident_decode=_resident(job, "decode", decode, plan.swapped),
        )
        least = replace(kept, scratch_allowance=_allowance(job, kept, host_available))
        if _unfit(job, least, host_available) is None:
            break
        keep = peaks.BLOCKS - _leaving(least) + 1
    return least


def _keeping(
    job: _Job, phase: Phase, tiles: Tiles | None, pinned: bool, swapped: int, keep: int
) -> tuple[bool, Tiles | None]:
    """Whether a pass of the job's encode or decode at or under `tiles`, in the planner's order
    (_fitting: untiled, then the largest tiles down), keeps `keep` of the DiT's blocks on the GPU
    or more (_resident; None, the DiT as its own phase has it, keeps every block that isn't
    swapped), and the largest that does (_dit_kept); a pinned pass's tiles alone."""
    passes = [tiles] if pinned else _fitting(job, phase, swapped)
    for option in passes[passes.index(tiles) :]:
        resident = _resident(job, phase, option, swapped)
        if (peaks.BLOCKS - swapped if resident is None else resident) >= keep:
            return True, option
    return False, None


def _shrunk_said(job: _Job, plan: Plan) -> str:
    """What a refusal for the system memory adds where the plan's tiles were shrunk to keep the
    DiT on the GPU in the VAE's phases and it still takes more than is there (_dit_kept): the
    tiles its figures are of, and what of the DiT they keep."""
    encode = _resident_said(plan.resident_encode, plan.swapped)
    decode = _resident_said(plan.resident_decode, plan.swapped)
    return (
        f", its VAE's tiles shrunk to keep the DiT on the GPU: the encode"
        f" {_tiles_said(plan.encode_tiles)}, with {encode}; the decode"
        f" {_tiles_said(plan.decode_tiles)}, with {decode}"
    )


def _pin_s_part(job: _Job, plan: Plan, pins: Pins, say: Say) -> str:
    """What a resume's refusal adds where swap_blocks is pinned and the DiT's phase is what
    doesn't fit: the pin's part of the shortfall said, and what the recorded shape is short of
    by itself, every block swapped and the VAE's weights away. The pin isn't refused as on a
    new job there (_around), the shape fitting under no count of swapped blocks, and the
    shortfall counted with the blocks it pins would pass for the shape's: a recorded window of
    21 latents at 1080p with 22 GiB free is 1.07 GiB short, and 15.41 with --swap-blocks 0.
    Empty elsewhere: no pin, another phase refused first, which no swapped block changes (its
    pass with the DiT away), or the system memory."""
    if pins.swap_blocks is None:
        return ""
    predicted = _peaks(job, plan)
    first = next((phase for phase in PHASES if predicted.need(phase) + job.margin > job.free), None)
    if first != "dit":
        return ""
    least = _dit_need(job, plan.window, peaks.BLOCKS, vae_in_dit=False) + job.margin
    there = _hundredths(job.free, up=False)
    short = _hundredths(least, up=True) - there
    part = _hundredths(predicted.need("dit") + job.margin, up=True) - there - short
    if part <= 0 or short <= 0:
        return ""
    pin = Setting("swap_blocks", str(pins.swap_blocks))
    return (
        f", {part / 100:.2f} GiB of them {say(pin)}'s: with every block of"
        f" the DiT swapped and the VAE off the GPU, the recorded plan is {short / 100:.2f} GiB"
        " short"
    )


def _tile_sizes(padded: tuple[int, int]) -> range:
    """The sides of the tiles the planner makes for padded frames, the largest first: multiples
    of TILE_STEP from the largest under the frame's longer side, a tile that covers the frame
    being no tiling (covers), or from TILE_MAX on a frame longer than it, down to TILE_MIN; none
    for a frame that small."""
    largest = min(peaks.TILE_MAX, (max(padded) - 1) // peaks.TILE_STEP * peaks.TILE_STEP)
    return range(largest, peaks.TILE_MIN - 1, -peaks.TILE_STEP)


def _vae_peak(
    job: _Job, phase: Phase, tiles: Tiles | None, resident: int | None, swapped: int
) -> int:
    """The predicted peak of the job's encode or decode (encode_peak, decode_peak), whatever
    its shots' lengths: no VAE pass keeps anything of the shot on the GPU."""
    if phase == "encode":
        return peaks.encode_peak(
            job.padded, tiles, resident, swapped, job.compile_vae, job.dit_model
        )
    return peaks.decode_peak(
        job.padded,
        job.output,
        tiles,
        resident,
        swapped,
        job.split,
        job.compile_vae,
        job.dit_model,
        frames=peaks.SLICE,
        tail=job.tail,
    )


def _vae_reserve(job: _Job, phase: Phase, peak: int, resident: int | None, swapped: int) -> int:
    """What the plan leaves the job's encode or decode beyond its `peak`, in bytes: the pass's
    reserve (peaks.reserve), by its P, the peak over the models resident in the phase
    (peaks.vae_working), and the allocator the reading names; VAE_ROOM under torch's own
    allocator without the segments and on a driver that refuses, as before."""
    working = peaks.vae_working(peak, resident, swapped, job.dit_model)
    return peaks.reserve(phase, working, job.reading.allocator, job.reading.margin, job.vae_floor)


def _vae_need(
    job: _Job, phase: Phase, tiles: Tiles | None, resident: int | None, swapped: int
) -> int:
    """What the planner judges the job's encode or decode by against the budget, in bytes: its
    predicted peak and its reserve (_vae_reserve), wherever it decides whether a pass fits and
    what of the DiT stays there, as _unfit judges the plan made. The plan records the peak alone
    (_peaks)."""
    peak = _vae_peak(job, phase, tiles, resident, swapped)
    return peak + _vae_reserve(job, phase, peak, resident, swapped)


def _fitting(job: _Job, phase: Phase, swapped: int) -> list[Tiles | None]:
    """The passes of the job's encode or decode that fit the budget with the DiT away, in the
    planner's order: untiled, then tiled from the largest tiles down, each with its room
    (_vae_need). `swapped` changes nothing to it: a phase that keeps nothing of the DiT frees
    its slots too (dit_on_gpu)."""
    passes = (None, *(Tiles(size) for size in _tile_sizes(job.padded)))
    return [
        tiles
        for tiles in passes
        if _vae_need(job, phase, tiles, 0, swapped) <= job.budget
        # With the VAE compiled, only the passes within the shapes measured (in_compile_range):
        # the largest tiles that fit and are measured, smaller where the compiled peak needs it.
        and (not job.compile_vae or peaks.in_compile_range(job.padded, phase, tiles))
    ]


def _pass(job: _Job, phase: Phase, pin: int | None, swapped: int) -> Tiles | None:
    """The tiles of the job's encode or decode (plan's choice 3), None untiled; --encode-tile or
    --decode-tile, `pin`, taken exactly or refused, the refusal naming the largest tile that
    fits. The job's floor holds already (_floor), so some pass fits (_least_pass); and no tile
    pinned covers the frame, which plan refuses first.

    A pass pinned, tiled or not, is held to the room the planner's own is (VAE_ROOM), and
    refused without it: every VAE pass is (DESIGN.md, Memory planner, Budget).

    PROVISIONAL (implementation, 2026-10-11): DESIGN.md has the planner choose no tile above
    TILE_MAX, the largest measured, and doesn't say what a pin above it gets. It is taken
    exactly when it fits by its fit, beyond the sizes measured as it is: a pin is the user's,
    and the search alone never chooses one, nor does a refusal name one. The log warns of it,
    once per pass (planner.beyond_measured)."""
    fitting = _fitting(job, phase, swapped)
    if pin is None:
        if not fitting:
            # Only with the VAE compiled: the passes within the shapes measured don't fit (the
            # untiled pass or the smallest tile may, outside them), or the frame is too small
            # for a tile and its pass can't be untiled.
            raise CompileRefused(
                f"the VAE compiled has no {phase} pass within the shapes measured"
                f" ({peaks.compile_range_said(phase)}) that fits the free VRAM at"
                f" {peaks.COMPILE_PEAK} times its plain peak",
                phase=phase,
            )
        return fitting[0]
    asked = Setting(f"{phase}_tile", str(pin))
    tiles = Tiles(pin) if pin else None
    if job.compile_vae and not peaks.in_compile_range(job.padded, phase, tiles):
        raise PinRefused(
            lambda say: (
                f"{say(asked)}: the VAE compiled has the {phase} {_tiles_said(tiles)} beyond the"
                f" shapes measured ({peaks.compile_range_said(phase)})"
            ),
            pin=asked,
            why="compile",
            phase=phase,
        )
    peak = _vae_peak(job, phase, tiles, 0, swapped)
    reserve = _vae_reserve(job, phase, peak, 0, swapped)
    need = peak + reserve + job.margin
    if need <= job.free:
        return tiles
    # The untiled pass alone fits where a frame little larger than the smallest tile takes less
    # untiled than in that tile (_least_pass).
    largest = next((fit.size for fit in fitting if fit is not None), None)
    # With the VAE compiled, the untiled pass is named only where it is within the shapes
    # measured (in_compile_range): a refusal never advises a pin the compiled plan refuses.
    fits = Setting(f"{phase}_tile", str(0 if largest is None else largest))

    room = _reserve_part(job, phase, reserve)

    def refused(say: Say) -> str:
        if largest is not None:
            advice = f"the largest tile that fits is {say(fits)}"
        elif job.compile_vae and not peaks.in_compile_range(job.padded, phase, None):
            advice = f"no tile within the shapes measured fits ({peaks.compile_range_said(phase)})"
        else:
            advice = f"no tile fits, the untiled {phase} does, {say(fits)}"
        return (
            f"{say(asked)}: the {phase} {_tiles_said(tiles)} takes {_over(need)} GiB of free"
            f" VRAM, the DiT off the GPU{room}, and {_there(job)}: {_to_free(job, need)}"
            f"{advice}"
        )

    raise PinRefused(
        refused,
        pin=asked,
        why="memory",
        need=int(need),
        there=int(job.free),
        phase=phase,
        instead=fits,
    )


# What a refusal adds after the free VRAM a VAE pass takes, its room in it (VAE_ROOM, or the last
# rung's, Plan.vae_floor).
def _room_held(job: _Job) -> str:
    room = VAE_ROOM if job.vae_floor is None else job.vae_floor
    return f", {room / peaks.GIB:.2f} of them the room a VAE pass is held to"


def _reserve_part(job: _Job, phase: Phase, reserve: int) -> str:
    """What a refusal adds after the free VRAM a phase takes, in words, `reserve` bytes of it
    being what the plan leaves beyond the phase's peak: a VAE pass's room under torch's own
    allocator without the segments and on a driver that refuses (_room_held), nothing for a
    window there (its margin is inside the figure counted), the step's reserve elsewhere
    (peaks.reserve)."""
    if job.reading.allocator == memory.PLAIN:
        return "" if phase == "dit" else _room_held(job)
    return f", {reserve / peaks.GIB:.2f} of them the reserve left beyond its peak"


def _reserve_fact(job: _Job, reserve: int) -> int | None:
    """The reserve a refusal carries (MemoryShort.reserve), in bytes: None on a driver that
    refuses; the flat memory.RESERVE under torch's own allocator without the segments, which
    is what the plan took off the figure; the step's own `reserve` (peaks.reserve) elsewhere."""
    if not job.reading.held:
        return None
    return memory.RESERVE if job.reading.allocator == memory.PLAIN else reserve


def _resident(job: _Job, phase: Phase, tiles: Tiles | None, swapped: int) -> int | None:
    """What of the DiT is left on the GPU in the job's encode or decode, in `tiles`
    (Plan.resident_encode, Plan.resident_decode; dit_on_gpu): None, the DiT staying as its own
    phase has it, when that fits, nothing then moving between the phases; else as many of the
    blocks that aren't swapped as fit, with its other parameters, from all of them down, the
    slots freed (with no block swapped, all of them is what None has there, which didn't fit);
    0, nothing of it, when one block and the other parameters don't fit. What fits leaves the
    pass its room (_vae_need).

    The most that fits is the least to bring back for the DiT's phase, each block one copy from
    its pinned memory, the whole DiT 0.29 s (milestone 3, 2026-10-11), and the fewest blocks
    pinned (_leaving)."""
    if _vae_need(job, phase, tiles, None, swapped) <= job.budget:
        return None
    for resident in range(peaks.BLOCKS - swapped, 0, -1):
        if _vae_need(job, phase, tiles, resident, swapped) <= job.budget:
            return resident
    return 0


def _floor(job: _Job) -> tuple[Phase, int] | None:
    """The phase that takes the most where every phase takes its least, and what it takes there,
    when that is over the budget, no plan then fitting the job (DESIGN.md, Memory planner: a size
    that doesn't fit is refused, never degraded); else None. The least: the smallest window the
    job can run, every block of the DiT swapped and the VAE's weights away (_dit_need), whatever
    its shots' lengths; the encode and the decode each in the pass
    that takes the least (_least_pass), the DiT away, whatever the plan swaps: a phase that
    keeps nothing of it frees its slots too (dit_on_gpu). What a phase takes is its peak, and
    for the encode and the decode their room with it (_vae_need)."""
    least: dict[Phase, int] = {
        "encode": _vae_need(job, "encode", _least_pass(job, "encode"), 0, peaks.BLOCKS),
        "dit": _dit_need(job, job.smallest, peaks.BLOCKS, vae_in_dit=False),
        "decode": _vae_need(job, "decode", _least_pass(job, "decode"), 0, peaks.BLOCKS),
    }
    phase = max(PHASES, key=lambda phase: least[phase])
    return (phase, least[phase]) if least[phase] > job.budget else None


def _least_pass(job: _Job, phase: Phase) -> Tiles | None:
    """The pass of the job's encode or decode that takes the least, the DiT away: None, the
    untiled pass, or tiles; untiled on a frame too small for a tile. The least of three, which
    tests/test_shot_plan.py holds to the least of every pass the planner may make, over many
    sizes: the smallest tiles, the least of all in most frames; the untiled pass, under them on
    a frame little larger than one: 512x528, where a square tile takes its measured peak
    (MEASURED) and the untiled pass its fit, or 496x528, whose tiles, cut off, take
    CUT_OFF_EXTRA over their rule; and the smallest tiles the refit serves (REFITTED), under
    the smallest of all on a strip over 1024 px long and under 74 px high for the encode, 134
    for the decode, where the two tiled fits cross."""
    sizes = _tile_sizes(job.padded)
    least = dict.fromkeys(size for size in (*sizes[-1:], peaks.REFITTED) if size in sizes)
    passes = [None, *(Tiles(size) for size in least)]
    return min(passes, key=lambda tiles: _vae_need(job, phase, tiles, 0, peaks.BLOCKS))


def _too_large(job: _Job, phase: Phase, least: int) -> MemoryShort:
    """The floor's refusal (_floor), counted in free VRAM, never in card sizes (DESIGN.md, Memory
    planner): the free VRAM the output size takes, what is there for it (_there), and the largest
    output size that fits in it, by its short side, a multiple of 16, the target scaled in
    proportion and the floor checked there with the same shots. The resolution is named first,
    and the size as about that: it is the target's scaled, not the source's, which the command
    line scales, and the two can differ by 2 px (a 1920x1080 source refused at --resolution 600:
    794x448 here, 796x448 by --resolution 448)."""
    height, width = job.output
    if phase == "dit":
        latents = _longest(job.lengths, job.smallest)
        how = (
            f"its smallest window, {_count(latents, 'latent')}, with every block of the DiT"
            " swapped and the VAE off the GPU"
        )
        room = _dit_reserve(job, job.smallest)
    else:
        tiled = _tiles_said(_least_pass(job, phase))
        if not _tile_sizes(job.padded):
            tiled += ", too small for a tile"
        how = f"its {phase} {tiled}, with the DiT off the GPU"
        least_pass = _vae_peak(job, phase, _least_pass(job, phase), 0, peaks.BLOCKS)
        room = _vae_reserve(job, phase, least_pass, 0, peaks.BLOCKS)
    # The room as what is left of the total once the peak and the margin are said, each rounded
    # as the message has it, so that its figures add up. No margin on a run held to the card
    # (_Job.margin), and none said.
    total, peak = _hundredths(least + job.margin, up=True), _hundredths(least - room, up=True)
    rest = (total - peak - _hundredths(job.margin, up=True)) / 100
    if job.reading.allocator == memory.PLAIN:
        held = f", {rest:.2f} of room a VAE pass is held to" if room else ""
    else:
        held = f", {rest:.2f} of reserve beyond its peak"
    margin = f", and {_gib(job.margin)} of margin" if job.margin else ""
    fitting: tuple[int, tuple[int, int]] | None = None
    target_height, target_width = job.target
    for short in range((min(job.target) - 1) // 16 * 16, 0, -16):
        if target_height <= target_width:
            scaled = (short, short * target_width // target_height)
        else:
            scaled = (short * target_height // target_width, short)
        try:
            check_target(scaled, short)
        except Refusal:
            break  # too small for the padding, as every smaller one is
        smaller = replace(job, target=scaled)
        if _floor(smaller) is None:
            fitting = short, smaller.output
            break

    def message(say: Say) -> str:
        if fitting is None:
            advice = "no output size fits there"
        else:
            short, (fits_height, fits_width) = fitting
            advice = (
                f"the largest output that fits there is {say(Setting('resolution', str(short)))},"
                f" about {fits_width}x{fits_height}"
            )
        return (
            f"an output of {width}x{height} takes {total / 100:.2f} GiB of free VRAM ({how}:"
            f" a peak of {peak / 100:.2f} GiB{held}{margin}), and"
            f" {_there(job)}: {_to_free(job, least + job.margin)}{advice}"
        )

    need = least + job.margin
    return MemoryShort(
        message,
        need=int(need),
        there=int(job.free),
        short=int(need) - int(job.free),
        freeable=_can_free(job, need),
        reserve=_reserve_fact(job, room),
        phase=phase,
        instead=None if fitting is None else Setting("resolution", str(fitting[0])),
    )


def _leaving(plan: Plan) -> int:
    """The DiT's blocks that leave the GPU in some phase of the plan, each then waiting in a
    pinned copy: its swapped ones, and those the VAE's phases send away."""
    stay = peaks.BLOCKS - plan.swapped
    there = (stay if resident is None else resident for resident in _residents(plan))
    return peaks.BLOCKS - min(stay, *there)


def _residents(plan: Plan) -> tuple[int | None, int | None]:
    return plan.resident_encode, plan.resident_decode


def _pinned(plan: Plan, dit_model: Architecture, held: bool = False) -> int:
    """The pinned system memory of the plan, in bytes: a copy of each part of the weights that
    leaves the GPU in some phase, of its own size rounded up to a page (blockswap.HostCopy;
    DESIGN.md, BlockSwap, rewritten), as the loader makes them (model.load_models: the parts of
    blockswap.dit_parts, and the VAE): each block of the DiT that is swapped or that a VAE phase
    doesn't keep; the DiT's other parameters when a VAE phase keeps nothing of it; the VAE's
    weights when they leave the DiT's phase; the DiT's in the format of its file (dit_model).
    In fp16, 15.19 GiB for the 36 blocks, 15.35 with the other parameters, 15.81 with the VAE
    (measured, milestone 3, 2026-10-11); in int8, by its headers, 7.61, 7.76 and 8.23. None of
    them with `held` (plan): the copies exist already, taken by the models kept in system
    memory, and this plan's run takes none more."""
    if held:
        return 0
    parts = list(peaks.leaving_blocks(dit_model, _leaving(plan)))
    parts += [peaks.other_bytes(dit_model)] if 0 in _residents(plan) else []
    parts += [] if plan.vae_in_dit else [peaks.VAE_WEIGHTS]
    return sum(-(-part // mmap.PAGESIZE) * mmap.PAGESIZE for part in parts)


def _left(plan: Plan, other: str, vae: str) -> str:
    """What of the weights leaves the GPU in some phase of the plan, pinned (_pinned), in words:
    the DiT's blocks counted, its other parameters and the VAE named as given; empty when
    nothing leaves."""
    leaving = _leaving(plan)
    left = [_count(leaving, "block")] if leaving else []
    left += [other] if 0 in _residents(plan) else []
    left += [] if plan.vae_in_dit else [vae]
    return " and ".join(filter(None, (", ".join(left[:-1]), *left[-1:])))


def _peaks(job: _Job, plan: Plan) -> Peaks:
    """What the plan is predicted to take for the job.

    Its system memory: the process's own during the shots, the pinned copies of what leaves the
    GPU, what the run holds of a shot at the moment it holds the most (_shot_held), and what a
    tiled shot's scratch may take of it, the plan's allowance (Plan.scratch_allowance): their
    sum. PROVISIONAL (implementation, 2026-10-12): a sum, as DESIGN.md counts the allowance
    after the rest (Memory planner, Host RAM), though a scratch is never open at the moment a
    shot holds the most: beside it the run holds one latent's worth of the shot, a tiled
    encode's sums or the latent a tiled decode reads (host_moments), so that the sum is over
    what any moment takes by the rest of the shot's part, 1 byte a padded pixel and a latent at
    least for a shot in one window. The load of the weights isn't counted apart: it reads their
    file at its own offsets, never mapped, the parts that leave the GPU straight into their
    pinned copies and every other tensor through one buffer the size of the largest
    (blockswap.read), so that it peaks at what it leaves, the copies counted here and about 1.1
    GiB of the process's own, under HOST_RESIDENT's 2.4: 1.1 GiB with nothing pinned, 7.5 with
    6.38 GiB pinned, 16.3 with the 36 blocks' 15.19 (DESIGN.md, Memory planner, Host RAM;
    BlockSwap, rewritten, the loader). Mapped, as until 2026-10-16, a load peaked at 16.3 to
    16.8 GiB whatever the plan pinned, which no plan could count. PROVISIONAL (implementation,
    2026-10-11): a scratch over its allowance goes whole to its memory-mapped file
    (scratch.Scratch), and the allowance is still counted, as DESIGN.md counts it, for the page
    cache its mapping works in."""
    latents = _longest(job.lengths, plan.window)
    held_by_shot, moment, holder = _shot_held(job, plan)
    encode = _vae_peak(job, "encode", plan.encode_tiles, plan.resident_encode, plan.swapped)
    decode = _vae_peak(job, "decode", plan.decode_tiles, plan.resident_decode, plan.swapped)
    reserves: dict[Phase, int] = {
        "encode": _vae_reserve(job, "encode", encode, plan.resident_encode, plan.swapped),
        "dit": _dit_reserve(job, plan.window),
        "decode": _vae_reserve(job, "decode", decode, plan.resident_decode, plan.swapped),
    }
    return Peaks(
        job.free,
        job.budget,
        encode,
        _dit_peak(job, plan.window, plan.swapped, plan.vae_in_dit),
        decode,
        HOST_RESIDENT
        + _pinned(plan, job.dit_model, job.held)
        + held_by_shot
        + plan.scratch_allowance,
        _kept(job, plan),
        latents,
        peaks.tokens(job.padded, 1),
        job.reading,
        job.attention,
        held_by_shot,
        moment,
        holder,
        dit_model=job.dit_model,
        reserves=reserves,
    )


def _shot_held(job: _Job, plan: Plan) -> tuple[int, str, int]:
    """What the run holds in system memory of one of the job's shots under the plan's cap, in
    bytes, at the moment it holds the most, that moment (host_moments), the first in the shot's
    order where two hold as much, and the shot's length in latents: the job's shot that holds
    the most (_most_held), which the plan counts with the pinned copies, the run holding one
    shot's tensors at a time.

    As the run is, a shot in one window holds the most through that window, its latent and its
    noise beside the window's output, 1.5 bytes a padded pixel and a latent: 3.03 MiB a latent
    at 1080p (3,179,520 bytes), 12.04 at 4K (12,625,920); a long shot in many windows as its
    decode merges them, its windows beside the merged latent, 1 byte a padded pixel and a
    latent and the latents its windows share: a shot of 1,000 latents takes 2.09 GiB at 1080p
    in windows of 20, and 10.47 at 4K in windows of 5. Until 2026-10-12 the check read the
    latent whole, 1.75 bytes a padded pixel and a latent, and a merge copied the shot at each
    join: 3.45 GiB for that shot at 1080p and 14.36 at 4K (shot_pass.finite, shot_pass._merge)."""
    return _most_held(job.padded, job.lengths, plan.window, job.in_memory)


@cache
def _most_held(
    padded: tuple[int, int], lengths: tuple[int, ...], cap: int | None, in_memory: bool = False
) -> tuple[int, str, int]:
    """(bytes, moment, latents): what the run holds in system memory of the shot that holds the
    most among shots of `lengths` latents (ascending) under a cap, the moment it does and that
    shot's length (_shot_held, host_moments). The longest shot's, but where a shorter one holds
    more: the most a shot holds can fall as the shot grows past the cap, its windows then
    shorter than the shot: 12 latents in one window hold 36 latents' worth, 13 in two windows
    of 8 and 7 hold 34 (the CLI's tests/test_planner.py). Of two that hold as much, the longer.

    No shot holds more than 3 latents' worth a latent and MIXING x SHARED more (its latent and
    noise beside a window no longer than itself; its windows, which share SHARED latents and
    are 2 x SHARED + 1 long at least, with the merged latent and a join's mix): the shots are
    read from the longest down, until none left can hold more than the most found. With in_memory
    (host_moments) the windows are kept whole, which a shot holds 4 latents' worth a latent at
    the most, and MIXING x SHARED more at the merge: its latent, its noise (or the merged
    latent), and the windows, L + SHARED x (n - 1) latents, which are 5 latents long at least
    and so fewer than 2 L."""
    height, width = padded
    latent = LATENT_CHANNELS * LATENT_VALUE * (height // LATENT_SIDE) * (width // LATENT_SIDE)
    most = (0, MOMENTS[0], lengths[-1])
    for latents in reversed(lengths):
        if latent * ((4 if in_memory else 3) * latents + MIXING * SHARED) <= most[0]:
            break
        moments = host_moments(padded, latents, cap, in_memory)
        moment = max(MOMENTS, key=lambda moment: moments[moment])
        # The encode's end and the decode never hold the most: a shot's windows hold a window
        # more than the first, which holds twice the second (host_moments). No sentence says
        # them (_MOMENT_SAID).
        assert moment in _MOMENT_SAID, moment
        if moments[moment] > most[0]:
            most = (moments[moment], moment, latents)
    return most


def _kept(job: _Job, plan: Plan) -> int:
    """The scratch of the job's longest shot under the plan, in bytes (scratch); 0 when neither
    pass is tiled. One pass's: the encode's and the decode's are the same size, and never open
    together."""
    tiled = plan.encode_tiles is not None or plan.decode_tiles is not None
    return peaks.scratch(job.padded, job.frames) if tiled else 0


def _allowance(job: _Job, plan: Plan, host_available: int | None) -> int:
    """What a tiled shot's scratch may take of system memory under the plan, in bytes
    (Plan.scratch_allowance), a larger one going to its file: half of what the host has
    available once the pinned copies and what the run holds of a shot at the most (_shot_held)
    are counted, and no more than the longest tiled shot needs (DESIGN.md, Memory planner, Host
    RAM: provisional, until measured on a consumer host); 0 when no pass is tiled. It shapes no
    bit.

    PROVISIONAL (implementation, 2026-10-11), two choices DESIGN.md doesn't make. No more either
    than what the host has left once the process's own memory is counted with them
    (HOST_RESIDENT), down to nothing, every scratch then going to its file: so that a job is
    refused for the system memory the process, its pinned copies and its longest shot take
    (_unfit), never for an allowance: half of what they leave is more than the process leaves
    of it where they leave under 4.8 GiB. And with what the host has available unknown (None),
    the longest tiled shot's scratch, ALLOWANCE_UNKNOWN_HOST at most.

    With the job's in_memory (plan) there is no file to go to: the allowance is the whole scratch,
    whatever the host has, and the host's memory is judged with it (_unfit)."""
    kept = _kept(job, plan)
    if job.in_memory:
        return kept
    if host_available is None:
        return min(kept, ALLOWANCE_UNKNOWN_HOST)
    left = host_available - _pinned(plan, job.dit_model, job.held) - _shot_held(job, plan)[0]
    return max(0, min(kept, left // 2, left - HOST_RESIDENT))


def _unfit(job: _Job, plan: Plan, host_available: int | None) -> str | None:
    """Why the plan doesn't fit the job's memory: the first phase over the budget, a VAE
    phase judged with its room as the planner chose it (Peaks.need), or the system memory it
    takes over what is available, the process's own, the pinned copies, what the run holds of
    a shot at the most and the scratch's allowance, each named with what it takes where that is a
    hundredth of a GiB or more; None when it fits. A plan of the planner's own is never refused
    for its allowance, which is nothing where the first three leave nothing (_allowance). The
    shot named is the one that holds the most (Peaks.holder)."""
    peaks = _peaks(job, plan)
    blocks = _count(plan.swapped, "block")
    vae = "" if plan.vae_in_dit else " and the VAE off the GPU"
    dit = f"a window of {_count(peaks.window, 'latent')}"
    whats = {
        "encode": f"the encode, {_tiles_said(plan.encode_tiles)}, with"
        f" {_resident_said(plan.resident_encode, plan.swapped)},",
        "dit": f"the DiT's phase, {dit} with {blocks} swapped{vae},",
        "decode": f"the decode, {_tiles_said(plan.decode_tiles)}, with"
        f" {_resident_said(plan.resident_decode, plan.swapped)},",
    }
    for phase in PHASES:
        need = peaks.need(phase) + job.margin
        if need > job.free:
            room = _reserve_part(job, phase, peaks.reserves[phase])
            room = f"{room}," if room else ""
            freed = ", which would have to be freed on the card" if _freeable(job, need) else ""
            return (
                f"{whats[phase]} takes {_over(need)} GiB of free VRAM{room} and {_there(job)},"
                f" {_short(need, job.free)} GiB short{freed}"
            )
    if host_available is not None and peaks.host > host_available:
        # Each part as the plan's lines say it, and the total their sum as said, so that the
        # message's figures add up (_host_hundredths).
        process, pinned, held, allowance = _host_hundredths(
            plan, peaks.shot, job.dit_model, job.held
        )
        taken = [(process, "for the process")]
        if pinned:
            left = _left(plan, "the DiT's other parameters", "the VAE's weights")
            taken.append((pinned, f"pinned for {left} that leave the GPU"))
        if held:
            # The shot that holds the most: the longest, but where a shorter one in fewer
            # windows does (_most_held).
            if peaks.holder == job.lengths[-1]:
                shot = f"{_count(job.frames, 'frame')} ({_count(job.lengths[-1], 'latent')})"
                whose = f"the longest shot, {shot},"
            else:
                shot = _count(peaks.holder, "latent")
                whose = f"a shot of {shot}, which holds more than the longest,"
            taken.append((held, f"for {whose} {_MOMENT_SAID[peaks.moment]}"))
        if allowance:
            taken.append((allowance, "for a tiled shot's scratch"))
        total = process + pinned + held + allowance
        # What is there is rounded down, and said a hundredth under the total at least: the
        # parts being rounded to the nearest, a host within 0.02 GiB of what the plan takes
        # would else read as having it.
        there = min(_hundredths(host_available, up=False), total - 1)
        said = ", ".join(f"{count / 100:.2f} {what}" for count, what in taken)
        return (
            f"the plan takes {total / 100:.2f} GiB of system memory ({said}) and"
            f" {there / 100:.2f} GiB are available, {(total - there) / 100:.2f} GiB short"
        )
    return None


def _shortfall(job: _Job, plan: Plan, host_available: int | None) -> Shortfall | None:
    """What `plan` is short of, as _unfit finds it (the same order: the first phase over the
    budget, then the system memory), in the parts a refusal words apart (Shortfall); None where
    it fits."""
    names = {"encode": "the encode", "dit": "the DiT's phase", "decode": "the decode"}
    phase = _refusing(job, plan, host_available)
    if phase == "host":
        return Shortfall("", "", host=True)
    if phase:
        need = _peaks(job, plan).need(phase) + job.margin
        return Shortfall(names[phase], _short(need, job.free), freeable=_can_free(job, need))
    return None


def _refusing(job: _Job, plan: Plan, host_available: int | None) -> Phase | Literal["host", ""]:
    """The phase `plan` first doesn't fit in, as _unfit finds it (the same order): "encode", "dit"
    or "decode" when its peak is over the budget, "host" when the system's memory is what falls
    short, else empty."""
    peaks = _peaks(job, plan)
    for phase in PHASES:
        if peaks.need(phase) + job.margin > job.free:
            return phase
    if host_available is not None and peaks.host > host_available:
        return "host"
    return ""


def _host_hundredths(
    plan: Plan, shot: int, dit_model: Architecture, held: bool = False
) -> tuple[int, int, int, int]:
    """What the plan takes of system memory, as its line and a refusal for it say it (planner.said,
    _unfit), in hundredths of a GiB: the process's own (HOST_RESIDENT), the pinned copies
    (_pinned), what the run holds of a shot at the most, `shot` bytes (Peaks.shot), and the
    scratch's allowance (Plan.scratch_allowance), each to the nearest, as the plan's other
    figures are; the total either says is their sum, so that its figures add up.
    One plan's pinned memory then reads the same in both, 15.81 GiB with every part of the
    weights away, DESIGN.md's figure (Memory planner, Host RAM): rounded up, as what is taken of
    the GPU's memory is, a refusal would say 15.82 where the plan's line says 15.81."""
    return (
        _nearest(HOST_RESIDENT),
        _nearest(_pinned(plan, dit_model, held)),
        _nearest(shot),
        _nearest(plan.scratch_allowance),
    )


# A shot's moments in system memory (MOMENTS) as the plan's line and a refusal name the one
# that holds the most, after the shot: the three that can (_most_held).
_MOMENT_SAID = {
    "check": "as its latent is checked after its encode",
    "windows": "at its longest window, its latent and its noise beside the window's output",
    "merge": "as its decode merges its windows",
}


def _held_under(reading: Reading) -> int:
    """The free memory a held run is held under, in bytes: the smaller of what its driver
    reports free to the process and of what the whole card has free; the driver's figure where
    the card's wasn't read (seedvr2x/memory.py)."""
    if reading.card is None:
        return reading.reported
    return min(reading.reported, reading.card)


def _named(reading: Reading, said: Callable[[float], str], driver: bool = True) -> str:
    """Which of a held reading's figures is the free memory, in words for a line that has just
    given it as "N GiB free", with the other figure: the words of seedvr2x/memory.py's own log
    lines. `said` puts a figure in GiB, as the line around it does.

    driver: whether the driver's figure, where the whole card's, NVML's, is the smaller and so
    the free memory, is said beside it. The plan's card line leaves it out (design's ruling,
    baton 3, 2026-10-30: "its driver reports N GiB free to this process" under --verbose; the
    line keeps NVML's free and why the run is held), and planner.card_detail says it at debug; a
    refusal (_there) keeps it."""
    if reading.card is None:
        return "as its driver reports it to this process (the whole card's, NVML's, wasn't read)"
    if reading.card <= reading.reported:
        if not driver:
            return "on the whole card by NVML"
        reported = said(reading.reported)
        return f"on the whole card by NVML (its driver reports {reported} GiB free to this process)"
    return (
        "as its driver reports it to this process (the whole card has"
        f" {said(reading.card)} GiB free by NVML)"
    )


def _early_said(reading: Reading, said: Callable[[float], str]) -> str:
    """An early reading's figures in words (memory.early): what NVML reads free on the whole card
    and what is counted off it for the CUDA context this process is yet to make. `said` puts a
    figure in GiB, as the line around it does."""
    assert reading.nvml is not None
    return (
        f"NVML's {said(reading.nvml)} GiB free on the whole card, less {said(reading.context)} GiB"
        " counted for this process's CUDA context to come"
    )


def _there(job: _Job) -> str:
    """What is there for the plan, as a refusal says it after what something takes: on a driver
    that refuses, what it reports free; on a run held to the card, what the plan counts on, of
    which figure, the driver's or the whole card's, with the other (_named). Rounded down, as
    what is there is."""
    reading = job.reading
    if reading.nvml is not None:
        # An early reading (memory.early): NVML's figure and the context counted off it, not a
        # driver's report, which no seedvr2x process has yet been given.
        if not reading.held:
            return f"{_early_said(reading, _under)}, leaves {_under(job.free)} GiB"
        return f"the plan counts on {_under(job.free)} GiB of {_early_said(reading, _under)}"
    if not reading.held:
        return f"the driver reports {_under(job.free)} GiB free"
    return (
        f"the plan counts on {_under(job.free)} GiB of the {_under(_held_under(reading))} GiB"
        f" free {_named(reading, _under)}, the run held to the card's own memory"
    )


def _to_free(job: _Job, need: float) -> str:
    """What a held run's refusal says before what fits: the memory that would have to be freed
    on the card for what was asked, `need` bytes of free VRAM, to fit what the plan counts on,
    by the message's own figures (_short); where no memory freed on the card gives it
    (_freeable), what it is short of, and no more. Nothing on a driver that refuses, whose
    refusals say what they said."""
    if not job.reading.held:
        return ""
    if not _freeable(job, need):
        return f"{_short(need, job.free)} GiB short; "
    return f"{_short(need, job.free)} GiB more would have to be freed on the card for it; "


def _freeable(job: _Job, need: float) -> bool:
    """Whether memory freed on the card, other programs closed, can give a held run's plan the
    `need` bytes of free VRAM it is refused for: only where the whole card's figure is the one
    the plan was made under, the smaller of the two, and only up to the driver's, which the plan
    is held under from there on (memory.reading). Elsewhere a refusal that said "N GiB more
    would have to be freed on the card" gave advice nobody can follow (a review's finding,
    2026-10-12): the driver's figure the smaller, or the card's not read, it is the process's
    own while the card has room, and other programs' memory isn't in it; and a request 16 GiB
    over a card that has 2 in use is short of more than the card holds."""
    reading = job.reading
    if reading.nvml is not None:
        # PROVISIONAL (impl-v010, 2026-10-29: the refusal's rule with NVML's used for the driver's
        # figure, which an early reading hasn't asked): what quitting programs can give back is
        # what they hold of the card by NVML (memory.card_state), there being no seedvr2x CUDA
        # context on it yet, as a held run's is what the driver reports free beyond the card's;
        # where used wasn't read, the early reading cannot tell, and does not say it.
        return reading.nvml_used is None or need - job.free <= reading.nvml_used
    if not reading.held or reading.card is None or reading.card >= reading.reported:
        return False
    return need - job.free <= reading.reported - reading.card


def _tiles_said(tiles: Tiles | None) -> str:
    """A pass as a refusal says it: untiled, or in its tiles."""
    return "untiled" if tiles is None else f"in tiles of {tiles.size} px"


def _resident_said(resident: int | None, swapped: int) -> str:
    """What of the DiT a VAE phase has on the GPU, in words, as dit_on_gpu counts it: nothing; a
    number of its blocks, with its other parameters; or, nothing moving, what its own phase
    has, the blocks that aren't swapped and the slots."""
    stay = peaks.BLOCKS - swapped
    if resident == 0:
        return "the DiT off the GPU"
    if resident is not None:
        return f"{resident} of the DiT's blocks on the GPU"
    if not swapped:
        return f"the DiT's {peaks.BLOCKS} blocks on the GPU"
    slots = _count(min(swapped, peaks.SLOTS), "slot")
    if not stay:
        return f"the DiT's {slots} on the GPU"
    return f"the DiT's {_count(stay, 'unswapped block')} and its {slots} on the GPU"


def _count(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _gib(count: float) -> str:
    """Bytes in GiB, to 2 decimals."""
    return f"{count / peaks.GIB:.2f}"


def _nearest(count: float) -> int:
    """Bytes in hundredths of a GiB, to the nearest, as _gib says them."""
    return round(float(_gib(count)) * 100)


def _over(count: float) -> str:
    """Bytes in GiB, to 2 decimals, rounded up: what something takes of the GPU's memory."""
    return f"{_hundredths(count, up=True) / 100:.2f}"


def _under(count: float) -> str:
    """Bytes in GiB, to 2 decimals, rounded down: what is there, so that what takes more than
    is there never reads as taking the same."""
    return f"{_hundredths(count, up=False) / 100:.2f}"


def _short(need: float, there: float) -> str:
    """What `need` bytes are short of in `there`, in GiB to 2 decimals: the difference of the two
    as _over and _under say them, so that a message's figures add up."""
    return f"{(_hundredths(need, up=True) - _hundredths(there, up=False)) / 100:.2f}"


def _hundredths(count: float, *, up: bool) -> int:
    """Bytes in hundredths of a GiB, rounded up or down, a figure that is one already kept."""
    exact = count / peaks.GIB * 100
    return math.ceil(exact - 1e-6) if up else math.floor(exact + 1e-6)
