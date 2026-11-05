"""seedvr2x/errors.py: a refusal's message in the library's own words and in an application's
(Setting, plain, Refusal.words), and the plan's refusals by kind (the base PlanRefused and its
subclasses): their facts, their kind set by the class, reworded."""

from pathlib import Path

import pytest
from library_fixtures import cli_say, cli_words
from test_weights import as_dtype, write

from seedvr2x import frames, picture, shot_plan, still_plan, weights
from seedvr2x.errors import (
    SETTINGS,
    CompileRefused,
    HostMemoryError,
    HostShort,
    InputTooLarge,
    JobError,
    MemoryShort,
    ModelChanged,
    ModelDamaged,
    ModelError,
    ModelMissing,
    ModelNoRoom,
    ModelNotPinned,
    ModelsRefused,
    ModelUnknown,
    ModelUnreachable,
    ModelUnreadable,
    PictureRefused,
    PinRefused,
    PlanRefused,
    Refusal,
    Setting,
    TargetTooSmall,
    aside,
    plain,
)

KINDS = (PinRefused, MemoryShort, HostShort, InputTooLarge, CompileRefused)


def test_a_setting_is_named_by_the_library_and_only_those() -> None:
    assert Setting("window", "6") == Setting("window", "6") and Setting("compile").value == ""
    assert {"window", "swap_blocks", "encode_tile", "decode_tile", "resolution"} <= set(SETTINGS)
    assert {"dit_model", "compile"} <= set(SETTINGS)
    for name, meaning in SETTINGS.items():
        assert meaning and Setting(name)
    with pytest.raises(ValueError, match="not one of the settings"):
        Setting("window-size")


def test_plain_words_name_no_option_and_carry_the_value() -> None:
    assert plain(Setting("window", "6")) == "window 6"
    assert plain(Setting("swap_blocks", "30")) == "swapped blocks 30"
    assert plain(Setting("encode_tile", "0")) == "encode tile 0"
    assert plain(Setting("decode_tile", "512")) == "decode tile 512"
    assert plain(Setting("resolution", "720")) == "resolution 720"
    assert plain(Setting("resolution")) == "resolution"
    assert plain(Setting("dit_model", "f.gguf")) == "DiT file f.gguf"
    assert plain(Setting("compile")) == "compiling"
    for name in SETTINGS:
        for value in ("", "7"):
            assert "--" not in plain(Setting(name, value))
            # The command line's words are the option, whatever the library says.
            assert cli_say(Setting(name, value)).startswith("--" + name.replace("_", "-"))


def test_a_message_given_as_a_function_is_worded_by_the_application() -> None:
    window = Setting("window", "6")
    refusal = Refusal(lambda say: f"{say(window)}: too long")
    assert str(refusal) == "window 6: too long" and refusal.words(plain) == str(refusal)
    assert refusal.words(cli_say) == "--window 6: too long" == cli_words(refusal)
    assert refusal.words(lambda setting: f"<{setting.name}>") == "<window>: too long"
    # A plain string is itself, whatever the words; the parts of picture.py's refusals stay.
    said = Refusal("no such thing", kind="raw", reason="it is", way="do this")
    assert said.words(cli_say) == str(said) == "no such thing"
    assert (said.kind, said.reason, said.way) == ("raw", "it is", "do this")
    assert Refusal("x").kind == ""
    assert isinstance(JobError("x"), Refusal)


def test_the_plan_s_refusals_are_of_a_kind_and_a_plan_refusal() -> None:
    for kind in KINDS:
        error = kind("a message")
        assert isinstance(error, PlanRefused) and isinstance(error, JobError)
        assert (error.phase, error.instead, error.shortfall) == (
            ("host" if kind is HostShort else ""),
            None,
            None,
        )
    assert {kind("x").kind for kind in KINDS} == {"pin", "memory", "host", "input", "compile"}
    # The names the command line and the tests catch the planners' refusals by.
    assert shot_plan.PlanError is PlanRefused and still_plan.PlanRefusal is PlanRefused
    assert shot_plan.Shortfall.__module__ == "seedvr2x.errors"


def test_a_refusal_reworded_keeps_its_kind_and_facts() -> None:
    window = Setting("window", "6")
    first = MemoryShort(
        lambda say: f"a window of 9 takes too much; {say(window)} fits",
        need=10,
        there=7,
        short=3,
        freeable=True,
        reserve=5,
        phase="dit",
        instead=window,
    )
    words = first.words
    again = first.reworded(lambda say: f"{words(say)}; and more", instead=None)
    assert type(again) is MemoryShort and again is not first
    assert (again.need, again.there, again.short, again.freeable, again.reserve) == (
        10,
        7,
        3,
        True,
        5,
    )
    assert (again.phase, again.kind, again.instead) == ("dit", "memory", None)
    assert str(again) == "a window of 9 takes too much; window 6 fits; and more"
    assert again.words(cli_say) == "a window of 9 takes too much; --window 6 fits; and more"
    assert again.args == (str(again),)
    # The first is as it was.
    assert first.instead == window and str(first) == "a window of 9 takes too much; window 6 fits"
    assert again.reworded("plain").words(cli_say) == "plain"


def test_a_reworded_refusal_is_worded_after_the_except_clause_that_made_it_has_ended() -> None:
    # A closure over an except clause's name breaks once the clause ends (Python unbinds it):
    # the planners reword a refusal inside one and the application words it later.
    window = Setting("window", "6")
    try:
        raise MemoryShort(lambda say: f"too long; {say(window)} fits", need=2, there=1)
    except MemoryShort as error:
        words = error.words
        again = error.reworded(lambda say: f"{words(say)}; and more")
    assert "error" not in locals()
    assert again.words(cli_say) == "too long; --window 6 fits; and more"
    assert (
        str(again) == "too long; window 6 fits; and more"
        and isinstance(again, MemoryShort)
        and again.need == 2
    )


def test_the_model_files_settings_have_plain_words_and_the_apps_options() -> None:
    assert {"vae_model", "model_dir", "cuts"} <= set(SETTINGS)
    assert plain(Setting("vae_model", "v.safetensors")) == "VAE file v.safetensors"
    assert plain(Setting("model_dir")) == "the model directory"
    assert plain(Setting("cuts")) == "the cut list"
    assert cli_say(Setting("vae_model", "v")) == "--vae-model v"
    assert cli_say(Setting("model_dir")) == "--model-dir"
    assert cli_say(Setting("cuts")) == "--cuts"
    # A clause that only points at an application's option is left out of the library's words.
    assert aside(cli_say, " (--x)") == " (--x)" and aside(plain, " (--x)") == ""
    assert aside(plain, " (--x)", " (y)") == " (y)"


def test_the_model_errors_are_of_a_kind_and_carry_their_facts() -> None:
    kinds = {
        ModelMissing: "missing",
        ModelUnknown: "unknown",
        ModelDamaged: "damaged",
        ModelChanged: "changed",
        ModelNotPinned: "not-pinned",
        ModelUnreachable: "unreachable",
        ModelNoRoom: "no-room",
        ModelUnreadable: "unreadable",
    }
    for kind, slug in kinds.items():
        error = kind("a message", role="dit", file="f.safetensors")
        assert isinstance(error, ModelError) and isinstance(error, Refusal)
        assert (error.kind, error.role, error.file, str(error)) == (
            slug,
            "dit",
            "f.safetensors",
            "a message",
        )
        assert (error.expected, error.found, error.settings, error.command) == (None, None, (), "")
    assert ModelError("x").kind == "" and ModelUnreachable("x", why="offline").why == "offline"
    assert (ModelNoRoom("x", need=2, free=1).need, ModelNoRoom("x", need=2, free=1).free) == (2, 1)
    # Several, each part's words one line; the host's memory and the pictures' have kinds too.
    several = ModelsRefused([ModelMissing("a"), ModelDamaged("b")])
    assert several.kind == "several" and str(several) == "a\nb" and several.words(cli_say) == "a\nb"
    assert HostMemoryError("x", need=3, available=1).kind == "host-memory"
    assert isinstance(PictureRefused("x", kind="raw"), Refusal)


def test_the_files_refusals_are_raised_by_their_kind_and_name_no_option(tmp_path: Path) -> None:
    def refused(path: Path, role: weights.Role) -> ModelError:
        with pytest.raises(ModelError) as caught:
            weights.check(path, role)
        assert "--" not in str(caught.value).replace(str(path), "")
        return caught.value

    gone = refused(tmp_path / "gone.safetensors", "dit")
    assert isinstance(gone, ModelMissing) and (gone.role, gone.file) == ("dit", "gone.safetensors")
    empty = tmp_path / "empty.safetensors"
    empty.write_bytes(b"")
    cut = refused(empty, "vae")
    assert isinstance(cut, ModelDamaged) and (cut.figure, cut.found, cut.role) == ("size", 0, "vae")
    assert isinstance(refused(tmp_path, "dit"), ModelUnreadable)
    vae = write(tmp_path / "vae.safetensors", as_dtype("vae", "F16"))
    wrong = refused(vae, "dit")
    # A VAE given as the DiT: the setting is a fact, the option worded by the app alone.
    assert isinstance(wrong, ModelUnknown) and wrong.settings == (Setting("dit_model"),)
    assert "given as the DiT:" in str(wrong) and "(--dit-model)" in cli_words(wrong)
    assert "given as the DiT (--dit-model):" in cli_words(wrong)
    # check_models joins the files refused, the advice and the unnamed detector worded alike.
    with pytest.raises(ModelsRefused) as several:
        weights.check_models(tmp_path / "gone", empty, None, tmp_path / "no")
    assert [type(part) for part in several.value.parts] == [
        ModelMissing,
        ModelDamaged,
        ModelMissing,
    ]
    assert "--" not in str(several.value).replace(str(tmp_path), "")
    assert "--model-dir (--cuts, a cut list, runs without them)" in cli_words(several.value)


def test_the_picture_s_refusals_carry_their_facts_and_the_resolution_its_setting() -> None:
    with pytest.raises(PictureRefused) as caught:
        picture.read(b"not a picture at all", "FILE")
    assert caught.value.kind == "not-a-picture" and (caught.value.pixels, caught.value.frames) == (
        None,
        None,
    )
    with pytest.raises(TargetTooSmall) as small:
        frames.check_target((16, 16), 16)
    assert small.value.kind == "too-small" and small.value.way == "resolution 16"
    assert str(small.value).startswith("resolution 16: ") and "--" not in str(small.value)
    assert cli_words(small.value).startswith("--resolution 16: ")
