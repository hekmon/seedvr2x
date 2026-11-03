"""The server's start-up (main.py; DESIGN.md, The image server, Its start): the options and their
defaults, --max-size, --listen and --max-input parsed or refused, the pins refused beside more
than one picture at a time, the API key (--api-key over SEEDVR2X_API_KEY, the source logged and
the key never, an empty one refused, none warned of), and the environment set before torch is
imported (the allocator's setting, Hugging Face's telemetry, torch's C++ log level).

`main.main` runs with `main.serve` replaced by a recorder, so that the start goes no further
than the options it hands on, with no GPU; one test runs the real `serve` with a card that
refuses at its construction (main returns 1 before anything is loaded). The environment is the
test's own: the variables main sets are cleared first through monkeypatch, so that they are put
back as they were. memory.startup asks no driver (the autouse `no_driver_asked`)."""

import logging
import os
import signal
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import main
import service as services
from metrics import Metrics
from seedvr2x import memory
from seedvr2x.errors import Refusal
from seedvr2x.still_plan import Pins

# pyright: reportPrivateUsage=false

WATCHED = (
    "HF_HUB_DISABLE_TELEMETRY",
    "TORCH_CPP_LOG_LEVEL",
    main.KEY_VARIABLE,
    *memory.VARIABLES,
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # setenv then delenv records each variable's original state, so that what main sets is
    # undone at the end of the test.
    for name in WATCHED:
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)
    yield


@dataclass
class Served:
    options: services.Options
    key: str | None
    listen: tuple[str, int]
    machine: Any
    metrics: Metrics


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[Served]:
    calls: list[Served] = []

    def record(options: Any, key: Any, listen: Any, machine: Any, found_metrics: Any) -> int:
        calls.append(Served(options, key, listen, machine, found_metrics))
        return 0

    monkeypatch.setattr(main, "serve", record)
    return calls


def run(*argv: str) -> int:
    return main.main(list(argv))


# ---- The parser ----------------------------------------------------------------------------------


def test_the_defaults() -> None:
    args = main.parser().parse_args([])
    assert args.listen == ("127.0.0.1", 8390)
    assert args.max_size == (3840, 2160)
    assert args.max_input == 50_000_000
    assert args.max_concurrent == 1
    assert args.queue == 8
    assert args.model_dir is None
    assert args.dit_model == "seedvr2x_ema_7b_sharp_fp16.safetensors"
    assert args.vae_model == "seedvr2x_ema_vae_fp16.safetensors"
    assert args.encode_tile is None and args.decode_tile is None and args.swap_blocks is None
    assert args.api_key is None
    assert args.verbose is False


def test_the_default_port_is_not_comfyuis() -> None:
    assert main.PORT == 8390 != 8188


def test_the_options_reach_serve_with_their_defaults(served: list[Served]) -> None:
    assert run() == 0
    (call,) = served
    assert call.options == services.Options(
        (3840, 2160),
        50_000_000,
        1,
        8,
        None,
        main.DIT,
        main.VAE,
        Pins(None, None, None),
    )
    assert call.key is None
    assert call.listen == ("127.0.0.1", 8390)
    assert call.machine is services.Card
    assert isinstance(call.metrics, Metrics)


def test_every_option_reaches_serve(served: list[Served], tmp_path: Path) -> None:
    code = run(
        "--listen=0.0.0.0:9000",
        "--max-size=1920x1080",
        "--max-input=1000000",
        "--max-concurrent=1",
        "--queue=0",
        f"--model-dir={tmp_path}",
        "--dit-model=a.safetensors",
        "--vae-model=b.safetensors",
        "--encode-tile=512",
        "--decode-tile=0",
        "--swap-blocks=4",
        "--api-key=secret",
    )
    assert code == 0
    (call,) = served
    assert call.listen == ("0.0.0.0", 9000)
    assert call.key == "secret"
    assert call.options == services.Options(
        (1920, 1080), 1_000_000, 1, 0, tmp_path, "a.safetensors", "b.safetensors", Pins(512, 0, 4)
    )


@pytest.mark.parametrize(
    ("text", "size"),
    [
        ("3840x2160", (3840, 2160)),
        ("2160x3840", (2160, 3840)),
        ("32x32", (32, 32)),
        ("1920x1080", (1920, 1080)),
        (" 640x480 ", (640, 480)),
        ("999998x32", (999998, 32)),
    ],
)
def test_max_size_is_parsed_as_given(text: str, size: tuple[int, int]) -> None:
    assert main.parser().parse_args(["--max-size", text]).max_size == size


def test_max_size_refuses_digits_that_are_not_ascii() -> None:
    with pytest.raises(SystemExit):
        main.parser().parse_args(
            ["--max-size", "\u0663\u0668\u0664\u0660x\u0662\u0661\u0666\u0660"]
        )


def test_listen_refuses_a_port_in_digits_that_are_not_ascii() -> None:
    with pytest.raises(SystemExit):
        main.parser().parse_args(["--listen", "h:\u0661\u0662"])


@pytest.mark.parametrize(
    "text",
    [
        "3841x2160",  # odd width
        "3840x2161",  # odd height
        "3841x2161",
        "30x2160",  # a side under 32
        "3840x30",
        "16x16",
        "0x0",
        "abc",
        "3840",
        "3840x",
        "x2160",
        "3840*2160",
        "1x2x3",
        "-3840x2160",
        "3840x2160x1",
        "1000000x1000000",  # more than 6 digits
        "",
    ],
)
def test_max_size_refused(text: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main.parser().parse_args(["--max-size", text])
    assert error.value.code == 2
    assert "--max-size" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("given", "bound"),
    [("2160x3840", (3840, 2160)), ("3840x2160", (3840, 2160)), ("1080x1920", (1920, 1080))],
)
def test_main_orders_the_bound_long_side_first(
    served: list[Served], given: str, bound: tuple[int, int]
) -> None:
    assert run("--max-size", given) == 0
    assert served[0].options.bound == bound


@pytest.mark.parametrize(
    ("text", "listen"),
    [
        ("127.0.0.1:8390", ("127.0.0.1", 8390)),
        ("0.0.0.0:80", ("0.0.0.0", 80)),
        ("localhost:9000", ("localhost", 9000)),
        ("[::1]:8390", ("::1", 8390)),
        ("[::]:65535", ("::", 65535)),
        (":8390", ("127.0.0.1", 8390)),
    ],
)
def test_listen_forms(text: str, listen: tuple[str, int]) -> None:
    assert main.parser().parse_args(["--listen", text]).listen == listen


@pytest.mark.parametrize(
    "text",
    [
        "localhost",
        "8390",
        "127.0.0.1:0",
        "127.0.0.1:65536",
        "host:abc",
        "host:",
        "",
        "h:-1",
    ],
)
def test_listen_refused(text: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main.parser().parse_args(["--listen", text])
    assert error.value.code == 2
    assert "HOST:PORT" in capsys.readouterr().err


@pytest.mark.parametrize(
    "option, text",
    [
        ("--max-input", "0"),
        ("--max-input", "-5"),
        ("--max-input", "1.5"),
        ("--max-input", "abc"),
        ("--max-concurrent", "0"),
        ("--max-concurrent", "-1"),
        ("--queue", "-1"),
        ("--queue", "x"),
        ("--encode-tile", "100"),
        ("--decode-tile", "520"),
        ("--encode-tile", "-64"),
        ("--swap-blocks", "-1"),
        ("--swap-blocks", "\u0663"),
    ],
)
def test_numbers_refused(option: str, text: str) -> None:
    with pytest.raises(SystemExit) as error:
        main.parser().parse_args([option, text])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "option, text",
    [("--queue", "0"), ("--encode-tile", "0"), ("--decode-tile", "512"), ("--encode-tile", "576")],
)
def test_numbers_taken(option: str, text: str) -> None:
    main.parser().parse_args([option, text])


# ---- Pins and limits refused before serve --------------------------------------------------------


@pytest.mark.parametrize(
    "pin", [["--swap-blocks", "0"], ["--encode-tile", "512"], ["--decode-tile", "0"]]
)
def test_pins_beside_several_pictures_are_refused(
    pin: list[str], served: list[Served], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    assert run("--max-concurrent", "2", *pin) == 1
    assert served == []
    assert "--max-concurrent 2" in caplog.text


def test_pins_with_one_picture_are_taken(served: list[Served]) -> None:
    assert run("--max-concurrent", "1", "--swap-blocks", "3") == 0
    assert served[0].options.pins == Pins(None, None, 3)


def test_several_pictures_without_pins_are_taken(served: list[Served]) -> None:
    assert run("--max-concurrent", "4") == 0
    assert served[0].options.count == 4


def test_a_max_input_over_pillows_limit_is_refused(
    served: list[Served], caplog: pytest.LogCaptureFixture
) -> None:
    from PIL import Image

    limit = 2 * (Image.MAX_IMAGE_PIXELS or 0)
    assert limit > 0
    assert run("--max-input", str(limit + 1)) == 1
    assert served == []
    assert "Pillow" in caplog.text
    assert run("--max-input", str(limit)) == 0
    assert len(served) == 1


def test_a_card_that_refuses_ends_the_start_with_1(monkeypatch: pytest.MonkeyPatch) -> None:
    # The real serve, with the card refusing at its construction: no GPU is asked for, and the
    # signals main takes are left alone.
    class NoCard:
        def __init__(self) -> None:
            raise Refusal("no CUDA device: stand-in")

    monkeypatch.setattr(signal, "signal", lambda *args: None)
    monkeypatch.setattr(services, "Card", NoCard)
    assert run() == 1


# ---- The API key ---------------------------------------------------------------------------------

KEY = "k3y-that-must-not-be-logged"


def test_the_option_wins_over_the_variable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(main.KEY_VARIABLE, "from-the-variable")
    caplog.set_level(logging.DEBUG)
    assert main._key("from-the-option") == "from-the-option"
    assert "--api-key" in caplog.text


def test_the_variable_is_taken_without_the_option(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(main.KEY_VARIABLE, KEY)
    caplog.set_level(logging.DEBUG)
    assert main._key(None) == KEY
    assert main.KEY_VARIABLE in caplog.text


@pytest.mark.parametrize("source", ["option", "variable"])
def test_the_key_is_never_in_a_log_record(
    source: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    if source == "variable":
        monkeypatch.setenv(main.KEY_VARIABLE, KEY)
        main._key(None)
    else:
        main._key(KEY)
    assert caplog.records
    for record in caplog.records:
        assert KEY not in record.getMessage()
        assert KEY not in str(record.args)
        assert KEY not in str(record.__dict__)
    assert KEY not in caplog.text


def test_the_key_is_never_logged_by_the_whole_start(
    served: list[Served], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    assert run("--api-key", KEY, "-v") == 0
    assert served[0].key == KEY
    assert KEY not in caplog.text


@pytest.mark.parametrize("empty", ["", "   ", "\t\n"])
@pytest.mark.parametrize("source", ["option", "variable"])
def test_an_empty_key_is_refused(
    source: str, empty: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    if source == "variable":
        monkeypatch.setenv(main.KEY_VARIABLE, empty)
        given = None
    else:
        given = empty
    with pytest.raises(SystemExit) as error:
        main._key(given)
    assert error.value.code == 1
    assert "empty" in caplog.text


def test_an_empty_option_is_refused_even_with_a_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(main.KEY_VARIABLE, KEY)
    with pytest.raises(SystemExit):
        main._key("")


@pytest.mark.parametrize("key", [f" {KEY}", f"{KEY} ", f"{KEY}\n", f"\t{KEY}"])
@pytest.mark.parametrize("source", ["option", "variable"])
def test_a_key_with_a_blank_at_an_end_is_refused_not_stripped(
    source: str, key: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    if source == "variable":
        monkeypatch.setenv(main.KEY_VARIABLE, key)
        given = None
    else:
        given = key
    with pytest.raises(SystemExit) as error:
        main._key(given)
    assert error.value.code == 1
    assert "blank" in caplog.text
    assert KEY not in caplog.text


@pytest.mark.parametrize(
    "key", ["clé-secrète", "a key", "key\x01", "key\x7f", "日本語", "tab\tkey"]
)
@pytest.mark.parametrize("source", ["option", "variable"])
def test_a_key_that_is_not_printable_ascii_is_refused(
    source: str, key: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    if source == "variable":
        monkeypatch.setenv(main.KEY_VARIABLE, key)
        given = None
    else:
        given = key
    with pytest.raises(SystemExit) as error:
        main._key(given)
    assert error.value.code == 1
    assert "printable ASCII" in caplog.text
    assert key not in caplog.text


def test_no_key_is_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    assert main._key(None) is None
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "no API key" in warnings[0].getMessage()
    assert main.KEY_VARIABLE in warnings[0].getMessage()


def test_an_empty_key_ends_the_start_before_serve(
    served: list[Served], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(main.KEY_VARIABLE, " ")
    with pytest.raises(SystemExit) as error:
        run()
    assert error.value.code == 1
    assert served == []


# ---- The environment -----------------------------------------------------------------------------


def test_the_allocator_is_expandable_when_the_user_set_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main._environment(False)
    assert os.environ[memory.VARIABLES[0]] == memory.EXPANDABLE
    assert memory.VARIABLES[1] not in os.environ


@pytest.mark.parametrize("name", memory.VARIABLES)
def test_a_users_allocator_stands(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(name, "backend:native,garbage_collection_threshold:0.8")
    main._environment(False)
    assert os.environ[name] == "backend:native,garbage_collection_threshold:0.8"
    other = next(each for each in memory.VARIABLES if each != name)
    # The default is written only where the user set neither.
    assert other not in os.environ


@pytest.mark.parametrize("name", memory.VARIABLES)
def test_an_empty_allocator_setting_is_removed(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(name, "")
    main._environment(False)
    if name == memory.VARIABLES[0]:
        # Removed, and the default put in its place.
        assert os.environ[name] == memory.EXPANDABLE
    else:
        assert memory.VARIABLES[0] in os.environ
        assert name not in os.environ


def test_an_empty_first_variable_does_not_hide_the_second(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(memory.VARIABLES[0], "")
    monkeypatch.setenv(memory.VARIABLES[1], "backend:native")
    main._environment(False)
    assert memory.VARIABLES[0] not in os.environ
    assert os.environ[memory.VARIABLES[1]] == "backend:native"


@pytest.mark.parametrize("name", memory.VARIABLES)
def test_cuda_malloc_async_warns(
    name: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(name, "backend:cudaMallocAsync")
    main._environment(False)
    assert os.environ[name] == "backend:cudaMallocAsync"
    warnings = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert any("cudaMallocAsync" in record.getMessage() for record in warnings)


def test_no_warning_for_the_default_allocator(caplog: pytest.LogCaptureFixture) -> None:
    main._environment(False)
    assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []


def test_hugging_faces_telemetry_is_off_unless_the_user_says(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main._environment(False)
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")
    main._environment(False)
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "0"


def test_torchs_cpp_log_level_is_error_without_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    main._environment(False)
    assert os.environ["TORCH_CPP_LOG_LEVEL"] == "ERROR"
    monkeypatch.setenv("TORCH_CPP_LOG_LEVEL", "INFO")
    main._environment(False)
    assert os.environ["TORCH_CPP_LOG_LEVEL"] == "INFO"


def test_torchs_cpp_log_level_is_untouched_with_verbose() -> None:
    main._environment(True)
    assert "TORCH_CPP_LOG_LEVEL" not in os.environ
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_main_sets_the_environment_before_it_hands_on(served: list[Served]) -> None:
    assert run() == 0
    assert os.environ[memory.VARIABLES[0]] == memory.EXPANDABLE
    assert os.environ["TORCH_CPP_LOG_LEVEL"] == "ERROR"
    assert run("-v") == 0
