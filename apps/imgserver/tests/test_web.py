"""The server's HTTP over real sockets on localhost, on the CPU with the stand-in models
(serving.py): every endpoint, header, error and state DESIGN.md (The image server) gives, the
admission and the queue, the key, /health and /metrics, and a picture's file against what the
library makes of the same picture."""

import io
import re
import socket
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from prometheus_client.parser import text_string_to_metric_families
from serving import (
    INPUT,
    Gate,
    HeldStart,
    Live,
    library_file,
    live,
    png_of,
    quiet_driver,
    until,
)

import answers
import reservation
import service as services
import web
from seedvr2x import model, picture, png, still, waiting
from seedvr2x import record as records
from seedvr2x.peaks import GIB

TIMING = re.compile(r'queue;dur=[\d.]+;desc="[^"]*", upscale;dur=[\d.]+;desc="[^"]*"')


@pytest.fixture(scope="module")
def shared(cpu_models: model.Models, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Live]:
    """One server for the tests that need no fresh one: started once, shut at the module's end."""
    with pytest.MonkeyPatch.context() as patches:
        quiet_driver(patches)
        with live(patches, tmp_path_factory.mktemp("shared"), cpu_models) as server:
            yield server


@pytest.fixture
def server(shared: Live) -> Iterator[Live]:
    """The shared server, its failure hook taken off after each test."""
    yield shared
    shared.machine.fail = None


def ihdr(data: bytes) -> tuple[int, int]:
    """A PNG's (bit depth, colour type), from its IHDR."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return data[24], data[25]


def image_of(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def scrape(server: Live) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    """/metrics read back with prometheus_client's parser: (sample name, sorted labels) -> value."""
    status, headers, body = server.request("GET", "/metrics")
    assert status == 200
    assert headers["content-type"].startswith("text/plain; version=0.0.4")
    found: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
    for family in text_string_to_metric_families(body.decode()):
        for sample in family.samples:
            found[(sample.name, tuple(sorted(sample.labels.items())))] = sample.value
    return found


def metric(server: Live, name: str, **labels: str) -> float:
    return scrape(server)[(f"seedvr2x_imgserver_{name}", tuple(sorted(labels.items())))]


def raw(
    server: Live,
    headers: list[tuple[str, str]],
    body: bytes | None = None,
    method: str = "POST",
    path: str = "/upscale",
) -> tuple[int, dict[str, str], bytes]:
    """A request with exactly the headers given, in order and as often as given (http.client's
    request() adds a Content-Length to a POST and merges nothing)."""
    connection = server.connect()
    try:
        connection.putrequest(method, path, skip_accept_encoding=True)
        for name, value in headers:
            connection.putheader(name, value)
        connection.endheaders(body)
        answer = connection.getresponse()
        return (
            answer.status,
            {name.lower(): value for name, value in answer.getheaders()},
            answer.read(),
        )
    finally:
        connection.close()


# ---- 200 ----------------------------------------------------------------------------------


def test_a_picture_is_upscaled_with_its_headers(server: Live) -> None:
    """A PNG in, an 8-bit PNG out; the seed given comes back with both sizes and the timings."""
    status, headers, body = server.upscale(png_of(40, 24), 48, seed=5)
    assert status == 200, body
    assert headers["content-type"] == "image/png"
    assert headers["content-length"] == str(len(body))
    assert headers["seedvr2x-seed"] == "5"
    assert headers["seedvr2x-source-size"] == "40x24"
    assert headers["seedvr2x-output-size"] == "80x48"
    assert TIMING.fullmatch(headers["server-timing"]), headers["server-timing"]
    image = image_of(body)
    assert (image.mode, image.size) == ("RGB", (80, 48))
    assert ihdr(body) == (8, 2)


def test_a_seed_absent_is_drawn_at_random_within_the_range(server: Live) -> None:
    seeds: list[int] = []
    for _ in range(2):
        status, headers, body = server.upscale(png_of(40, 24), 48)
        assert status == 200, body
        seeds.append(int(headers["seedvr2x-seed"]))
    assert all(0 <= seed <= answers.SEED_MAX for seed in seeds)
    assert seeds[0] != seeds[1]


def test_the_seed_returned_makes_the_picture_again(server: Live) -> None:
    data = png_of(40, 24, seed=1)
    _, headers, first = server.upscale(data, 48)
    _, _, again = server.upscale(data, 48, seed=int(headers["seedvr2x-seed"]))
    assert again == first


def test_a_16_bit_picture_comes_back_at_16_bits(server: Live) -> None:
    pixels = np.random.default_rng(3).random((24, 40, 3), dtype=np.float32)
    data = png.write(pixels, bits=16, grey=False, icc=None, chunks=())
    status, _, body = server.upscale(data, 48, seed=1)
    assert status == 200, body
    assert ihdr(body) == (16, 2)
    assert image_of(body).size == (80, 48)


def test_a_16_bit_grey_picture_comes_back_grey_at_16_bits(server: Live) -> None:
    pixels = np.random.default_rng(4).random((24, 40, 3), dtype=np.float32)
    data = png.write(pixels, bits=16, grey=True, icc=None, chunks=())
    status, _, body = server.upscale(data, 48, seed=1)
    assert status == 200, body
    assert ihdr(body) == (16, 0)
    assert image_of(body).mode == "I;16"


def test_a_grey_picture_comes_back_grey(server: Live) -> None:
    status, _, body = server.upscale(png_of(40, 24, mode="L"), 48, seed=1)
    assert status == 200, body
    assert ihdr(body) == (8, 0)
    assert image_of(body).mode == "L"


def test_an_opaque_alpha_is_dropped(server: Live) -> None:
    status, _, body = server.upscale(png_of(40, 24, mode="RGBA"), 48, seed=1)
    assert status == 200, body
    assert ihdr(body) == (8, 2)
    assert image_of(body).mode == "RGB"


def test_a_picture_turned_by_its_exif_orientation_comes_back_upright(server: Live) -> None:
    """Orientation 6 stored as 40x24: sizes as stored for the source, swapped for the output, no
    EXIF left in the file."""
    status, headers, body = server.upscale(png_of(40, 24, orientation=6), 48, seed=1)
    assert status == 200, body
    assert headers["seedvr2x-source-size"] == "40x24"
    assert headers["seedvr2x-output-size"] == "48x80"
    image = image_of(body)
    assert image.size == (48, 80)
    assert len(image.getexif()) == 0


def test_the_output_carries_the_record_and_the_ai_mark(server: Live) -> None:
    status, _, body = server.upscale(png_of(40, 24), 48, seed=9)
    assert status == 200, body
    info = image_of(body).info
    xmp = info["XML:com.adobe.xmp"]
    assert "seedvr2x-imgserver" in xmp
    assert records.DIGITAL_SOURCE_TYPE in xmp
    assert "<seedvr2x:seed>9</seedvr2x:seed>" in xmp
    assert str(info["Software"]).startswith("seedvr2x-imgserver ")


def test_the_file_is_the_librarys_bit_for_bit(server: Live, cpu_models: model.Models) -> None:
    """The CPU half of one picture through the server bit for bit the CLI's (whose own still path
    is held to still.upscale by the CLI's tests/test_still_cli.py)."""
    data = png_of(40, 24, seed=2)
    status, _, body = server.upscale(data, 48, seed=7)
    assert status == 200, body
    assert body == library_file(server, cpu_models, data, 48, 7)


def test_the_orientation_and_a_grey_picture_are_the_librarys_bit_for_bit(
    server: Live, cpu_models: model.Models
) -> None:
    data = png_of(40, 24, seed=2, mode="L", orientation=3)
    status, _, body = server.upscale(data, 48, seed=7)
    assert status == 200, body
    assert body == library_file(server, cpu_models, data, 48, 7)


# ---- WebP and Accept ------------------------------------------------------------------------


def test_a_webp_is_answered_where_asked(server: Live, cpu_models: model.Models) -> None:
    data = png_of(40, 24, seed=2)
    status, headers, body = server.upscale(data, 48, seed=7, headers={"Accept": "image/webp"})
    assert status == 200, body
    assert headers["content-type"] == "image/webp"
    assert image_of(body).format == "WEBP"
    assert image_of(body).size == (80, 48)
    assert body == library_file(server, cpu_models, data, 48, 7, as_webp=True)


def sixteen_bit() -> bytes:
    pixels = np.random.default_rng(3).random((24, 40, 3), dtype=np.float32)
    return png.write(pixels, bits=16, grey=False, icc=None, chunks=())


def test_a_picture_webp_cannot_hold_gets_a_png_where_the_client_takes_it(server: Live) -> None:
    status, headers, body = server.upscale(
        sixteen_bit(), 48, seed=1, headers={"Accept": "image/webp, image/png"}
    )
    assert status == 200, body
    assert headers["content-type"] == "image/png"
    assert ihdr(body)[0] == 16


def test_a_picture_webp_cannot_hold_is_refused_where_the_client_takes_no_png(
    server: Live,
) -> None:
    status, _, body = server.upscale(sixteen_bit(), 48, headers={"Accept": "image/webp"})
    assert status == 406
    assert "image/png" in body.decode()


def test_a_406_names_the_two_formats_the_server_answers(server: Live) -> None:
    status, headers, body = server.upscale(
        b"not a picture at all", 48, headers={"Accept": "image/jpeg"}
    )
    assert status == 406
    assert "accept-post" not in headers
    text = body.decode()
    assert "image/png" in text
    assert "image/webp" in text


def test_an_accept_the_server_cannot_meet_is_refused_before_the_body_matters(
    server: Live,
) -> None:
    status, _, body = server.upscale(b"not a picture at all", 48, headers={"Accept": "image/jpeg"})
    assert status == 406
    assert "image/png" in body.decode()


# ---- OPTIONS, 405 ---------------------------------------------------------------------------

ACCEPT_POST = "image/png, image/jpeg, image/webp, image/tiff, image/bmp, image/gif"
# The headers every answer carries, whatever it says.
ALWAYS = ("date", "server")


def test_the_media_types_cover_every_format_the_library_reads() -> None:
    assert set(answers.MEDIA_TYPES) == set(picture.FORMATS)
    assert answers.ACCEPT_POST == ACCEPT_POST


def test_options_says_what_upscale_takes(server: Live) -> None:
    status, headers, body = server.request("OPTIONS", "/upscale")
    assert status == 204
    assert body == b""
    for name in ALWAYS:
        headers.pop(name)
    # No Content-Length on a 204 (RFC 9110, 8.6), and no cross-origin header: that is the
    # reverse proxy's.
    assert headers == {"allow": "POST, OPTIONS", "accept-post": ACCEPT_POST}


def test_options_takes_a_query_and_a_body_it_ignores(server: Live) -> None:
    status, headers, body = server.request("OPTIONS", "/upscale?x=1", b"ignored")
    assert (status, body) == (204, b"")
    assert headers["accept-post"] == ACCEPT_POST


def test_options_is_behind_the_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    with live(monkeypatch, tmp_path, cpu_models, key="secret") as server:
        for given in (
            [],
            [("Authorization", "Bearer nope")],
            [("Authorization", "Basic secret")],
        ):
            status, headers, body = raw(server, given, method="OPTIONS")
            assert status == 401, given
            assert headers["www-authenticate"] == "Bearer"
            assert "API key" in body.decode()
            assert "accept-post" not in headers
        status, headers, body = raw(server, [("Authorization", "Bearer secret")], method="OPTIONS")
        assert (status, body) == (204, b"")
        assert headers["accept-post"] == ACCEPT_POST
        # A path that isn't /upscale is a 404 whatever the key says, as for POST.
        assert raw(server, [], method="OPTIONS", path="/nothing")[0] == 404
        assert metric(server, "requests_total", status="401") == 3
        assert metric(server, "requests_total", status="204") == 1


def test_options_on_another_path_is_a_404_and_on_a_star_a_501(server: Live) -> None:
    for path in ("/", "/health", "/metrics", "/nothing"):
        status, headers, _ = server.request("OPTIONS", path)
        assert status == 404, path
        assert "allow" not in headers
        assert "accept-post" not in headers
    status, headers, body = raw(server, [], method="OPTIONS", path="*")
    assert status == 501
    assert "accept-post" not in headers
    assert "OPTIONS" in body.decode()


def test_options_carries_no_cross_origin_header_whatever_the_origin_asks(server: Live) -> None:
    _, headers, _ = server.request(
        "OPTIONS",
        "/upscale",
        headers={
            "Origin": "https://example.org",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert not [name for name in headers if name.startswith("access-control-")]


def test_get_on_upscale_is_a_405_that_allows_post_and_options(server: Live) -> None:
    status, headers, body = server.request("GET", "/upscale")
    assert status == 405
    assert headers["allow"] == "POST, OPTIONS"
    assert "accept-post" not in headers
    assert "POST and OPTIONS" in body.decode()


# ---- 400 ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("headers", "word"),
    [
        ({}, "Seedvr2x-Resolution: missing"),
        ({"Seedvr2x-Resolution": "big"}, "Seedvr2x-Resolution"),
        ({"Seedvr2x-Resolution": "0"}, "Seedvr2x-Resolution"),
        ({"Seedvr2x-Resolution": "48", "Seedvr2x-Seed": str(answers.SEED_MAX + 1)}, "largest seed"),
        ({"Seedvr2x-Resolution": "48", "Seedvr2x-Seed": "-1"}, "Seedvr2x-Seed"),
        ({"Seedvr2x-Resolution": "48", "Seedvr2x-Foo": "1"}, "Seedvr2x-Foo"),
    ],
)
def test_a_header_wrong_or_unknown_is_a_400(
    server: Live, headers: dict[str, str], word: str
) -> None:
    status, _, body = server.request("POST", "/upscale", png_of(40, 24), headers)
    assert status == 400
    assert word in body.decode()


def test_a_setting_given_twice_is_a_400(server: Live) -> None:
    sent = [("Seedvr2x-Resolution", "48"), ("Seedvr2x-Resolution", "96"), ("Content-Length", "1")]
    status, _, body = raw(server, sent, b"x")
    assert status == 400
    assert "twice" in body.decode()


def test_a_content_length_that_is_no_number_is_a_400(server: Live) -> None:
    # The last two: 21 digits, and 5,000 (past Python's limit on int() of a string, 4,300).
    for lengths in (["12abc"], ["5", "5"], ["1" * 21], ["9" * 5000]):
        sent = [("Seedvr2x-Resolution", "48"), *(("Content-Length", each) for each in lengths)]
        status, _, _ = raw(server, sent)
        assert status == 400


# ---- 401 ------------------------------------------------------------------------------------


def test_the_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    with live(monkeypatch, tmp_path, cpu_models, key="secret") as server:
        data = png_of(40, 24)
        length = ("Content-Length", str(len(data)))
        resolution = ("Seedvr2x-Resolution", "48")
        refused = {
            "none": [resolution, length],
            "wrong": [resolution, length, ("Authorization", "Bearer nope")],
            "scheme": [resolution, length, ("Authorization", "Basic secret")],
            "two": [
                resolution,
                length,
                ("Authorization", "Bearer secret"),
                ("Authorization", "Bearer secret"),
            ],
            "no token": [resolution, length, ("Authorization", "Bearer")],
        }
        for name, headers in refused.items():
            status, answer, body = raw(server, headers, data)
            assert status == 401, name
            assert answer["www-authenticate"] == "Bearer", name
            assert "API key" in body.decode(), name
        status, _, body = raw(server, [*refused["none"], ("Authorization", "Bearer secret")], data)
        assert status == 200, body
        # The key is no business of the probes'.
        assert server.request("GET", "/health")[0] == 200
        assert server.request("GET", "/metrics")[0] == 200
        assert metric(server, "requests_total", status="401") == len(refused)


def test_the_key_is_checked_before_a_byte_of_the_body_is_waited_for(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    """The headers of a POST with a large Content-Length and no body, no key: the 401 comes at
    once, not after the body that never arrives."""
    with live(monkeypatch, tmp_path, cpu_models, key="secret") as server:
        connection = server.connect(timeout=5)
        try:
            started = time.monotonic()
            connection.putrequest("POST", "/upscale", skip_accept_encoding=True)
            connection.putheader("Seedvr2x-Resolution", "48")
            connection.putheader("Content-Length", "10000000")
            connection.endheaders()
            answer = connection.getresponse()
            body = answer.read()
            assert answer.status == 401, body
            assert time.monotonic() - started < web.DRAIN_SECONDS
        finally:
            connection.close()


# ---- 411, 413 -------------------------------------------------------------------------------


def test_a_body_without_a_length_is_a_411(server: Live) -> None:
    status, _, body = raw(server, [("Seedvr2x-Resolution", "48")])
    assert status == 411
    assert "Content-Length" in body.decode()


def test_a_chunked_body_is_a_411(server: Live) -> None:
    sent = [("Seedvr2x-Resolution", "48"), ("Transfer-Encoding", "chunked")]
    status, _, body = raw(server, sent)
    assert status == 411
    assert "Transfer-Encoding" in body.decode()


def test_a_body_over_the_limit_is_a_413_the_client_reads(
    server: Live, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headers and 3 MiB of a body declared a byte over the limit: the answer is read, though
    the body is far from complete (it is never read, only dropped up to web.DRAIN_BYTES, which
    keeps the connection from resetting before the client has read the refusal)."""

    def not_read(*args: Any) -> None:
        raise AssertionError("the body of a refused request was read as a picture")

    monkeypatch.setattr(services.Service, "prepare", not_read)
    limit = reservation.body_limit(INPUT)
    head = (
        f"POST /upscale HTTP/1.0\r\nSeedvr2x-Resolution: 48\r\nContent-Length: {limit + 1}\r\n\r\n"
    )
    with socket.create_connection(server.running.address, timeout=30) as client:
        client.sendall(head.encode() + b"x" * (3 * 1024**2))
        received = b""
        while b"\r\n\r\n" not in received:
            chunk = client.recv(4096)
            assert chunk, "the server closed without an answer"
            received += chunk
        head_part, _, rest = received.partition(b"\r\n\r\n")
        length = int(re.search(rb"Content-Length: (\d+)", head_part).group(1))  # type: ignore[union-attr]
        while len(rest) < length:
            rest += client.recv(4096)
    assert head_part.startswith(b"HTTP/1.0 413")
    assert f"{limit:,}" in rest.decode()


# ---- 422 ------------------------------------------------------------------------------------


def test_a_transparent_picture_is_refused_in_the_servers_words(server: Live) -> None:
    pixels = np.random.default_rng(0).integers(0, 256, (24, 40, 4), dtype=np.uint8)
    pixels[:4, :4, 3] = 0
    out = io.BytesIO()
    Image.fromarray(pixels, "RGBA").save(out, "PNG")
    status, headers, body = server.upscale(out.getvalue(), 48)
    text = body.decode()
    assert status == 422
    # The picture is one the server takes, refused for its content: no Accept-Post.
    assert "accept-post" not in headers
    assert "transparent" in text
    assert "Flatten it onto a background" in text
    assert "ffmpeg" not in text.lower()
    assert "--" not in text


def test_a_picture_over_the_input_limit_is_refused_before_it_is_decoded(
    server: Live, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Known by the header alone: picture.read, which decodes, fails the test if called."""

    def decoded(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("an oversized picture was decoded")

    monkeypatch.setattr(picture, "read", decoded)
    status, _, body = server.upscale(png_of(200, 100), 48)
    assert 200 * 100 > INPUT
    assert status == 422
    assert f"{200 * 100:,} pixels (200x100)" in body.decode()
    assert f"{INPUT:,}" in body.decode()


def test_an_output_over_the_largest_size_names_the_resolution_that_fits(server: Live) -> None:
    data = png_of(40, 24)
    status, _, body = server.upscale(data, 200)
    assert status == 422
    found = re.search(r"Seedvr2x-Resolution (\d+) at the most", body.decode())
    assert found is not None, body
    fits = int(found.group(1))
    status, headers, body = server.upscale(data, fits, seed=1)
    assert status == 200, body
    sides = [int(side) for side in headers["seedvr2x-output-size"].split("x")]
    assert max(sides) <= 128 and min(sides) <= 96
    assert server.upscale(data, fits + 1)[0] == 422


def test_a_resolution_too_small_for_the_padding_is_refused(server: Live) -> None:
    status, _, body = server.upscale(png_of(40, 24), 8)
    assert status == 422
    assert "too small" in body.decode()
    assert "larger Seedvr2x-Resolution" in body.decode()


def test_an_empty_body_is_a_422(server: Live) -> None:
    status, headers, body = server.upscale(b"", 48)
    assert status == 422
    assert "accept-post" not in headers
    assert "empty" in body.decode()


def test_bytes_that_are_no_picture_are_a_422(server: Live) -> None:
    status, headers, body = server.upscale(b"this is not a picture, though it is bytes", 48)
    text = body.decode()
    assert status == 422
    assert "PNG, JPEG" in text
    assert "ffmpeg" not in text.lower()
    # The formats the server takes, as OPTIONS says them.
    assert headers["accept-post"] == ACCEPT_POST


# ---- 404, 501 -------------------------------------------------------------------------------


def test_another_path_is_a_404(server: Live) -> None:
    assert server.request("GET", "/nothing")[0] == 404
    assert server.request("POST", "/nothing", b"x")[0] == 404


def test_other_methods_are_a_501(server: Live) -> None:
    for method in ("PUT", "DELETE", "PATCH"):
        assert server.request(method, "/upscale", b"x")[0] == 501


# ---- 503 ------------------------------------------------------------------------------------


def test_a_server_starting_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    machine = HeldStart(cpu_models)
    with live(monkeypatch, tmp_path, cpu_models, ready=False, machine=machine) as server:
        server.starting.start()
        try:
            assert machine.reached.wait(120)
            status, _, body = server.request("GET", "/health")
            assert (status, body.strip()) == (503, b"starting")
            status, headers, body = server.upscale(png_of(40, 24))
            assert status == 503
            assert headers["retry-after"] == str(web.RETRY_AFTER)
            assert "starting" in body.decode()
            assert metric(server, "ready") == 0
        finally:
            machine.go.set()
        server.wait_ready()
        status, _, body = server.request("GET", "/health")
        assert (status, body.strip()) == (200, b"ready")
        assert metric(server, "ready") == 1


def test_a_wait_past_its_bound_says_stalled_and_ready_again_once_it_ends(
    server: Live, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/health is 503 stalled while a wait on the GPU is past its bound (waiting.stalled, set by
    the library's wait), 200 ready again after it, with no wait made: the flag is the library's,
    tested in core/tests/test_waiting.py."""
    status, _, body = server.request("GET", "/health")
    assert (status, body.strip()) == (200, b"ready")
    monkeypatch.setattr(waiting, "stalled", lambda: True)
    status, headers, body = server.request("GET", "/health")
    assert (status, body.strip()) == (503, b"stalled")
    assert headers["content-type"].startswith("text/plain")
    # The server takes pictures still, and says ready in /metrics: it is waiting, not stopped.
    assert server.upscale(png_of(40, 24))[0] == 200
    assert metric(server, "ready") == 1
    monkeypatch.setattr(waiting, "stalled", lambda: False)
    status, _, body = server.request("GET", "/health")
    assert (status, body.strip()) == (200, b"ready")


@pytest.mark.parametrize(
    ("longest", "expected"), [(12.5, 12.5), (None, 0.0), (0.0, 0.0)], ids=["12.5 s", "none", "zero"]
)
def test_the_metrics_carry_the_longest_wait_on_the_gpu(
    server: Live, monkeypatch: pytest.MonkeyPatch, longest: float | None, expected: float
) -> None:
    monkeypatch.setattr(waiting, "longest", lambda: longest)
    assert metric(server, "gpu_wait_seconds") == expected


def test_a_full_queue_is_a_503_with_retry_after(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    """One picture running (held on a gate), one waiting, the third turned away."""
    with live(monkeypatch, tmp_path, cpu_models, count=1, queue=1) as server:
        gate = Gate()
        server.machine.fail = gate
        admitted = server.service.admitted
        assert admitted is not None
        with ThreadPoolExecutor(2) as pool:
            running = pool.submit(server.upscale, png_of(40, 24, 1), 48, 1)
            assert gate.entered.wait(60)
            waiting = pool.submit(server.upscale, png_of(40, 24, 2), 48, 2)
            until(lambda: admitted.waiting == 1, "the second picture never waited")
            assert metric(server, "pictures_running") == 1
            assert metric(server, "pictures_waiting") == 1
            status, headers, body = server.upscale(png_of(40, 24, 3), 48)
            assert status == 503
            assert headers["retry-after"] == str(web.RETRY_AFTER)
            assert "queue is full" in body.decode()
            gate.release.set()
            assert running.result(60)[0] == 200
            assert waiting.result(60)[0] == 200
        assert gate.calls == 2
        assert metric(server, "requests_total", status="503") == 1


def test_connections_past_the_cap_are_a_503(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    """A cap of one connection (one picture, no queue, no extra): one held idle fills it."""
    monkeypatch.setattr(web, "EXTRA_CONNECTIONS", 0)
    with live(monkeypatch, tmp_path, cpu_models, count=1, queue=0) as server:
        with socket.create_connection(server.running.address) as idle:
            status, headers, body = server.request("GET", "/health")
            assert status == 503
            assert headers["retry-after"] == str(web.RETRY_AFTER)
            assert "connections" in body.decode()
            idle.close()
        until(
            lambda: server.request("GET", "/health")[0] == 200,
            "the cap never freed after the idle connection closed",
        )


def test_a_picture_past_count_and_queue_is_a_503_before_its_body_is_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    """One picture running (held on a gate), no queue: a second POST finds no place, and neither
    its body is read nor a picture prepared (the host's plan counts count + queue pictures)."""
    reads: list[int] = []
    prepared: list[int] = []
    body, prepare = web.Handler._body, services.Service.prepare  # pyright: ignore[reportPrivateUsage]

    def counted_body(self: Any, limit: int) -> bytes | None:
        reads.append(limit)
        return body(self, limit)

    def counted_prepare(self: Any, *args: Any) -> Any:
        prepared.append(1)
        return prepare(self, *args)

    monkeypatch.setattr(web.Handler, "_body", counted_body)
    monkeypatch.setattr(services.Service, "prepare", counted_prepare)
    with live(monkeypatch, tmp_path, cpu_models, count=1, queue=0) as server:
        gate = Gate()
        server.machine.fail = gate
        with ThreadPoolExecutor(1) as pool:
            running = pool.submit(server.upscale, png_of(40, 24, 1), 48, 1)
            assert gate.entered.wait(60)
            assert len(reads) == 1
            status, headers, body_seen = server.upscale(png_of(40, 24, 2), 48)
            assert status == 503
            assert headers["retry-after"] == str(web.RETRY_AFTER)
            assert "queue is full" in body_seen.decode()
            assert len(reads) == 1 and len(prepared) == 1
            gate.release.set()
            assert running.result(60)[0] == 200
        # The place is given back once the request ends: the next picture is taken.
        assert server.upscale(png_of(40, 24, 3), 48)[0] == 200
        assert len(reads) == 2 and len(prepared) == 2


# ---- Deadlines ----------------------------------------------------------------------------


def test_a_head_that_stops_halfway_is_cut_at_its_deadline(
    server: Live, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(web, "HEAD_SECONDS", 0.5)
    caplog.set_level("INFO", logger="imgserver.web")
    # The clock starts before the connection: the server's timer starts once it has accepted it
    # (web.Handler.setup), and this thread may be held up after that, before or after its send.
    started = time.monotonic()
    with socket.create_connection(server.running.address, timeout=10) as client:
        client.sendall(b"POST /upsc")
        # The server shuts the connection down: an end of file (or a reset), well before TIMEOUT.
        try:
            assert client.recv(1) == b""
        except ConnectionResetError:
            pass
        assert 0.4 < time.monotonic() - started < 5
    assert "head missed" in caplog.text


def test_a_head_sent_in_time_is_not_cut_later(
    server: Live, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The timer is cancelled once the head is parsed: a picture that takes longer than the head's
    deadline to make is answered."""
    monkeypatch.setattr(web, "HEAD_SECONDS", 0.5)
    gate = Gate()
    server.machine.fail = gate
    with ThreadPoolExecutor(1) as pool:
        running = pool.submit(server.upscale, png_of(40, 24, 1), 48, 1)
        assert gate.entered.wait(60)
        time.sleep(1.0)
        gate.release.set()
        assert running.result(60)[0] == 200


def test_a_body_trickled_too_slowly_is_dropped_unanswered(
    server: Live, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """2,000 bytes declared at 1,000 B/s and 0.5 s: 2.5 s to arrive, a few bytes in 1 s."""
    monkeypatch.setattr(web, "BODY_SECONDS", 0.5)
    monkeypatch.setattr(web, "BODY_RATE", 1_000)
    caplog.set_level("INFO", logger="imgserver.web")
    head = b"POST /upscale HTTP/1.0\r\nSeedvr2x-Resolution: 48\r\nContent-Length: 2000\r\n\r\n"
    with socket.create_connection(server.running.address, timeout=10) as client:
        client.sendall(head)
        started = time.monotonic()
        for _ in range(3):
            client.sendall(b"x")
            time.sleep(0.3)
        try:
            assert client.recv(1) == b""
        except ConnectionResetError:
            pass
        assert 2.0 < time.monotonic() - started < 8
    assert "missed its deadline" in caplog.text
    assert server.request("GET", "/health")[0] == 200


# ---- 500 ------------------------------------------------------------------------------------


def failing_once(error: BaseException) -> Callable[[], None]:
    """A `StandIn.fail` hook raising `error` on its first call only."""
    calls: list[int] = []

    def hook() -> None:
        calls.append(1)
        if len(calls) == 1:
            raise error

    return hook


def test_an_out_of_memory_is_a_500_and_the_reservation_is_made_again(
    server: Live, caplog: pytest.LogCaptureFixture
) -> None:
    # The reservation is made again where torch holds less than it with no picture running
    # (service._held): the stand-in's allocator gives a GiB back on the failure, as torch's does
    # for an allocation it can't otherwise serve.
    made = server.service.made
    assert made is not None
    machine, before = server.machine, len(server.machine.reserves)
    oom = failing_once(MemoryError("out of memory"))

    def gives_back() -> None:
        machine.lost = GIB
        oom()

    machine.fail = gives_back
    with caplog.at_level("INFO", logger="imgserver.service"):
        status, _, body = server.upscale(png_of(40, 24), 48, seed=1)
    assert status == 500
    assert "ran out of the GPU's memory" in body.decode()
    assert machine.reserves[before:] == [made.reservation]
    assert machine.reserved() == made.reservation
    assert "the reservation held again" in caplog.text
    assert server.request("GET", "/health")[0] == 200
    assert server.upscale(png_of(40, 24), 48, seed=1)[0] == 200


def test_a_picture_not_finite_is_a_500_and_the_server_goes_on(server: Live) -> None:
    server.machine.fail = failing_once(still.NonFinite("the decode holds NaN"))
    status, _, body = server.upscale(png_of(40, 24), 48, seed=1)
    assert status == 500
    assert "NaN" in body.decode()
    assert server.request("GET", "/health")[0] == 200
    assert server.upscale(png_of(40, 24), 48, seed=1)[0] == 200


def test_another_error_is_a_500_and_the_server_goes_on(server: Live) -> None:
    server.machine.fail = failing_once(ValueError("a bug"))
    status, _, body = server.upscale(png_of(40, 24), 48, seed=1)
    assert status == 500
    assert "server goes on" in body.decode()
    assert "a bug" not in body.decode()
    assert server.request("GET", "/health")[0] == 200
    assert server.upscale(png_of(40, 24), 48, seed=1)[0] == 200


def test_a_cuda_error_fails_the_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpu_models: model.Models,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with live(monkeypatch, tmp_path, cpu_models) as server:
        server.machine.fail = failing_once(RuntimeError("CUDA error: an illegal memory access"))
        with caplog.at_level("INFO", logger="imgserver.service"):
            status, _, body = server.upscale(png_of(40, 24), 48)
        assert status == 500
        # The answer is a fixed sentence: torch's text is the log's, never the client's.
        assert "CUDA error" in body.decode() and "which error" in body.decode()
        assert "illegal memory access" not in body.decode()
        assert "illegal memory access" in caplog.text
        assert "the GPU's state is untrusted" in caplog.text
        status, _, body = server.request("GET", "/health")
        assert (status, body.strip()) == (503, b"failed")
        status, headers, body = server.upscale(png_of(40, 24), 48)
        assert status == 503
        assert "retry-after" not in headers
        assert "failed" in body.decode()
        assert metric(server, "ready") == 0


# ---- A client gone --------------------------------------------------------------------------


def test_a_client_gone_while_its_picture_waits_is_dropped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    with live(monkeypatch, tmp_path, cpu_models, count=1, queue=2) as server:
        gate = Gate()
        server.machine.fail = gate
        admitted = server.service.admitted
        assert admitted is not None
        with ThreadPoolExecutor(1) as pool:
            running = pool.submit(server.upscale, png_of(40, 24, 1), 48, 1)
            assert gate.entered.wait(60)
            data = png_of(40, 24, 2)
            head = (
                f"POST /upscale HTTP/1.0\r\nSeedvr2x-Resolution: 48\r\n"
                f"Content-Length: {len(data)}\r\n\r\n"
            )
            leaver = socket.create_connection(server.running.address)
            leaver.sendall(head.encode() + data)
            until(lambda: admitted.waiting == 1, "the second picture never waited")
            leaver.close()
            until(lambda: admitted.waiting == 0, "the picture of a client gone still waits", 10)
            gate.release.set()
            assert running.result(60)[0] == 200
        # Only the first picture ever reached the models.
        assert gate.calls == 1
        assert admitted.running == 0
        assert server.request("GET", "/health")[0] == 200


# ---- /metrics -------------------------------------------------------------------------------


def test_metrics_after_a_few_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    with live(monkeypatch, tmp_path, cpu_models) as server:
        made = server.service.made
        assert made is not None
        data = png_of(40, 24)
        assert server.upscale(data, 48, seed=1)[0] == 200
        assert server.upscale(data, 48, seed=2)[0] == 200
        assert server.upscale(data, "big")[0] == 400
        assert server.upscale(b"", 48)[0] == 422
        assert server.request("GET", "/health")[0] == 200
        found = scrape(server)

        def value(name: str, **labels: str) -> float:
            return found[(f"seedvr2x_imgserver_{name}", tuple(sorted(labels.items())))]

        assert value("requests_total", status="200") == 3
        assert value("requests_total", status="400") == 1
        assert value("requests_total", status="422") == 1
        # The two pictures made, not the warm-up's.
        assert value("upscale_seconds_count") == 2
        assert value("wait_seconds_count") == 2
        assert value("upscale_seconds_bucket", le="+Inf") == 2
        assert value("output_pixels_total") == 2 * 80 * 48
        assert value("vram_reservation_bytes") == made.reservation
        assert value("pictures_running") == 0
        assert value("pictures_waiting") == 0
        assert value("ready") == 1
        info = [labels for (name, labels) in found if name == "seedvr2x_imgserver_info"]
        assert dict(info[0])["version"] == metadata.version(services.APP)
        assert dict(info[0])["dit_model"]


def test_the_model_files_hash_is_the_records(server: Live) -> None:
    """The record's SHA-256s are those of the stand-in files (empty)."""
    status, _, body = server.upscale(png_of(40, 24), 48, seed=3)
    assert status == 200
    assert sha256(b"").hexdigest() in image_of(body).info["XML:com.adobe.xmp"]
