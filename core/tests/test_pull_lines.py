"""A download's failures on each line of huggingface_hub the library takes (seedvr2x/pull.py's
_download; core/pyproject.toml): below 1.0, 0.34 to 0.36, the line ComfyUI's transformers 4.x takes,
then 1.x and 2.x, whose errors are the same classes on the same bases (2.4.0's errors.py has
1.33.0's, read on 2026-11-13, its HTTP library httpx2, httpx's fork, where 1.x's is httpx).

On the CPU, huggingface_hub's two calls replaced (test_pull.py's Hub): each failure refused by its
kind and in its words, raised as the line's own classes raise it, the installed line's real
classes and the other line's stand-in (stand_in), which the installed line's real classes hold to
(test_the_stand_in_has_the_installed_lines_classes). The CLI's environment runs 1.x (1.33.0), the
floors' environment 0.x (0.34.0): AGENTS.md, Tests. hf-xet before its own refusal classes (1.1.3,
the least 0.34 to 0.36 take) and 2.x's httpx2 are held apart."""

import importlib
import shutil
import sys
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from typing import Any

import huggingface_hub
import pytest
from library_fixtures import cli_words
from test_pull import (
    DIT,
    ELSEWHERE,
    REVISION,
    UNAVAILABLE,
    VAE,
    Hub,
    hub,  # noqa: F401
    pins,  # noqa: F401
)

from seedvr2x import pull
from seedvr2x.errors import ModelUnreachable
from seedvr2x.weights import ModelError

# The line installed: "0" below 1.0, "1" from 1.0 (1.x and 2.x).
INSTALLED = "0" if int(huggingface_hub.__version__.split(".")[0]) < 1 else "1"
LINES = ("0", "1")

# The classes a download meets, with their bases by name, as each line has them: 0.36.2's
# errors.py:40-299 (0.34.0's the same), 1.33.0's and 2.4.0's errors.py:90-486.
BASES = {
    "0": {
        "HfHubHTTPError": ("HTTPError",),
        "FileMetadataError": ("OSError",),
        "RepositoryNotFoundError": ("HfHubHTTPError",),
        "GatedRepoError": ("RepositoryNotFoundError",),
        "DisabledRepoError": ("HfHubHTTPError",),
        "RevisionNotFoundError": ("HfHubHTTPError",),
        "EntryNotFoundError": ("HfHubHTTPError",),
        "LocalEntryNotFoundError": ("EntryNotFoundError", "FileNotFoundError", "ValueError"),
    },
    "1": {
        "HfHubHTTPError": ("HTTPError", "OSError"),
        "FileMetadataError": ("OSError",),
        "RepositoryNotFoundError": ("HfHubHTTPError",),
        "GatedRepoError": ("RepositoryNotFoundError",),
        "DisabledRepoError": ("HfHubHTTPError",),
        "RevisionNotFoundError": ("HfHubHTTPError",),
        "EntryNotFoundError": ("Exception",),
        "RemoteEntryNotFoundError": ("HfHubHTTPError", "EntryNotFoundError"),
        "LocalEntryNotFoundError": ("FileNotFoundError", "EntryNotFoundError"),
    },
}


def stand_in(line: str) -> ModuleType:
    """huggingface_hub.errors as `line` has it, BASES' classes on a stand-in of its HTTP library's
    HTTPError: requests', an OSError carrying its response, None by default, before 1.0 (requests
    2.34.2: exceptions.py:20-31, 66); httpx's, an Exception, from 1.0, the response required."""
    module = ModuleType("huggingface_hub.errors")
    if line == "0":

        class OldHTTPError(OSError):
            def __init__(self, *args: object, response: Any = None, request: Any = None) -> None:
                super().__init__(*args)
                self.response = response
                self.request = request

        class OldHub(OldHTTPError):
            # 0.36.2, errors.py:70: the response optional, None by default.
            def __init__(self, message: str, response: Any = None, **_: object) -> None:
                super().__init__(message, response=response)

        http_error: type[Exception] = OldHTTPError
        hub_error: type[Exception] = OldHub
    else:

        class NewHTTPError(Exception):
            pass

        class NewHub(NewHTTPError, OSError):
            def __init__(self, message: str, *, response: Any, **_: object) -> None:
                super().__init__(message)
                self.response = response

        http_error, hub_error = NewHTTPError, NewHub
    http_error.__name__ = "HTTPError"
    hub_error.__name__ = "HfHubHTTPError"
    classes: dict[str, type[BaseException]] = {
        "HTTPError": http_error,
        "HfHubHTTPError": hub_error,
        "OSError": OSError,
        "FileNotFoundError": FileNotFoundError,
        "ValueError": ValueError,
        "Exception": Exception,
    }
    for name, bases in BASES[line].items():
        if name not in classes:
            classes[name] = type(name, tuple(classes[each] for each in bases), {})
    if line == "0":
        local = classes["LocalEntryNotFoundError"]

        # 0.36.2, errors.py:296-297: the message alone, no response.
        def made(self: BaseException, message: str) -> None:
            super(local, self).__init__(message, response=None)  # pyright: ignore[reportCallIssue]

        local.__init__ = made  # pyright: ignore[reportAttributeAccessIssue]
    for name, each in classes.items():
        setattr(module, name, each)
    return module


@pytest.fixture(params=LINES, ids=[f"huggingface_hub-{line}.x" for line in LINES])
def line(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The errors module of the line asked: the installed line's real one, or the other line's
    stand-in put in its place, which pull imports at its call."""
    if request.param == INSTALLED:
        return importlib.import_module("huggingface_hub.errors")
    errors = stand_in(request.param)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    return errors


def is_old(errors: ModuleType) -> bool:
    """Whether errors is a line before 1.0's, which has no RemoteEntryNotFoundError."""
    return not hasattr(errors, "RemoteEntryNotFoundError")


def answer(errors: ModuleType, status: int) -> Any:
    """The Hub's answer of `status`, as the line's HTTP library makes it: the installed line's real
    Response (requests' before 1.0, set by attribute; httpx's or httpx2's), or a stand-in's."""
    if not hasattr(errors, "Response"):
        return SimpleNamespace(status_code=status, headers={}, request=None)
    response_class = vars(errors)["Response"]
    if response_class.__module__.startswith("requests"):
        response = response_class()
        response.status_code = status
        return response
    library = importlib.import_module(response_class.__module__.split(".")[0])
    return response_class(status, request=library.Request("HEAD", "https://huggingface.co"))


def http(errors: ModuleType, name: str, status: int, message: str) -> BaseException:
    """The line's error class `name` with the Hub's answer of `status`, as hf_raise_for_status
    raises it."""
    return vars(errors)[name](message, response=answer(errors, status))


def local(errors: ModuleType, cause: BaseException | None) -> BaseException:
    """The line's LocalEntryNotFoundError, as the library raises it when the file's HEAD request
    failed (0.36.2: file_download.py:1665-1669; 1.33.0: 1959-1965), cause its own."""
    error = vars(errors)["LocalEntryNotFoundError"](
        "An error happened while trying to locate the file on the Hub"
    )
    error.__cause__ = cause
    return error


def network(errors: ModuleType, message: str) -> BaseException:
    """The network's error on the line: requests' ConnectionError, an OSError, before 1.0;
    httpx's ConnectError from 1.0 (httpx2's on 2.x is its fork's, held below)."""
    if is_old(errors):
        return importlib.import_module("requests").ConnectionError(message)
    return pytest.importorskip("httpx").ConnectError(message)


def lost(errors: ModuleType, message: str) -> BaseException:
    """The network lost during a plain download, past the library's retries: requests'
    ConnectionError before 1.0 (0.36.2: file_download.py http_get), httpx's ReadTimeout from 1.0."""
    if is_old(errors):
        return importlib.import_module("requests").ConnectionError(message)
    return pytest.importorskip("httpx").ReadTimeout(message)


def entry(errors: ModuleType) -> BaseException:
    """The Hub's "entry not found": EntryNotFoundError itself before 1.0, RemoteEntryNotFoundError
    from 1.0."""
    name = "EntryNotFoundError" if is_old(errors) else "RemoteEntryNotFoundError"
    return http(errors, name, 404, "404 Client Error. Entry Not Found")


NAME_RESOLUTION = "[Errno -3] Temporary failure in name resolution"
CASES: list[tuple[str, Callable[[ModuleType], BaseException], str, str]] = [
    (
        "unreachable",
        lambda errors: local(errors, network(errors, NAME_RESOLUTION)),
        "unreachable",
        f"{DIT}: not in the cache, and https://huggingface.co couldn't be reached to download it"
        f" ({NAME_RESOLUTION}): run again once it can be, {ELSEWHERE}",
    ),
    (
        # Before 1.0 a LocalEntryNotFoundError is an HfHubHTTPError with no response: read for a
        # refusal, it is no 401 or 403.
        "unreachable-no-cause",
        lambda errors: local(errors, None),
        "unreachable",
        f"{DIT}: not in the cache, and https://huggingface.co couldn't be reached to download it"
        " (An error happened while trying to locate the file on the Hub): run again once it can"
        f" be, {ELSEWHERE}",
    ),
    (
        "forbidden-head",
        lambda errors: local(errors, http(errors, "HfHubHTTPError", 403, "403 Forbidden: None.")),
        "refused",
        f"{DIT}: the Hub refused it (403 Forbidden: None.): {UNAVAILABLE}",
    ),
    (
        "entry",
        entry,
        "not-found",
        f"{DIT}: not in hekmon/seedvr2x at {REVISION}, the revision this seedvr2x pins (the Hub:"
        " entry not found): download it elsewhere and give --model-dir with its directory",
    ),
    (
        "revision",
        lambda errors: http(
            errors, "RevisionNotFoundError", 404, "404 Client Error. Revision Not Found"
        ),
        "not-found",
        f"{DIT}: hekmon/seedvr2x has no revision {REVISION}, the one this seedvr2x pins (the Hub:"
        " revision not found): download the files elsewhere and give --model-dir with their"
        " directory",
    ),
    (
        "repository",
        lambda errors: http(
            errors, "RepositoryNotFoundError", 401, "401 Client Error. Repository Not Found"
        ),
        "not-found",
        f"{DIT}: the Hub has no repository hekmon/seedvr2x, or not a public one (the Hub:"
        " repository not found): download the files elsewhere and give --model-dir with their"
        " directory",
    ),
    (
        "too-many-requests",
        lambda errors: http(errors, "HfHubHTTPError", 429, "429 Client Error: Too Many Requests"),
        "failed",
        f"{DIT}: the Hub refused its download (429 Client Error: Too Many Requests): run again,"
        f" which downloads it again, {ELSEWHERE}",
    ),
    (
        "unauthorized",
        lambda errors: http(errors, "HfHubHTTPError", 401, "401 Client Error: Unauthorized"),
        "refused",
        f"{DIT}: the Hub refused it (401 Client Error: Unauthorized): {UNAVAILABLE}",
    ),
    (
        "disabled",
        lambda errors: http(
            errors, "DisabledRepoError", 403, "Access to this resource is disabled."
        ),
        "refused",
        f"{DIT}: the Hub refused it (Access to this resource is disabled.): {UNAVAILABLE}",
    ),
    (
        "network-lost",
        lambda errors: lost(errors, "The read operation timed out"),
        "failed",
        f"{DIT}: its download failed (The read operation timed out): run again, which downloads"
        f" it again, {ELSEWHERE}",
    ),
]


@pytest.mark.usefixtures("pins")
@pytest.mark.parametrize(
    ("raised", "why", "said"),
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_each_failure_is_refused_on_each_line(
    monkeypatch: pytest.MonkeyPatch,
    hub: Hub,  # noqa: F811
    line: ModuleType,
    raised: Callable[[ModuleType], BaseException],
    why: str,
    said: str,
) -> None:
    # Every failure the line raises refused by its kind, in the words of every line: none escapes
    # as another error (0.34 to 0.36 raised ImportError for RemoteEntryNotFoundError, and an
    # AttributeError reading the status of a LocalEntryNotFoundError's missing response).
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=10**12))
    hub.error = raised(line)
    with pytest.raises(ModelError) as error:
        pull.resolve(None, DIT, VAE)
    assert isinstance(error.value, ModelUnreachable) and error.value.why == why
    assert cli_words(error.value) == said
    assert "--" not in str(error.value)
    assert [call["filename"] for call in hub.calls] == [DIT]


def test_the_stand_in_has_the_installed_lines_classes() -> None:
    # The stand-in of the installed line is its real classes: each one's bases by name, and its
    # HTTP library's HTTPError an OSError (requests') or not (httpx's, httpx2's), as the real one.
    real = importlib.import_module("huggingface_hub.errors")
    fake = stand_in(INSTALLED)
    for name, bases in BASES[INSTALLED].items():
        assert tuple(each.__name__ for each in vars(real)[name].__bases__) == bases, name
        assert tuple(each.__name__ for each in vars(fake)[name].__bases__) == bases, name
    assert issubclass(vars(real)["HTTPError"], OSError) == issubclass(fake.HTTPError, OSError)
    assert hasattr(real, "RemoteEntryNotFoundError") == (INSTALLED == "1")
    real_local = vars(real)["LocalEntryNotFoundError"]("not here")
    fake_local = fake.LocalEntryNotFoundError("not here")
    assert getattr(real_local, "response", "none") == getattr(fake_local, "response", "none")


@pytest.mark.usefixtures("pins")
def test_an_os_error_is_refused_whatever_hf_xet_has(
    monkeypatch: pytest.MonkeyPatch,
    hub: Hub,  # noqa: F811
) -> None:
    # hf-xet 1.1.3, the least 0.34 to 0.36 take, has neither XetObjectNotFoundError nor
    # XetAuthenticationError: an OSError of a download, hf_xet imported, is refused as a failed
    # download, never an AttributeError.
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=10**12))
    monkeypatch.setitem(sys.modules, "hf_xet", ModuleType("hf_xet"))
    hub.error = OSError("IO Error: Connection reset by peer (os error 104)")
    with pytest.raises(ModelError) as error:
        pull.resolve(None, DIT, VAE)
    assert isinstance(error.value, ModelUnreachable) and error.value.why == "failed"
    assert cli_words(error.value) == (
        f"{DIT}: its download failed (IO Error: Connection reset by peer (os error 104)): run"
        f" again, which downloads it again, {ELSEWHERE}"
    )


@pytest.mark.usefixtures("pins")
def test_httpx2s_request_errors_are_refused(
    monkeypatch: pytest.MonkeyPatch,
    hub: Hub,  # noqa: F811
) -> None:
    # 2.x makes its requests with httpx2, whose errors are its own classes, not httpx's (2.4.0:
    # errors.py:6): one of its RequestErrors, the network lost, refused as a failed download.
    httpx2 = ModuleType("httpx2")

    class RequestError(Exception):
        pass

    class ReadTimeout(RequestError):
        pass

    vars(httpx2).update(RequestError=RequestError, ReadTimeout=ReadTimeout)
    monkeypatch.setitem(sys.modules, "httpx2", httpx2)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=10**12))
    hub.error = ReadTimeout("The read operation timed out")
    with pytest.raises(ModelError) as error:
        pull.resolve(None, DIT, VAE)
    assert isinstance(error.value, ModelUnreachable) and error.value.why == "failed"
    assert cli_words(error.value) == (
        f"{DIT}: its download failed (The read operation timed out): run again, which downloads"
        f" it again, {ELSEWHERE}"
    )
