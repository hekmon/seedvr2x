"""The server's flat layout (DESIGN.md, The image server; Repository and apps, step 2): the
modules at the app's root, each listed in pyproject's `py-modules` (setuptools builds a flat
layout from that list alone), no package folder beside them but `tests/`, and each module's name
resolving to the app's own file in this environment, so that no dependency's `web`, `main` or
`metrics` shadows the server's. Held as the CLI's tests/test_names.py holds its own."""

import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent


def listed() -> list[str]:
    with (APP / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)["tool"]["setuptools"]["py-modules"]


def stems() -> list[str]:
    return sorted(path.stem for path in APP.glob("*.py"))


def test_the_list_is_the_modules_at_the_root() -> None:
    # py-modules is what the wheel ships: a module left out of it imports from the source tree
    # and fails in the installed server (the Docker image).
    assert listed() == stems()
    assert listed() == sorted(listed())


def test_the_modules_are_the_ones_the_server_has() -> None:
    assert set(stems()) >= {"admission", "answers", "main", "metrics", "reservation", "service"}


def test_no_folder_but_the_tests_is_a_package() -> None:
    # A package folder at the root would be one more name for setuptools to list (`packages`),
    # which the server doesn't use.
    folders = [
        path.name
        for path in APP.iterdir()
        if path.is_dir() and (path / "__init__.py").exists() and path.name != "tests"
    ]
    assert folders == []
    assert not (APP / "src").exists()


@pytest.mark.parametrize("name", stems())
def test_each_module_is_the_apps_own_file(name: str) -> None:
    spec = importlib.util.find_spec(name)
    assert spec is not None, name
    assert spec.origin is not None
    assert Path(spec.origin).resolve() == (APP / f"{name}.py").resolve()


@pytest.mark.parametrize("name", stems())
def test_no_module_is_named_as_the_standard_librarys(name: str) -> None:
    assert name not in sys.stdlib_module_names
