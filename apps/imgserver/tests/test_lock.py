"""The server's lock against the CLI's (DESIGN.md, Repository and apps: "The image server's lock
holds the same versions of what they share, a test checking it"): the same models, the same
CUDA support packages, so that what was tested and measured in the CLI's environment is the
server's.

Every package that both locks list has the same version and the same source, and the server's
`[project] dependencies` equal the CLI's, pins included, but for the distribution itself.

To bring a moved lock back: copy the CLI's `uv.lock` over the server's, then `uv lock` in
`apps/imgserver/`, uv keeping every version it can and changing only what the server's
dependency groups and name need (prometheus-client in dev, the distribution's own entry); check
that this test passes, and that `uv lock --check` does."""

import tomllib
from pathlib import Path
from typing import Any

import pytest

APPS = Path(__file__).resolve().parent.parent.parent
SERVER, CLI = APPS / "imgserver", APPS / "seedvr2x"


def load(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def packages(app: Path) -> dict[str, dict[str, Any]]:
    return {each["name"]: each for each in load(app / "uv.lock")["package"]}


def dependencies(app: Path) -> list[str]:
    return load(app / "pyproject.toml")["project"]["dependencies"]


def test_the_locks_are_there() -> None:
    assert (SERVER / "uv.lock").is_file()
    assert (CLI / "uv.lock").is_file()


def test_every_package_in_both_locks_has_the_same_version() -> None:
    server, cli = packages(SERVER), packages(CLI)
    shared = sorted(set(server) & set(cli))
    # Most of the lock is shared: torch, the CUDA packages, the library and its dependencies.
    assert len(shared) > 50
    assert {"torch", "torchvision", "diffusers", "pillow", "seedvr2x-core"} <= set(shared)
    different = {
        name: (server[name].get("version"), cli[name].get("version"))
        for name in shared
        if server[name].get("version") != cli[name].get("version")
    }
    assert different == {}, "the server's lock moved from the CLI's: see this file's docstring"


def test_every_package_in_both_locks_has_the_same_source() -> None:
    server, cli = packages(SERVER), packages(CLI)
    different = {
        name: (server[name].get("source"), cli[name].get("source"))
        for name in sorted(set(server) & set(cli))
        if server[name].get("source") != cli[name].get("source")
    }
    assert different == {}


def test_the_servers_dependencies_are_the_clis() -> None:
    # Pins included: torch==2.14.1 and the rest.
    assert sorted(dependencies(SERVER)) == sorted(dependencies(CLI))
    assert any("==" in each for each in dependencies(SERVER))


def test_the_constraints_and_the_sources_are_the_clis() -> None:
    server, cli = (
        load(SERVER / "pyproject.toml")["tool"]["uv"],
        load(CLI / "pyproject.toml")["tool"]["uv"],
    )
    assert server["constraint-dependencies"] == cli["constraint-dependencies"]
    assert server["sources"] == cli["sources"]
    assert server["index"] == cli["index"]


@pytest.mark.parametrize("app", [SERVER, CLI], ids=["server", "cli"])
def test_each_lock_holds_its_own_distribution(app: Path) -> None:
    name = load(app / "pyproject.toml")["project"]["name"]
    assert packages(app)[name]["source"] == {"editable": "."}
