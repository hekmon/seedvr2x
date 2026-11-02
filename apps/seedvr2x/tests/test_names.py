"""The CLI's top-level names (DESIGN.md, Repository and apps, step 2): the command is flat, its
modules at the app's root (`main`, `shot`, `run`, `media/`, ...) with no package folder, each one
global to the environment, where a package folder's would not be. That is held, not hoped for:

- the list (manifest.MODULES, PACKAGES), which the code hash and the environment record use, is
  the files at the app's root, and pyproject.toml's py-modules and packages;
- no name is the standard library's, or a built-in module's;
- in this environment, each name resolves to the CLI's own file, and nothing else provides it:
  no directory on sys.path has another module of the name, and no other installed distribution
  claims it, so that a dependency that brings a `run` or a `media` fails here, never replaces
  ours silently. (The CLI is never installed into a shared Python.)"""

import importlib.machinery
import importlib.metadata
import importlib.util
import sys
import tomllib
from pathlib import Path

import manifest

APP = Path(manifest.__file__).resolve().parent
NAMES = (*manifest.MODULES, *manifest.PACKAGES)


def own_files(name: str, app: Path) -> set[Path]:
    """The CLI's own file for a module `name` at the app's root `app`: its module, or its
    package's folder and __init__. Nothing else under the app is ours: the environment the CLI
    ships in is apps/seedvr2x/.venv, and a dependency's module in its site-packages sits under
    the app's root too."""
    app = app.resolve()
    return {app / f"{name}.py", app / name, app / name / "__init__.py"}


def foreign_providers(name: str, entries: list[str], app: Path) -> list[str]:
    """What on the path `entries` provides a module `name` other than the CLI's own file at the
    app's root `app` (own_files): the entry and where it finds it (a namespace package's
    directories included)."""
    own = own_files(name, app)
    found: list[str] = []
    for entry in entries:
        spec = importlib.machinery.PathFinder.find_spec(name, [entry])
        if spec is None:
            continue
        places = [spec.origin] if spec.origin else list(spec.submodule_search_locations or [])
        for place in places:
            if place is None or Path(place).resolve() not in own:
                found.append(f"{entry}: {place}")
    return found


def test_the_modules_are_the_files_at_the_apps_root_and_the_pyprojects_list() -> None:
    setuptools = tomllib.loads((APP / "pyproject.toml").read_text())["tool"]["setuptools"]
    assert list(manifest.MODULES) == sorted(path.stem for path in APP.glob("*.py"))
    assert list(manifest.MODULES) == setuptools["py-modules"] == sorted(setuptools["py-modules"])
    assert list(manifest.PACKAGES) == setuptools["packages"]
    # Each package is a folder with its __init__, and the only folder of code at the root.
    assert all((APP / package / "__init__.py").is_file() for package in manifest.PACKAGES)
    folders = {
        path.name for path in APP.iterdir() if path.is_dir() and (path / "__init__.py").is_file()
    }
    assert folders == set(manifest.PACKAGES)


def test_no_name_is_the_standard_librarys() -> None:
    taken = {
        name
        for name in NAMES
        if name in sys.stdlib_module_names or name in sys.builtin_module_names
    }
    assert taken == set()


def test_each_name_resolves_to_the_clis_own_file() -> None:
    for name in NAMES:
        spec = importlib.util.find_spec(name)
        assert spec is not None and spec.origin is not None, name
        assert Path(spec.origin).resolve() in own_files(name, APP), (name, spec.origin)


def test_nothing_else_provides_a_name() -> None:
    entries = [entry for entry in sys.path if entry]
    foreign = {name: foreign_providers(name, entries, APP) for name in NAMES}
    assert {name: found for name, found in foreign.items() if found} == {}
    distributions = importlib.metadata.packages_distributions()
    claimed = {name: distributions.get(name) for name in NAMES}
    assert {name: d for name, d in claimed.items() if d and set(d) != {"seedvr2x"}} == {}


def test_a_second_provider_is_found(tmp_path: Path) -> None:
    # The check bites: a module, a package and a namespace package of the CLI's names on a path
    # are found, and the CLI's own files are not.
    (tmp_path / "run.py").write_text("")
    (tmp_path / "media").mkdir()
    (tmp_path / "media" / "__init__.py").write_text("")
    (tmp_path / "plan").mkdir()
    (tmp_path / "plan" / "a.py").write_text("")
    entries = [str(tmp_path)]
    assert foreign_providers("run", entries, APP) == [f"{tmp_path}: {tmp_path / 'run.py'}"]
    assert len(foreign_providers("media", entries, APP)) == 1
    assert foreign_providers("plan", entries, APP) == [f"{tmp_path}: {tmp_path / 'plan'}"]
    assert foreign_providers("shot", entries, APP) == []
    # The app's own root on the path provides nothing foreign.
    assert [foreign_providers(name, [str(APP)], APP) for name in NAMES] == [[]] * len(NAMES)
    # Nor does a path that has none of the names.
    assert foreign_providers("run", [str(tmp_path / "nowhere")], APP) == []
    # A dependency's module in the app's own environment is found: the venv sits under the
    # app's root, and only the app's own file is ours.
    app = tmp_path / "app"
    site = app / ".venv" / "lib" / "site-packages"
    site.mkdir(parents=True)
    (app / "run.py").write_text("")
    (site / "run.py").write_text("")
    assert foreign_providers("run", [str(app), str(site)], app) == [f"{site}: {site / 'run.py'}"]
