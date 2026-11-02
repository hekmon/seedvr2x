"""The library imports nothing of the command line's (DESIGN.md, Repository and apps: the
library is a collection of bricks, and the CLI assembles them; a test holds the rule). The CLI is
flat, its modules top-level names (`main`, `shot`, `media`, ...: apps/seedvr2x/pyproject.toml's
py-modules and packages), so a library file that imports one, by an import statement, a relative
import climbing out of its package or a module named by a string, imports the CLI.

Two ways, each on the CPU. A static scan of every library file (core/seedvr2x/, the vendored code
and TransNetV2's included; `imports_of_the_cli`, held on its own to synthetic sources, one per
form, as tests/test_ends_with_us.py holds its scan), which names the file and the line. And a
fresh process that imports every library module, then lists the modules loaded from the CLI's
directory: none. That one sees what no scan can, an import made through a third party or by a
name built at run time. The process runs with `-I` and the library's directory put on its path
by the code it is given: `-I` drops PYTHONPATH and the working directory, so that nothing else
can stand for the package.

The CLI's names are read from the repository, so the test skips where apps/seedvr2x isn't there
(a copy of core/ alone)."""

import ast
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parents[1]
PACKAGE = CORE / "seedvr2x"
APP = CORE.parent / "apps" / "seedvr2x"
# Calls that take a module's name as a string: import_module, __import__, find_spec.
BY_NAME = ("import_module", "__import__", "find_spec")


def cli_names() -> frozenset[str]:
    """The CLI's top-level names, its modules and packages, from its pyproject.toml."""
    pyproject = APP / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip(f"needs the CLI's pyproject.toml ({pyproject}): not a copy of core/ alone")
    setuptools = tomllib.loads(pyproject.read_text())["tool"]["setuptools"]
    return frozenset([*setuptools["py-modules"], *setuptools["packages"]])


def resolved(module: str, package: bool, level: int, name: str | None) -> str | None:
    """The absolute name of a relative import of `name` at `level` dots from `module` (a
    package's own __init__ if `package`), None if it climbs out of the top."""
    base = module.split(".")
    if not package:
        base = base[:-1]
    keep = len(base) - (level - 1)
    if keep < 1:
        return None
    return ".".join([*base[:keep], *([name] if name else [])])


def string_of(call: ast.Call, index: int, keyword: str) -> str | None:
    """The argument of a call at `index` or named `keyword`, if it is a string literal."""
    for given in (
        call.args[index] if len(call.args) > index else None,
        *(k.value for k in call.keywords if k.arg == keyword),
    ):
        if isinstance(given, ast.Constant) and isinstance(given.value, str):
            return given.value
    return None


def imports_of_the_cli(
    source: str, module: str, names: frozenset[str], package: bool = False
) -> list[tuple[int, str]]:
    """Each (line, name) at which `source`, the code of the library module `module` (a package's
    __init__ if `package`), imports one of the CLI's top-level `names`: an import or a
    from-import by an absolute name; a relative import, resolved against `module`, which a
    relative import climbing out of the package is a finding of too; a call of import_module,
    __import__ or find_spec with a string literal naming one. Anywhere in the code, a function's
    included."""
    found: list[tuple[int, str]] = []

    def check(line: int, name: str | None) -> None:
        if name is None:
            found.append((line, "a relative import out of the package"))
        elif name.partition(".")[0] in names:
            found.append((line, name))

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                check(node.lineno, alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = resolved(module, package, node.level, node.module)
                # A relative import stays in the package, whose names are none of the CLI's:
                # only one climbing out of it is a finding.
                if base is None:
                    check(node.lineno, None)
            else:
                check(node.lineno, node.module)
        elif isinstance(node, ast.Call):
            function = node.func
            called = function.attr if isinstance(function, ast.Attribute) else None
            called = function.id if isinstance(function, ast.Name) else called
            if called not in BY_NAME:
                continue
            named = string_of(node, 0, "name")
            if named is None:
                continue
            if named.startswith("."):
                anchor = string_of(node, 1, "package")
                if anchor is None:
                    continue
                named = resolved(
                    anchor, True, len(named) - len(named.lstrip(".")), named.lstrip(".")
                )
            check(node.lineno, named)
    return sorted(set(found), key=lambda each: (each[0], each[1]))


def library_modules() -> dict[str, Path]:
    """Every module of core/seedvr2x/, the vendored code and TransNetV2's included, by dotted
    name, its __init__s named for their package."""
    found: dict[str, Path] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = path.relative_to(CORE).with_suffix("").parts
        if "__pycache__" in parts:
            continue
        found[".".join(parts[:-1] if parts[-1] == "__init__" else parts)] = path
    return found


NAMES = frozenset({"main", "shot", "run", "planner", "media"})

# The scan, held on synthetic sources.


def test_an_absolute_import_of_the_cli_is_found() -> None:
    source = "import os\nimport shot\nimport media.ffmpeg as ff\nimport seedvr2x.model\n"
    assert imports_of_the_cli(source, "seedvr2x.peaks", NAMES) == [(2, "shot"), (3, "media.ffmpeg")]
    assert imports_of_the_cli("from run import decode_shot", "seedvr2x.x", NAMES) == [(1, "run")]


def test_a_from_import_of_the_cli_is_found() -> None:
    assert imports_of_the_cli("from media import ffmpeg\n", "seedvr2x.peaks", NAMES) == [
        (1, "media")
    ]
    assert imports_of_the_cli("from media.ffmpeg import check\n", "seedvr2x.peaks", NAMES) == [
        (1, "media.ffmpeg")
    ]
    assert imports_of_the_cli("import main as cli\n", "seedvr2x.peaks", NAMES) == [(1, "main")]


def test_a_relative_import_is_the_librarys_unless_it_climbs_out() -> None:
    # A relative import resolves into the library's own package, whose names are none of the
    # CLI's, even when one is spelt like one of them (a library `run` module, say).
    assert imports_of_the_cli("from . import run\n", "seedvr2x.peaks", NAMES) == []
    assert imports_of_the_cli("from .run import go\n", "seedvr2x.peaks", NAMES) == []
    assert imports_of_the_cli("from .. import run\n", "seedvr2x.vendor.example", NAMES) == []
    assert imports_of_the_cli("from ... import x\n", "seedvr2x.peaks", NAMES) == [
        (1, "a relative import out of the package")
    ]
    # In a package's own __init__, one dot is the package itself, two climb out of it.
    assert imports_of_the_cli("from . import x\n", "seedvr2x", NAMES, package=True) == []
    assert imports_of_the_cli("from .. import x\n", "seedvr2x", NAMES, package=True) == [
        (1, "a relative import out of the package")
    ]


def test_an_aliased_import_is_found() -> None:
    assert imports_of_the_cli("import planner as p\n", "seedvr2x.peaks", NAMES) == [(1, "planner")]
    assert imports_of_the_cli("from media import rate as r\n", "seedvr2x.peaks", NAMES) == [
        (1, "media")
    ]


def test_an_import_inside_a_function_is_found() -> None:
    source = "def late():\n    if True:\n        from planner import plan\n"
    assert imports_of_the_cli(source, "seedvr2x.peaks", NAMES) == [(3, "planner")]


@pytest.mark.parametrize(
    "call",
    [
        'importlib.import_module("main")',
        'import_module("main")',
        'il.import_module(name="main")',
        'importlib.util.find_spec("main")',
        'find_spec("main")',
        '__import__("main")',
        'importlib.import_module("main.x")',
    ],
)
def test_a_module_named_by_a_string_is_found(call: str) -> None:
    (line, name), *_ = imports_of_the_cli(f"x = {call}\n", "seedvr2x.peaks", NAMES)
    assert line == 1 and name.partition(".")[0] == "main"


@pytest.mark.parametrize(
    ("call", "climbs"),
    [
        ('importlib.import_module(".run", "seedvr2x")', []),
        ('importlib.import_module(".run", package="seedvr2x")', []),
        ('importlib.import_module("..run", "seedvr2x")', True),
        ('importlib.import_module("..run", package="seedvr2x")', True),
    ],
)
def test_a_module_named_by_a_relative_string_is_resolved(call: str, climbs: bool) -> None:
    # Relative to the package it is given: the library's own, unless it climbs out of it.
    expected = [(1, "a relative import out of the package")] if climbs else []
    assert imports_of_the_cli(f"x = {call}\n", "seedvr2x.peaks", NAMES) == expected


def test_what_is_the_librarys_or_not_the_clis_is_fine() -> None:
    source = (
        "import os\n"
        "import torch\n"
        "from seedvr2x import clock, frames\n"
        "from seedvr2x import model, tiling\n"
        "from seedvr2x.model import Models\n"
        "from seedvr2x.vendor.core.infer import VideoDiffusionInfer\n"
        "from seedvr2x.transnetv2.transnetv2_pytorch import TransNetV2\n"
        "from . import weights\n"
        "from .. import sampling\n"
        "from seedvr2x import __doc__\n"
        'importlib.import_module("seedvr2x.pull")\n'
        'importlib.import_module("torch")\n'
        "importlib.import_module(name)\n"
    )
    # From a vendored module's depth, so that one dot and two both stay in the library.
    assert imports_of_the_cli(source, "seedvr2x.vendor.example", NAMES) == []


def test_the_cli_names_are_read_from_its_pyproject() -> None:
    names = cli_names()
    assert {"main", "shot", "run", "manifest", "media"} <= names
    assert "seedvr2x" not in names and "tests" not in names


# The rule.


def test_no_library_file_imports_the_cli() -> None:
    names = cli_names()
    modules = library_modules()
    # The walk reaches the bricks, the vendored code and TransNetV2's: not an empty one.
    assert {"seedvr2x.model", "seedvr2x.vendor.core.infer", "seedvr2x.transnetv2"} <= set(modules)
    findings = [
        f"{path.relative_to(CORE.parent)}:{line}: imports {name}"
        for module, path in modules.items()
        for line, name in imports_of_the_cli(
            path.read_text(), module, names, package=path.name == "__init__.py"
        )
    ]
    assert findings == []


IMPORT_ALL = """
import importlib, json, sys
sys.path.insert(0, {core!r})
for name in {names!r}:
    importlib.import_module(name)
print(json.dumps({{
    name: getattr(module, "__file__", None) for name, module in sys.modules.items()
}}))
"""


def test_importing_the_whole_library_loads_nothing_of_the_cli() -> None:
    # A fresh process, seedvr2x from this tree alone (the module's docstring): every library
    # module imported, then the files of every module loaded looked at: none is one of the CLI's.
    cli = cli_names()
    names = sorted(library_modules())
    code = IMPORT_ALL.format(core=str(CORE), names=names)
    done = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=300
    )
    assert done.returncode == 0, done.stderr[-3000:]
    loaded: dict[str, str | None] = json.loads(done.stdout.splitlines()[-1])
    assert set(names) <= set(loaded)
    assert loaded["seedvr2x"] is not None and Path(loaded["seedvr2x"]).is_relative_to(CORE)
    # The CLI's files are the app's root modules and packages; the app's directory holds its
    # environment too (apps/seedvr2x/.venv), whose files are none of ours.
    ours = [
        name
        for name, file in loaded.items()
        if file
        and Path(file).is_relative_to(APP)
        and Path(file).relative_to(APP).parts[0].removesuffix(".py") in cli
    ]
    assert ours == []
