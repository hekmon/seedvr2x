"""The library imports nothing of the command line's (DESIGN.md, Repository and apps: the
library is a collection of bricks, and the CLI assembles them; a test holds the rule). This is
the rule's form while both live in one package: the library is the modules listed below
(LIBRARY, and the vendored code), everything else in seedvr2x is the command line's, and no
library module may import one, by an import statement, a relative import or a module named by
a string. It becomes the split's own test when the command line moves out of this package.

Two ways, each on the CPU. A static scan of every library file (`imports_of_the_rest`, held on
its own to synthetic sources, one per form, as tests/test_ends_with_us.py holds its scan), which
names the file and the line. And a fresh process that imports every library module, the
vendored models' too, then lists the seedvr2x modules loaded: none is outside the library. That
one sees what no scan can, an import made through a third party or by a name built at run time.
The process runs with `-I` and the source directory put on its path by the code it is given:
`-I` drops PYTHONPATH and the working directory, so that nothing else can stand for the
package, and the editable install of the CLI's environment, which may point at another tree,
cannot either."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).parent.parent / "src"
PACKAGE = SOURCE / "seedvr2x"

# The library's modules (DESIGN.md, Repository and apps; the layout of today: the bricks at the
# package's top and in runtime/). `seedvr2x.runtime` is the package marker both layers share.
LIBRARY = frozenset(
    {
        "seedvr2x",
        "seedvr2x.clock",
        "seedvr2x.errors",
        "seedvr2x.files",
        "seedvr2x.frames",
        "seedvr2x.passes",
        "seedvr2x.peaks",
        "seedvr2x.sampling",
        "seedvr2x.runtime",
        "seedvr2x.runtime.blockswap",
        "seedvr2x.runtime.colour",
        "seedvr2x.runtime.compiling",
        "seedvr2x.runtime.detector",
        "seedvr2x.runtime.memory",
        "seedvr2x.runtime.model",
        "seedvr2x.runtime.pull",
        "seedvr2x.runtime.tilegrid",
        "seedvr2x.runtime.tiling",
        "seedvr2x.runtime.weights",
    }
)
# The vendored code is the library's, whole.
LIBRARY_TREES = ("seedvr2x.vendor", "seedvr2x.transnetv2")
# Calls that take a module's name as a string: import_module, __import__, find_spec.
BY_NAME = ("import_module", "__import__", "find_spec")


def in_library(module: str) -> bool:
    """Whether the seedvr2x module `module` is the library's."""
    return module in LIBRARY or any(
        module == tree or module.startswith(f"{tree}.") for tree in LIBRARY_TREES
    )


def is_module(name: str) -> bool:
    """Whether the dotted name is a module or a package of the seedvr2x on disk."""
    path = SOURCE.joinpath(*name.split("."))
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


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


def imports_of_the_rest(source: str, module: str, package: bool = False) -> list[tuple[int, str]]:
    """Each (line, module) at which `source`, the code of the seedvr2x module `module` (a
    package's __init__ if `package`), imports a seedvr2x module outside the library: an import
    or a from-import, whose names count as modules where they are seedvr2x's (`from seedvr2x
    import cli`); a relative import, resolved against `module`; a call of import_module,
    __import__ or find_spec with a string literal naming one. Anywhere in the code, a function's
    included. A relative import climbing out of the package is a finding too."""
    found: list[tuple[int, str]] = []

    def check(line: int, name: str | None) -> None:
        if name is None:
            found.append((line, "a relative import out of the package"))
        elif (name == "seedvr2x" or name.startswith("seedvr2x.")) and not in_library(name):
            found.append((line, name))

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                check(node.lineno, alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = resolved(module, package, node.level, node.module)
            else:
                base = node.module
            check(node.lineno, base)
            # The names a package gives as its submodules: from seedvr2x import cli.
            if base is not None and in_library(base):
                for alias in node.names:
                    if is_module(f"{base}.{alias.name}"):
                        check(node.lineno, f"{base}.{alias.name}")
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
    return sorted(set(found))


def modules_of(directory: Path) -> dict[str, Path]:
    """Every module under directory (the package seedvr2x's), by dotted name, its __init__s
    named for their package."""
    found: dict[str, Path] = {}
    for path in sorted(directory.rglob("*.py")):
        parts = path.relative_to(SOURCE).with_suffix("").parts
        if parts[-1] == "__main__":
            continue
        found[".".join(parts[:-1] if parts[-1] == "__init__" else parts)] = path
    return found


def library_modules() -> dict[str, Path]:
    return {name: path for name, path in modules_of(PACKAGE).items() if in_library(name)}


# The scan, held on synthetic sources.


def test_an_absolute_import_of_the_rest_is_found() -> None:
    source = "import os\nimport seedvr2x.cli\nimport seedvr2x.media.ffmpeg as ff\n"
    assert imports_of_the_rest(source, "seedvr2x.peaks") == [
        (2, "seedvr2x.cli"),
        (3, "seedvr2x.media.ffmpeg"),
    ]
    assert imports_of_the_rest("from seedvr2x.runtime.shot import shot_layout", "seedvr2x.x") == [
        (1, "seedvr2x.runtime.shot")
    ]


def test_a_from_import_of_a_submodule_of_the_rest_is_found() -> None:
    assert imports_of_the_rest("from seedvr2x import cli\n", "seedvr2x.peaks") == [
        (1, "seedvr2x.cli")
    ]
    assert imports_of_the_rest("from seedvr2x.runtime import shot, model\n", "seedvr2x.peaks") == [
        (1, "seedvr2x.runtime.shot")
    ]
    assert imports_of_the_rest("from seedvr2x import media\n", "seedvr2x.peaks") == [
        (1, "seedvr2x.media")
    ]


def test_a_relative_import_of_the_rest_is_resolved_and_found() -> None:
    assert imports_of_the_rest("from . import shot\n", "seedvr2x.runtime.model") == [
        (1, "seedvr2x.runtime.shot")
    ]
    assert imports_of_the_rest("from .shot import shot_layout\n", "seedvr2x.runtime.model") == [
        (1, "seedvr2x.runtime.shot")
    ]
    assert imports_of_the_rest("from .. import cli\n", "seedvr2x.runtime.model") == [
        (1, "seedvr2x.cli")
    ]
    assert imports_of_the_rest("from ..media import ffmpeg\n", "seedvr2x.runtime.model") == [
        (1, "seedvr2x.media")
    ]
    # In a package's own __init__, one dot is the package itself.
    assert imports_of_the_rest("from . import shot\n", "seedvr2x.runtime", package=True) == [
        (1, "seedvr2x.runtime.shot")
    ]
    assert imports_of_the_rest("from ... import x\n", "seedvr2x.peaks") == [
        (1, "a relative import out of the package")
    ]


def test_an_aliased_import_is_found() -> None:
    assert imports_of_the_rest("import seedvr2x.cli as c\n", "seedvr2x.peaks") == [
        (1, "seedvr2x.cli")
    ]
    assert imports_of_the_rest("from seedvr2x import cli as c\n", "seedvr2x.peaks") == [
        (1, "seedvr2x.cli")
    ]


def test_an_import_inside_a_function_is_found() -> None:
    source = "def late():\n    if True:\n        from seedvr2x.runtime.planner import plan\n"
    assert imports_of_the_rest(source, "seedvr2x.peaks") == [(3, "seedvr2x.runtime.planner")]


@pytest.mark.parametrize(
    "call",
    [
        'importlib.import_module("seedvr2x.cli")',
        'import_module("seedvr2x.cli")',
        'il.import_module(name="seedvr2x.cli")',
        'importlib.util.find_spec("seedvr2x.cli")',
        'find_spec("seedvr2x.cli")',
        '__import__("seedvr2x.cli")',
        'importlib.import_module(".cli", "seedvr2x")',
        'importlib.import_module(".cli", package="seedvr2x")',
    ],
)
def test_a_module_named_by_a_string_is_found(call: str) -> None:
    assert imports_of_the_rest(f"x = {call}\n", "seedvr2x.peaks") == [(1, "seedvr2x.cli")]


def test_what_is_the_librarys_or_not_seedvr2xs_is_fine() -> None:
    source = (
        "import os\n"
        "import torch\n"
        "from seedvr2x import clock, frames\n"
        "from seedvr2x.runtime import model, tiling\n"
        "from seedvr2x.runtime.model import Models\n"
        "from seedvr2x.vendor.core.infer import VideoDiffusionInfer\n"
        "from seedvr2x.transnetv2.transnetv2_pytorch import TransNetV2\n"
        "from . import weights\n"
        "from .. import sampling\n"
        "from seedvr2x import __doc__\n"
        'importlib.import_module("seedvr2x.runtime.pull")\n'
        'importlib.import_module("torch")\n'
        "importlib.import_module(name)\n"
    )
    assert imports_of_the_rest(source, "seedvr2x.runtime.pull") == []


def test_the_library_list_is_whole_modules() -> None:
    # The list names modules that are there, and the shared package marker.
    assert [name for name in LIBRARY if not is_module(name)] == []
    assert not in_library("seedvr2x.cli") and not in_library("seedvr2x.media.files")
    assert not in_library("seedvr2x.runtime.shot") and not in_library("seedvr2x.vendorish")
    assert in_library("seedvr2x.vendor.core.infer") and in_library("seedvr2x.transnetv2")


# The rule.


def test_no_library_file_imports_the_rest() -> None:
    modules = library_modules()
    # The list reaches the vendored code and the bricks: not an empty walk.
    assert len(modules) > len(LIBRARY) and "seedvr2x.vendor.core.infer" in modules
    assert set(LIBRARY) <= set(modules)
    findings = [
        f"{path.relative_to(SOURCE.parent)}:{line}: imports {name}"
        for module, path in modules.items()
        for line, name in imports_of_the_rest(
            path.read_text(), module, package=path.name == "__init__.py"
        )
    ]
    assert findings == []


IMPORT_ALL = """
import importlib, json, sys
sys.path.insert(0, {source!r})
for name in {names!r}:
    importlib.import_module(name)
print(json.dumps(sorted(m for m in sys.modules if m == "seedvr2x" or m.startswith("seedvr2x."))))
"""


def test_importing_the_whole_library_loads_none_of_the_rest() -> None:
    # A fresh process, seedvr2x from this tree alone (the module's docstring): every library
    # module imported, the vendored models too, then the seedvr2x modules loaded listed.
    names = sorted(library_modules())
    code = IMPORT_ALL.format(source=str(SOURCE), names=names)
    done = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=300
    )
    assert done.returncode == 0, done.stderr[-3000:]
    loaded: list[str] = json.loads(done.stdout.splitlines()[-1])
    assert set(names) <= set(loaded)
    assert [name for name in loaded if not in_library(name)] == []
