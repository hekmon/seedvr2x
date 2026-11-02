#!/usr/bin/env python3
"""Copy, diff and check the vendored model code in src/seedvr2x/vendor/.

The model code is numz's ComfyUI-SeedVR2_VideoUpscaler at 4490bd1, itself ByteDance's SeedVR at
e4de8c2 with numz's changes, copied path for path: numz's src/X is vendor/X, and its configs and
text embeddings sit at the root of vendor/ (AGENTS.md, The vendored model layer). Upstream files
are read from the git objects of the pinned commits, never from a work tree, so a moved submodule
can't shift the reference.

Usage, from seedvr2x/:
  uv run tools/vendor.py copy [--numz DIR] [--force]   write numz's files into vendor/
  uv run tools/vendor.py diff [--bytedance] [--raw] [--full] [PATH...]
  uv run tools/vendor.py check [--numz DIR]

copy refuses to overwrite a file that differs from numz unless --force: those are our changes.
diff compares vendor/ with numz, or with ByteDance (--bytedance), whose absolute imports are first
rewritten in numz's relative form unless --raw, so that the import style isn't reported as a
change, and lists the numz files dropped on purpose. check verifies that a file carries the marker
exactly when it differs from numz, that the original header is kept above it, that binary files
are untouched, that no dropped file is back, that every relative import resolves inside vendor/,
which keeps numz's runtime out, and that ruff finds no syntax error or undefined name that numz's
own files don't have.
"""

import argparse
import ast
import difflib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

NUMZ_COMMIT = "4490bd1f482e026674543386bb2a4d176da245b9"
BYTEDANCE_COMMIT = "e4de8c24441a67e1b7df56abea10645059bb1185"

PROJECT = Path(__file__).resolve().parents[1]
VENDOR = PROJECT / "src" / "seedvr2x" / "vendor"
NUMZ = PROJECT.parent / "upstream" / "seedvr2-numz"
BYTEDANCE = PROJECT.parent / "upstream" / "seedvr-bytedance"

MARKER = "# Modified for seedvr2x:"
# Syntax errors and undefined names: what an edit can break. numz's own files already have some
# (names a function deletes after the lambdas using them, a class nothing builds), so only those
# it doesn't have count.
LINT_RULES = "E9,F63,F7,F82"
BINARY_SUFFIXES = (".pt",)

# The files vendored, as numz paths: the import closure of the DiTs (7B, 3B), the VAE, the
# diffusion sampler, VideoDiffusionInfer and the input transforms, plus the config loader
# (common/config.py) and set_seed (common/seed.py), the configs and the text embeddings.
FILES: tuple[str, ...] = (
    "configs_3b/main.yaml",
    "configs_7b/main.yaml",
    "neg_emb.pt",
    "pos_emb.pt",
    "src/__init__.py",
    "src/common/__init__.py",
    "src/common/cache.py",
    "src/common/config.py",
    "src/common/diffusion/__init__.py",
    "src/common/diffusion/config.py",
    "src/common/diffusion/samplers/base.py",
    "src/common/diffusion/samplers/euler.py",
    "src/common/diffusion/schedules/base.py",
    "src/common/diffusion/schedules/lerp.py",
    "src/common/diffusion/timesteps/base.py",
    "src/common/diffusion/timesteps/sampling/trailing.py",
    "src/common/diffusion/types.py",
    "src/common/diffusion/utils.py",
    "src/common/distributed/__init__.py",
    "src/common/distributed/basic.py",
    "src/common/half_precision_fixes.py",
    "src/common/logger.py",
    "src/common/seed.py",
    "src/core/__init__.py",
    "src/core/infer.py",
    "src/data/image/transforms/area_resize.py",
    "src/data/image/transforms/divisible_crop.py",
    "src/data/image/transforms/na_resize.py",
    "src/data/image/transforms/side_resize.py",
    "src/models/dit_3b/attention.py",
    "src/models/dit_3b/embedding.py",
    "src/models/dit_3b/mlp.py",
    "src/models/dit_3b/mm.py",
    "src/models/dit_3b/modulation.py",
    "src/models/dit_3b/na.py",
    "src/models/dit_3b/nablocks/__init__.py",
    "src/models/dit_3b/nablocks/attention/__init__.py",
    "src/models/dit_3b/nablocks/attention/mmattn.py",
    "src/models/dit_3b/nablocks/mmsr_block.py",
    "src/models/dit_3b/nadit.py",
    "src/models/dit_3b/normalization.py",
    "src/models/dit_3b/patch/__init__.py",
    "src/models/dit_3b/patch/patch_v1.py",
    "src/models/dit_3b/rope.py",
    "src/models/dit_3b/window.py",
    "src/models/dit_7b/attention.py",
    "src/models/dit_7b/blocks/__init__.py",
    "src/models/dit_7b/blocks/mmdit_window_block.py",
    "src/models/dit_7b/embedding.py",
    "src/models/dit_7b/mlp.py",
    "src/models/dit_7b/mm.py",
    "src/models/dit_7b/modulation.py",
    "src/models/dit_7b/na.py",
    "src/models/dit_7b/nablocks/__init__.py",
    "src/models/dit_7b/nablocks/mmsr_block.py",
    "src/models/dit_7b/nadit.py",
    "src/models/dit_7b/normalization.py",
    "src/models/dit_7b/patch.py",
    "src/models/dit_7b/rope.py",
    "src/models/dit_7b/window.py",
    "src/models/video_vae_v3/modules/attn_video_vae.py",
    "src/models/video_vae_v3/modules/causal_inflation_lib.py",
    "src/models/video_vae_v3/modules/context_parallel_lib.py",
    "src/models/video_vae_v3/modules/global_config.py",
    "src/models/video_vae_v3/modules/types.py",
    "src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml",
)

# numz files deliberately no longer vendored, which numz's model code imports: the
# sequence-parallel code, an identity on one GPU (DESIGN.md, Vendored model code), removed with
# its calls.
DROPPED: tuple[str, ...] = (
    "src/common/distributed/advanced.py",
    "src/common/distributed/ops.py",
)

# numz path prefix to ByteDance path prefix, first match wins (provenance.md, Lineage).
BYTEDANCE_PATHS: tuple[tuple[str, str], ...] = (
    ("src/models/dit_7b/", "models/dit/"),
    ("src/models/dit_3b/", "models/dit_v2/"),
    ("src/core/infer.py", "projects/video_diffusion_sr/infer.py"),
    ("src/", ""),
    ("", ""),
)

# ByteDance's absolute module prefixes and numz's modules for them, for the import rewrite.
BYTEDANCE_MODULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("models", "dit_v2"), ("src", "models", "dit_3b")),
    (("models", "dit"), ("src", "models", "dit_7b")),
    (("projects", "video_diffusion_sr", "infer"), ("src", "core", "infer")),
    (("common",), ("src", "common")),
    (("data",), ("src", "data")),
    (("models",), ("src", "models")),
)
ABSOLUTE_IMPORT = re.compile(r"^(\s*)from\s+((?:common|data|models|projects)(?:\.\w+)*)\s+import\b")


def vendor_path(numz_path: str) -> str:
    """Path under vendor/ of a numz file: src/X is X, the root files stay at the root."""
    return numz_path.removeprefix("src/")


def bytedance_path(numz_path: str) -> str:
    for numz_prefix, bytedance_prefix in BYTEDANCE_PATHS:
        if numz_path.startswith(numz_prefix):
            return bytedance_prefix + numz_path.removeprefix(numz_prefix)
    raise AssertionError("the last prefix matches every path")


def git_show(repo: Path, commit: str, path: str) -> bytes | None:
    """The file at path in commit, or None when the commit has no such file."""
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{path}"], capture_output=True, check=False
    )
    return result.stdout if result.returncode == 0 else None


def check_commit(repo: Path, commit: str) -> None:
    result = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"{repo}: commit {commit} not found (submodule not checked out?)")


def is_binary(path: str) -> bool:
    return path.endswith(BINARY_SUFFIXES)


def header_lines(text: str) -> list[str]:
    """The leading comment block of a file: its licence header, empty when it has none."""
    lines = text.splitlines(keepends=True)
    count = 0
    while count < len(lines) and lines[count].startswith("#"):
        count += 1
    return lines[:count]


def to_numz_imports(text: str, numz_path: str) -> str:
    """Rewrite ByteDance's absolute imports in the relative form numz uses at numz_path."""
    package = numz_path.split("/")[:-1]

    def relative(match: re.Match[str]) -> str:
        module = tuple(match.group(2).split("."))
        for bytedance_prefix, numz_prefix in BYTEDANCE_MODULES:
            if module[: len(bytedance_prefix)] == bytedance_prefix:
                target = list(numz_prefix + module[len(bytedance_prefix) :])
                break
        else:
            return match.group(0)
        common = 0
        while common < min(len(package), len(target)) and package[common] == target[common]:
            common += 1
        dots = "." * (len(package) - common + 1)
        rest = ".".join(target[common:])
        return f"{match.group(1)}from {dots}{rest} import"

    return "".join(ABSOLUTE_IMPORT.sub(relative, line) for line in text.splitlines(keepends=True))


@dataclass
class Comparison:
    path: str  # numz path
    status: str  # identical, modified, missing (from vendor/), absent (from upstream), dropped
    added: int = 0
    removed: int = 0
    diff: str = ""


def compare(upstream: bytes | None, ours: bytes | None, numz_path: str, label: str) -> Comparison:
    if ours is None:
        return Comparison(numz_path, "missing")
    if upstream is None:
        return Comparison(numz_path, "absent")
    if upstream == ours:
        return Comparison(numz_path, "identical")
    if is_binary(numz_path):
        return Comparison(numz_path, "modified")
    lines = list(
        difflib.unified_diff(
            upstream.decode().splitlines(keepends=True),
            ours.decode().splitlines(keepends=True),
            f"{label}/{numz_path}",
            f"vendor/{vendor_path(numz_path)}",
        )
    )
    added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
    if not lines:  # equal text, different bytes (line endings): count as modified
        return Comparison(numz_path, "modified")
    return Comparison(numz_path, "modified", added, removed, "".join(lines))


def read_ours(numz_path: str) -> bytes | None:
    path = VENDOR / vendor_path(numz_path)
    return path.read_bytes() if path.is_file() else None


def selected(paths: list[str], files: tuple[str, ...] = FILES) -> list[str]:
    """files (FILES by default), or those of them under the given vendor/ paths."""
    if not paths:
        return list(files)
    wanted = [p.rstrip("/").split("vendor/", 1)[-1] for p in paths]
    return [
        f
        for f in files
        if any(vendor_path(f) == w or vendor_path(f).startswith(w + "/") for w in wanted)
    ]


def cmd_copy(numz: Path, force: bool) -> int:
    check_commit(numz, NUMZ_COMMIT)
    refused: list[str] = []
    for numz_path in FILES:
        data = git_show(numz, NUMZ_COMMIT, numz_path)
        if data is None:
            raise SystemExit(f"{numz_path}: not in numz at {NUMZ_COMMIT[:7]}")
        dest = VENDOR / vendor_path(numz_path)
        if dest.is_file() and dest.read_bytes() != data and not force:
            refused.append(vendor_path(numz_path))
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    for path in refused:
        print(f"not overwritten, differs from numz: vendor/{path}", file=sys.stderr)
    print(f"{len(FILES) - len(refused)} files written from numz {NUMZ_COMMIT[:7]}")
    return 1 if refused else 0


def cmd_diff(paths: list[str], bytedance: bool, raw: bool, full: bool) -> int:
    repo, commit, label = (
        (BYTEDANCE, BYTEDANCE_COMMIT, "bytedance") if bytedance else (NUMZ, NUMZ_COMMIT, "numz")
    )
    check_commit(repo, commit)
    results: list[Comparison] = []
    for numz_path in selected(paths):
        upstream_path = bytedance_path(numz_path) if bytedance else numz_path
        upstream = git_show(repo, commit, upstream_path)
        if bytedance and upstream is not None and not raw and not is_binary(numz_path):
            upstream = to_numz_imports(upstream.decode(), numz_path).encode()
        results.append(compare(upstream, read_ours(numz_path), numz_path, label))
    dropped = [Comparison(numz_path, "dropped") for numz_path in selected(paths, DROPPED)]
    counts = {s: sum(1 for r in results if r.status == s) for s in ("identical", "modified")}
    print(
        f"vendor/ against {label} {commit[:7]}: {len(results)} files, "
        f"{counts['identical']} identical, {counts['modified']} modified; {len(dropped)} dropped"
    )
    for r in results + dropped:
        if r.status == "identical":
            continue
        where = (
            f"  ({label}: {bytedance_path(r.path)})" if bytedance and r.status != "absent" else ""
        )
        lines = f"  +{r.added} -{r.removed}" if r.status == "modified" else ""
        print(f"  {r.status:9} vendor/{vendor_path(r.path)}{lines}{where}")
    if full:
        for r in results:
            if r.diff:
                print()
                print(r.diff, end="")
    return 0


def module_exists(parts: list[str]) -> bool:
    """Whether vendor/ holds the module or package named by parts (relative to vendor/)."""
    base = VENDOR.joinpath(*parts)
    return base.with_suffix(".py").is_file() or base.is_dir()


def unresolved_imports(source: str, vendor_file: str) -> Iterator[str]:
    """The relative imports of a vendored file that don't resolve inside vendor/."""
    package = vendor_file.split("/")[:-1]
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ImportFrom) or node.level == 0:
            continue
        if node.level - 1 > len(package):
            yield f"line {node.lineno}: {'.' * node.level}{node.module or ''} leaves vendor/"
            continue
        base = package[: len(package) - (node.level - 1)]
        target = base + (node.module.split(".") if node.module else [])
        if node.module and not module_exists(target):
            yield f"line {node.lineno}: {'.' * node.level}{node.module} not in vendor/"
        elif not node.module:
            for alias in node.names:
                if not module_exists([*target, alias.name]) and not module_exists(target):
                    yield f"line {node.lineno}: {'.' * node.level} import {alias.name}"


def lint_findings(root: Path) -> Counter[tuple[str, str, str]]:
    """ruff's error findings under root, by file, rule and message: line numbers, which our edits
    shift, are left out."""
    ruff = [sys.executable, "-m", "ruff", "check", "--isolated", "--exit-zero"]
    result = subprocess.run(
        [*ruff, "--select", LINT_RULES, "--output-format", "json", str(root)],
        capture_output=True,
        text=True,
        check=True,
    )
    return Counter(
        (Path(f["filename"]).relative_to(root).as_posix(), f["code"], f["message"])
        for f in json.loads(result.stdout)
    )


def new_lint_errors(numz: Path) -> list[str]:
    """The lint errors of vendor/ that numz's own copy of the files doesn't have."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for numz_path in FILES:
            data = git_show(numz, NUMZ_COMMIT, numz_path)
            if numz_path.endswith(".py") and data is not None:
                dest = base / vendor_path(numz_path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
        upstream = lint_findings(base)
    new = lint_findings(VENDOR) - upstream
    return [f"vendor/{path}: {code} {message}" for path, code, message in sorted(new.elements())]


def cmd_check(numz: Path) -> int:
    check_commit(numz, NUMZ_COMMIT)
    problems: list[str] = []
    expected = {vendor_path(f) for f in FILES}
    dropped = {vendor_path(f) for f in DROPPED}
    for numz_path in FILES:
        where = f"vendor/{vendor_path(numz_path)}"
        upstream = git_show(numz, NUMZ_COMMIT, numz_path)
        ours = read_ours(numz_path)
        if upstream is None:
            problems.append(f"{where}: not in numz at {NUMZ_COMMIT[:7]}")
            continue
        if ours is None:
            problems.append(f"{where}: missing")
            continue
        if is_binary(numz_path):
            if ours != upstream:
                problems.append(f"{where}: binary file differs from numz")
            continue
        text, header = ours.decode(), header_lines(upstream.decode())
        lines = text.splitlines(keepends=True)
        marked = len(lines) > len(header) and lines[len(header)].startswith(MARKER)
        if lines[: len(header)] != header:
            problems.append(f"{where}: original header changed")
        if ours != upstream and not marked:
            problems.append(f"{where}: differs from numz but has no '{MARKER}' line")
        if ours == upstream and MARKER in text:
            problems.append(f"{where}: marker in an unmodified file")
        if numz_path.endswith(".py"):
            problems += [f"{where}: {p}" for p in unresolved_imports(text, vendor_path(numz_path))]
    for path in sorted(VENDOR.rglob("*")):
        relative = path.relative_to(VENDOR).as_posix()
        if not path.is_file() or "__pycache__" in path.parts or relative in expected:
            continue
        if relative in dropped:
            problems.append(f"vendor/{relative}: dropped on purpose (DROPPED), but back")
        else:
            problems.append(f"vendor/{relative}: not a numz file")
    problems += new_lint_errors(numz)
    for problem in problems:
        print(problem)
    print(f"vendor/: {len(FILES)} files checked, {len(problems)} problems")
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    commands = parser.add_subparsers(dest="command", required=True)
    copy = commands.add_parser("copy", help="write numz's files into vendor/")
    copy.add_argument("--numz", type=Path, default=NUMZ, help="numz repository (git)")
    copy.add_argument("--force", action="store_true", help="overwrite our changes too")
    diff = commands.add_parser("diff", help="compare vendor/ with numz or ByteDance")
    diff.add_argument("--bytedance", action="store_true", help="compare with ByteDance")
    diff.add_argument("--raw", action="store_true", help="keep ByteDance's absolute imports")
    diff.add_argument("--full", action="store_true", help="print the unified diffs")
    diff.add_argument("paths", nargs="*", help="vendor/ files or directories (default: all)")
    check = commands.add_parser("check", help="verify markers, headers and imports")
    check.add_argument("--numz", type=Path, default=NUMZ, help="numz repository (git)")
    args = parser.parse_args()
    if args.command == "copy":
        return cmd_copy(args.numz, args.force)
    if args.command == "diff":
        return cmd_diff(args.paths, args.bytedance, args.raw, args.full)
    return cmd_check(args.numz)


if __name__ == "__main__":
    raise SystemExit(main())
