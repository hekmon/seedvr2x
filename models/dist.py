# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""The rest of seedvr2x's Hugging Face repository, around the model files in models/dist/.

    uv run models/dist.py [--out DIR]

Run after the scripts making the files (seedvr2_fp16.py, transnetv2_weights.py). Writes, in --out
(default models/dist):
- LICENSE: the Apache License 2.0, downloaded from apache.org and checked by its SHA-256;
- NOTICE: each file's origin, copyright, licence and change, from its own safetensors metadata;
- README.md: the model card, hf/README.md beside this script, with the table of the files filled in;
- SHA256SUMS: every file of the directory but itself, as sha256sum writes it.
A file under the MIT licence must have its licence beside it, as NAME.LICENSE for NAME.safetensors.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from string import Template

from common import DIST, HERE, fetch, log, read_header, sha256_file

LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0.txt"
LICENSE_SIZE = 11358
LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
KEYS = ("license", "copyright", "source", "source_url", "source_sha256", "change", "conversion")
LICENCES = {"apache-2.0": "the Apache License, Version 2.0", "mit": "the MIT License"}
HEAD = """seedvr2x model files
Copyright 2026 Edouard Hur

These files are unofficial conversions made for seedvr2x, not their authors' releases. Each
entry gives a file's origin, its copyright and licence, and what was changed.
"""


def short(text: str) -> str:
    """text with its SHA-256s cut to 12 characters and its commit hashes to 7: the card's table
    links to the full revision, and NOTICE and the metadata give every hash in full."""
    text = re.sub(r"\b([0-9a-f]{12})[0-9a-f]{52}\b", "\\1…", text)
    return re.sub(r"\b([0-9a-f]{7})[0-9a-f]{33}\b", "\\1", text)


def licence_file(out: Path, name: str, licence: str) -> str:
    if licence == "apache-2.0":
        return "LICENSE"
    own = name.removesuffix(".safetensors") + ".LICENSE"
    if not (out / own).exists():
        raise SystemExit(f"{name}: {licence}, but no {own} beside it")
    return own


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DIST)
    a = ap.parse_args()
    out: Path = a.out
    if leftovers := sorted(p.name for p in out.glob("*.part")):
        raise SystemExit(f"{out}: unfinished files {leftovers}")
    fetch(LICENSE_URL, out / "LICENSE", LICENSE_SHA256, LICENSE_SIZE)
    files = []
    for path in sorted(out.glob("*.safetensors")):
        md = read_header(path)[0].get("__metadata__", {})
        if missing := [k for k in KEYS if k not in md]:
            raise SystemExit(f"{path.name}: metadata without {missing}")
        if md["license"] not in LICENCES:
            raise SystemExit(f"{path.name}: licence {md['license']}")
        log(f"{path.name}: SHA-256")
        files.append((path.name, path.stat().st_size, sha256_file(path), md))
    if not files:
        raise SystemExit(f"{out}: no .safetensors file")
    notice = [HEAD]
    for name, _, _, md in files:
        notice.append(
            f"{name}\n"
            f"  Made from: {md['source']}\n"
            f"    {md['source_url']}\n"
            f"    SHA-256: {md['source_sha256']}\n"
            f"  {md['copyright']}, under {LICENCES[md['license']]} "
            f"({licence_file(out, name, md['license'])}).\n"
            f"  Changed: {md['change']}.\n"
        )
    (out / "NOTICE").write_text("\n".join(notice))
    rows = [
        f"| `{name}` | {size:,} | `{sha}` | [{short(md['source'])}]({md['source_url']}) "
        f"| {short(md['change'])} |"
        for name, size, sha, md in files
    ]
    card = Template((HERE / "hf" / "README.md").read_text()).substitute(files="\n".join(rows))
    (out / "README.md").write_text(card)
    known = {name: sha for name, _, sha, _ in files}
    sums = [
        f"{known.get(p.name) or sha256_file(p)}  {p.name}\n"
        for p in sorted(out.iterdir())
        if p.is_file() and p.name != "SHA256SUMS"
    ]
    (out / "SHA256SUMS").write_text("".join(sums))
    log(f"{out}: LICENSE, NOTICE, README.md and SHA256SUMS written; {len(sums)} files summed")
    for line in sums:
        print(line, end="")


if __name__ == "__main__":
    main()
