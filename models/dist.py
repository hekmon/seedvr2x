# /// script
# requires-python = ">=3.12"
# dependencies = ["gguf==0.19.0"]
# ///
"""The rest of seedvr2x's Hugging Face repository, around the model files in models/dist/.

    uv run models/dist.py [--out DIR]

Run after the scripts making the files (seedvr2_fp16.py, transnetv2_weights.py, and phase 2's).
Writes, in --out (default models/dist):
- LICENSE: the Apache License 2.0, downloaded from apache.org and checked by its SHA-256;
- NOTICE: each file's origin, copyright, licence and change, from its own metadata (a safetensors
  file's, or a GGUF file's: GGUF_KEYS), and ggml's credit when GGUF files are there;
- README.md: the model card, hf/README.md beside this script, with the table of the files filled in;
- SHA256SUMS: every file of the directory but itself, as sha256sum writes it.
A file under the MIT licence must have its licence beside it, as NAME.LICENSE for NAME.safetensors
or NAME.gguf.

An importance matrix (NAME.imatrix.safetensors, gpu/imatrix_hook.py's file: the name and the
format IMATRIX_FORMAT in its metadata go together) is no model: NOTICE and the card's table list
it after the models, from its own metadata (the runs it sums) and from the GGUF files made with it
(seedvr2_gguf_dyn.py's), whose copyright and licence are its. Those name it in their metadata by
its name and SHA-256: each must find it in --out, under that name and with those bytes, and it
must be named by one of them.

The upload is one directory holding every file. Phase 2's files were made outside models/dist,
in directories of their own until GPU runs validated them, so every file is hard-linked into it
(one filesystem: nothing is copied). Never link NOTICE, README.md or SHA256SUMS: this script
rewrites them, which would rewrite the files they are linked to. With M the directory holding the
scripts' outputs (on the box, its model directory: models/dist, phase2, phase2-dyn-v3 for
the GGUF files made with an importance matrix, gpu/imatrix/v3 for the importance matrices):

    U=$M/upload; mkdir $U
    ln $M/models/dist/{LICENSE,transnetv2.LICENSE,transnetv2.safetensors} $U/
    ln $M/models/dist/seedvr2x_ema_{7b,7b_sharp,vae}_fp16.safetensors $U/
    ln $M/phase2/seedvr2x_ema_7b{,_sharp}_{fp8_scaled,int8_convrot,nvfp4}.safetensors $U/
    ln $M/phase2/seedvr2x_ema_7b{,_sharp}_{Q4_K,Q8_0}.gguf $U/
    ln $M/phase2-dyn-v3/seedvr2x_ema_7b{,_sharp}_{dyn,Q4_K_imatrix}.gguf $U/
    ln $M/gpu/imatrix/v3/seedvr2_ema_7b{,_sharp}_fp16.imatrix.safetensors $U/
    uv run models/dist.py --out $U && (cd $U && sha256sum -c SHA256SUMS)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from string import Template

import gguf
from common import DIST, HERE, fetch, log, read_header, sha256_file

LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0.txt"
LICENSE_SIZE = 11358
LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
KEYS = ("license", "copyright", "source", "source_url", "source_sha256", "change", "conversion")
# a GGUF file's keys for KEYS, as seedvr2_gguf.py and seedvr2_gguf_dyn.py write them
GGUF_KEYS = {
    "license": "general.license",
    "copyright": "seedvr2x.copyright",
    "source": "seedvr2x.source",
    "source_url": "general.source.url",
    "source_sha256": "seedvr2x.source_sha256",
    "change": "seedvr2x.change",
    "conversion": "seedvr2x.conversion",
}
# the importance matrix a GGUF file was made with, as seedvr2_gguf_dyn.py writes it
GGUF_IMATRIX = "seedvr2x.imatrix"
IMATRIX = ".imatrix.safetensors"  # an importance matrix's name ends so
IMATRIX_FORMAT = "seedvr2x-imatrix-1"  # and its metadata's format is gpu/imatrix_hook.py's
IMATRIX_TENSOR = re.compile(  # per block matrix, its inputs' sums of squares and token count
    r"^blocks\.\d+\.(attn\.proj_(qkv|out)\.(txt|vid)|mlp\.(txt|vid)\.proj_(in|out))"
    r"\.weight\.(in_sum2|counts)$"
)
LICENCES = {"apache-2.0": "the Apache License, Version 2.0", "mit": "the MIT License"}
HEAD = """seedvr2x model files
Copyright 2026 Edouard Hur

These files are unofficial conversions made for seedvr2x, not their authors' releases. Each
entry gives a file's origin, its copyright and licence, and what was changed.
"""
GGML = """The GGUF files were quantized by ggml's own quantizer (ggml / llama.cpp, MIT License,
Copyright (c) 2023-2026 The ggml authors), built from llama.cpp at {commits} and called
unchanged; no ggml code is in the files.
"""
LLAMA_CPP = re.compile(r"\bllama\.cpp ([0-9a-f]{40})\b")  # the commit a GGUF file's change names


def short(text: str) -> str:
    """text with its SHA-256s cut to 12 characters and its commit hashes to 7: the card's table
    links to the full revision, and NOTICE and the metadata give every hash in full."""
    text = re.sub(r"\b([0-9a-f]{12})[0-9a-f]{52}\b", "\\1…", text)
    return re.sub(r"\b([0-9a-f]{7})[0-9a-f]{33}\b", "\\1", text)


def licence_file(out: Path, name: str, licence: str) -> str:
    if licence == "apache-2.0":
        return "LICENSE"
    own = Path(name).stem + ".LICENSE"
    if not (out / own).exists():
        raise SystemExit(f"{name}: {licence}, but no {own} beside it")
    return own


def metadata(path: Path) -> dict[str, str]:
    """A file's metadata: a safetensors file's __metadata__, or a GGUF file's GGUF_KEYS under KEYS'
    names, and its GGUF_IMATRIX as "imatrix"."""
    if path.suffix == ".gguf":
        fields = gguf.GGUFReader(path).fields
        keys = {**GGUF_KEYS, "imatrix": GGUF_IMATRIX}
        return {k: fields[g].contents() for k, g in keys.items() if g in fields}
    return read_header(path)[0].get("__metadata__", {})


def importance(path: Path, md: dict[str, str]) -> bool:
    """Whether path is an importance matrix rather than a model, by its name and by its metadata's
    format, which must agree; an importance matrix's tensors checked: per block matrix, its inputs'
    sums of squares (float64 [in]) and token count (int64 [1])."""
    named = path.name.endswith(IMATRIX)
    if named != (md.get("format") == IMATRIX_FORMAT):
        raise SystemExit(
            f"{path.name}: format {md.get('format')!r}; an importance matrix is named "
            f"*{IMATRIX} and has the format {IMATRIX_FORMAT!r}, a model neither"
        )
    if named:
        t = {k: v for k, v in read_header(path)[0].items() if k != "__metadata__"}
        sums = {
            k.removesuffix(".in_sum2")
            for k, v in t.items()
            if k.endswith(".in_sum2") and v["dtype"] == "F64" and len(v["shape"]) == 1
        }
        counts = {
            k.removesuffix(".counts")
            for k, v in t.items()
            if k.endswith(".counts") and v["dtype"] == "I64" and v["shape"] == [1]
        }
        if not sums or sums != counts or len(t) != 2 * len(sums):
            raise SystemExit(f"{path.name}: not an in_sum2 (float64) and a count per matrix")
        if not all(map(IMATRIX_TENSOR.match, t)):
            raise SystemExit(f"{path.name}: other tensors than the blocks' matrices'")
    return named


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DIST)
    a = ap.parse_args()
    out: Path = a.out
    if leftovers := sorted(p.name for p in out.glob("*.part")):
        raise SystemExit(f"{out}: unfinished files {leftovers}")
    fetch(LICENSE_URL, out / "LICENSE", LICENSE_SHA256, LICENSE_SIZE)
    files, imatrices = [], []
    for path in sorted([*out.glob("*.safetensors"), *out.glob("*.gguf")]):
        md = metadata(path)
        if importance(path, md):
            log(f"{path.name}: SHA-256")
            imatrices.append((path.name, path.stat().st_size, sha256_file(path), md))
            continue
        if missing := [k for k in KEYS if k not in md]:
            raise SystemExit(f"{path.name}: metadata without {missing}")
        if md["license"] not in LICENCES:
            raise SystemExit(f"{path.name}: licence {md['license']}")
        im = json.loads(md.get("imatrix") or "{}")
        if im.get("kind") == "file" and not (out / im["file"]).is_file():
            raise SystemExit(f"{path.name}: made with {im['file']}, which {out} lacks")
        log(f"{path.name}: SHA-256")
        files.append((path.name, path.stat().st_size, sha256_file(path), md))
    if not files:
        raise SystemExit(f"{out}: no .safetensors or .gguf file")
    notice = [HEAD]
    commits = set()
    for name, _, _, md in files:
        if name.endswith(".gguf"):
            if not (m := LLAMA_CPP.search(md["change"])):
                raise SystemExit(f"{name}: its change names no llama.cpp commit")
            commits.add(m.group(1)[:7])
    if commits:
        notice.append(GGML.format(commits=", ".join(sorted(commits))))
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
    rows = [
        f"| `{name}` | {size:,} | `{sha}` | [{short(md['source'])}]({md['source_url']}) "
        f"| {short(md['change'])} |"
        for name, size, sha, md in files
    ]
    # the importance matrices, after the models: each with the GGUF files made with it
    made_with: dict[str, list[tuple[str, dict]]] = {}
    imatrix_sha = {name: sha for name, _, sha, _ in imatrices}
    for name, _, _, md in files:
        im = json.loads(md.get("imatrix") or "{}")
        if im.get("kind") == "file":
            if imatrix_sha.get(im["file"]) != im["sha256"]:
                raise SystemExit(
                    f"{name}: made with {im['file']} of SHA-256 {im['sha256']}, "
                    f"not the one in {out}"
                )
            made_with.setdefault(im["file"], []).append((name, md))
    for name, size, sha, md in imatrices:
        if not (users := made_with.get(name)):
            raise SystemExit(f"{name}: no GGUF file of {out} was made with it, by name and SHA-256")
        if len(rights := {(u["copyright"], u["license"]) for _, u in users}) != 1:
            raise SystemExit(f"{name}: the GGUF files made with it differ in copyright or licence")
        ((holder, licence),) = rights
        model, runs = json.loads(md["model"]), json.loads(md["runs"])
        clips = [Path(r["input"]).name for r in runs]
        numz = ", ".join(sorted({r["numz_commit"] for r in runs}))
        layers = sum(k.endswith(".in_sum2") for k in read_header(out / name)[0])
        made = (
            f"{len(runs)} runs of {model['name']} ({model['size']:,} bytes, numz's float16 file) "
            f"in numz's SeedVR2 at {numz}, on {len(set(clips))} calibration clips: "
            f"{', '.join(clips)}"
        )
        what = (
            f"an importance matrix, not a model: for each of the {layers} attention and MLP "
            "matrices of the blocks, per input channel, the sum over the runs' tokens of the "
            "input's square (in_sum2, float64) and the count of tokens (counts); in_sum2 / counts, "
            "each channel's mean square, is llama.cpp's imatrix. Collected by seedvr2x's "
            "models/gpu/imatrix_hook.py; models/seedvr2_gguf_dyn.py made "
            f"{' and '.join(u for u, _ in users)} with it, whose metadata name it by this name "
            "and its SHA-256"
        )
        notice.append(
            f"{name}\n"
            f"  Made from: {made}\n"
            f"  {holder}, under {LICENCES[licence]} ({licence_file(out, name, licence)}).\n"
            f"  What it is: {what}.\n"
        )
        rows.append(f"| `{name}` | {size:,} | `{sha}` | {short(made)} | {short(what)} |")
    (out / "NOTICE").write_text("\n".join(notice))
    card = Template((HERE / "hf" / "README.md").read_text()).substitute(files="\n".join(rows))
    (out / "README.md").write_text(card)
    known = {name: sha for name, _, sha, _ in files + imatrices}
    sums = [
        f"{known.get(p.name) or sha256_file(p)}  {p.name}\n"
        for p in sorted(out.iterdir())
        if p.is_file() and p.name != "SHA256SUMS"
    ]
    (out / "SHA256SUMS").write_text("".join(sums))
    log(f"{len(files)} model files, {len(imatrices)} importance matrices")
    log(f"{out}: LICENSE, NOTICE, README.md and SHA256SUMS written; {len(sums)} files summed")
    for line in sums:
        print(line, end="")


if __name__ == "__main__":
    main()
