#!/usr/bin/env python3
"""Whether a card gives the same bits for the same work, seedvr2x left out: a DiT window's kind of
work at its sizes chained BLOCKS times, on fixed inputs made once from a seeded generator, each
pass's output hashed and compared with the first pass's.

One pass: bf16 matrix products of a 27-latent 1080p window's tokens (223,560 x 3,072 by 3,072 x
3,072, by 3,072 x 12,288 and back), FlashAttention 2's varlen forward over attention windows of
the 7B's sizes (7 x 16 x 27 tokens, 24 heads of 128) and the elementwise work between, chained as
blocks are (each output the next input, renormalised). One JSON line a pass: its digest, whether
it is the first's, its time.

  python repeat_chain.py --passes 40 --out FILE.jsonl

Run under small_card.py (`run --keep 0 -- repeat_chain.py ...`, with `ballast --card whole`
watching a desktop's share), seedvr2x's venv with FlashAttention 2 installed. Written for
impl-place-2's study of the other bits (small-cards.md, 2026-11-07): 4 passes of 40 gave other
bits on the RTX 5090 that day."""

import argparse
import hashlib
import json
import time

import torch
from flash_attn import flash_attn_varlen_func

TOKENS = 27 * 69 * 120
DIM, HIDDEN, HEADS, HEAD = 3072, 12288, 24, 128
WINDOW = 7 * 16 * 27
BLOCKS = 36


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passes", type=int, default=40)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    generator = torch.Generator(device="cpu").manual_seed(7)

    def made(*shape: int, scale: float) -> torch.Tensor:
        return (torch.randn(*shape, generator=generator) * scale).to(device, torch.bfloat16)

    x0 = made(TOKENS, DIM, scale=1.0)
    qkv = made(DIM, 3 * DIM, scale=DIM**-0.5)
    proj = made(DIM, DIM, scale=DIM**-0.5)
    up = made(DIM, HIDDEN, scale=DIM**-0.5)
    down = made(HIDDEN, DIM, scale=HIDDEN**-0.5)
    windows = TOKENS // WINDOW
    lengths = [WINDOW] * windows + ([TOKENS - windows * WINDOW] if TOKENS % WINDOW else [])
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    first = None
    out = open(args.out, "a")
    for number in range(args.passes):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            x = x0.clone()
            for _ in range(BLOCKS):
                q, k, v = (x @ qkv).view(TOKENS, 3, HEADS, HEAD).unbind(1)
                attended = flash_attn_varlen_func(
                    q.contiguous(), k.contiguous(), v.contiguous(), cu, cu, WINDOW, WINDOW
                )
                x = x + attended.reshape(TOKENS, DIM) @ proj
                x = x + torch.nn.functional.gelu(x @ up, approximate="tanh") @ down
                x = x / x.float().pow(2).mean(-1, keepdim=True).add(1e-6).sqrt().to(x.dtype)
                del q, k, v, attended
        torch.cuda.synchronize(device)
        wall = time.perf_counter() - started
        digest = hashlib.sha256(x.view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:12]
        first = first or digest
        line = {"pass": number, "digest": digest, "equal": digest == first, "s": round(wall, 2)}
        print(json.dumps(line), flush=True)
        out.write(json.dumps(line) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
