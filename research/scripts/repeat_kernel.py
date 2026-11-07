#!/usr/bin/env python3
"""One kernel repeated on the same inputs, each result compared on the card with the first,
seedvr2x left out. For each result that differs: how many values, the largest distance in units
in the last place (bfloat16's bits read as integers of one sign), the most bits that differ in
one value, and where (the first 8 places).

  python repeat_kernel.py --kernel gemm|attention|both --seconds 180 --out FILE.jsonl

gemm: (33,120 x 3,072) @ (3,072 x 12,288) in bfloat16, the 7B's MLP shape at 4 latents of 1080p
tokens; attention: FlashAttention 2's varlen forward over a 27-latent window's tokens cut in the
7B's attention windows (3,024 tokens, 24 heads of 128). A summary line every 1,000 calls. Run as
repeat_chain.py is. Written for impl-place-2's study of the other bits (small-cards.md,
2026-11-07): 5 of 18,898 GEMMs gave other values on the RTX 5090 that day, none of 5,634
attention calls."""

import argparse
import json
import time

import torch

DIM, HIDDEN, HEADS, HEAD = 3072, 12288, 24, 128
TOKENS, WINDOW = 27 * 69 * 120, 7 * 16 * 27


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=("gemm", "attention", "both"), default="both")
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    generator = torch.Generator(device="cpu").manual_seed(11)
    out = open(args.out, "a")

    def say(line: dict) -> None:
        out.write(json.dumps({"t": round(time.time(), 3), **line}) + "\n")
        out.flush()

    def made(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator).to(device, torch.bfloat16)

    kernels = []
    if args.kernel in ("gemm", "both"):
        a, b = made(4 * 69 * 120, DIM), made(DIM, HIDDEN) * DIM**-0.5
        kernels.append(("gemm", lambda: a @ b))
    if args.kernel in ("attention", "both"):
        from flash_attn import flash_attn_varlen_func

        q, k, v = (made(TOKENS, HEADS, HEAD) for _ in range(3))
        windows = TOKENS // WINDOW
        lengths = [WINDOW] * windows + ([TOKENS - windows * WINDOW] if TOKENS % WINDOW else [])
        cu = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device
        )
        kernels.append(
            ("attention", lambda: flash_attn_varlen_func(q, k, v, cu, cu, WINDOW, WINDOW))
        )
    for name, run in kernels:
        with torch.no_grad():
            first = run().view(torch.int16)
            started, calls, other = time.monotonic(), 0, 0
            while time.monotonic() - started < args.seconds:
                got = run().view(torch.int16)
                calls += 1
                differ = got != first
                if bool(differ.any()):
                    other += 1
                    at = differ.nonzero()
                    a_, b_ = first[differ].int(), got[differ].int()
                    # bfloat16 bits as one-signed integers: the distance is in units in the last
                    # place where the signs agree.
                    ordered = lambda x: torch.where(x < 0, -32768 - x, x)  # noqa: E731
                    ulp = (ordered(a_) - ordered(b_)).abs()
                    bits = (a_ ^ b_).bitwise_and(0xFFFF)
                    popcount = sum(((bits >> i) & 1) for i in range(16))
                    say(
                        {
                            "kernel": name,
                            "call": calls,
                            "values": int(differ.sum()),
                            "ulp_most": int(ulp.max()),
                            "ulp_one": int((ulp == 1).sum()),
                            "bits_most": int(popcount.max()),
                            "where": at[:8].tolist(),
                        }
                    )
                if calls % 1000 == 0:
                    say({"kernel": name, "calls": calls, "other": other})
            torch.cuda.synchronize(device)
            say({"kernel": name, "calls": calls, "other": other, "done": True,
                 "s": round(time.monotonic() - started, 1)})


if __name__ == "__main__":
    main()
