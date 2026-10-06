"""Our GGUF files through numz's own GGUF loader, on the CPU: the load test of DESIGN.md's phase 2.

Run with numz's environment, not uv (it imports numz's code), CUDA hidden:

    cd NUMZ && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \\
        /path/to/models/numz_gguf_check.py NUMZ FILE.gguf...

NUMZ is a checkout of numz's ComfyUI-SeedVR2_VideoUpscaler at 4490bd1 with its environment. For
each file, numz's functions only (src/core/model_loader.py, src/optimization/gguf_*.py):
- _load_gguf_state: every quantized tensor decoded by numz (dequantize_tensor, float32) equal, bit
  for bit, to gguf-py's decoder; every float16 tensor equal to the file's;
- _load_model_weights on the 7B DiT built on the meta device (configs_7b/main.yaml): nothing left
  on meta, the 288 block matrices as numz's GGUFQuantizedLinear;
- the forward numz runs: its float16 decoding, against gguf-py's values rounded once to float16,
  and one layer's output under bfloat16 autocast against F.linear with that weight.
Prints a JSON report; exits 1 if a check fails.
"""

import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
NUMZ = os.path.abspath(sys.argv[1])
sys.path.insert(0, NUMZ)

import gguf  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from src.common.config import create_object, load_config  # noqa: E402
from src.core import model_loader as ml  # noqa: E402
from src.optimization.gguf_dequant import dequantize_tensor  # noqa: E402
from src.optimization.gguf_ops import GGUFQuantizedLinear  # noqa: E402
from src.utils.debug import Debug  # noqa: E402

dbg = Debug(enabled=False)
cfg = load_config(os.path.join(NUMZ, "configs_7b", "main.yaml"))


def check(path: str) -> dict:
    r: dict = {"file": path, "torch": torch.__version__, "cuda_visible": torch.cuda.is_available()}
    want = {}
    for t in gguf.GGUFReader(path).tensors:
        a = np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), np.float32)
        want[t.name] = (
            t.tensor_type.name,
            torch.from_numpy(a.reshape(tuple(int(d) for d in reversed(t.shape)))),
        )
    state = ml._load_gguf_state(path, torch.device("cpu"), dbg)
    types: dict = {}
    for k, v in state.items():
        qt, g = want[k]
        row = types.setdefault(
            qt,
            {
                "tensors": 0,
                "numz_equal_gguf_py": 0,
                "fp16_path_share_equal": 0.0,
                "fp16_path_max_abs_over_max": 0.0,
            },
        )
        row["tensors"] += 1
        if isinstance(v, ml.GGUFTensor):
            a32 = dequantize_tensor(v, torch.float32, torch.float32)
            row["numz_equal_gguf_py"] += bool(torch.equal(a32, g))
            a16 = dequantize_tensor(v, torch.float16, torch.float16)
            row["fp16_path_share_equal"] += float((a16 == g.to(torch.float16)).float().mean())
            row["fp16_path_max_abs_over_max"] = max(
                row["fp16_path_max_abs_over_max"],
                float((a16.float() - g).abs().max() / g.abs().max()),
            )
        else:
            row["numz_equal_gguf_py"] += bool(torch.equal(v.float(), g))
    for row in types.values():
        row["fp16_path_share_equal"] /= row["tensors"]
    r["state"] = types
    with torch.device("meta"):
        model = create_object(cfg.dit.model)
    shapes = set(model.state_dict())
    r["names_equal_model"] = set(state) == shapes
    model = ml._load_model_weights(model, path, torch.device("cpu"), True, "DiT", "", dbg, None)
    r["left_on_meta"] = [n for n, p in model.named_parameters() if p.is_meta] + [
        n for n, b in model.named_buffers() if b.is_meta
    ]
    ql = [(n, m) for n, m in model.named_modules() if isinstance(m, GGUFQuantizedLinear)]
    r["gguf_linears"] = len(ql)
    n, m = ql[0]
    x = torch.randn(
        4, m.in_features, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(0)
    )
    with torch.autocast("cpu", torch.bfloat16):
        y = m(x)
    w = m.dequantize_weight(None, torch.float16).to(torch.bfloat16)
    y_ref = F.linear(x, w, None if m.bias is None else m.bias.to(torch.bfloat16))
    r["forward"] = {
        "layer": n,
        "out_dtype": str(y.dtype),
        "equal_f_linear": bool(torch.equal(y, y_ref)),
    }
    r["pass"] = (
        all(row["numz_equal_gguf_py"] == row["tensors"] for row in types.values())
        and r["names_equal_model"]
        and not r["left_on_meta"]
        and r["gguf_linears"] == 288
        and r["forward"]["equal_f_linear"]
        and not r["cuda_visible"]
    )
    return r


results = [check(p) for p in sys.argv[2:]]
print(json.dumps(results, indent=1))
sys.exit(0 if all(r["pass"] for r in results) else 1)
