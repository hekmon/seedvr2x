# 23. Help text and log messages that don't match the code

| | |
|---|---|
| Severity | UX-doc |
| Status | from code |
| Affected options | `--output`, `--output_format`, `--batch_size`, `--cache_dit`, `--cache_vae`, `--prepend_frames`, `--temporal_overlap`, `--dit_offload_device`, `--tensor_offload_device`, `--blocks_to_swap`, `--debug` |
| Version | SeedVR2 `4490bd1` (v2.5.24) |

## Summary

A set of help strings and log lines in `inference_cli.py` describe behaviour the code doesn't
have. Each is small; together they send users the wrong way. Items that are real bugs have their
own file and are only cross-referenced here.

## The list

| Where | Text | What the code does | Fix |
|---|---|---|---|
| `--output` help, `inference_cli.py:1350-1351` | "default: auto-generated in 'output/' directory" | Next to the input: `<stem>_upscaled.mp4`, `<stem>_upscaled/` or `<stem>_upscaled.png`; for a directory input, a sibling `<dir>_upscaled/` (`generate_output_path`, `inference_cli.py:376-421`). The comment at `inference_cli.py:1638` ("handles None gracefully with 'outputs' default") is wrong too | "default: next to the input, with an `_upscaled` suffix" |
| `--output_format`, `inference_cli.py:1352` | `choices=["mp4", "png", None]`, shown as `{mp4,png,None}` | `type=str`: the string `None` isn't the value `None`, so `--output_format None` is rejected | `choices=["mp4", "png"]` (the default `None` still means auto) |
| `--batch_size` help, `inference_cli.py:1375` | "must follow 4n+1: 1, 5, 9, 13, 17, 21,..." | Not validated: any value is accepted and each batch is padded to the next 4n+1 with mirrored frames, dropped after decode (batch 8 computes 9 frames per 8) | "4n+1 recommended; other values are padded to the next 4n+1 (wasted compute)" |
| `--cache_dit` / `--cache_vae` help, `inference_cli.py:1471-1476` | "Requires --dit_offload_device" / "--vae_offload_device"; "Works with single-GPU directory processing or multi-GPU streaming" | The offload device defaults to `cpu` when caching (`_parse_offload_device`, `inference_cli.py:252-254`; info line at `1542-1556`); caching also works for single-GPU streaming (`--chunk_size`, `inference_cli.py:1670-1671`); a single file without streaming ignores it | "Keeps the model between files of a directory or `--chunk_size` chunks; the offload device defaults to cpu" |
| `--prepend_frames` help, `inference_cli.py:1391` | "(auto-removed)" | Only removed with several GPUs | Fix the code: [05](05-prepend-frames-not-removed.md) |
| `--temporal_overlap` help, `inference_cli.py:1393` | "for smooth blending" | 1, 2 and 4 don't blend; between `--chunk_size` chunks nothing is blended | Fix the code: [06](06-temporal-overlap-blend-weights.md), [07](07-chunk-overlap-not-blended.md) |
| `--dit_offload_device` help, `inference_cli.py:1413-1414` | "Frees VRAM between phases" | Without caching, the DiT is deleted after Phase 2 anyway and is never on the GPU during the VAE phases: on its own the option saves nothing and costs +0.7 s ([vram.md](../docs/vram.md#offload-devices)); it matters for `--cache_dit` and as the BlockSwap prerequisite | "Where the DiT is parked when cached; required for BlockSwap" |
| `--tensor_offload_device` help, `inference_cli.py:1417-1418` | "'none' (keep on GPU)" | Latents stay on the GPU, but decoded frames still go to the CPU (`decode_all_batches` falls back to `cpu`, `src/core/generation_phases.py:868-873`) | "'none': keep latents on the GPU" |
| `--blocks_to_swap` / `--swap_io_components` help, `inference_cli.py:1422-1427` | "Requires --dit_offload_device" | True, but the `ValueError` is raised in `configure_runner` (`src/optimization/blockswap.py:112-115`, via `model_configuration.py:799`), after the input has been read; and `--swap_io_components` without swapping every block leaks the DiT ([12](12-swap-io-dit-leak.md)) | Validate in `main()` before reading the input; mention the leak until it is fixed |
| `--10bit` help, `inference_cli.py:1356-1358` | "Save 10-bit video ... (reduces banding)" | 8-bit frames encoded as 10-bit; ignored with `opencv` | [19](19-10bit-output-is-8-bit.md) |
| `--compile_dynamo_cache_size_limit` help, `inference_cli.py:1464-1465` | "Max cached compiled versions per function" | No effect | [20](20-dynamo-cache-size-limit-no-effect.md) |
| `--attention_mode` help, `inference_cli.py:1450` | "'sageattn_3' (Blackwell GPUs)" | Runs SageAttention 2's Triton varlen kernel on every DiT call | [18](18-attention-modes-misleading.md) |
| `--compile_vae` help, `inference_cli.py:1454` | "15-25% speedup" | True for time (−16–19% measured), silent on the ≈ 2× VAE memory | [14](14-compile-vae-doubles-memory.md) |
| `--compile_mode` help, `inference_cli.py:1458-1459` | offers `reduce-overhead` and `max-autotune` | Both crash under the CLI's default allocator | [02](02-cuda-graph-compile-modes-crash.md) |
| `--debug` log, `inference_cli.py:1566` | "Using device index 0 inside script (mapped to selected GPU)" | For a single `--cuda_device N`, the script uses `cuda:N` and the mask has no effect | [21](21-cuda-device-nonzero-mask.md) |
| BlockSwap log, `src/optimization/blockswap.py:445-448` | "Total VRAM saved: 5.10MB" with GGUF models | Counts parameters only; GGUF weights are buffers. The measured peak drops by 4.2 GiB for 36 Q4_K_M blocks ([vram.md](../docs/vram.md#blockswap)) | Count buffers too (`get_module_memory_mb`, `blockswap.py:142`) |
| BlockSwap summary, `src/core/generation_phases.py:764-793` | "BlockSwap overhead: N ms", "Block swaps: avg … ms" | Time the whole wrapped forward, compute included: not the swap cost | Rename to "swapped-block forward time", or time the `.to()` calls only |
| `release_model_memory` log, `src/optimization/memory_manager.py:576-577` | "Released memory from N params and M buffers" | Frees nothing | [13](13-release-memory-frees-nothing.md) |

## Impact

Users set options that do nothing (`--compile_dynamo_cache_size_limit`, `--10bit` with OpenCV),
avoid ones that are fine (any `--batch_size`), add needless options (`--dit_offload_device` alone),
look for output in a directory that doesn't exist, and read memory logs that are wrong.

## Possible fix

Update the help strings as in the table (one commit, no behaviour change), and fix the log lines
listed. The behaviour bugs are tracked in their own files.

Test: `python inference_cli.py --help` review; `--output_format None` no longer listed.

## References

- [cli-flags.md, Options](../docs/cli-flags.md#options)
- [cli-flags.md, Ignored and overridden combinations](../docs/cli-flags.md#ignored-and-overridden-combinations)
