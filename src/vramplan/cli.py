"""Command-line interface for vramplan."""

from __future__ import annotations

import argparse

from . import __version__
from .estimate import estimate, fits, max_context
from .specs import lookup


def _fmt(gib: float) -> str:
    return f"{gib:.2f} GiB"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="vramplan",
        description="Estimate GPU VRAM for LLM inference (weights + KV cache + activations).",
    )
    p.add_argument("model", help="model name, e.g. Qwen2.5-7B or Llama-3.1-8B")
    p.add_argument("--context", type=int, default=8192, help="context length in tokens (default 8192)")
    p.add_argument("--batch", type=int, default=1, help="batch size (default 1)")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16", "fp8", "int8", "int4"],
                   help="weight dtype / quantization (default bf16)")
    p.add_argument("--kv-dtype", default=None, help="KV cache dtype (default: same as --dtype)")
    p.add_argument("--gpu", type=float, default=None, help="GPU memory in GiB; if set, report fit + max context")
    p.add_argument("--version", action="version", version=f"vramplan {__version__}")
    args = p.parse_args(argv)

    try:
        spec = lookup(args.model)
    except KeyError as e:
        p.error(str(e))

    b = estimate(spec, args.context, args.batch, weight_dtype=args.dtype, kv_dtype=args.kv_dtype)

    print(f"model       : {spec.name}  ({spec.num_params/1e9:.2f}B params)")
    print(f"context     : {args.context} tokens, batch {args.batch}, dtype {args.dtype}")
    print("-" * 46)
    print(f"  weights     : {_fmt(b.weights)}")
    print(f"  KV cache    : {_fmt(b.kv_cache)}")
    print(f"  activations : {_fmt(b.activations)}")
    print(f"  overhead    : {_fmt(b.overhead)}")
    print(f"  TOTAL       : {_fmt(b.total)}")

    if args.gpu:
        ok = fits(b, args.gpu)
        print("-" * 46)
        print(f"fits in {args.gpu:.0f} GiB?  {'YES' if ok else 'NO'}")
        if not ok:
            print(f"  shortfall   : {_fmt(b.total - args.gpu)}")
        mc = max_context(spec, args.gpu, args.batch, weight_dtype=args.dtype, kv_dtype=args.kv_dtype)
        print(f"max context (batch {args.batch}) on {args.gpu:.0f} GiB : ~{mc:,} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
