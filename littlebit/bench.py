import argparse
import json
import math
import platform
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn.functional import linear
from transformers import AutoModelForCausalLM, AutoTokenizer

from .linear import KERNELS, LBLinear, kernel_index
from .loader import load_pretrained, load_reference
from .reference import UnpackedLinear

SHAPES = {
    "Llama-2-7B": [(4096, 4096), (4096, 11008), (11008, 4096)],
    "Qwen2.5-0.5B": [(896, 896), (896, 128), (896, 4864), (4864, 896)],
}
TARGETS = (1.0, 0.55, 0.3, 0.1)
VARIANTS = ("packed", "unpacked", "dense-fp32", "dense-bf16")


def _schedule(names: tuple[str, ...], repeats: int) -> Iterator[str]:
    for rep in range(repeats):
        shifted = rep % len(names)
        yield from names[shifted:] + names[:shifted]


def _summary(samples: list[float]) -> dict[str, Any]:
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "samples": samples,
    }


def _metadata(threads: int) -> dict[str, Any]:
    data = {}
    cpu_file = Path("/proc/cpuinfo")
    if cpu_file.is_file():
        first = cpu_file.read_text().split("\n\n", 1)[0]
        data = dict(
            tuple(part.strip() for part in line.split(":", 1)) for line in first.splitlines()
        )
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        commit = head.stdout.strip() or None
    except FileNotFoundError:
        commit = None
    return {
        "cpu": data.get("model name", platform.processor()),
        "flags": data.get("flags", "").split(),
        "threads": threads,
        "torch": torch.__version__,
        "commit": commit,
    }


def _packed_random(rows: int, width: int, rng: torch.Generator) -> Tensor:
    value = torch.randint(
        -(2**31), 2**31, (rows, (width + 31) // 32), dtype=torch.int32, generator=rng
    )
    if width % 32:
        value[:, -1] &= (1 << (width % 32)) - 1
    return value


def _example(width: int, out: int, rank: int, rng: torch.Generator) -> dict[str, Tensor]:
    fields = {}
    for suffix in ("", "_R"):
        fields.update(
            {
                f"U{suffix}_packed": _packed_random(out, rank, rng),
                f"V{suffix}_packed": _packed_random(rank, width, rng),
                f"u1{suffix}": torch.ones(1, out),
                f"u2{suffix}": torch.ones(1, rank),
                f"v1{suffix}": torch.full((1, rank), 1 / math.sqrt(width * rank)),
                f"v2{suffix}": torch.ones(1, width),
            }
        )
    return fields


@torch.inference_mode()
def _layers(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    rng = torch.Generator().manual_seed(0)
    names = VARIANTS
    for model, shapes in SHAPES.items():
        for width, out in shapes:
            dense = torch.randn(out, width, generator=rng) / math.sqrt(width)
            dense_bf16 = dense.bfloat16()
            for eff_bit in TARGETS:
                budget = (eff_bit * width * out / 2 - 16 * (width + out)) / (width + out + 16)
                rank = min(width, out, max(8, 8 * math.floor(budget / 8)))
                weights = _example(width, out, rank, rng)
                packed = LBLinear(
                    width, out, rank, weights, kernel=args.kernel, threads=args.threads
                )
                unpacked = UnpackedLinear(width, rank, weights)
                for tokens in args.batch:
                    x = torch.randn(tokens, width, generator=rng)
                    x_bf16 = x.bfloat16()
                    calls: dict[str, Callable[[], Tensor]] = {
                        "packed": lambda: packed(x),
                        "unpacked": lambda: unpacked(x),
                        "dense-fp32": lambda: linear(x, dense),
                        "dense-bf16": lambda: linear(x_bf16, dense_bf16),
                    }
                    samples: dict[str, list[float]] = {name: [] for name in names}
                    for call in calls.values():
                        call()
                    for name in _schedule(names, args.repeats):
                        call = calls[name]
                        start = time.perf_counter()
                        for _ in range(args.iterations):
                            call()
                        samples[name].append((time.perf_counter() - start) * 1000 / args.iterations)
                    rows.append(
                        {
                            "published_model_shape": model,
                            "in": width,
                            "out": out,
                            "rank": rank,
                            "tokens": tokens,
                            "target_bpw": eff_bit,
                            "logical_bpw": (
                                2 * (rank * (width + out + 16) + 16 * (width + out)) / (width * out)
                            ),
                            "kernel": args.kernel,
                            "milliseconds": {
                                name: _summary(values) for name, values in samples.items()
                            },
                        }
                    )
    return rows


@torch.inference_mode()
def _model_worker(params: dict[str, Any]) -> dict[str, float]:
    torch.set_num_threads(params["threads"])
    torch.set_num_interop_threads(1)
    kind = params["variant"]
    if kind == "packed":
        model = load_pretrained(
            params["checkpoint"], kernel=params["kernel"], threads=params["threads"]
        )
    elif kind == "unpacked":
        model = load_reference(params["checkpoint"])
    else:
        dtype = torch.float32 if kind == "dense-fp32" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(params["source"], dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(params["source"])
    inputs = tokenizer(params["prompt"], return_tensors="pt")
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    count = params["tokens"]

    def generate(n: int) -> Tensor:
        return model.generate(
            **inputs, min_new_tokens=n, max_new_tokens=n, do_sample=False, pad_token_id=pad
        )

    generate(2)
    start = time.perf_counter()
    output = generate(count)
    seconds = time.perf_counter() - start
    new_tokens = output.shape[-1] - inputs.input_ids.shape[-1]
    return {
        "tokens_per_second": new_tokens / seconds,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "prompt_tokens": inputs.input_ids.shape[-1],
        "new_tokens": new_tokens,
    }


def _models(args: argparse.Namespace) -> dict[str, Any]:
    workers = {name: [] for name in VARIANTS}
    params = {
        "threads": args.threads,
        "source": args.source,
        "checkpoint": str(args.checkpoint),
        "kernel": args.kernel,
        "tokens": args.tokens,
        "prompt": args.prompt,
    }
    for name in _schedule(VARIANTS, args.repeats):
        call = json.dumps(params | {"variant": name})
        proc = subprocess.run(
            [sys.executable, "-m", "littlebit.bench", "_worker", call],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        workers[name].append(json.loads(proc.stdout))
    return params | {
        "results": {
            name: {
                "tokens_per_second": _summary([v["tokens_per_second"] for v in runs]),
                "peak_rss_mib": _summary([v["peak_rss_mib"] for v in runs]),
                "runs": runs,
            }
            for name, runs in workers.items()
        }
    }


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "_worker":
        print(json.dumps(_model_worker(json.loads(sys.argv[2]))))
        return
    parser = argparse.ArgumentParser(description="Benchmark LittleBit inference on CPU")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--kernel", choices=KERNELS, default="auto")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    commands = parser.add_subparsers(dest="mode", required=True)
    layers = commands.add_parser("layers")
    layers.add_argument("--batch", type=int, nargs="+", default=[1, 16])
    layers.add_argument("--iterations", type=int, default=3)
    models = commands.add_parser("model")
    models.add_argument("source")
    models.add_argument("checkpoint", type=Path)
    models.add_argument("--tokens", type=int, default=16)
    models.add_argument("--prompt", default="The first time I saw the ocean, I noticed")
    args = parser.parse_args()
    kernel_index(args.kernel, args.threads)
    if args.repeats < 5:
        parser.error("--repeats must be at least 5")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.mode == "layers":
        if min(args.batch) < 1 or args.iterations < 1:
            parser.error("--batch and --iterations must be positive")
        results = _layers(args)
    else:
        if args.tokens < 1:
            parser.error("--tokens must be positive")
        results = _models(args)
    data = _metadata(args.threads) | {"mode": args.mode, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
