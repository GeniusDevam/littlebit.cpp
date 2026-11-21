import argparse
import json
from pathlib import Path

from .checkpoint import read_checkpoint


def info() -> None:
    parser = argparse.ArgumentParser(description="Inspect a LittleBit checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    checkpoint = read_checkpoint(args.checkpoint)
    rows = []
    for name, layer in checkpoint.layers.items():
        paths = 2 if layer.residual else 1
        logical = paths * (
            layer.rank * (layer.width + layer.out + 16) + 16 * (layer.width + layer.out)
        )
        packed = paths * (
            32 * (layer.out * ((layer.rank + 31) // 32) + layer.rank * ((layer.width + 31) // 32))
            + 16 * (layer.width + layer.out + 2 * layer.rank)
        )
        rows.append(
            {
                "name": name,
                "in": layer.width,
                "out": layer.out,
                "rank": layer.rank,
                "paths": paths,
                "logical_bpw": logical / (layer.width * layer.out),
                "padded_bpw": packed / (layer.width * layer.out),
            }
        )
    data = {
        "weight_file_bytes": sum(file.stat().st_size for file in checkpoint.files),
        "quantization": checkpoint.metadata,
        "layers": rows,
    }
    if args.json:
        print(json.dumps(data, indent=2))
        return
    print(f"{len(rows)} projections, {data['weight_file_bytes']:,} bytes of weight files")
    print("layer | in x out | rank | paths | logical bpw | padded bpw")
    for row in rows:
        print(
            f"{row['name']} | {row['in']} x {row['out']} | {row['rank']} | {row['paths']} | "
            f"{row['logical_bpw']:.4f} | {row['padded_bpw']:.4f}"
        )
