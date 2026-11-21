import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import Tensor

_DTYPES = {
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}
_INFORMATION = ("_eff_bit_target", "_split_dim_final", "_eff_bit_actual")


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype | str


@dataclass(frozen=True)
class Projection:
    width: int
    out: int
    rank: int
    residual: bool
    fields: dict[str, TensorSpec]


@dataclass(frozen=True)
class Checkpoint:
    layers: dict[str, Projection]
    metadata: dict[str, Any]
    weights: dict[str, Tensor] | None
    files: list[Path]


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _catalog(
    path: Path,
) -> tuple[dict[str, TensorSpec], dict[str, Tensor], dict[str, Tensor] | None, list[Path]]:
    index = path / "model.safetensors.index.json"
    single = path / "model.safetensors"
    legacy = path / "model_state.pt"
    if single.is_file():
        files = [single]
    elif index.is_file():
        split = read_json(index).get("weight_map")
        if not isinstance(split, dict) or not split:
            raise ValueError(f"invalid weight map in {index}")
        names = set(split.values())
        if any(not isinstance(name, str) or Path(name).name != name for name in names):
            raise ValueError("checkpoint shard names must be local file names")
        files = [path / name for name in sorted(names)]
    elif legacy.is_file():
        try:
            weights = torch.load(legacy, map_location="cpu", weights_only=True, mmap=True)
        except RuntimeError as error:
            if "mmap can only be used" not in str(error):
                raise
            weights = torch.load(legacy, map_location="cpu", weights_only=True)
        if not isinstance(weights, dict) or not all(
            isinstance(key, str) and isinstance(value, Tensor) for key, value in weights.items()
        ):
            raise ValueError("model_state.pt must contain a tensor state dictionary")
        specs = {k: TensorSpec(tuple(v.shape), v.dtype) for k, v in weights.items()}
        shapes = {k: v for k, v in weights.items() if k.endswith("_shape")}
        return specs, shapes, weights, [legacy]
    else:
        raise FileNotFoundError(f"no LittleBit weight files in {path}")

    specs = {}
    shapes = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as shard:
            for key in shard.keys():
                if key in specs or (len(files) > 1 and split.get(key) != file.name):
                    raise ValueError(f"incorrect shard for {key}")
                header = shard.get_slice(key)
                dtype = header.get_dtype()
                specs[key] = TensorSpec(tuple(header.get_shape()), _DTYPES.get(dtype, dtype))
                if key.endswith("_shape") and specs[key] == TensorSpec((2,), torch.int64):
                    shapes[key] = shard.get_tensor(key)
    if not single.is_file() and specs.keys() != split.keys():
        raise ValueError("checkpoint index does not match its shards")
    return specs, shapes, None, files


def _read_projection(
    name: str, specs: dict[str, TensorSpec], shapes: dict[str, Tensor]
) -> Projection:
    def shape(key: str) -> tuple[int, int]:
        tensor = shapes.get(f"{name}.{key}")
        if tensor is None or tensor.dtype != torch.int64 or tuple(tensor.shape) != (2,):
            raise ValueError(f"invalid or missing shape tensor: {name}.{key}")
        a, b = tensor.tolist()
        if not (0 < a <= 2**31 - 1 and 0 < b <= 2**31 - 1):
            raise ValueError(f"invalid dimensions in {name}.{key}")
        return a, b

    out, rank = shape("U_shape")
    v_rank, width = shape("V_shape")
    if v_rank != rank:
        raise ValueError(f"inconsistent U and V ranks in {name}")
    residual = any(
        f"{name}.{stem}_R{tail}" in specs
        for stem, tail in (
            ("U", "_packed"),
            ("V", "_packed"),
            ("U", "_shape"),
            ("V", "_shape"),
            ("u1", ""),
            ("u2", ""),
            ("v1", ""),
            ("v2", ""),
        )
    )
    if residual and (shape("U_R_shape") != (out, rank) or shape("V_R_shape") != (rank, width)):
        raise ValueError(f"inconsistent residual dimensions in {name}")

    fields = {}
    for suffix in ("", "_R") if residual else ("",):
        fields.update(
            {
                f"U{suffix}_packed": TensorSpec((out, (rank + 31) // 32), torch.int32),
                f"V{suffix}_packed": TensorSpec((rank, (width + 31) // 32), torch.int32),
                f"U{suffix}_shape": TensorSpec((2,), torch.int64),
                f"V{suffix}_shape": TensorSpec((2,), torch.int64),
                f"u1{suffix}": TensorSpec((1, out), torch.bfloat16),
                f"u2{suffix}": TensorSpec((1, rank), torch.bfloat16),
                f"v1{suffix}": TensorSpec((1, rank), torch.bfloat16),
                f"v2{suffix}": TensorSpec((1, width), torch.bfloat16),
            }
        )
    for key, expected in fields.items():
        if specs.get(f"{name}.{key}") != expected:
            raise ValueError(f"invalid or missing tensor: {name}.{key}, expected {expected}")
    bias = specs.get(f"{name}.bias")
    if bias is not None:
        if (
            bias.shape != (out,)
            or not isinstance(bias.dtype, torch.dtype)
            or not bias.dtype.is_floating_point
        ):
            raise ValueError(f"invalid bias in {name}")
        fields["bias"] = bias
    for key in _INFORMATION:
        value = specs.get(f"{name}.{key}")
        if value is not None:
            fields[key] = value
    return Projection(width, out, rank, residual, fields)


def read_checkpoint(path: Path, overrides: dict[str, Any] | None = None) -> Checkpoint:
    config = read_json(path / "config.json")
    metadata = {
        key: config[key]
        for key in (
            "quant_func",
            "eff_bit",
            "split_dim",
            "residual",
            "kv_factor",
            "min_split_dim",
            "quant_mod",
            "use_itq",
            "itq_n_iter",
        )
        if key in config
    }
    config_path = path / "littlebit_config.json"
    if config_path.is_file():
        metadata.update(read_json(config_path))
    metadata.update(overrides or {})
    specs, shapes, weights, files = _catalog(path)
    names = {
        key.removesuffix(suffix)
        for key in specs
        for suffix in (".U_packed", ".V_packed", ".U_shape", ".V_shape")
        if key.endswith(suffix)
    }
    if not names:
        raise ValueError("checkpoint has no packed LittleBit projections")
    layers = {name: _read_projection(name, specs, shapes) for name in sorted(names)}
    if "residual" in metadata and any(
        layer.residual != metadata["residual"] for layer in layers.values()
    ):
        raise ValueError("residual tensors disagree with checkpoint configuration")
    return Checkpoint(layers, metadata, weights, files)
