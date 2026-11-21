import argparse
import json
import math
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


def choose_rank(width: int, out: int, eff_bit: float, residual: bool) -> int:
    paths = 2 if residual else 1
    per_path = eff_bit * width * out / paths
    limit = min(width, out)
    budget = (per_path - 16 * (width + out)) / (width + out + 16)
    return min(limit, max(8, int(min(budget, limit) // 8) * 8))


def pack_signs(factor: Tensor) -> Tensor:
    padded = torch.nn.functional.pad(factor < 0, (0, -factor.shape[1] % 32))
    shifts = torch.arange(32, dtype=torch.int32)
    words = padded.reshape(factor.shape[0], -1, 32).int() << shifts
    return words.sum(dim=-1, dtype=torch.int32)


def magnitude_scales(factor: Tensor) -> tuple[Tensor, Tensor]:
    magnitude = factor.abs()
    # Power iteration finds the leading rank-one fit without a full second SVD.
    latent = torch.ones(factor.shape[1], dtype=factor.dtype)
    for _ in range(12):
        outer = magnitude @ latent
        latent = magnitude.T @ outer
        latent = latent / latent.norm().clamp_min(torch.finfo(factor.dtype).tiny)
    outer = magnitude @ latent
    root = outer.norm().sqrt().clamp_min(torch.finfo(factor.dtype).tiny)
    return (outer / root).unsqueeze(0).bfloat16(), (latent * root).unsqueeze(0).bfloat16()


def dual_svid(
    weight: Tensor, rank: int, suffix: str, *, reconstruct: bool
) -> tuple[dict[str, Tensor], Tensor | None]:
    u, singular, v = torch.svd_lowrank(weight, q=min(rank + 8, *weight.shape), niter=4)
    root = singular[:rank].sqrt()
    u = u[:, :rank] * root
    v = v[:, :rank] * root
    h, u2 = magnitude_scales(u)
    g, v1 = magnitude_scales(v)
    tensors = {
        f"U{suffix}_packed": pack_signs(u),
        f"V{suffix}_packed": pack_signs(v.T),
        f"U{suffix}_shape": torch.tensor((weight.shape[0], rank), dtype=torch.int64),
        f"V{suffix}_shape": torch.tensor((rank, weight.shape[1]), dtype=torch.int64),
        f"u1{suffix}": h,
        f"u2{suffix}": u2,
        f"v1{suffix}": v1,
        f"v2{suffix}": g,
    }
    estimate = None
    if reconstruct:
        u_sign = torch.where(u < 0, -1.0, 1.0)
        v_sign = torch.where(v < 0, -1.0, 1.0)
        estimate = (u_sign * (u2.float() * v1.float())) @ v_sign.T
        estimate.mul_(h.float().T).mul_(g.float())
    return tensors, estimate


@torch.inference_mode()
def synthetic_state(model: PreTrainedModel, eff_bit: float, residual: bool) -> dict[str, Tensor]:
    if not math.isfinite(eff_bit) or eff_bit <= 0:
        raise ValueError("eff_bit must be finite and positive")
    if model.config.model_type in {"phi3", "phi3small"}:
        raise ValueError("Phi-3 fused QKV is outside the supported checkpoint format")
    state = dict(model.state_dict())
    head = model.get_output_embeddings()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name == "lm_head" or module is head:
            continue
        rank = choose_rank(module.in_features, module.out_features, eff_bit, residual)
        weight = state.pop(f"{name}.weight").float()
        fields, estimate = dual_svid(weight, rank, "", reconstruct=residual)
        state.update({f"{name}.{key}": value for key, value in fields.items()})
        if estimate is not None:
            fields, _ = dual_svid(weight - estimate, rank, "_R", reconstruct=False)
            state.update({f"{name}.{key}": value for key, value in fields.items()})

    if head is not None and head.weight is model.get_input_embeddings().weight:
        for name, module in model.named_modules():
            if module is head:
                state.pop(f"{name}.weight", None)
    for key, value in state.items():
        if value.is_floating_point() and value.element_size() > 2:
            value = value.bfloat16()
        state[key] = value.contiguous()
    return state


def write_checkpoint(
    model: PreTrainedModel,
    output: Path,
    eff_bit: float,
    residual: bool,
    seed: int = 0,
    tokenizer_source: str | None = None,
) -> None:
    if output.exists():
        raise FileExistsError(output)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        tensors = synthetic_state(model, eff_bit, residual)
    metadata = {
        "quant_func": "sign",
        "eff_bit": eff_bit,
        "split_dim": None,
        "residual": residual,
        "kv_factor": 1.0,
        "min_split_dim": 8,
        "quant_mod": "littlebit",
        "use_itq": False,
        "itq_n_iter": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".littlebit-") as stage:
        checkpoint = Path(stage) / "checkpoint"
        checkpoint.mkdir()
        save_file(tensors, checkpoint / "model.safetensors", metadata={"format": "pt"})
        config = model.config.to_dict() | metadata | {"use_cache": True}
        for name, data in (
            ("config.json", config),
            ("littlebit_config.json", metadata),
        ):
            (checkpoint / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        if model.generation_config is not None:
            model.generation_config.save_pretrained(checkpoint)
        if tokenizer_source is not None:
            AutoTokenizer.from_pretrained(tokenizer_source).save_pretrained(checkpoint)
        checkpoint.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a synthetic Dual-SVID checkpoint")
    parser.add_argument("source", help="Hugging Face model ID or local directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--eff-bit", type=float, required=True)
    parser.add_argument("--residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not math.isfinite(args.eff_bit) or args.eff_bit <= 0 or args.threads < 1:
        parser.error("--eff-bit and --threads must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    torch.set_num_threads(args.threads)
    model = AutoModelForCausalLM.from_pretrained(args.source, dtype="auto")
    write_checkpoint(model, args.output, args.eff_bit, args.residual, args.seed, args.source)


if __name__ == "__main__":
    main()
