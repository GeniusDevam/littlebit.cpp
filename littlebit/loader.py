from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig, PreTrainedModel
from transformers.quantizers import HfQuantizer
from transformers.quantizers.auto import register_quantization_config, register_quantizer
from transformers.utils.quantization_config import QuantizationConfigMixin

from .checkpoint import Projection, read_checkpoint
from .linear import LBLinear, kernel_index
from .reference import UnpackedLinear


@register_quantization_config("littlebit")
class LittleBitConfig(QuantizationConfigMixin):
    def __init__(
        self,
        layers: dict[str, Projection] | None = None,
        kernel: str = "auto",
        threads: int = 1,
        reference_dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        self.quant_method = "littlebit"
        self.layers = layers
        self.kernel = kernel
        self.threads = threads
        self.reference_dtype = reference_dtype

    def to_dict(self) -> dict[str, Any]:
        return {"quant_method": self.quant_method}


class _PackedBuffers(nn.Module):
    def __init__(self, shape: Projection) -> None:
        super().__init__()
        for name, spec in shape.fields.items():
            if not isinstance(spec.dtype, torch.dtype):
                raise ValueError(f"unsupported dtype for {name}: {spec.dtype}")
            self.register_buffer(name, torch.empty(spec.shape, dtype=spec.dtype, device="meta"))


@register_quantizer("littlebit")
class LittleBitQuantizer(HfQuantizer):
    quantization_config: LittleBitConfig

    def is_serializable(self) -> bool:
        return False

    @property
    def is_trainable(self) -> bool:
        return False

    def _process_model_before_weight_loading(self, model: PreTrainedModel, **kwargs: Any) -> None:
        layers = self.quantization_config.layers
        if layers is None:
            raise ValueError("use littlebit.load_pretrained to read the checkpoint shapes")
        output = model.get_output_embeddings()
        linear = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and name != "lm_head" and module is not output
        }
        if linear.keys() != layers.keys():
            missing = sorted(linear.keys() - layers.keys())
            extra = sorted(layers.keys() - linear.keys())
            raise ValueError(
                f"packed projections differ from the model: missing={missing}, extra={extra}"
            )
        for name, shape in layers.items():
            module = linear[name]
            if (
                shape.width != module.in_features
                or shape.out != module.out_features
                or ("bias" in shape.fields) != (module.bias is not None)
            ):
                raise ValueError(f"packed projection {name} does not match its linear layer")
            model.set_submodule(name, _PackedBuffers(shape))

    def _process_model_after_weight_loading(
        self, model: PreTrainedModel, **kwargs: Any
    ) -> PreTrainedModel:
        config = self.quantization_config
        for name, shape in config.layers.items():
            weights = dict(model.get_submodule(name).named_buffers())
            if config.reference_dtype is None:
                layer = LBLinear(
                    shape.width,
                    shape.out,
                    shape.rank,
                    weights,
                    kernel=config.kernel,
                    threads=config.threads,
                )
            else:
                layer = UnpackedLinear(
                    shape.width, shape.rank, weights, dtype=config.reference_dtype
                )
            model.set_submodule(name, layer.eval())
        return model


def _load(
    path: str | Path,
    kernel: str,
    threads: int,
    reference_dtype: torch.dtype | None,
    quantization: dict[str, Any] | None,
    dtype: torch.dtype,
) -> PreTrainedModel:
    kernel_index(kernel, threads)
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("dtype must be float32, float16 or bfloat16")
    path = Path(path).resolve()
    checkpoint = read_checkpoint(path, quantization)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    if config.model_type in {"phi3", "phi3small"}:
        raise ValueError(
            "checkpoints with split Phi-3 QKV projections are outside this format subset"
        )
    config.quantization_config = vars(
        LittleBitConfig(checkpoint.layers, kernel, threads, reference_dtype)
    )
    options = {
        "config": config,
        "dtype": dtype,
        "local_files_only": True,
        "output_loading_info": True,
    }
    if checkpoint.weights is None:
        model, loading = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=False, **options
        )
    else:
        with torch.device("meta"):
            model_class = type(AutoModelForCausalLM.from_config(config, trust_remote_code=False))
        if (path / "generation_config.json").is_file():
            options["generation_config"] = GenerationConfig.from_pretrained(
                path, local_files_only=True
            )
        model, loading = model_class.from_pretrained(None, state_dict=checkpoint.weights, **options)
    if any(loading.values()):
        raise ValueError(f"checkpoint did not load cleanly: {loading}")
    return model.eval().requires_grad_(False)


def load_pretrained(
    path: str | Path,
    *,
    kernel: str = "auto",
    threads: int = 1,
    quantization: dict[str, Any] | None = None,
    dtype: torch.dtype = torch.float32,
) -> PreTrainedModel:
    """Load a local LittleBit CPU model with ordinary weights in dtype."""
    return _load(path, kernel, threads, None, quantization, dtype)


def load_reference(path: str | Path, *, dtype: torch.dtype = torch.float32) -> PreTrainedModel:
    return _load(path, "auto", 1, dtype, None, torch.float32)
