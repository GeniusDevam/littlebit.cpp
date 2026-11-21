import torch
from torch import Tensor, nn
from torch.nn.functional import linear

from .linear import path_tensors


def unpack_signs(packed: Tensor, cols: int) -> Tensor:
    shifts = torch.arange(32, dtype=torch.int32)
    bits = ((packed.unsqueeze(-1) >> shifts) & 1).flatten(1)
    return (1 - 2 * bits[:, :cols].float()).contiguous()


class UnpackedLinear(nn.Module):
    """Float reference and unpack-on-load benchmark baseline."""

    def __init__(
        self, width: int, rank: int, tensors: dict[str, Tensor], dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        self.paths = 2 if "U_R_packed" in tensors else 1
        self.dtype = dtype
        for index in range(self.paths):
            suffix = "_R" if index else ""
            u, v, h, latent, g = path_tensors(tensors, suffix)
            if dtype == torch.float64:
                latent = (
                    tensors[f"u2{suffix}"].double() * tensors[f"v1{suffix}"].double()
                ).flatten()
            self.register_buffer(f"u{index}", unpack_signs(u, rank).to(dtype), persistent=False)
            self.register_buffer(f"v{index}", unpack_signs(v, width).to(dtype), persistent=False)
            self.register_buffer(f"h{index}", h.to(dtype), persistent=False)
            self.register_buffer(f"l{index}", latent.to(dtype), persistent=False)
            self.register_buffer(f"g{index}", g.to(dtype), persistent=False)
        self.register_buffer("bias", tensors.get("bias"), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        inputs = x.to(self.dtype)
        y = self._path(inputs, 0)
        if self.paths == 2:
            y = y + self._path(inputs, 1)
        if self.bias is not None:
            y = y + self.bias
        return y.to(x.dtype)

    def _path(self, x: Tensor, index: int) -> Tensor:
        u = self.get_buffer(f"u{index}")
        v = self.get_buffer(f"v{index}")
        h = self.get_buffer(f"h{index}")
        latent = self.get_buffer(f"l{index}")
        g = self.get_buffer(f"g{index}")
        return linear(linear(x * g, v) * latent, u) * h
