from pathlib import Path

import torch
from torch import Tensor, nn

KERNELS = ("auto", "scalar", "avx2", "avx512", "lut", "batched")


def kernel_index(kernel: str, threads: int) -> int:
    index = KERNELS.index(kernel)
    if not 1 <= threads <= 32:
        raise ValueError("threads must be in [1, 32]")
    if not torch.ops.littlebit.kernel_available(index):
        raise ValueError(f"kernel is unavailable on this CPU: {kernel}")
    return index


def _load_library() -> None:
    library = next(Path(__file__).parent.glob("_native*.so"), None)
    if library is None:
        raise ImportError(
            "LittleBit extension is missing. Install the package with a CPU build of PyTorch."
        )
    torch.ops.load_library(str(library))


_load_library()


def path_tensors(
    tensors: dict[str, Tensor], suffix: str
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (
        tensors[f"U{suffix}_packed"],
        tensors[f"V{suffix}_packed"],
        tensors[f"u1{suffix}"].flatten().float(),
        (tensors[f"u2{suffix}"].float() * tensors[f"v1{suffix}"].float()).flatten(),
        tensors[f"v2{suffix}"].flatten().float(),
    )


class LBLinear(nn.Module):
    """One packed LittleBit projection, with float32 compute on CPU."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        tensors: dict[str, Tensor],
        *,
        kernel: str = "auto",
        threads: int = 1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.kernel = kernel_index(kernel, threads)
        self.threads = threads
        paths = list(path_tensors(tensors, ""))
        if "U_R_packed" in tensors:
            paths.extend(path_tensors(tensors, "_R"))
        bias = tensors.get("bias")
        self._layer = torch.classes.littlebit.Layer(
            in_features,
            out_features,
            rank,
            [weight.detach().contiguous() for weight in paths],
            None if bias is None else bias.detach().float().contiguous(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.ops.littlebit.forward(x, self._layer, self.kernel, self.threads)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}"
