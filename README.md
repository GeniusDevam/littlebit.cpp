# littlebit.cpp

Fast CPU kernels and a small runtime for **sub-1-bit LittleBit and LittleBit-2 language models**, the extreme compression method from Samsung Research (NeurIPS 2025, ICML 2026) that stores LLM weights at 0.1 to 1 bit per weight.

> **This is a research project.** It is independent and experimental, and it is not affiliated with or endorsed by Samsung. Expect breaking changes, rough edges and numbers that move as the work goes on.

## Why this exists

LittleBit replaces every linear layer `W` of a transformer with two small sign matrices and three scale vectors:

```
W  ≈  diag(h) · U± · diag(l) · V±ᵀ · diag(g)        (plus an optional residual path of the same shape)
```

At 0.1 bits per weight that is roughly 31x smaller than FP16. The catch is that the reference code unpacks those bits back into full matrices when it loads a model and then runs ordinary PyTorch matmuls. The file on disk is tiny; the model in memory and the speed are not.

littlebit.cpp keeps the weights packed and computes on the bits directly:

```
y = h ⊙ U±( l ⊙ V±( g ⊙ x ) )
```

Multiplying by a ±1 matrix needs only additions and subtractions, and the two matrices are low-rank, so each layer does a small fraction of the work of a dense layer.

## Status

Early. The checkpoint format is documented and the kernel design is written up. Code lands next. This README will only ever quote numbers that the benchmark scripts in this repo actually produced.

| Component | Status |
|---|---|
| Checkpoint format notes (LittleBit and LittleBit-2) | done, see [docs/FORMAT.md](docs/FORMAT.md) |
| Kernel and runtime design | done, see [docs/DESIGN.md](docs/DESIGN.md) |
| Portable C kernels (single token and batched) | planned, first milestone |
| x86 AVX2 / AVX-512 kernels | planned, first milestone |
| PyTorch layer and Hugging Face loader | planned, first milestone |
| Benchmarks vs dense FP32/BF16 and vs unpack-then-matmul | planned, first milestone |
| ARM NEON and int8 dot-product kernels | planned |
| Standalone C++ runtime (no Python) | planned |
| CUDA / Metal | later |

## Goals

- Load checkpoints written by the official LittleBit training code as they are. LittleBit and LittleBit-2 share one inference format, so both work with the same kernels.
- Keep weights packed in memory and compute on the bits.
- Correctness first: every kernel is tested against a plain floating-point reference.
- Honest benchmarks: speed, memory and output agreement, with the CPU model and flags recorded.

## How the kernels work, briefly

Each LittleBit layer is two stages: a sign matrix of shape `r × in` followed by one of shape `out × r`, with element-wise scaling in between. For a row of signs `s` and an input `x`:

```
s · x  =  Σ x  −  2 · Σ_{j : s_j = −1} x_j
```

So a dot product is a sum plus a masked sum. Two strategies are implemented and picked per shape:

- **sign-flip SIMD**: expand weight bits into lane masks and flip the sign bit of the activations, then accumulate (AVX2, AVX-512, NEON).
- **lookup tables**: precompute partial sums for small groups of activations and index them with the weight bits, the approach popularised by T-MAC and bitnet.cpp.

Details, layouts and the test plan are in [docs/DESIGN.md](docs/DESIGN.md).

## Related work

- LittleBit: [arXiv:2506.13771](https://arxiv.org/abs/2506.13771). LittleBit-2: [arXiv:2603.00042](https://arxiv.org/abs/2603.00042).
- Official training code: [SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit) (CC BY-NC 4.0). No code from it is used or redistributed here; this project is a separate implementation based on the papers and on the format notes in `docs/FORMAT.md`.
- Other sub-1-bit methods: NanoQuant, DBF, BTC-LLM. Runtimes for other low-bit formats: llama.cpp, bitnet.cpp, T-MAC.
- Sister project: [littlebit-gguf](https://github.com/GeniusDevam/littlebit-gguf) runs LittleBit models inside llama.cpp using its existing 1-bit tensor type.

## License

MIT, see [LICENSE](LICENSE). Model weights keep their own licenses.

## Citation

If this is useful in your work, please cite the original papers:

```bibtex
@inproceedings{lee2025littlebit,
  title     = {LittleBit: Ultra Low-Bit Quantization via Latent Factorization},
  author    = {Lee, Banseok and Kim, Dongkyu and You, Youngcheon and Kim, Youngmin},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}

@inproceedings{lee2026littlebit2,
  title     = {LittleBit-2: Maximizing the Spectral Energy Gain in Sub-1-Bit LLMs via Latent Geometry Alignment},
  author    = {Lee, Banseok and Kim, Youngmin},
  booktitle = {International Conference on Machine Learning},
  year      = {2026}
}
```
