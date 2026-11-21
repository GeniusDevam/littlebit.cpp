# littlebit.cpp

Fast CPU kernels and a small runtime for **sub-1-bit LittleBit and LittleBit-2 language models**, the extreme compression method from Samsung Research (NeurIPS 2025, ICML 2026) that stores LLM weights at 0.1 to 1 bit per weight.

> **This is a research project.** It is independent and experimental, and it is not affiliated with or endorsed by Samsung. Expect breaking changes, rough edges and numbers that move as the work goes on.

## Why this exists

LittleBit replaces every linear layer `W` of a transformer with two small sign matrices and three scale vectors:

```
W  ≈  diag(h) · U± · diag(l) · V±ᵀ · diag(g)        (plus an optional residual path of the same shape)
```

The target bit rate applies to the compressed projections. Row padding, stored scales, embeddings and the output head also contribute to checkpoint size. The reference runtime unpacks the sign matrices for PyTorch matmuls.

littlebit.cpp keeps the weights packed and computes on the bits directly:

```
y = h ⊙ U±( l ⊙ V±( g ⊙ x ) )
```

Multiplying by a ±1 matrix needs only additions and subtractions, and the two matrices are low-rank, so each layer does a small fraction of the work of a dense layer.

## Status

| Component | Status |
|---|---|
| C11 layer API, scalar reference, lookup table, FP32 prefill | Implemented |
| x86 AVX2 / AVX-512F | Implemented with runtime dispatch |
| PyTorch layer and Hugging Face causal language model loader | CPU inference, including `generate()` |
| Synthetic checkpoints, correctness tests, benchmark commands | Implemented |
| ARM NEON, int8 activations, MSVC | Planned |
| Standalone C++ runtime, CUDA / Metal | Planned |

## Install

Build the C library and run its tests on Linux or Ubuntu under WSL:

```sh
cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu --parallel 2
ctest --test-dir build/cpu --output-on-failure
```

For Python, install a CPU build of PyTorch into a virtual environment first. The extension uses the
PyTorch headers and libraries from that environment. CMake and a C/C++ compiler are required.

```sh
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'setuptools>=69' wheel ninja
MAX_JOBS=1 python -m pip install --no-build-isolation -e '.[test]'
python -m pytest
```

## Use

```python
from littlebit import load_pretrained
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("checkpoints/sample")
model = load_pretrained("checkpoints/sample", threads=4)
inputs = tokenizer("The sky is blue because", return_tensors="pt")
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=8)[0]))
```

The loader supports local safetensors, sharded safetensors and tensor-only `model_state.pt` checkpoints.
It uses FP32 arithmetic in packed layers. Phi-3's split fused projection is outside this revision.
`LBLinear` can also be created directly from the packed arrays and scales.
The C ABI in [include/littlebit.h](include/littlebit.h) copies the input weights and accepts token-major
FP32 activations. Use `lb_layer_forward_ex` for per-call kernel and thread settings.

Create a test checkpoint with the original paper's Dual-SVID initialization. The target bit rate is
rounded to a valid rank. Use `--no-residual` for one path. These checkpoints are for runtime
verification. Their model quality has not been trained or evaluated.

```sh
python tools/make_synthetic.py HuggingFaceTB/SmolLM2-135M checkpoints/sample --eff-bit 0.1
littlebit-info checkpoints/sample
python -m pytest -o addopts='' -m slow -s
```

## Benchmarks

The layer suite uses Llama-2-7B and Qwen2.5-0.5B projection shapes. The model suite compares
packed and unpacked execution of the same synthetic checkpoint with the original dense model
in FP32 and BF16. Generation includes the prompt pass and reports the rate of newly generated
tokens. Model peak RSS includes loading and inference in a fresh process. Samples alternate
implementations and are saved to ignored JSON files.

```sh
littlebit-bench --threads 4 --output bench-results/layers.json layers
littlebit-bench --threads 4 --output bench-results/smollm2.json model HuggingFaceTB/SmolLM2-135M checkpoints/sample
```

Use `--kernel scalar`, `--kernel avx2`, `--kernel avx512`, `--kernel lut` or `--kernel batched`
to force the packed kernel. The loader accepts the same names via `kernel=`.

## How the kernels work

Each LittleBit layer is two stages: a sign matrix of shape `r × in` followed by one of shape `out × r`, with element-wise scaling in between. For a row of signs `s` and an input `x`:

```
s · x  =  Σ x  −  2 · Σ_{j : s_j = −1} x_j
```

So a dot product is a sum plus a masked sum. Two strategies are implemented and picked per shape:

- **sign-flip SIMD**: expand weight bits into lane masks and flip the sign bit of the activations, then accumulate (AVX2 and AVX-512F).
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
