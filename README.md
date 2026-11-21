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
python -m pip install 'setuptools>=77' wheel ninja
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
rounded to a valid rank. Use `--no-residual` for one path. Use these untrained checkpoints to verify inference correctness.

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

### Measured results

Intel Core i7-8550U in Ubuntu under WSL2, 4 threads, kernel `auto`.
Relevant CPU flags `fma avx f16c bmi1 avx2 bmi2`. The runs used PyTorch `2.10.0+cpu` and
source commit `2eb4c46479ee`. Each cell shows the median (minimum–maximum)
of five interleaved samples.

Generation used a ten-token prompt and sixteen new tokens. The packed and unpacked variants
used the same synthetic weights at a target of 0.1 BPW.

| Model | Variant | New tokens/s | Peak RSS (MiB) |
|---|---|---:|---:|
| SmolLM2-135M | Packed | 24.32 (9.42–25.63) | 507.7 (507.5–507.9) |
| SmolLM2-135M | Unpacked FP32 | 22.68 (18.81–23.06) | 528.8 (528.7–529.2) |
| SmolLM2-135M | Dense FP32 | 10.61 (9.72–11.07) | 1088.5 (1088.4–1088.6) |
| SmolLM2-135M | Dense BF16 | 10.52 (9.85–13.65) | 617.8 (617.3–618.9) |
| Qwen2.5-0.5B | Packed | 9.83 (8.34–10.55) | 1125.4 (1125.3–1125.6) |
| Qwen2.5-0.5B | Unpacked FP32 | 9.37 (7.94–9.76) | 1297.9 (1297.1–1298.2) |
| Qwen2.5-0.5B | Dense FP32 | 4.13 (0.77–5.33) | 2970.6 (2376.6–3077.0) |
| Qwen2.5-0.5B | Dense BF16 | 6.50 (6.13–7.45) | 1386.7 (1386.4–1386.8) |

Example projection latencies in milliseconds. The complete suite covers all projection shapes
and bit targets listed in the design.

| Model shape | In × out | Target BPW | Rank | Tokens | Packed | Unpacked FP32 | Dense FP32 | Dense BF16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-2-7B | 4096 × 4096 | 0.10 | 80 | 1 | 0.667 (0.639–0.769) | 0.862 (0.779–0.947) | 4.136 (4.114–4.183) | 2.135 (2.102–2.399) |
| Llama-2-7B | 4096 × 4096 | 0.10 | 80 | 16 | 1.502 (1.242–1.627) | 2.298 (1.664–3.718) | 16.191 (16.050–16.464) | 9.952 (9.056–10.324) |
| Qwen2.5-0.5B | 896 × 896 | 0.10 | 8 | 1 | 0.061 (0.053–0.089) | 0.068 (0.062–0.073) | 0.223 (0.174–0.260) | 0.235 (0.225–0.238) |
| Qwen2.5-0.5B | 896 × 896 | 0.10 | 8 | 16 | 0.646 (0.635–0.681) | 0.663 (0.634–0.729) | 0.275 (0.265–0.302) | 0.633 (0.606–0.681) |


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
