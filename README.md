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
It uses FP32 arithmetic in packed layers. Pass `dtype=torch.bfloat16` to keep ordinary model
weights in BF16. Phi-3's split fused projection is outside this revision.
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

Intel Core i7-8550U in Ubuntu under WSL2 with AVX2, PyTorch `2.10.0+cpu` and four
configured threads. The baseline is `2439fd5`. Values are medians (minimum–maximum)
over five samples. Generation included a ten-token prompt and sixteen new tokens.
Packed and unpacked used the same synthetic checkpoint at a target of 0.1 BPW.

| Model | Variant | Before tok/s | After tok/s | After peak RSS (MiB) |
|---|---|---:|---:|---:|
| SmolLM2-135M | Packed | 23.09 (10.25–24.20) | 24.41 (19.09–25.19) | 507.7 (507.5–507.8) |
| SmolLM2-135M | Unpacked FP32 | 13.80 (11.42–21.17) | 21.29 (19.92–21.39) | 528.7 (528.4–528.9) |
| SmolLM2-135M | Dense BF16 | 9.45 (7.55–11.97) | 12.07 (11.31–13.22) | 617.5 (617.4–617.6) |
| Qwen2.5-0.5B | Packed | 8.58 (6.17–10.36) | 12.80 (11.97–13.50) | 1125.4 (1122.5–1125.4) |
| Qwen2.5-0.5B | Unpacked FP32 | 7.58 (5.98–9.16) | 8.57 (7.06–9.27) | 1297.7 (1297.6–1298.0) |
| Qwen2.5-0.5B | Dense BF16 | 5.92 (4.19–7.83) | 7.27 (5.65–7.77) | 1386.6 (1378.4–1386.7) |

Example projection latencies in milliseconds. The packed and unpacked variants include
both sign stages per path, plus scaling and call overhead.

| In × out | Target BPW | Rank | Tokens | Packed before | Packed after | Unpacked FP32 after | Dense BF16 after |
|---|---:|---:|---:|---:|---:|---:|---:|
| 896 × 896 | 0.55 | 104 | 1 | 0.671 (0.640–0.685) | 0.088 (0.082–0.098) | 0.721 (0.686–0.735) | 0.194 (0.184–0.213) |
| 896 × 896 | 0.10 | 8 | 16 | 0.697 (0.656–0.751) | 0.164 (0.156–0.217) | 0.720 (0.669–0.778) | 0.657 (0.616–0.802) |
| 4096 × 4096 | 0.10 | 80 | 1 | 0.999 (0.945–1.070) | 0.205 (0.182–0.242) | 0.814 (0.784–0.904) | 2.118 (2.072–2.183) |
| 4096 × 4096 | 0.10 | 80 | 16 | 2.600 (2.137–4.120) | 1.307 (1.107–1.716) | 2.391 (1.748–3.083) | 13.088 (10.226–15.715) |
| 11008 × 4096 | 1.00 | 1472 | 16 | 34.138 (29.234–42.419) | 17.002 (15.601–18.079) | 17.983 (17.225–19.044) | 33.286 (29.279–39.138) |
| 11008 × 4096 | 0.55 | 800 | 16 | 27.871 (20.584–44.778) | 8.609 (8.203–10.014) | 10.672 (10.129–12.061) | 36.775 (33.034–42.060) |
| 4096 × 11008 | 1.00 | 1472 | 16 | 28.492 (24.320–42.736) | 18.983 (14.800–19.297) | 21.365 (19.918–24.432) | 37.008 (33.898–48.726) |

A four-thread empty dispatch took 0.175 ms. For the one-token, rank-104
`V` stage at width 896, automatic dispatch went from 0.197 ms
to 0.010 ms by selecting one thread. AVX2 prefill reuses each sign mask
across a pair of tokens.

### Trained Qwen3-8B checkpoint

[LittleBitLLM/littlebit-qwen3-8b-0.55bpw, revision `6b2c47a`](https://huggingface.co/LittleBitLLM/littlebit-qwen3-8b-0.55bpw/tree/6b2c47aa9707e624e2e737d16d64a66be8090ae9)
is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
The checkpoint remains local and is not redistributed. The run used `load_pretrained`
with `dtype=torch.bfloat16`, retaining the separate embedding and output tables in BF16.

These CPU runs used the laptop above with 3.8 GiB of WSL RAM and 1.0 GiB of swap.
Peak RSS was 1.24 GiB to load, 2.48 GiB during the throughput run and 2.81 GiB
during perplexity. The perplexity values are cumulative after each 2,048-token WikiText-2
window, using the same token IDs and second-half scoring rule as littlebit-gguf r3.
Measured window times ranged from 6.9 to 9.5 minutes.

| Runtime and table format | Window 1 | Window 2 | Window 3 | Window 4 |
|---|---:|---:|---:|---:|
| littlebit.cpp, BF16 | 23.4962 | 25.9472 | 27.8435 | 26.2756 |
| littlebit-gguf r3, F16 | 23.4913 | 25.9303 | 27.8360 | 26.2586 |

Throughput is mean ± sample standard deviation over five repetitions. Prefill starts with
a fresh cache. Generation starts with one input token and produces 32 new tokens greedily.
The GGUF measurements use llama-bench, whose benchmark inputs and timing boundaries differ.
[Littlebit-gguf r3](https://github.com/GeniusDevam/littlebit-gguf/tree/r3)

| Runtime and table format | Prefill 128 (tok/s) | Generation 32 (tok/s) |
|---|---:|---:|
| littlebit.cpp, BF16 | 7.71 ± 0.74 | 2.22 ± 0.16 |
| littlebit-gguf r3, Q4_K / Q6_K | 8.47 ± 1.52 | 5.11 ± 1.60 |
| littlebit-gguf r3, F16 | 5.04 ± 2.46 | 4.80 ± 0.42 |

## How the kernels work

Each LittleBit layer is two stages: a sign matrix of shape `r × in` followed by one of shape `out × r`, with element-wise scaling in between. For a row of signs `s` and an input `x`:

```
s · x  =  Σ x  −  2 · Σ_{j : s_j = −1} x_j
```

Automatic dispatch times the supported kernels and compares one thread with the configured
thread count. It caches the choice per stage shape and samples at most 16 tokens during tuning.
The scalar kernel is available as a reference. The other kernels use the following approaches.

- **sign-flip SIMD**: expand weight bits into lane masks and flip the sign bit of the activations, then accumulate (AVX2 and AVX-512F).
- **lookup tables**: precompute partial sums for small groups of activations and index them with the weight bits, the approach popularised by T-MAC and bitnet.cpp.
- **blocked prefill**: expand a small tile of weights to integer signs and reuse it across several tokens.

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
