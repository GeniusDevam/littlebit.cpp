# Design

## Scope

littlebit.cpp runs LittleBit / LittleBit-2 checkpoints (format in [FORMAT.md](FORMAT.md)) with the weights kept packed. It does not train or compress models; the official code does that.

First milestone: CPU kernels, a C API, a PyTorch layer and a Hugging Face loader, tests and benchmarks. Later: ARM NEON, int8 dot-product kernels, a standalone C++ runtime, GPU backends.

## The layer

Per path, with `x` of shape `[n, in]`:

```
y = h ⊙ ( U± · ( l ⊙ ( V± · ( g ⊙ x ) ) ) )      l = v1 ⊙ u2
```

Cost per token: `r·in + out·r` sign multiplies (adds) and `in + r + out` scalings, versus `in·out` multiply-adds for a dense layer. A residual path doubles the sign work. The two stages have very different shapes: `V±` is short and wide (`r` rows of `in` columns), `U±` is tall and narrow (`out` rows of `r` columns), with `r` anywhere from 8 to about 1500. Kernels are chosen per stage.

## Weight layout

The loader reads the packed int32 arrays exactly as stored and validates them against the shape tensors. It then repacks once into a kernel-friendly layout (for example rows interleaved in groups of 4 or 8 so one pass over `x` serves several rows, and LUT-ordered nibbles for the lookup kernel). The original arrays can still be produced from the repacked form, which the tests check.

Scales are converted from bf16 to fp32 at load. `l` is precomputed. Where it is exact to do so, scales may be folded (for example `l` into the first stage's output), but the default path follows the reference order so results are easy to compare.

## Kernels

1. **Reference (scalar C)**: unpacks bits and accumulates in double. Ground truth for tests, never used for speed.
2. **Sign-flip SIMD (AVX2, AVX-512, later NEON)**: for each byte of weight bits, expand to lane masks (a 256-entry mask table or `pdep`/shuffle tricks), XOR the float sign bit of `x`, accumulate. Process 4 to 8 rows per pass so `x` stays in registers or L1.
3. **Lookup table (T-MAC style)**: split `x` into groups of 4 (16 entries) or 8 (256 entries) and precompute every signed partial sum once per token. Each group of weight bits then becomes one table index. Pays off when many rows reuse the same `x`, which is the `U±` stage, and needs care when `r` is small.
4. **Batched (prefill)**: for `n > 1`, unpack tiles of signs to int8 and run blocked GEMM, either in fp32 or with per-token int8 activations and VNNI / dot-product instructions. Any int8 path must report its error against the fp32 path.

Dispatch: detect CPU features at runtime; for each layer shape, time the candidate kernels once at load (or use a cached choice) and keep the fastest. A flag forces a specific kernel for testing.

## Threading

Split output rows across a thread pool (or OpenMP). Each thread owns a contiguous block of rows, so results are deterministic for a fixed thread count. Single-token decode on small layers should fall back to fewer threads when scheduling costs more than the work.

## Precision

fp32 activations in and out, fp32 accumulation. Scales in fp32. Optional fp16/bf16 activations later. Expected error versus the reference is rounding only (summation order), so tests use tight tolerances.

## API

- **C**: `lb_layer_create(...)` from the raw arrays (packed U, V, scales, optional residual, optional bias, shapes), `lb_layer_forward(layer, x, n_tokens, y)`, `lb_layer_free(...)`, plus kernel selection and thread controls. Stable, plain C ABI so other runtimes can call it.
- **Python**: a `torch` C++ extension providing `littlebit.LBLinear` (CPU, inference only) and `littlebit.load_pretrained(path, ...)`. The loader builds the model skeleton from `config.json` with `transformers` on the meta device, swaps every `nn.Linear` except `lm_head` for `LBLinear` with the stored shapes, loads the packed tensors without unpacking them, and returns a model that works with `generate()`. It never imports the official training code.
- **CLI**: `littlebit-bench` for per-layer and end-to-end measurements, `littlebit-info` to print a checkpoint's layers, ranks and bits per weight.

## Testing

- Kernel tests against the reference: `in`/`out` multiples of 32 and not, `r` in {8, 24, 64, 128, 1000+}, `n` in {1, 3, 16, 64}, residual on/off, bias on/off, negative scales, every kernel and thread count.
- Packing tests: hand-built bit patterns that pin down LSB-first order, the bit-0 = +1 convention and row padding.
- **Synthetic checkpoints**: a script that turns a real small Hugging Face model (for example SmolLM2-135M or Qwen2.5-0.5B) into a format-correct LittleBit checkpoint using the paper's initialisation (truncated SVD, signs of the factors, rank-1 fits of their magnitudes for the scales, optional residual from the error), with no training. The model's quality is meaningless; it exists to check that littlebit.cpp and a plain float implementation of the same weights produce the same logits.
- Optional interop test: if a checkout of the official code is available at a pinned commit, compare against its layer on the same tensors. Skipped otherwise. That code is never copied into this repository.

## Benchmarks

Per-layer time for Llama-2-7B and Qwen2.5-0.5B shapes at 1.0, 0.55, 0.3 and 0.1 bits per weight. End-to-end tokens per second and peak memory for three setups: the dense model in fp32/bf16, unpack-then-matmul (what the reference does), and littlebit.cpp. Every result records the CPU model, flags, thread count and commit. Only numbers produced by these scripts go into the README.

## Milestones

1. CPU kernels (scalar, AVX2, AVX-512), C API, PyTorch layer and loader, tests, benchmarks.
2. ARM NEON, int8 dot-product batched kernels, Windows (MSVC) builds.
3. Standalone C++ runtime for Llama/Qwen-family models, no Python needed.
4. GPU backends (CUDA, Metal) and a WebAssembly build for in-browser demos.
