# LittleBit checkpoint format

Notes on the files written by the official LittleBit training code ([SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit)), from studying how its checkpoints are saved and loaded. They describe the format only. No code from that repository is part of this project.

LittleBit and LittleBit-2 write the **same** format. LittleBit-2 changes only how training is initialised (`use_itq: true`), so a runtime does not need to tell them apart.

## Files in a checkpoint directory

| File | Contents |
|---|---|
| `config.json` | The base Hugging Face model config, plus the quantization fields below and `use_cache: true` |
| `littlebit_config.json` | The quantization fields on their own |
| `model.safetensors` | All tensors. Large models may be sharded, with `model.safetensors.index.json`. Older checkpoints may use `model_state.pt` |
| tokenizer files | As saved by Hugging Face |

Quantization fields: `quant_func`, `eff_bit`, `split_dim`, `residual`, `kv_factor`, `min_split_dim`, `quant_mod`, `use_itq`, `itq_n_iter`. Older checkpoints without `littlebit_config.json` need these supplied by the user.

## Which layers are compressed

Every `nn.Linear` in the model except `lm_head`. Token embeddings, norms and `lm_head` are stored normally (16-bit). In Llama/Qwen-style models that means `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` in every block. `kv_factor` only changes the rank chosen for `k_proj`/`v_proj` at training time; at load time read ranks from the shape tensors. Phi-3 models are a special case (the fused QKV projection is split first) and are out of scope for now.

## Tensors for one compressed layer

For a layer with prefix `P` (for example `model.layers.0.mlp.up_proj`), `in` input features, `out` output features and rank `r`:

| Name | dtype | Shape | Meaning |
|---|---|---|---|
| `P.U_packed` | int32 | `[out, ceil(r/32)]` | sign matrix `U±`, `out × r` |
| `P.U_shape` | int64 | `[2]` | `(out, r)` |
| `P.V_packed` | int32 | `[r, ceil(in/32)]` | sign matrix `V±`, `r × in` |
| `P.V_shape` | int64 | `[2]` | `(r, in)` |
| `P.u1` | bf16 | `[1, out]` | output scale `h` |
| `P.u2` | bf16 | `[1, r]` | half of the rank scale |
| `P.v1` | bf16 | `[1, r]` | other half of the rank scale |
| `P.v2` | bf16 | `[1, in]` | input scale `g` |
| `P.bias` | as saved | `[out]` | only if the original layer had a bias (for example Qwen2 q/k/v) |

The rank scale is `l = v1 ⊙ u2` (element-wise). Scales can be negative.

If `residual` is true, a second path is stored with the same names plus `_R`: `U_R_packed, U_R_shape, V_R_packed, V_R_shape, u1_R, u2_R, v1_R, v2_R`. Its rank is the same as the primary path's.

Informational buffers may also appear: `_eff_bit_target`, `_split_dim_final`, `_eff_bit_actual` (cast to bf16 on save). They are not needed for inference.

## Bit packing

- Matrices are packed row by row into 32-bit words, **least significant bit first**: column `c` of row `i` is bit `c % 32` of word `c // 32` in row `i`.
- **Bit 0 means +1, bit 1 means −1.**
- Each row is padded to a multiple of 32 columns with +1 (bit 0).
- Words are stored as signed int32; read them as uint32. On a little-endian machine, byte `k` of a word holds columns `32w + 8k` to `32w + 8k + 7`, least significant bit first.

## Forward pass

The reference computes each path in this order (`x` has shape `[tokens, in]`):

```
t = x ⊙ v2            # input scale g
t = t @ V±ᵀ           # [tokens, r]
t = t ⊙ (v1 ⊙ u2)     # rank scale l
t = t @ U±ᵀ           # [tokens, out]
y = t ⊙ u1            # output scale h
```

With `residual`, the `_R` path is computed the same way and added. The bias, if any, is added last.

## Bits per weight

How the reference counts a layer's size (`a = in`, `b = out`, `r` = rank, scales counted at 16 bits):

- with residual: `(2·r·(a + b + 16) + 32·(a + b)) / (a·b)`
- without: `(r·(a + b + 16) + 16·(a + b)) / (a·b)`

Embeddings and `lm_head` are not included in that figure, and in small models they are most of the file.
