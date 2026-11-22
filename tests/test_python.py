import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM

from littlebit import LBLinear, load_pretrained
from littlebit.checkpoint import read_checkpoint
from littlebit.linear import KERNELS
from littlebit.loader import load_reference
from littlebit.reference import UnpackedLinear, unpack_signs
from tools.make_synthetic import choose_rank, pack_signs, write_checkpoint


@pytest.fixture(scope="session", autouse=True)
def torch_threads() -> None:
    torch.set_num_threads(2)


def test_packing() -> None:
    u = torch.ones(2, 33)
    u[0, [0, 31, 32]] = -1
    u[1, [7, 24]] = -1
    integers = pack_signs(u)
    assert integers.dtype == torch.int32
    assert integers.tolist() == [[-2147483647, 1], [16777344, 0]]
    torch.testing.assert_close(unpack_signs(integers, 33), u)
    assert choose_rank(1024, 1024, 0.1, True) == 8
    assert choose_rank(32, 7, 10, False) == 7


@pytest.mark.parametrize("kernel", KERNELS)
def test_forward(kernel: str) -> None:
    if not torch.ops.littlebit.kernel_available(KERNELS.index(kernel)):
        pytest.skip("kernel unavailable on this CPU")
    gen = torch.Generator().manual_seed(4)
    width, out, rank = 35, 21, 24
    state = {"bias": torch.randn(out, generator=gen)}
    for suffix in ("", "_R"):
        state[f"U{suffix}_packed"] = pack_signs(torch.randn(out, rank, generator=gen))
        state[f"V{suffix}_packed"] = pack_signs(torch.randn(rank, width, generator=gen))
        for key, size in (("u1", out), ("v1", rank), ("u2", rank), ("v2", width)):
            state[f"{key}{suffix}"] = torch.randn(1, size, generator=gen) * 0.2
    packed = LBLinear(width, out, rank, state, kernel=kernel, threads=4)
    reference = UnpackedLinear(width, rank, state)
    x = torch.randn(5, 3, width, generator=gen).transpose(0, 1)
    torch.testing.assert_close(packed(x), reference(x), atol=1e-6, rtol=1e-5)
    assert packed(x.bfloat16()).dtype == torch.bfloat16
    assert packed(x[:0]).shape == (0, 5, out)
    with pytest.raises(RuntimeError, match="inference only"):
        packed(x.requires_grad_())


@pytest.fixture
def toy(tmp_path: Path) -> Path:
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=43,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        pad_token_id=0,
        eos_token_id=2,
    )
    torch.manual_seed(27)
    path = tmp_path / "toy"
    write_checkpoint(Qwen2ForCausalLM(config), path, 0.13, True)
    return path


@pytest.mark.parametrize("layout", ["single", "sharded", "legacy"])
@torch.inference_mode()
def test_load_and_generate(toy: Path, layout: str) -> None:
    file = toy / "model.safetensors"
    if layout == "legacy":
        torch.save(load_file(file), toy / "model_state.pt")
        file.unlink()
        (toy / "littlebit_config.json").unlink()
    elif layout == "sharded":
        state = load_file(file)
        index = {}
        for n in range(2):
            shard = {key: state[key] for key in list(state)[n::2]}
            filename = f"model-0000{n + 1}-of-00002.safetensors"
            save_file(shard, toy / filename, metadata={"format": "pt"})
            index.update(dict.fromkeys(shard, filename))
        (toy / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": index, "metadata": {}}), encoding="utf-8"
        )
        file.unlink()

    model = load_pretrained(toy, kernel="scalar")
    reference = load_reference(toy)
    assert isinstance(model.model.layers[0].self_attn.q_proj, LBLinear)
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    input_ids = torch.tensor([[1, 3, 4, 5]], dtype=torch.int64)
    torch.testing.assert_close(
        model(input_ids).logits, reference(input_ids).logits, atol=1e-6, rtol=1e-5
    )
    generated = model.generate(input_ids, min_new_tokens=2, max_new_tokens=2)
    assert generated.shape == (1, 6)
    assert torch.equal(generated[:, :4], input_ids)


@torch.inference_mode()
def test_load_bf16_untied_tables(tmp_path: Path) -> None:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=2,
    )
    path = tmp_path / "untied"
    write_checkpoint(Qwen3ForCausalLM(config).bfloat16(), path, 1.0, True)
    model = load_pretrained(path, dtype=torch.bfloat16)
    state = load_file(path / "model.safetensors")
    for weight, key in (
        (model.get_input_embeddings().weight, "model.embed_tokens.weight"),
        (model.get_output_embeddings().weight, "lm_head.weight"),
    ):
        assert weight.dtype == torch.bfloat16
        torch.testing.assert_close(weight, state[key], rtol=0, atol=0)
    assert model.get_input_embeddings().weight is not model.get_output_embeddings().weight
    assert isinstance(model.model.layers[0].self_attn.q_proj, LBLinear)
    ids = torch.tensor([[1, 3, 4, 5]])
    assert torch.isfinite(model(ids).logits).all()
    generated = model.generate(
        ids, attention_mask=torch.ones_like(ids), min_new_tokens=2, max_new_tokens=2
    )
    assert generated.shape == (1, 6)


def test_malformed_shape(toy: Path) -> None:
    file = toy / "model.safetensors"
    state = load_file(file)
    key = "model.layers.0.self_attn.q_proj.U_shape"
    state[key][1] += 1
    file.unlink()
    save_file(state, file, metadata={"format": "pt"})
    with pytest.raises(ValueError, match="inconsistent U and V"):
        read_checkpoint(toy)


def test_residual_off(tmp_path: Path) -> None:
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=17,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    path = tmp_path / "single_path"
    write_checkpoint(Qwen2ForCausalLM(config), path, 1.0, False)
    assert not any(layer.residual for layer in read_checkpoint(path).layers.values())
    assert load_pretrained(path)(torch.tensor([[1]])).logits.shape == (1, 1, 32)


@pytest.mark.slow
@torch.inference_mode()
def test_official_layer(toy: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("LITTLEBIT_TEST_OFFICIAL") != "1":
        pytest.skip("set LITTLEBIT_TEST_OFFICIAL=1 to fetch the pinned reference")
    commit = "42d658b0c79f76450b34b6a3547462c7cdc3e1a0"
    commands = (
        ("init", "-q"),
        ("fetch", "-q", "--depth=1", "https://github.com/SamsungLabs/LittleBit.git", commit),
        ("checkout", "-q", "--detach", "FETCH_HEAD"),
    )
    with TemporaryDirectory(prefix="littlebit-reference-") as checkout:
        try:
            for command in commands:
                subprocess.run(
                    ["git", *command], cwd=checkout, check=True, capture_output=True, timeout=60
                )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            pytest.skip(f"pinned reference unavailable: {error}")
        monkeypatch.syspath_prepend(checkout)
        # Load the two reference files without the training package's circular imports.
        for directory in ("modules", "utils"):
            name = f"quantization.{directory}"
            package = ModuleType(name)
            package.__path__ = [str(Path(checkout) / "quantization" / directory)]
            monkeypatch.setitem(sys.modules, name, package)
        module = importlib.import_module("quantization.modules.littlebit")
        unpacker = importlib.import_module("quantization.utils.binary_packer").binary_unpacker
    official = module.LittleBitLinear()

    prefix = "model.layers.0.self_attn.q_proj."
    fields = {
        name.removeprefix(prefix): value
        for name, value in load_file(toy / "model.safetensors").items()
        if name.startswith(prefix)
    }
    out, rank = fields["U_shape"].tolist()
    _, width = fields["V_shape"].tolist()
    official.in_features = width
    official.out_features = out
    official.residual = True
    official._binarized = True
    for name in ("U", "V", "U_R", "V_R"):
        signs = unpacker(fields[f"{name}_packed"], tuple(fields[f"{name}_shape"].tolist()))
        setattr(official, name, signs)
    for name in ("u1", "u2", "v1", "v2", "u1_R", "u2_R", "v1_R", "v2_R", "bias"):
        setattr(official, name, fields[name].float())
    x = torch.randn(3, width, generator=torch.Generator().manual_seed(19))
    packed = LBLinear(width, out, rank, fields, kernel="scalar")
    torch.testing.assert_close(packed(x), official(x), atol=1e-6, rtol=1e-5)
