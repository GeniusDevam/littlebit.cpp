import json
import os
from pathlib import Path

import pytest
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from littlebit import load_pretrained
from littlebit.loader import load_reference
from tools.make_synthetic import write_checkpoint

PROMPTS = (
    "The sky is blue because",
    "A short story about a small bird",
    "The next number in the sequence 1, 2, 3 is",
)
MODELS = {
    "smollm2": "HuggingFaceTB/SmolLM2-135M",
    "qwen2": "Qwen/Qwen2.5-0.5B",
}


@torch.inference_mode()
def _logits(model: PreTrainedModel, inputs: list[Tensor]) -> Tensor:
    rows = []
    for input_ids in inputs:
        result = model(input_ids=input_ids, use_cache=True)
        rows.append(result.logits[0])
        result = model(
            input_ids=input_ids[:, -1:],
            past_key_values=result.past_key_values,
            use_cache=True,
        )
        rows.append(result.logits[0])
    return torch.cat(rows)


def _compare(path: Path, source: str, name: str, variant: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(source)
    inputs = [tokenizer(text, return_tensors="pt").input_ids for text in PROMPTS]
    packed = load_pretrained(path, threads=4)
    got = _logits(packed, inputs)
    with torch.inference_mode():
        output = packed.generate(
            inputs[0],
            do_sample=False,
            min_new_tokens=2,
            max_new_tokens=2,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            attention_mask=torch.ones_like(inputs[0]),
        )
    assert output.shape[-1] == inputs[0].shape[-1] + 2
    del packed
    expected = _logits(load_reference(path), inputs)
    agreement = (got.argmax(dim=-1) == expected.argmax(dim=-1)).sum().item()
    result = {
        "model": name,
        "variant": variant,
        "checkpoint": str(path),
        "tokens": len(got),
        "top1_matches": agreement,
        "max_abs_diff": (got - expected).abs().max().item(),
    }
    print(json.dumps(result))
    report = os.environ.get("LITTLEBIT_MODEL_RESULTS")
    if report:
        with Path(report).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result) + "\n")
    assert agreement == len(got)
    torch.testing.assert_close(got, expected, atol=0.003, rtol=0.0003)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("name", "eff_bit"), [(name, bit) for name in MODELS for bit in (1.0, 0.1)]
)
def test_published_models(tmp_path: Path, name: str, eff_bit: float) -> None:
    torch.set_num_threads(4)
    source = os.environ.get(f"LITTLEBIT_{name.upper()}_SOURCE", MODELS[name])
    data = Path(os.environ.get("LITTLEBIT_TEST_DATA", tmp_path))
    path = data / f"{name}-{eff_bit}"
    if not path.is_dir():
        dense = AutoModelForCausalLM.from_pretrained(source, dtype="auto")
        write_checkpoint(dense, path, eff_bit, True, tokenizer_source=source)
        del dense
    _compare(path, source, name, f"synthetic-{eff_bit}")


@pytest.mark.slow
def test_external_checkpoint() -> None:
    path = os.environ.get("LITTLEBIT_INTEROP_CHECKPOINT")
    if not path:
        pytest.skip("set LITTLEBIT_INTEROP_CHECKPOINT to a LittleBit directory with a tokenizer")
    torch.set_num_threads(4)
    _compare(Path(path), path, "external", "interop")
