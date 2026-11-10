"""D1/D2/F2: decoder equals HF Llama; cached == uncached; speculative == greedy.
Uses the small 135M checkpoint where possible so the suite runs in minutes on CPU."""
import pytest
import torch

from trialmatch.config import DRAFT_MODEL, TARGET_MODEL, model_path
from trialmatch.generation import greedy_kv, greedy_no_cache, speculative_greedy
from trialmatch.model import load_decoder

needs_models = pytest.mark.skipif(not (model_path(DRAFT_MODEL) / "model.safetensors").exists(),
                                  reason="models not downloaded")


@pytest.fixture(scope="module")
def small():
    torch.set_num_threads(8)
    return load_decoder(str(model_path(DRAFT_MODEL)))


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_path(DRAFT_MODEL)))


@needs_models
def test_logits_match_huggingface(small, tok):
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(str(model_path(DRAFT_MODEL)), torch_dtype=torch.float32).eval()
    ids = tok("A 58-year-old woman with chest pain. She denies smoking.", return_tensors="pt").input_ids
    with torch.no_grad():
        ref = hf(ids).logits
        for be in ("sdpa", "reference", "tiled"):
            assert (small(ids, backend=be) - ref).abs().max() < 1e-3


@needs_models
@pytest.mark.parametrize("L,N", [(8, 12), (40, 24), (120, 40)])
def test_kv_cache_equals_no_cache(small, tok, L, N):
    text = "Patient note: 0. A 45-year-old man with type 2 diabetes and hypertension presents with chest pain. " * 6
    p = tok(text).input_ids[:L]
    a = greedy_no_cache(small, p, N).tokens
    for kind in ("static", "dynamic"):
        assert greedy_kv(small, p, N, cache_kind=kind).tokens == a
    assert greedy_kv(small, p, N, backend="tiled").tokens == a


@needs_models
def test_chunked_prefill_matches_full(small, tok):
    p = tok("The quick brown fox jumps over the lazy dog again and again.").input_ids
    with torch.no_grad():
        full = small(torch.tensor([p]))
        c = small.new_cache(64)
        parts = [small(torch.tensor([p[:5]]), cache=c), small(torch.tensor([p[5:9]]), cache=c),
                 small(torch.tensor([p[9:]]), cache=c)]
    assert (torch.cat(parts, 1) - full).abs().max() < 1e-4


@needs_models
def test_cache_crop_then_regenerate(small, tok):
    """Roll-back used by speculative decoding: crop and re-feed gives identical logits."""
    p = tok("Exclusion criteria: pregnancy, renal failure, prior stroke.").input_ids
    with torch.no_grad():
        c = small.new_cache(64)
        small(torch.tensor([p]), cache=c)
        c.crop(len(p) - 3)
        again = small(torch.tensor([p[-3:]]), cache=c)
        full = small(torch.tensor([p]))[:, -3:]
    assert (again - full).abs().max() < 1e-4


@needs_models
@pytest.mark.skipif(not (model_path(TARGET_MODEL) / "model.safetensors").exists(), reason="target missing")
@pytest.mark.parametrize("k", [1, 3, 5])
def test_speculative_equals_greedy(small, tok, k):
    target = load_decoder(str(model_path(TARGET_MODEL)))
    p = tok("Inclusion criteria: adults aged 18 to 65 with stable angina. The patient is").input_ids
    g = greedy_kv(target, p, 30).tokens
    s = speculative_greedy(target, small, p, 30, k=k)
    assert s.tokens == g
    assert s.stats["accepted"] <= s.stats["drafted"]


@needs_models
def test_speculative_with_self_draft_accepts_everything(small, tok):
    p = tok("Exclusion criteria include pregnancy and").input_ids
    s = speculative_greedy(small, small, p, 20, k=4)
    assert s.tokens == greedy_kv(small, p, 20).tokens
    assert s.stats["acceptance_rate"] == 1.0
