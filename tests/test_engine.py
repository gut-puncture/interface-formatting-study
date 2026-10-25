"""Exactness of the engine on tiny random models of the three architectures (CPU, float32).

Covers: prefix-cache equivalence (incl. sliding-window layers shorter than the prefix),
right-padding equivalence, identity patching, last-layer patching, row independence,
sublayer capture/patch, and greedy generation on a cache.
"""
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from keyhole import config as C
from keyhole.engine import Engine, Plan

TINY = {
    "q4": dict(num_hidden_layers=4, hidden_size=64, intermediate_size=128, num_attention_heads=4,
               num_key_value_heads=2, head_dim=16),
    "olmo": dict(num_hidden_layers=4, hidden_size=64, intermediate_size=128, num_attention_heads=4,
                 num_key_value_heads=4, sliding_window=8,
                 layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]),
    "g12": dict(num_hidden_layers=6, hidden_size=64, intermediate_size=128, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, global_head_dim=32, sliding_window=8,
                layer_types=["sliding_attention"] * 5 + ["full_attention"]),
}


def tiny_engine(key):
    m = C.MODELS[key]
    cfg = AutoConfig.from_pretrained(m["repo"], revision=m["rev"])
    tc = cfg.get_text_config()
    for k, v in TINY[key].items():
        setattr(tc, k, v)
    tc.vocab_size = 512
    for k in ("pad_token_id", "bos_token_id", "eos_token_id"):
        if isinstance(getattr(tc, k, None), int) and getattr(tc, k) >= 512:
            setattr(tc, k, 0)
    if getattr(cfg, "vision_config", None) is not None:
        vc = cfg.vision_config
        for k, v in dict(num_hidden_layers=1, hidden_size=32, intermediate_size=64, num_attention_heads=2,
                         num_key_value_heads=2, head_dim=16).items():
            if hasattr(vc, k):
                setattr(vc, k, v)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32, attn_implementation="sdpa")
    tok = AutoTokenizer.from_pretrained(m["repo"], revision=m["rev"])
    return Engine(key, device="cpu", dtype=torch.float32, model=model, tok=tok)


@pytest.fixture(scope="module", params=list(TINY))
def eng(request):
    return tiny_engine(request.param)


def rnd(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, 500, (n,), generator=g).tolist()


def full_logits(eng, ids, keep):
    out = eng.model(input_ids=torch.tensor([ids]), use_cache=False, logits_to_keep=torch.tensor(keep))
    return out.logits.float()[0]


def test_prefix_cache_equivalence(eng):
    pre, suf = rnd(37, 1), rnd(11, 2)          # prefix longer than the sliding window (8)
    keep = list(range(len(suf)))
    lg, _ = eng.forward(eng.prefix_cache(pre), [suf], keep)
    ref = full_logits(eng, pre + suf, [len(pre) + k for k in keep])
    assert torch.allclose(lg[0], ref, atol=1e-4, rtol=1e-4)


def test_cache_not_mutated(eng):
    pre, suf = rnd(30, 21), rnd(6, 22)
    cache = eng.prefix_cache(pre)
    a, _ = eng.forward(cache, [suf], [5])
    eng.generate(cache, [suf], max_new=3)
    b, _ = eng.forward(cache, [suf], [5])
    assert torch.allclose(a, b, atol=1e-5)


def test_right_padding(eng):
    pre = rnd(20, 3)
    rows = [rnd(9, 4), rnd(5, 5), rnd(7, 6)]
    cache = eng.prefix_cache(pre)
    lg, _ = eng.forward(cache, rows, list(range(9)))
    for i, r in enumerate(rows):
        one, _ = eng.forward(cache, [r], list(range(len(r))))
        assert torch.allclose(lg[i, :len(r)], one[0], atol=1e-4, rtol=1e-4)


def test_identity_and_row_independence(eng):
    pre, suf = rnd(25, 7), rnd(10, 8)
    cache = eng.prefix_cache(pre)
    span = list(range(4))
    cap = {("r", l): (torch.zeros(4, dtype=torch.long), torch.tensor(span)) for l in range(eng.L)}
    base, p = eng.forward(cache, [suf], [9], Plan(capture=cap))
    rows = [suf] * eng.L
    patch = {("r", l): (torch.full((4,), l), torch.tensor(span), p.got[("r", l)]) for l in range(eng.L)}
    lg, _ = eng.forward(cache, rows, [9], Plan(patch=patch))
    assert torch.allclose(lg[:, 0], base[0, 0].expand(eng.L, -1), atol=1e-4, rtol=1e-4)


def test_last_layer_final_position_patch(eng):
    pre_a, suf_a = rnd(20, 9), rnd(8, 10)
    pre_b, suf_b = rnd(31, 11), rnd(12, 12)
    ca, cb = eng.prefix_cache(pre_a), eng.prefix_cache(pre_b)
    L = eng.L - 1
    la, p = eng.forward(ca, [suf_a], [7], Plan(capture={("r", L): (torch.tensor([0]), torch.tensor([7]))}))
    lb, _ = eng.forward(cb, [suf_b], [11], Plan(patch={("r", L): (torch.tensor([0]), torch.tensor([11]), p.got[("r", L)])}))
    assert torch.allclose(lb[0, 0], la[0, 0], atol=1e-4, rtol=1e-4)


def test_sublayer_identity(eng):
    pre, suf = rnd(18, 13), rnd(9, 14)
    cache = eng.prefix_cache(pre)
    keys = [(s, l) for s in ("attn", "mlp") for l in range(eng.L)]
    cap = {k: (torch.zeros(3, dtype=torch.long), torch.tensor([0, 1, 2])) for k in keys}
    base, p = eng.forward(cache, [suf], [8], Plan(capture=cap))
    patch = {k: (torch.zeros(3, dtype=torch.long), torch.tensor([0, 1, 2]), p.got[k]) for k in keys}
    lg, _ = eng.forward(cache, [suf], [8], Plan(patch=patch))
    assert torch.allclose(lg, base, atol=1e-4, rtol=1e-4)
    z = {("attn", 1): (torch.zeros(3, dtype=torch.long), torch.tensor([0, 1, 2]), torch.zeros_like(p.got[("attn", 1)]))}
    lz, _ = eng.forward(cache, [suf], [8], Plan(patch=z))
    assert not torch.allclose(lz, base, atol=1e-4)


def test_generate_matches_forward(eng):
    pre, suf = rnd(22, 15), rnd(6, 16)
    cache = eng.prefix_cache(pre)
    gen = eng.generate(cache, [suf, suf], max_new=5)
    assert gen[0] == gen[1]
    seq = list(suf)
    for t in gen[0]:
        lg, _ = eng.forward(cache, [seq], [len(seq) - 1])
        assert lg[0, 0].argmax().item() == t
        seq.append(t)
