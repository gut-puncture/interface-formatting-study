"""The batched trial builder gives exactly the same trials as the reference builder, on all four tokenizers."""
import pytest
from transformers import AutoTokenizer

from keyhole import config as C
from keyhole import prompts as P


@pytest.mark.parametrize("key", C.MODEL_ORDER)
def test_make_trials_matches_reference(key):
    m = C.MODELS[key]
    tok = AutoTokenizer.from_pretrained(m["repo"], revision=m["rev"])
    items = P.load_items()[::53]
    pool, assign = P.load_fillers("tb64")
    for cond in ["plain", "null_pre", "null_post", "json_S", "log_M", "email_L", "prose_S", "html_M"]:
        fast, err = P.make_trials(tok, items, cond, pool, assign)
        assert not err
        for it, tr in zip(items, fast):
            ref = P.make_trial(tok, it, cond, pool, assign)
            assert (tr.prefix_key, tr.prefix_ids, tr.suffix_ids, tr.span_len, tr.n_prompt, tr.tail_ids) == \
                   (ref.prefix_key, ref.prefix_ids, ref.suffix_ids, ref.span_len, ref.n_prompt, ref.tail_ids)
