"""Render every condition of an item and align its tokens (FINAL_SPEC sections 4.1-4.2, 5).

A trial's token ids split into a shared PREFIX (everything up to and including "Question:")
and an item-specific SUFFIX (the question span, the rest of the wrapper, the closing
instruction and the answer cue). All items in one filler share the prefix, so its KV cache
is computed once per group.
"""
import json
from dataclasses import dataclass

from keyhole import config as C
from keyhole.frozen import kh_wrappers as K


def load_items():
    with open(C.DATA / "items.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_fillers(layout="tb64"):
    pool = json.load(open(C.DATA / f"fillers_{layout}.json", encoding="utf-8"))
    assign = json.load(open(C.DATA / f"assign_{layout}.json"))
    return pool, assign


def block_text(item, cond, pool, assign):
    """Block text and the wrapper-only text (for leak checks) of one condition."""
    q = item["q"]
    if cond == "plain":
        blk = K.render_plain(q)
    elif cond in K.NULLS:
        blk = K.render_null(item["id"], q, K.NULLS[cond], item["aliases"])
    else:
        fam, dose = cond.rsplit("_", 1)
        F = pool[fam][assign[item["id"]][fam]]
        blk = K.render(F, q, dose)
    return blk, blk.replace(K.ANCHOR + q, "")


def filler_id(item, cond, assign):
    if cond == "plain" or cond in K.NULLS:
        return -1
    return assign[item["id"]][cond.rsplit("_", 1)[0]]


def full_text(tok, user, gen_prefix=""):
    s = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                add_generation_prompt=True, **C.TEMPLATE_KWARGS)
    return s + gen_prefix + C.ANSWER_CUE


@dataclass
class Trial:
    item_id: str
    cond: str
    prefix_key: str        # the prefix string; trials with equal keys share a KV cache
    prefix_ids: list
    suffix_ids: list       # span + rest of prompt, ends with the answer cue
    span_len: int          # span = suffix[:span_len]
    n_prompt: int          # total prompt tokens
    tail_ids: tuple        # ids from the closing instruction to the end (alignment check)
    wrapper_text: str

    @property
    def q_end(self):       # suffix-relative index of the last question token
        return self.span_len - 1

    @property
    def final(self):       # suffix-relative index of the answer-cue token (site B)
        return len(self.suffix_ids) - 1


class AlignError(Exception):
    pass


def encode(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def make_trial(tok, item, cond, pool, assign, gen_prefix=""):
    q = item["q"]
    blk, wtext = block_text(item, cond, pool, assign)
    if blk.count(K.ANCHOR + q) != 1:
        raise AlignError("question not unique")
    full = full_text(tok, K.user_content(blk), gen_prefix)
    for st in getattr(tok, "all_special_tokens", []):
        if st and st in K.user_content(blk):
            raise AlignError(f"special token text {st!r} inside user content")
    ids = encode(tok, full)
    a = full.index(K.ANCHOR + q) + len(K.ANCHOR) - 1     # just after "Question:"
    b = a + len(K.interior(q))
    t = full.rindex(K.TAIL)
    pa, pb, pt = encode(tok, full[:a]), encode(tok, full[:b]), encode(tok, full[:t])
    for p in (pa, pb, pt):
        if ids[:len(p)] != p:
            raise AlignError("prefix not token-aligned")
    if len(pb) <= len(pa):
        raise AlignError("empty span")
    bos = getattr(tok, "bos_token_id", None)
    if bos is not None and bos in ids[1:]:
        raise AlignError("extra BOS")
    return Trial(item["id"], cond, full[:a], pa, ids[len(pa):], len(pb) - len(pa), len(ids),
                 tuple(ids[len(pt):]), wtext)


def make_trials(tok, items, cond, pool, assign, gen_prefix=""):
    """Batched, faster make_trial for many items of one condition (identical output).
    Long prompts are encoded in one parallel call; the shared prefix is encoded once per group;
    the span and the closing part are checked against short encodings."""
    out, errors = [], []
    texts = []
    for it in items:
        blk, wtext = block_text(it, cond, pool, assign)
        texts.append((it, blk, wtext, full_text(tok, K.user_content(blk), gen_prefix)))
    enc_all = tok([t[3] for t in texts], add_special_tokens=False)["input_ids"]
    pref_cache = {}
    bos = getattr(tok, "bos_token_id", None)
    for (it, blk, wtext, full), ids in zip(texts, enc_all):
        q = it["q"]
        try:
            if blk.count(K.ANCHOR + q) != 1:
                raise AlignError("question not unique")
            a = full.index(K.ANCHOR + q) + len(K.ANCHOR) - 1
            b = a + len(K.interior(q))
            t = full.rindex(K.TAIL)
            key = full[:a]
            if key not in pref_cache:
                pref_cache[key] = encode(tok, key)
            pa = pref_cache[key]
            if ids[:len(pa)] != pa:
                raise AlignError("prefix not token-aligned")
            span = encode(tok, full[a:b])
            if ids[len(pa):len(pa) + len(span)] != span:
                raise AlignError("span not token-aligned")
            tail = encode(tok, full[t:])
            if ids[len(ids) - len(tail):] != tail:
                raise AlignError("closing part not token-aligned")
            if bos is not None and bos in ids[1:]:
                raise AlignError("extra BOS")
            out.append(Trial(it["id"], cond, key, pa, ids[len(pa):], len(span), len(ids), tuple(tail), wtext))
        except AlignError as e:
            errors.append((it["id"], cond, str(e)))
    return out, errors


def check_against_plain(trial, plain):
    """The question span and the closing part must be token-identical to the plain prompt."""
    if trial.suffix_ids[:trial.span_len] != plain.suffix_ids[:plain.span_len] or trial.span_len != plain.span_len:
        raise AlignError("span differs from plain")
    if trial.tail_ids != plain.tail_ids:
        raise AlignError("closing instruction differs from plain")
