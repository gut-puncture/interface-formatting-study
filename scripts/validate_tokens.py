"""CPU check of token alignment for every item x condition on each model's real tokenizer.

python scripts/validate_tokens.py [model_key ...]  -> writes data/token_check_<model>.json
"""
import collections
import json
import sys
import time

from transformers import AutoTokenizer

from keyhole import config as C
from keyhole import prompts as P


def main(keys):
    items = P.load_items()
    pool, assign = P.load_fillers("tb64")
    for key in keys:
        m = C.MODELS[key]
        tok = AutoTokenizer.from_pretrained(m["repo"], revision=m["rev"])
        t0 = time.time()
        bad, lens, q2end, prefixes = collections.Counter(), collections.defaultdict(list), [], collections.defaultdict(set)
        dropped = set()
        for it in items:
            try:
                plain = P.make_trial(tok, it, "plain", pool, assign)
            except P.AlignError as e:
                bad[("plain", str(e))] += 1; dropped.add(it["id"]); continue
            for cond in C.CONDITIONS[1:]:
                try:
                    tr = P.make_trial(tok, it, cond, pool, assign)
                    P.check_against_plain(tr, plain)
                except P.AlignError as e:
                    bad[(cond, str(e))] += 1; dropped.add(it["id"]); continue
                lens[cond].append(tr.n_prompt)
                q2end.append(len(tr.suffix_ids))
                prefixes[cond].add(tr.prefix_key)
        out = {"model": key, "items": len(items), "dropped": sorted(dropped), "bad": {f"{k[0]}|{k[1]}": v for k, v in bad.items()},
               "median_len": {c: sorted(v)[len(v) // 2] for c, v in lens.items()},
               "max_suffix": max(q2end), "n_prefix_groups": {c: len(v) for c, v in prefixes.items()},
               "seconds": round(time.time() - t0, 1)}
        json.dump(out, open(C.DATA / f"token_check_{key}.json", "w"), indent=1)
        print(key, "dropped", len(dropped), "bad", dict(bad), "max_suffix", out["max_suffix"],
              "median L", {c: out["median_len"][c] for c in out["median_len"] if c.endswith("_L")}, out["seconds"], "s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or C.MODEL_ORDER)
