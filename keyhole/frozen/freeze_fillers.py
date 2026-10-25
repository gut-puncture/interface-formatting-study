"""Build and freeze the filler pools + item->filler assignment (run once, CPU).
Usage: python -I freeze_fillers.py QWEN3_TOKENIZER_DIR items.jsonl OUT_DIR
Writes fillers_tb64.json, fillers_tb512.json (E1), assign_tb64.json, assign_tb512.json."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import kh_wrappers as K
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1]); ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"]) if s else 0
items = [json.loads(l) for l in open(sys.argv[2])]
for tb in (64, 512):
    pool = {f: [K.build_filler(f, i, ntok, tail_budget=tb) for i in range(K.N_FILLERS)] for f in K.FAMILIES}
    assert all(F is not None for v in pool.values() for F in v)
    assign = {it["id"]: {f: K.assign_filler(it["id"], f, pool[f], it["q"], it["aliases"], "L") for f in K.FAMILIES} for it in items}
    assert all(v is not None for a in assign.values() for v in a.values())
    json.dump(pool, open(f"{sys.argv[3]}/fillers_tb{tb}.json", "w"), ensure_ascii=False)
    json.dump(assign, open(f"{sys.argv[3]}/assign_tb{tb}.json", "w"))
    print(tb, "ok", {f: [F["pre_counts"] for F in v][:1] for f, v in pool.items()})
