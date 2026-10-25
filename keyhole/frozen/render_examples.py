"""Render the FINAL_SPEC example item in every condition: dose S in full, M and L abridged; token-count table.
Usage: python -I render_examples.py QWEN3_TOKENIZER_DIR items.jsonl OUT.md"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import kh_wrappers as K
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1]); ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"]) if s else 0
it = next(json.loads(l) for l in open(sys.argv[2]) if '"Mongolia"' in l)
q, al = it["q"], it["aliases"]
out = [f"# Rendered examples: item {it['id']}  q=`{q}`  aliases={al}\nUser-content strings (chat template is applied around them). Token counts: Qwen3 tokenizer, user content only. Filler id 7 of each family.\n"]
def add(n, blk, abridge=False):
    c = K.user_content(blk); lines = c.split("\n")
    if abridge and len(lines) > 70:
        qi = next(i for i, l in enumerate(lines) if K.ANCHOR + q in l)
        lines = lines[:12] + [f"[... {qi - 30 - 12} lines omitted ...]"] + lines[qi - 30:]
    out.append(f"## {n} ({ntok(c)} tok)\n```text\n" + "\n".join(lines) + "\n```\n")
add("plain", K.render_plain(q))
for n, k in K.NULLS.items(): add(n, K.render_null(it["id"], q, k, al))
rows = []
for f in K.FAMILIES:
    F = K.build_filler(f, 7, ntok, al)
    for d in K.DOSES:
        c = K.user_content(K.render(F, q, d)); rows.append(f"| {f} | {d} | {F['pre_counts'][d]} | {ntok(c)} | {ntok(c[c.index(K.ANCHOR + q):])} |")
        add(f"{f} dose {d}", K.render(F, q, d), abridge=(d != "S"))
out.insert(1, "| family | dose | leading units | user tokens | question-to-end tokens |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
open(sys.argv[3], "w").write("\n".join(out)); print("\n".join(rows))
