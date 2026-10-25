"""Keyhole v2 item builder -- FROZEN reference for FINAL_SPEC.md section 3.
Usage: python -I build_items.py RAW_DIR OUT_JSONL
RAW_DIR holds: fv_country-capital.json, lre_{landmark_in_country,company_hq,star_constellation,superhero_person}.json,
wendler_{fr,de,ru}_clean.csv (URLs in FINAL_SPEC.md). Standard library only; deterministic."""
import json, csv, re, sys, hashlib, random, unicodedata, collections, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kh_wrappers as K

SALT = "keyhole-v2"
def H(s): return int(hashlib.sha256(s.encode()).hexdigest(), 16)
def split_of(family, unit): return "dev" if H(f"{SALT}|{family}|{unit}") % 10 < 3 else "test"
def form_of(item_id): return ["Q", "I"][H(item_id) % 2]
def fold(s): return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).casefold()
def p4(s): return {w[:4] for w in re.findall(r"[a-z0-9]+", fold(s)) if len(w) >= 4}
SAFE = re.compile(r"^[A-Za-z0-9 ,.\-]+$")

FACT_T = {  # relation: (Q form, I form)
    "country_capital":     ("What is the capital city of {}?", "Name the capital city of {}."),
    "landmark_in_country": ("In which country is {} located?", "Name the country where {} is located."),
    "company_hq":          ("In which city is the company {} headquartered?", "Name the city where the company {} is headquartered."),
    "star_constellation":  ("Which constellation contains the star {}?", "Name the constellation that contains the star {}."),
    "superhero_person":    ("What is the real name of the superhero {}?", "Give the real name of the superhero {}."),
}
ARITH_T = {"add": ("What is {} plus {}?", "Compute {} plus {}."),
           "sub": ("What is {} minus {}?", "Compute {} minus {}."),
           "mul": ("What is {} times {}?", "Compute {} times {}.")}
LANG = {"fr": "French", "de": "German", "ru": "Russian"}
TRANS_T = ("What is the {L} word for {w}?", "Translate the English word {w} into {L}.")
ALIAS = {  # small fixed alias table (applied to gold objects)
    "United States": ["United States of America", "USA", "US", "U.S.", "U.S.A.", "America"],
    "United Kingdom": ["UK", "U.K.", "Britain", "Great Britain"],
    "Netherlands": ["Holland", "The Netherlands"], "Czech Republic": ["Czechia"],
    "People's Republic of China": ["China"], "Republic of Ireland": ["Ireland"],
    "Washington, D.C.": ["Washington", "Washington D.C.", "Washington DC"],
    "New York City": ["New York", "NYC"], "Ulaanbaatar": ["Ulan Bator"],
}
FACT_N, ARITH_N = 250, 150

def item(family, template_id, slot, unit, gold, aliases, tpl_pair):
    iid = f"{family}-{template_id}-{H(template_id + '|' + slot) % 10**8:08d}"
    form = form_of(iid)
    q = K.canonical_question(tpl_pair[0 if form == "Q" else 1].format(*slot.split("|")) if family != "TRANS"
                             else tpl_pair[0 if form == "Q" else 1].format(L=LANG[template_id], w=slot))
    if q is None: return None
    al = sorted({gold, *aliases, *ALIAS.get(gold, [])})
    return {"id": iid, "family": family, "template_id": template_id, "form": form, "q": q, "slot": slot,
            "gold": gold, "aliases": al, "unit": unit, "split": split_of(family, unit)}

def build(raw):
    out = []
    # FACT
    src = {"country_capital": [(d["input"], d["output"]) for d in json.load(open(f"{raw}/fv_country-capital.json"))]}
    for r in ["landmark_in_country", "company_hq", "star_constellation", "superhero_person"]:
        src[r] = [(d["subject"], d["object"]) for d in json.load(open(f"{raw}/lre_{r}.json"))["samples"]]
    for r, pairs in src.items():
        objs = collections.defaultdict(set)
        for s, o in pairs: objs[s.strip()].add(o.strip())
        keep = [(s, next(iter(o))) for s, o in objs.items() if len(o) == 1]           # unambiguous subjects only
        keep = [(s, o) for s, o in keep if SAFE.match(s) and not (p4(s) & p4(o)) and fold(o) not in fold(s)]
        keep.sort(key=lambda x: H(f"{r}|{x[0]}"))
        cap = max(1, int(0.10 * min(FACT_N, len(keep)))); c = collections.Counter(); n = 0
        for s, o in keep:
            if n >= FACT_N: break
            if c[o] >= cap: continue
            it = item("FACT", r, s, s, o, [], FACT_T[r])
            if it: out.append(it); c[o] += 1; n += 1
    # ARITH
    for op in ["add", "sub", "mul"]:
        rng = random.Random(f"{SALT}|ARITH|{op}"); seen = set(); n = 0
        while n < ARITH_N:
            if op == "add": a, b = rng.randint(12, 89), rng.randint(12, 89); ans = a + b
            elif op == "sub": a = rng.randint(30, 99); b = rng.randint(12, a - 11); ans = a - b
            else: a, b = rng.randint(12, 49), rng.randint(3, 9); ans = a * b
            key = (op, tuple(sorted((a, b))) if op != "sub" else (a, b))
            if key in seen or any(x % 10 == 0 or x in (0, 1, 10, 11) for x in (a, b)) or ans in (a, b) \
               or str(ans) in f"{a} {b}": continue
            seen.add(key)
            it = item("ARITH", op, f"{a}|{b}", f"{op}|{key[1][0]}|{key[1][1]}", str(ans), [], ARITH_T[op])
            if it: out.append(it); n += 1
    # TRANS
    for l in LANG:
        for row in csv.DictReader(open(f"{raw}/wendler_{l}_clean.csv", encoding="utf-8")):
            w, t = row["word_original"].strip(), row["word_translation"].strip()
            if not re.fullmatch(r"[a-z]+", w) or fold(t)[:4] == fold(w)[:4] or fold(t) in fold(w): continue
            it = item("TRANS", l, w, w, t, [], TRANS_T)
            if it: out.append(it)
    return out

if __name__ == "__main__":
    items = build(sys.argv[1])
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
    c = collections.Counter((i["family"], i["template_id"], i["split"]) for i in items)
    for k in sorted(c): print(k, c[k])
    print("total", len(items), collections.Counter(i["family"] for i in items), collections.Counter(i["form"] for i in items))
