"""Activation patching on failed wrapped runs (FINAL_SPEC section 10).

For a recipient item r that fails under wrapper condition c, run the failed wrapped prompt
with its own gold tokens g appended, once per patch row (layer-as-batch):
  BASE      no patch
  A@l       question-token residual states at layer l  <- r's plain run
  D@l       question-token residual states at layer l  <- a different known item's plain run
  B@l       final-position residual state at layer l   <- r's plain run
  AFULL     question-token states at every layer        <- r's plain run
  SUB(b,v)  attention ("attn"), MLP ("mlp") or both sublayer outputs at the question tokens
            in all layers of depth third b                <- r's plain run
"""
import hashlib
import math
import time

import pandas as pd
import torch

from keyhole import config as C
from keyhole import prompts as P
from keyhole.batching import max_rows
from keyhole.engine import Plan


def h(s):
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


# ------------------------------------------------------------------ pair selection
def select_pairs(beh, items, model, n_patch=C.N_PATCH):
    byid = {it["id"]: it for it in items}
    plain = beh[(beh.condition == "plain") & (beh.known == True)].set_index("item_id")
    lp_plain = plain["lp_g"].to_dict()
    span = plain["span_len"].to_dict()
    g = plain["g_ids"].apply(list).to_dict()
    w = beh[(beh.condition.isin(C.WRAPPED)) & (~beh.correct) & (beh.fclass.isin(["F1", "F2"]))].copy()
    w = w[w.item_id.isin(plain.index)]
    w["gap"] = w.item_id.map(lp_plain).astype("float64") - w.lp_g.astype("float64")
    w = w[w.gap >= C.FAIL_GAP]
    if w.empty:
        return [], 0
    known = list(plain.index)

    def donor(r, split):
        it = byid[r]
        for relax in (False, True):
            cands = [d for d in known if d != r and byid[d]["family"] == it["family"] and byid[d]["form"] == it["form"]
                     and byid[d]["split"] == split and span[d] == span[r] and byid[d]["unit"] != it["unit"]
                     and g[d][0] != g[r][0] and (relax or byid[d]["template_id"] == it["template_id"])]
            if cands:
                return min(cands, key=lambda d: h(f"{model}|{r}|{d}"))
        return None

    w["order"] = [h(f"{model}|{i}|{c}") for i, c in zip(w.item_id, w.condition)]
    w = w.sort_values("order")
    cells = {}
    for rec in w.itertuples():
        cells.setdefault((rec.family, rec.wrapper), []).append(rec)
    chosen, per_item, fams_used = [], {}, {}
    keys = sorted(cells)
    while len(chosen) < n_patch and any(cells[k] for k in keys):
        for k in keys:
            while cells[k]:
                rec = cells[k].pop(0)
                if per_item.get(rec.item_id, 0) >= C.MAX_PAIRS_PER_ITEM or rec.wrapper in fams_used.get(rec.item_id, set()):
                    continue
                d = donor(rec.item_id, rec.split)
                if d is None:
                    continue
                chosen.append({"pair_id": f"{model}|{rec.item_id}|{rec.condition}", "item_id": rec.item_id,
                               "condition": rec.condition, "donor_id": d, "split": rec.split})
                per_item[rec.item_id] = per_item.get(rec.item_id, 0) + 1
                fams_used.setdefault(rec.item_id, set()).add(rec.wrapper)
                break
            if len(chosen) >= n_patch:
                break
    return chosen, len(w)


# ------------------------------------------------------------------ one pair
_CACHE = {}


def bands(L):
    return {b: [l for l in range(L) if (3 * l) // L == b] for b in range(3)}


def run_pair(eng, pair, items_by_id, trials, gold, grid):
    """trials: dict (item_id, cond) -> Trial. gold: item_id -> g ids."""
    L = eng.L
    r, d, cond = pair["item_id"], pair["donor_id"], pair["condition"]
    pr, pd_, wr = trials[(r, "plain")], trials[(d, "plain")], trials[(r, cond)]
    g, gd = gold[r], gold[d]
    span = pr.span_len
    assert pd_.span_len == span == wr.span_len
    sp = torch.arange(span, device=eng.device)

    # 1. plain captures (recipient row 0, donor row 1); both share the plain prefix
    cap = {}
    zero = torch.zeros(span, dtype=torch.long, device=eng.device)
    for l in range(L):
        cap[("r", l)] = (torch.cat([zero, zero + 1]), torch.cat([sp, sp]))
        cap[("attn", l)] = (zero, sp)
        cap[("mlp", l)] = (zero, sp)
    cache_p = eng.prefix_cache(pr.prefix_ids)
    _, plan = eng.forward(cache_p, [pr.suffix_ids, pd_.suffix_ids], [0], Plan(capture=cap))
    resid_r = {l: plan.got[("r", l)][:span] for l in range(L)}
    resid_d = {l: plan.got[("r", l)][span:] for l in range(L)}
    sub_r = {(v, l): plan.got[(v, l)] for v in ("attn", "mlp") for l in range(L)}
    capf = {("r", l): (torch.zeros(1, dtype=torch.long, device=eng.device), eng.t([pr.final])) for l in range(L)}
    _, planf = eng.forward(cache_p, [pr.suffix_ids], [0], Plan(capture=capf))
    final_r = {l: planf.got[("r", l)] for l in range(L)}
    del cache_p

    # 2. patch rows on the failed wrapped run
    specs = [("BASE", -1, -1, "")]
    specs += [("A", l, -1, "") for l in grid] + [("D", l, -1, "") for l in grid] + [("B", l, -1, "") for l in grid]
    specs += [("AFULL", -1, -1, "")]
    specs += [("SUB", -1, b, v) for b in range(3) for v in ("attn", "mlp", "both")]
    band = bands(L)

    def patches(spec):
        site, l, b, v = spec
        if site == "A":
            return [(("r", l), sp, resid_r[l])]
        if site == "D":
            return [(("r", l), sp, resid_d[l])]
        if site == "B":
            return [(("r", l), eng.t([wr.final]), final_r[l])]
        if site == "AFULL":
            return [(("r", k), sp, resid_r[k]) for k in range(L)]
        if site == "SUB":
            vs = ("attn", "mlp") if v == "both" else (v,)
            return [((s, k), sp, sub_r[(s, k)]) for k in band[b] for s in vs]
        return []

    seq = wr.suffix_ids + g
    n, m = len(wr.suffix_ids), len(g)
    keep = list(range(n - 1, n - 1 + m))
    if _CACHE.get("key") != wr.prefix_key:          # pairs are sorted by prefix, so this is reused
        _CACHE.clear()
        if eng.device == "cuda":
            torch.cuda.empty_cache()
        _CACHE.update(key=wr.prefix_key, cache=eng.prefix_cache(wr.prefix_ids))
    cache_w = _CACHE["cache"]
    B = max_rows(eng, cache_w, positions=len(keep))
    out = []
    for i in range(0, len(specs), B):
        chunk = specs[i:i + B]
        merged = {}
        for row, spec in enumerate(chunk):
            for key, pos, vals in patches(spec):
                merged.setdefault(key, []).append((torch.full((len(pos),), row, device=eng.device), pos, vals))
        plan = Plan(patch={k: (torch.cat([a for a, _, _ in v]), torch.cat([b for _, b, _ in v]), torch.cat([c for _, _, c in v]))
                           for k, v in merged.items()})
        logits, _ = eng.forward(cache_w, [seq] * len(chunk), keep, plan)
        lp = torch.log_softmax(logits, -1)
        for row, spec in enumerate(chunk):
            lpg = sum(lp[row, j, g[j]].item() for j in range(m))
            out.append({"spec": spec, "lp_g": lpg, "logit_g1": logits[row, 0, g[0]].item(),
                        "argmax1": logits[row, 0].argmax().item(), "lp_donor_g1": lp[row, 0, gd[0]].item(),
                        "logits0": logits[row, 0]})
    base = out[0]
    w1 = base["argmax1"]
    recs = []
    for o in out:
        site, l, b, v = o["spec"]
        recs.append({"site": site, "layer": l, "band": b, "sub": v, "lp_g": o["lp_g"], "lp_g_base": base["lp_g"],
                     "logit_g1": o["logit_g1"], "logit_w1": o["logits0"][w1].item(), "w1": w1,
                     "lp_donor_g1": o["lp_donor_g1"], "argmax1": o["argmax1"]})
    return recs


def run(eng, pairs, items, trials, gold, beh, out_dir, layout, grid=None, log=print, deadline=None):
    t0 = time.time()
    grid = grid or list(range(eng.L))
    byid = {it["id"]: it for it in items}
    lp_plain = beh[beh.condition == "plain"].set_index("item_id")["lp_g"].to_dict()
    lp_beh = beh.set_index(["item_id", "condition"])["lp_g"].to_dict()
    rows, skipped = [], []
    for k, pair in enumerate(pairs):
        if deadline and time.time() > deadline:
            log(f"[{eng.key}] patch deadline reached after {k} pairs"); break
        recs = run_pair(eng, pair, byid, trials, gold, grid)
        base = recs[0]["lp_g"]
        if abs(base - lp_beh[(pair["item_id"], pair["condition"])]) > 0.05:
            skipped.append((pair["pair_id"], base, lp_beh[(pair["item_id"], pair["condition"])]))
            continue
        it = byid[pair["item_id"]]
        cond = pair["condition"]
        for r in recs:
            r.update({"model": eng.key, "pair_id": pair["pair_id"], "item_id": pair["item_id"], "donor_id": pair["donor_id"],
                      "family": it["family"], "template_id": it["template_id"], "form": it["form"], "split": pair["split"],
                      "condition": cond, "wrapper": cond.rsplit("_", 1)[0], "dose": cond.rsplit("_", 1)[1],
                      "lp_g_plain": lp_plain[pair["item_id"]], "layout": layout})
            r["R"] = (r["lp_g"] - r["lp_g_base"]) / (r["lp_g_plain"] - r["lp_g_base"])
        rows += recs
        if k % 20 == 0:
            log(f"[{eng.key}] patched {k + 1}/{len(pairs)} pairs ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / f"patch_{eng.key}_{layout}.parquet")
    log(f"[{eng.key}] patching done: {len(pairs) - len(skipped)} pairs kept, {len(skipped)} skipped, {time.time() - t0:.0f}s")
    return df, skipped
