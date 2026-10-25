"""Behaviour pass for one model (FINAL_SPEC section 7).

1. plain prompt for every item: greedy answer -> gold tokens g, known items;
2. every other condition for known items: teacher-forced log-prob of g, plus greedy
   generation when the teacher-forced answer is not reproduced;
3. residual states at the last question token for the geometry analysis.
"""
import json
import math
import time

import numpy as np
import pandas as pd
import torch

from keyhole import config as C
from keyhole import prompts as P
from keyhole import scoring as S
from keyhole.batching import iter_groups, max_rows
from keyhole.engine import Plan


def state_grid(L):
    return sorted(set(range(0, L, 2)) | {L - 1})


# ------------------------------------------------------------------ primitives
def teacher_force(eng, jobs, grid=None, log=None):
    """jobs: list of (trial, g). Returns dict id(trial) -> result; states if grid given."""
    res, states = {}, {}
    stop = eng.stop_ids()
    gmap = {id(t): g for t, g in jobs}
    for cache, group in iter_groups(eng, [t for t, _ in jobs]):
        group = sorted(group, key=lambda t: len(t.suffix_ids) + len(gmap[id(t)]))
        i = 0
        while i < len(group):
            B = max_rows(eng, cache, positions=24)
            batch = group[i:i + B]
            # keep the union of needed positions small: split if it grows past 48
            while len(batch) > 1:
                lo = len(batch[0].suffix_ids) - 1
                hi = len(batch[-1].suffix_ids) - 1 + len(gmap[id(batch[-1])])
                if hi - lo + 1 <= 48:
                    break
                batch = batch[:len(batch) // 2]
            rows = [t.suffix_ids + gmap[id(t)] for t in batch]
            need = sorted({len(t.suffix_ids) - 1 + j for t in batch for j in range(len(gmap[id(t)]) + 1)})
            pos_of = {p: k for k, p in enumerate(need)}
            plan = None
            if grid is not None:
                rws = torch.arange(len(batch), device=eng.device)
                qe = eng.t([t.q_end for t in batch])
                plan = Plan(capture={("r", l): (rws, qe) for l in grid})
            logits, plan = eng.forward(cache, rows, need, plan)
            for k, t in enumerate(batch):
                g = gmap[id(t)]
                n = len(t.suffix_ids)
                idx = [pos_of[n - 1 + j] for j in range(len(g) + 1)]
                lg = logits[k, idx]
                lp = torch.log_softmax(lg, -1)
                toks = [lp[j, g[j]].item() for j in range(len(g))]
                am = lg.argmax(-1).tolist()
                nxt = am[len(g)]
                nxt_txt = eng.tok.decode([nxt])
                tf_ok = am[:len(g)] == list(g) and (nxt in stop or not nxt_txt[:1].isalnum())
                res[id(t)] = {"lp_g": float(sum(toks)), "lp_g_tokens": toks, "tf_correct": bool(tf_ok),
                              "logit_g1": lg[0, g[0]].item(), "argmax1": am[0], "logit_argmax1": lg[0, am[0]].item()}
                if grid is not None:
                    states[id(t)] = torch.stack([plan.got[("r", l)][k] for l in grid]).to(torch.float16).cpu().numpy()
            i += len(batch)
    return res, states


def greedy(eng, trials):
    """Greedy continuation (<= MAX_NEW_TOKENS, stop at EOS/newline) for each trial."""
    out = {}
    for cache, group in iter_groups(eng, trials):
        by_len = {}
        for t in group:
            by_len.setdefault(len(t.suffix_ids), []).append(t)
        B = max_rows(eng, cache, positions=1, cap=32)
        for n, ts in by_len.items():
            for i in range(0, len(ts), B):
                chunk = ts[i:i + B]
                gens = eng.generate(cache, [t.suffix_ids for t in chunk])
                for t, g in zip(chunk, gens):
                    out[id(t)] = g
    return out


def gold_tokens(eng, gen_ids, aliases):
    """Shortest prefix of the greedy ids whose text is a correct answer, else None."""
    for k in range(1, len(gen_ids) + 1):
        if S.is_correct(eng.tok.decode(gen_ids[:k], skip_special_tokens=True), aliases):
            return list(gen_ids[:k])
    return None


# ------------------------------------------------------------------ the pass
def run(eng, items, layout, out_dir, log=print, conds=None):
    conds = conds or C.CONDITIONS[1:]
    t0 = time.time()
    pool, assign = P.load_fillers(layout)
    tok = eng.tok
    grid = state_grid(eng.L)
    rows, dropped = [], []
    byid = {it["id"]: it for it in items}

    plain = {}
    for it in items:
        try:
            plain[it["id"]] = P.make_trial(tok, it, "plain", pool, assign)
        except P.AlignError as e:
            dropped.append((it["id"], "plain", str(e)))
    gens = greedy(eng, list(plain.values()))
    gold = {}
    for iid, tr in plain.items():
        g = gold_tokens(eng, gens[id(tr)], byid[iid]["aliases"])
        if g is not None:
            gold[iid] = g
    tf_plain, st_plain = teacher_force(eng, [(plain[i], g) for i, g in gold.items()], grid)
    known = set()
    for iid, tr in plain.items():
        it, r = byid[iid], tf_plain.get(id(plain[iid]))
        text = tok.decode(gens[id(tr)], skip_special_tokens=True)
        g = gold.get(iid)
        is_known = g is not None and r is not None and math.exp(r["lp_g"]) >= C.KNOWN_P and len(g) <= C.MAX_G_TOKENS
        if is_known:
            known.add(iid)
        rows.append(_row(eng, it, "plain", tr, -1, g, r, gens[id(tr)], text, g is not None, None, is_known, ""))
    log(f"[{eng.key}] plain: {len(gold)} correct, {len(known)} known of {len(items)} ({time.time() - t0:.0f}s)")

    states = {(i, "plain"): st_plain[id(plain[i])] for i in known}
    for cond in conds:
        tc = time.time()
        trials = []
        for iid in sorted(known):
            it = byid[iid]
            try:
                tr = P.make_trial(tok, it, cond, pool, assign)
                P.check_against_plain(tr, plain[iid])
                trials.append(tr)
            except P.AlignError as e:
                dropped.append((iid, cond, str(e)))
        keep_states = cond in C.NULLS or cond.endswith("_L")
        tf, st = teacher_force(eng, [(t, gold[t.item_id]) for t in trials], grid if keep_states else None)
        need_gen = [t for t in trials if not tf[id(t)]["tf_correct"]]
        gn = greedy(eng, need_gen)
        for t in trials:
            it, r = byid[t.item_id], tf[id(t)]
            gids = gn.get(id(t), [])
            text = tok.decode(gids, skip_special_tokens=True) if gids else ""
            ok = r["tf_correct"] or S.is_correct(text, it["aliases"])
            fc = None if ok else S.failure_class(text, it["aliases"], it["family"], t.wrapper_text)
            rows.append(_row(eng, it, cond, t, P.filler_id(it, cond, assign), gold[t.item_id], r, gids, text, ok, fc, None, layout))
            if keep_states:
                states[(t.item_id, cond)] = st[id(t)]
        acc = np.mean([r["correct"] for r in rows[-len(trials):]]) if trials else float("nan")
        log(f"[{eng.key}] {cond}: n={len(trials)} acc={acc:.3f} gen={len(need_gen)} ({time.time() - tc:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / f"behavior_{eng.key}_{layout}.parquet")
    keys = sorted(states)
    np.save(out_dir / f"states_{eng.key}_{layout}.npy", np.stack([states[k] for k in keys]))
    pd.DataFrame({"item_id": [k[0] for k in keys], "condition": [k[1] for k in keys]}).to_parquet(
        out_dir / f"states_index_{eng.key}_{layout}.parquet")
    json.dump({"grid": grid, "dropped": dropped, "seconds": time.time() - t0},
              open(out_dir / f"behavior_meta_{eng.key}_{layout}.json", "w"))
    log(f"[{eng.key}] behaviour done in {time.time() - t0:.0f}s, dropped {len(dropped)}")
    return df, gold, plain


def _row(eng, it, cond, tr, fid, g, r, gids, text, correct, fclass, known, layout):
    fam, dose = (cond.rsplit("_", 1) if "_" in cond and cond not in C.NULLS else (cond, ""))
    return {"model": eng.key, "item_id": it["id"], "family": it["family"], "template_id": it["template_id"],
            "form": it["form"], "split": it["split"], "condition": cond, "wrapper": fam, "dose": dose,
            "filler_id": fid, "layout": layout, "n_prompt_tok": tr.n_prompt, "q2end_tok": len(tr.suffix_ids),
            "span_len": tr.span_len,
            "embedded_count": sum(P.K.embedded_count(a, tr.wrapper_text.lower()) for a in it["aliases"]) if cond != "plain" else 0,
            "g_ids": list(g) if g else [], "lp_g": r["lp_g"] if r else float("nan"),
            "lp_g_tokens": r["lp_g_tokens"] if r else [], "tf_correct": bool(r["tf_correct"]) if r else False,
            "gen_ids": list(gids), "gen_text": text, "correct": bool(correct), "fclass": fclass,
            "retained": bool(correct or (bool(text) and S.answer_present(text, it["aliases"]))),
            "logit_g1": r["logit_g1"] if r else float("nan"), "argmax1": r["argmax1"] if r else -1,
            "logit_argmax1": r["logit_argmax1"] if r else float("nan"), "known": known}
