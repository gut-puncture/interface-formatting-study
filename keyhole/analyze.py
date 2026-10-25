"""Every number in the paper, computed on CPU from the run outputs -> results/results.json.

python -m keyhole.analyze --run runs/main --out results
Headline numbers use test-split items; dev-split items choose the patching layer.
Intervals: 95% percentile bootstrap over clusters (the item's unit: subject, operand pair or word).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from keyhole import config as C
from keyhole import prompts as P

RNG = np.random.default_rng(C.SEED)


# ------------------------------------------------------------------ bootstrap
class Boot:
    """Cluster bootstrap with fixed resamples, so paired quantities share resamples."""

    def __init__(self, clusters, n=C.BOOTSTRAP):
        self.codes, self.uniq = pd.factorize(pd.Series(clusters))
        k = len(self.uniq)
        self.w = np.zeros((n, k))
        idx = RNG.integers(0, k, size=(n, k))
        for r in range(n):
            np.add.at(self.w[r], idx[r], 1)

    def mean(self, values):
        v = np.asarray(values, float)
        ok = ~np.isnan(v)
        s = np.bincount(self.codes[ok], weights=v[ok], minlength=len(self.uniq))
        c = np.bincount(self.codes[ok], minlength=len(self.uniq)).astype(float)
        bs = (self.w @ s) / np.maximum(self.w @ c, 1e-12)
        return {"mean": float(v[ok].mean()) if ok.any() else None, "lo": float(np.percentile(bs, 2.5)),
                "hi": float(np.percentile(bs, 97.5)), "n": int(ok.sum()), "p_le0": float((bs <= 0).mean())}


# What replaced a lost answer, judged on the first line of the reply (rules fixed after reading Qwen3-4B replies).
_SCOPE = re.compile(r"(provided|given|above|this) (text|data|context|information|content|logs?|messages?|emails?|thread|"
                    r"page|document|search results|results)|does not (contain|include|mention|provide)|"
                    r"not (mentioned|provided|included)|no (information|mention)|unrelated|not related|not a real|"
                    r"fictional|does not exist|not exist|is not a (known|recognized|valid)|not possible|"
                    r"cannot be determined|n.est pas", re.I)
_HIJACK = re.compile(r"\b(logs?|jobs?|errors?|timeouts?|timeouterror|traceback|exceptions?|"
                     r"the (email|thread|messages?|posts?|search|results|page|board|tool))\b", re.I)


def lost_type(text):
    """'task' = replies about the wrapper's content; 'scope' = says the text holds no answer; 'wrong' = another answer."""
    s = text.strip().split("\n")[0][:300]
    if _SCOPE.search(s):
        return "scope"
    if _HIJACK.search(s):
        return "task"
    return "wrong"


def winsor(x):
    return np.clip(np.asarray(x, float), -1.0, 2.0)


# ------------------------------------------------------------------ behaviour
def behaviour(beh, units):
    plain = beh[beh.condition == "plain"]
    known = plain[plain.known == True]
    out = {"n_items": int(len(plain)), "n_plain_correct": int(plain.correct.sum()), "n_known": int(len(known)),
           "n_known_dev": int((known.split == "dev").sum()), "n_known_test": int((known.split == "test").sum()),
           "known_by_family": known.groupby("family").size().to_dict()}
    test_ids = set(known[known.split == "test"].item_id)
    d = beh[beh.item_id.isin(test_ids) & (beh.condition != "plain")].copy()
    d["unit"] = d.item_id.map(units)
    conds = {}
    for cond, g in d.groupby("condition"):
        b = Boot(g.unit)
        lost = ~g.retained
        cls = g[lost].fclass.value_counts(normalize=False).to_dict()
        conds[cond] = {"retained": b.mean(g.retained.astype(float)), "correct": b.mean(g.correct.astype(float)),
                       "lost": b.mean(lost.astype(float)), "tokens": float(g.n_prompt_tok.mean()),
                       "n_lost": int(lost.sum()), "lost_classes": {k: int(v) for k, v in cls.items()},
                       "verbose": int((~g.correct & g.retained).sum())}
    out["conditions"] = conds
    # noise floor: both null wrappers pooled, paired by item
    nul = d[d.condition.isin(C.NULLS)]
    out["floor"] = Boot(nul.unit).mean(nul.retained.astype(float))
    # per task family x condition (retention), test items
    fam = {}
    for (f, cond), g in d.groupby(["family", "condition"]):
        fam.setdefault(f, {})[cond] = float(g.retained.mean())
    out["by_task_family"] = fam
    # what replaced the lost answers, per wrapper family (all doses pooled), test items
    lost = d[~d.retained & d.condition.isin(C.WRAPPED)].copy()
    lost["type"] = lost.gen_text.map(lost_type)
    lost["wrapper"] = lost.condition.str.rsplit("_", n=1).str[0]
    out["lost_types"] = {f: {t: int(n) for t, n in g.type.value_counts().items()} for f, g in lost.groupby("wrapper")}
    out["n_trials_by_wrapper"] = {f: int(d.condition.str.startswith(f + "_").sum()) for f in C.FAMILIES}
    # kept answers by how sure the model is of the answer in the plain prompt (test items, all doses)
    pp = np.exp(known.set_index("item_id").lp_g.astype(float))
    w = d[d.condition.isin(C.WRAPPED)].copy()
    w["p_plain"] = w.item_id.map(pp)
    w["bin"] = np.digitize(w.p_plain, CONF_EDGES[1:-1])
    w["wrapper"] = w.condition.str.rsplit("_", n=1).str[0]
    conf = {}
    for f, g in w.groupby("wrapper"):
        conf[f] = []
        for k in range(len(CONF_EDGES) - 1):
            gk = g[g.bin == k]
            conf[f].append({"bin": k, **Boot(gk.unit, n=2000).mean(gk.retained.astype(float))} if len(gk) else None)
    out["by_confidence"] = conf
    out["conf_bin_n"] = [int((w[w.wrapper == "json"].bin == k).sum()) for k in range(len(CONF_EDGES) - 1)]
    return out


# bins of the plain-prompt probability of the answer; known items have p >= 0.5
CONF_EDGES = [0.5, 0.8, 0.95, 0.99, 0.999, 1.0001]


# ------------------------------------------------------------------ patching
def patching(pt, units, L, beh):
    pt = pt.copy()
    pt["R"] = winsor(pt.R)
    pt["unit"] = pt.item_id.map(units)
    res = {}
    for split in ("dev", "test"):
        s = pt[pt.split == split]
        res[f"n_pairs_{split}"] = int(s.pair_id.nunique())
    grid = sorted(pt[pt.site == "A"].layer.unique())

    def wide(s, site):
        return s[s.site == site].pivot_table(index="pair_id", columns="layer", values="R")

    dev, test = pt[pt.split == "dev"], pt[pt.split == "test"]
    Ad, Dd = wide(dev, "A"), wide(dev, "D")
    diff_dev = (Ad - Dd).mean()
    lstar = int(diff_dev.idxmax())
    res["lstar"], res["depth_star"] = lstar, (lstar + 1) / L
    res["dev_diff_curve"] = {int(k): float(v) for k, v in diff_dev.items()}

    pairs_t = test[test.site == "BASE"][["pair_id", "unit", "family", "wrapper", "dose", "form"]].drop_duplicates("pair_id").set_index("pair_id")
    boot = Boot(pairs_t.unit.values)
    curves = {}
    for site in ("A", "D", "B"):
        w = wide(test, site).reindex(pairs_t.index)
        curves[site] = [{"layer": int(l), "depth": (l + 1) / L, **boot.mean(w[l].values)} for l in grid]
    res["curves"] = curves
    At, Dt = wide(test, "A").reindex(pairs_t.index), wide(test, "D").reindex(pairs_t.index)
    h1 = boot.mean((At[lstar] - Dt[lstar]).values)
    h1["pass"] = bool(h1["lo"] > 0 and h1["mean"] >= C.H1_MIN)
    res["H1"] = h1
    res["A_star"] = boot.mean(At[lstar].values)
    res["D_star"] = boot.mean(Dt[lstar].values)
    af = test[test.site == "AFULL"].set_index("pair_id").R.reindex(pairs_t.index)
    res["AFULL"] = boot.mean(af.values)
    # sublayer bands
    sub = {}
    for b in range(3):
        for v in ("attn", "mlp", "both"):
            x = test[(test.site == "SUB") & (test.band == b) & (test["sub"] == v)].set_index("pair_id").R.reindex(pairs_t.index)
            sub[f"{b}_{v}"] = boot.mean(x.values)
    bstar = (3 * lstar) // L
    res["sub"], res["bstar"] = sub, bstar
    xa = test[(test.site == "SUB") & (test.band == bstar)].pivot_table(index="pair_id", columns="sub", values="R").reindex(pairs_t.index)
    res["H2"] = {"attn": sub[f"{bstar}_attn"], "mlp": sub[f"{bstar}_mlp"],
                 "both_minus_max": boot.mean((xa["both"] - np.maximum(xa["attn"], xa["mlp"])).values)}
    # generality: A-D at l* by task family, wrapper, dose, form (test)
    gen = {}
    for col in ("family", "wrapper", "dose", "form"):
        gen[col] = {}
        for val, idx in pairs_t.groupby(col).groups.items():
            sub_b = Boot(pairs_t.loc[idx, "unit"].values, n=2000)
            gen[col][val] = sub_b.mean((At.loc[idx, lstar] - Dt.loc[idx, lstar]).values)
    res["by"] = gen
    # leave-one-family-out: choose l* on dev pairs of the other families
    lofo = {}
    for f in sorted(pairs_t.family.unique()):
        dv = dev[dev.family != f]
        dd = (wide(dv, "A") - wide(dv, "D")).mean()
        lf = int(dd.idxmax())
        idx = pairs_t.index[pairs_t.family == f]
        lofo[f] = {"layer": lf, **Boot(pairs_t.loc[idx, "unit"].values, n=2000).mean((At.loc[idx, lf] - Dt.loc[idx, lf]).values)}
    res["lofo"] = lofo
    # does the donor push its own answer? log p(donor's first token) under D vs base, at l*
    dl = test[(test.site == "D") & (test.layer == lstar)].set_index("pair_id").lp_donor_g1.reindex(pairs_t.index)
    bl = test[test.site == "BASE"].set_index("pair_id").lp_donor_g1.reindex(pairs_t.index)
    res["donor_g1_shift"] = boot.mean((dl - bl).values)
    # the same patches judged by the model's most likely next token: is it the first token of the target answer?
    plain = beh[(beh.condition == "plain") & (beh.known == True)]
    g1 = dict(zip(plain.item_id, [int(x[0]) for x in plain.g_ids]))
    t = test.copy()
    t["own1"] = (t.argmax1 == t.item_id.map(g1)).astype(float)
    t["donor1"] = (t.argmax1 == t.donor_id.map(g1)).astype(float)
    top1 = {}
    for name, site, col in [("A_own", "A", "own1"), ("D_donor", "D", "donor1"), ("D_own", "D", "own1"),
                            ("B_own", "B", "own1")]:
        w = t[t.site == site].pivot_table(index="pair_id", columns="layer", values=col).reindex(pairs_t.index)
        top1[name] = [{"layer": int(l), "depth": (l + 1) / L, **boot.mean(w[l].values)} for l in grid]
    for name, site in [("BASE_own", "BASE"), ("AFULL_own", "AFULL")]:
        top1[name] = boot.mean(t[t.site == site].set_index("pair_id").own1.reindex(pairs_t.index).values)
    # how often the donor's answer survives when the donor itself is asked inside the same wrapper
    ret = beh.set_index(["item_id", "condition"]).retained.to_dict()
    pairs_cond = test[test.site == "BASE"].drop_duplicates("pair_id").set_index("pair_id")
    dk = [float(ret.get((d, c), np.nan)) for d, c in zip(pairs_cond.donor_id.reindex(pairs_t.index),
                                                         pairs_cond.condition.reindex(pairs_t.index))]
    top1["donor_kept_in_wrapper"] = boot.mean(dk)
    # same measure as the patches: in the other question's own wrapped prompt, is its answer's first token on top?
    am = beh.set_index(["item_id", "condition"]).argmax1.to_dict()
    dt = [float(am.get((d, c), -1) == g1.get(d)) if (d, c) in am else np.nan
          for d, c in zip(pairs_cond.donor_id.reindex(pairs_t.index), pairs_cond.condition.reindex(pairs_t.index))]
    top1["donor_top1_in_wrapper"] = boot.mean(dt)
    # the same curves for log wrappers and for the other wrappers separately
    grp = {}
    for name, sel in [("log", pairs_t.wrapper == "log"), ("other", pairs_t.wrapper != "log")]:
        idx = pairs_t.index[sel]
        if len(idx) < 10:
            continue
        b2 = Boot(pairs_t.loc[idx, "unit"].values, n=2000)
        g = {}
        for nm, site, col in [("A_own", "A", "own1"), ("D_donor", "D", "donor1"), ("B_own", "B", "own1")]:
            w = t[t.site == site].pivot_table(index="pair_id", columns="layer", values=col).reindex(idx)
            g[nm] = [{"layer": int(l), "depth": (l + 1) / L, **b2.mean(w[l].values)} for l in grid]
        g["n"] = int(len(idx))
        grp[name] = g
    top1["by_group"] = grp
    res["top1"] = top1
    # where the question's information reaches the last position: first depth at which restoring the
    # last position gives back at least half of the lost answer (R, test pairs)
    bm = [r["mean"] for r in curves["B"]]
    res["handoff_depth"] = next((curves["B"][i]["depth"] for i, v in enumerate(bm) if v >= 0.5), None)
    return res


# ------------------------------------------------------------------ geometry
def geometry(run, key, layout, beh, lstar, L, units):
    st = np.load(run / f"states_{key}_{layout}.npy", mmap_mode="r")
    idx = pd.read_parquet(run / f"states_index_{key}_{layout}.parquet")
    meta = json.load(open(run / f"behavior_meta_{key}_{layout}.json"))
    grid = meta["grid"]
    X = np.asarray(st, dtype=np.float32)
    X = X - X.mean(0, keepdims=True)                 # centre per layer over all saved rows
    pos = {(r.item_id, r.condition): i for i, r in enumerate(idx.itertuples())}
    ret = beh.set_index(["item_id", "condition"]).retained.to_dict()
    split = beh[beh.condition == "plain"].set_index("item_id").split.to_dict()
    rows = []
    for (iid, cond), i in pos.items():
        if cond.endswith("_L") and cond.split("_")[0] in C.STRUCTURED and split.get(iid) == "test" and (iid, "plain") in pos:
            a, b = X[i], X[pos[(iid, "plain")]]
            cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
            rows.append((iid, cond, bool(ret[(iid, cond)]), cos))
    if not rows:
        return {}
    cos = np.stack([r[3] for r in rows])
    ok = np.array([r[2] for r in rows])
    unit = [units[r[0]] for r in rows]
    out = {"grid": grid, "depth": [(l + 1) / L for l in grid], "n_success": int(ok.sum()), "n_failure": int((~ok).sum())}
    out["success"] = [float(cos[ok, j].mean()) for j in range(len(grid))]
    out["failure"] = [float(cos[~ok, j].mean()) for j in range(len(grid))] if (~ok).any() else []
    j = int(np.argmin([abs(l - lstar) for l in grid])) if lstar is not None else None
    if j is not None and (~ok).any():
        bs, bf = Boot(np.array(unit)[ok], n=4000), Boot(np.array(unit)[~ok], n=4000)
        s, f = bs.mean(cos[ok, j]), bf.mean(cos[~ok, j])
        out["H3"] = {"layer": grid[j], "success": s, "failure": f, "diff": s["mean"] - f["mean"],
                     "pass": bool(s["lo"] > f["hi"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = json.load(open(run / "manifest.json"))
    layout = man.get("layout", "tb64")
    items = P.load_items()
    units = {it["id"]: f'{it["family"]}|{it["unit"]}' for it in items}
    res = {"layout": layout, "gate": man.get("gate"), "guard": man.get("guard"), "gpu": man.get("gpu"),
           "hours": man.get("hours"), "models": {}}
    behs = {}
    for key in C.MODEL_ORDER:
        bp = run / f"behavior_{key}_{layout}.parquet"
        if not bp.exists():
            continue
        beh = pd.read_parquet(bp)
        L = man["models"][key]["L"]
        r = {"name": C.MODELS[key]["name"], "L": L, "d": man["models"][key]["d"], "behaviour": behaviour(beh, units)}
        pp = run / f"patch_{key}_{layout}.parquet"
        lstar = None
        if pp.exists():
            pt = pd.read_parquet(pp)
            if len(pt):
                r["patching"] = patching(pt, units, L, beh)
                lstar = r["patching"]["lstar"]
        r["geometry"] = geometry(run, key, layout, beh, lstar, L, units)
        res["models"][key] = r
        behs[key] = beh
        print(key, "done", flush=True)
    res["pooled"] = pooled(behs, units)
    # the first model was also run with the question closer to the end (64 tokens of wrapper after it)
    alt = run / f"behavior_{C.MODEL_ORDER[0]}_tb64.parquet"
    if layout != "tb64" and alt.exists():
        res["q4_tb64"] = behaviour(pd.read_parquet(alt), units)
    json.dump(res, open(out / "results.json", "w"), indent=1)


def pooled(behs, units):
    """All models together: test items, clusters = model x unit."""
    rows = []
    for key, beh in behs.items():
        known = beh[(beh.condition == "plain") & (beh.known == True) & (beh.split == "test")]
        pp = np.exp(known.set_index("item_id").lp_g.astype(float))
        d = beh[beh.item_id.isin(set(known.item_id)) & beh.condition.isin(C.WRAPPED)].copy()
        d["p_plain"] = d.item_id.map(pp)
        d["unit"] = key + "|" + d.item_id.map(units)
        d["model"] = key
        rows.append(d)
    if not rows:
        return {}
    d = pd.concat(rows, ignore_index=True)
    d["wrapper"] = d.condition.str.rsplit("_", n=1).str[0]
    d["bin"] = np.digitize(d.p_plain, CONF_EDGES[1:-1])
    out = {"n_trials": int(len(d)), "kept": {}, "by_confidence": {}, "lost_types": {}, "lost": {}}
    for f, g in d.groupby("wrapper"):
        out["kept"][f] = Boot(g.unit, n=2000).mean(g.retained.astype(float))
        out["by_confidence"][f] = []
        for k in range(len(CONF_EDGES) - 1):
            gk = g[g.bin == k]
            out["by_confidence"][f].append({"bin": k, **Boot(gk.unit, n=2000).mean(gk.retained.astype(float))})
        lost = g[~g.retained]
        types = lost.gen_text.map(lost_type).value_counts()
        out["lost_types"][f] = {t: int(types.get(t, 0)) for t in ("task", "scope", "wrong")}
        out["lost"][f] = int(len(lost))
    out["conf_bin_n"] = [int((d[d.wrapper == "json"].bin == k).sum()) for k in range(len(CONF_EDGES) - 1)]
    # what happens to each wrapped prompt, by how sure the model is of the answer (all wrappers together)
    d["type"] = np.where(d.retained, "kept", d.gen_text.map(lost_type))
    out["type_by_confidence"] = {}
    units_, bins_ = d.unit.to_numpy(), d.bin.to_numpy()
    for t in ("kept", "task", "scope", "wrong"):
        x = (d.type == t).to_numpy(dtype=float)
        out["type_by_confidence"][t] = [{"bin": k, **Boot(units_[bins_ == k], n=2000).mean(x[bins_ == k])}
                                        for k in range(len(CONF_EDGES) - 1)]
    out["type_bin_n"] = [int((d.bin == k).sum()) for k in range(len(CONF_EDGES) - 1)]
    # the same, separately for log wrappers and for the other wrappers
    out["type_by_confidence_group"] = {}
    for name, sel in [("log", d.wrapper == "log"), ("other", d.wrapper != "log")]:
        sel = sel.to_numpy()
        gg = {}
        for t in ("kept", "task", "scope", "wrong"):
            x = (d.type == t).to_numpy(dtype=float)
            gg[t] = [{"bin": k, **Boot(units_[sel & (bins_ == k)], n=2000).mean(x[sel & (bins_ == k)])}
                     for k in range(len(CONF_EDGES) - 1)]
        gg["n"] = [int((sel & (bins_ == k)).sum()) for k in range(len(CONF_EDGES) - 1)]
        out["type_by_confidence_group"][name] = gg
    # noise floor by the same bins: share lost with one neutral sentence
    nrows = []
    for key, beh in behs.items():
        known = beh[(beh.condition == "plain") & (beh.known == True) & (beh.split == "test")]
        pp = np.exp(known.set_index("item_id").lp_g.astype(float))
        z = beh[beh.item_id.isin(set(known.item_id)) & beh.condition.isin(C.NULLS)].copy()
        z["bin"] = np.digitize(z.item_id.map(pp), CONF_EDGES[1:-1])
        z["unit"] = key + "|" + z.item_id.map(units)
        nrows.append(z)
    z = pd.concat(nrows, ignore_index=True)
    out["null_lost_by_confidence"] = [{"bin": k, **Boot(z.unit[z.bin == k].to_numpy(), n=2000).mean(
        (~z.retained[z.bin == k]).to_numpy(dtype=float))} for k in range(len(CONF_EDGES) - 1)]
    out["n_known_test"] = {key: int(((beh.condition == "plain") & (beh.known == True) & (beh.split == "test")).sum())
                           for key, beh in behs.items()}
    return out


if __name__ == "__main__":
    main()
