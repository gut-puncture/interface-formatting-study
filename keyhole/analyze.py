"""Every number in the paper, computed on CPU from the run outputs -> results/results.json.

python -m keyhole.analyze --run runs/main --out results
Headline numbers use test-split items; dev-split items choose the patching layer.
Intervals: 95% percentile bootstrap over clusters (the item's unit: subject, operand pair or word).
"""
import argparse
import json
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
    return out


# ------------------------------------------------------------------ patching
def patching(pt, units, L):
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
                r["patching"] = patching(pt, units, L)
                lstar = r["patching"]["lstar"]
        r["geometry"] = geometry(run, key, layout, beh, lstar, L, units)
        res["models"][key] = r
        print(key, "done", flush=True)
    json.dump(res, open(out / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
